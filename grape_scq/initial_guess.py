from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .result import Pulse
from .bases import fourier_basis, symmetric_fourier_basis, antisymmetric_fourier_basis

"""Initial-pulse-guess generators. Each takes the OptimizationProblem to
initialize a guess for and returns a Pulse (already fit to the problem's
analytic basis, if any).

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
"""


def zero(problem) -> Pulse:
    """The all-zero pulse."""
    return _fit(problem, np.zeros(problem.n_steps), np.zeros(problem.n_steps))


def constant(problem, theta: float = np.pi) -> Pulse:
    """A constant in-phase pulse implementing a rotation of angle `theta`
    (uy = 0)."""
    ux = theta * np.ones(problem.n_steps) / (2*np.pi*problem.reference_amplitude*problem.duration)
    uy = np.zeros(problem.n_steps)
    return _fit(problem, ux, uy)


def random_fourier(problem, n_coeffs: int = 3, rng: np.random.Generator | None = None) -> Pulse:
    """A random low-order Fourier series for both ux and uy."""
    rng = rng or np.random.default_rng()
    f = fourier_basis(problem.n_steps, n_coeffs)
    ux = f.T @ (2*rng.random(n_coeffs) - 1) / n_coeffs
    uy = f.T @ (2*rng.random(n_coeffs) - 1) / n_coeffs
    return _fit(problem, ux, uy)


def random_symmetric_antisymmetric(problem, n_coeffs: int = 3, rng: np.random.Generator | None = None) -> Pulse:
    """A random low-order Fourier series: ux symmetric, uy antisymmetric."""
    rng = rng or np.random.default_rng()
    fx = symmetric_fourier_basis(problem.n_steps, n_coeffs)
    fy = antisymmetric_fourier_basis(problem.n_steps, n_coeffs)
    ux = fx.T @ (2*rng.random(n_coeffs) - 1) / n_coeffs
    uy = fy.T @ (2*rng.random(n_coeffs) - 1) / n_coeffs
    return _fit(problem, ux, uy)


def cos_drag(problem, theta: float = np.pi) -> Pulse:
    """A DRAG-like pulse ux=A*sin(pi*t/T), uy=B*cos(pi*t/T), with A and B
    fit (via a small internal GRAPE optimization on a 2-level system) to
    realize a leakage-free rotation of angle `theta` about x, given the
    problem's anharmonicity and duration."""
    from .system import Transmon
    from .targets import GateTarget
    from .problem import OptimizationProblem
    from .optimizer import GrapeOptimizer

    n_steps = problem.n_steps
    tau = np.linspace(0, 1, n_steps)
    fx = np.sin(np.pi * tau)[None, :]
    fy = np.cos(np.pi * tau)[None, :]

    local_system = Transmon(
        n_levels=problem.system.n_levels,
        anharmonicity=float(np.mean(problem.system.anharmonicity)),
    )
    local_target = GateTarget(matrix=expm(-1j*theta*np.array([[0, 1], [1, 0]])/2), subspace=[0, 1])
    local_problem = OptimizationProblem(
        local_system, local_target, n_steps=n_steps, dt=problem.dt,
        reference_amplitude=problem.reference_amplitude, ux_basis=fx, uy_basis=fy,
    )

    a0 = 2*theta / (local_problem.reference_amplitude * local_problem.duration * 2*np.pi)
    b0 = a0 / 10
    result = GrapeOptimizer(display=False, max_iter=500).optimize(local_problem, Pulse([a0], [b0]))

    ux, uy = local_problem.waveform(result.pulse)
    return _fit(problem, ux, uy)


def _fit(problem, ux: np.ndarray, uy: np.ndarray) -> Pulse:
    """Fit a direct per-timestep waveform to `problem`'s analytic basis (if
    any); otherwise return it unchanged."""
    ax = _fit_coefficients(ux, problem.ux_basis) if problem.is_shaped_x else ux
    ay = _fit_coefficients(uy, problem.uy_basis) if problem.is_shaped_y else uy
    return Pulse(ax, ay)


def _fit_coefficients(target_waveform: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Least-squares coefficients c minimizing ||basis.T @ c - target_waveform||^2."""
    def cost_and_grad(c):
        f = basis.T @ c
        residual = f - target_waveform
        J = np.sum(residual**2) / target_waveform.size
        G = 2 * basis @ residual / target_waveform.size
        return J, G

    c0 = 1e-3 * np.ones(basis.shape[0])
    res = minimize(cost_and_grad, c0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 500, "gtol": 1e-12, "ftol": 1e-12, "maxfun": 1e12})
    return res.x
