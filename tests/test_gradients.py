import numpy as np
import pytest

from grape_scq import Transmon, GateTarget, StateTarget, OptimizationProblem, GrapeOptimizer
from grape_scq.bases import symmetric_fourier_basis, antisymmetric_fourier_basis


def _finite_diff_check(cost_fn, x0, eps=1e-6, rtol=2e-4, atol=1e-6):
    J, g = cost_fn(x0)
    gnum = np.zeros_like(g)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        Jp, _ = cost_fn(xp)
        Jm, _ = cost_fn(xm)
        gnum[i] = (float(np.asarray(Jp).ravel()[0]) - float(np.asarray(Jm).ravel()[0])) / (2 * eps)
    err = np.max(np.abs(g - gnum))
    denom = np.max(np.abs(g)) + np.max(np.abs(gnum)) + atol
    assert err / denom < rtol, f"analytic vs finite-diff mismatch: max abs err={err}"


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_gate_gradient_matches_finite_difference():
    system = Transmon(n_levels=3, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=15, dt=0.1e-9, reference_amplitude=10e6)

    optimizer = GrapeOptimizer()
    cost_fn = optimizer._cost_function(problem, energy_penalty_weight=0.0)
    x0 = 0.3 * np.random.randn(problem.n_x + problem.n_y)
    _finite_diff_check(cost_fn, x0)


def test_state_gradient_matches_finite_difference_multi_member():
    system = Transmon(n_levels=3, anharmonicity=-100e6, frequency=[5e9, 5.01e9, 4.99e9])
    target = StateTarget(initial=[1, 0, 0], target=[0, 1, 0])
    problem = OptimizationProblem(system, target, n_steps=12, dt=0.1e-9, reference_amplitude=10e6)

    optimizer = GrapeOptimizer()
    cost_fn = optimizer._cost_function(problem, energy_penalty_weight=0.0)
    x0 = 0.3 * np.random.randn(problem.n_x + problem.n_y)
    _finite_diff_check(cost_fn, x0)


def test_gate_gradient_with_analytic_pulse_shaping():
    n_steps = 20
    fx = symmetric_fourier_basis(n_steps, 2)
    fy = antisymmetric_fourier_basis(n_steps, 2)
    system = Transmon(n_levels=2, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=n_steps, dt=0.1e-9,
                                   reference_amplitude=10e6, ux_basis=fx, uy_basis=fy)

    optimizer = GrapeOptimizer()
    cost_fn = optimizer._cost_function(problem, energy_penalty_weight=0.0)
    x0 = 0.3 * np.random.randn(problem.n_x + problem.n_y)
    _finite_diff_check(cost_fn, x0)


def test_gate_gradient_with_energy_penalty():
    system = Transmon(n_levels=3, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=15, dt=0.1e-9, reference_amplitude=10e6)

    optimizer = GrapeOptimizer()
    cost_fn = optimizer._cost_function(problem, energy_penalty_weight=0.1)
    x0 = 0.3 * np.random.randn(problem.n_x + problem.n_y)
    _finite_diff_check(cost_fn, x0)
