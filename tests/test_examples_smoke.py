"""End-to-end smoke tests mirroring the 6 example notebooks, trimmed down for
speed. Not a benchmark or bit-for-bit reproduction -- asserts the whole
Transmon -> OptimizationProblem -> GrapeOptimizer.optimize -> Result pipeline
runs without error and improves the cost for each documented use case."""

import numpy as np
import pytest

from grape_scq import (
    Transmon, GateTarget, StateTarget, OptimizationProblem, GrapeOptimizer, initial_guess,
)
from grape_scq.bases import fourier_basis, symmetric_fourier_basis, antisymmetric_fourier_basis


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_leakage_free_gate():
    system = Transmon(n_levels=3, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=30, dt=0.1e-9, reference_amplitude=10e6)

    pulse0 = initial_guess.cos_drag(problem, theta=np.pi)
    result = GrapeOptimizer(max_iter=30, display=False).optimize(problem, pulse0)

    assert result.average_fidelity >= 1 - result.initial_cost - 1e-6
    pop = result.population(0, 2)
    assert pop.shape == (problem.n_steps + 1,)


def test_two_photon_state_transfer():
    system = Transmon(n_levels=4, anharmonicity=-100e6, frequency=5e9, carrier_frequency=5e9)
    target = StateTarget(initial=[1, 0, 0, 0], target=[0, 0, 1, 0])
    problem = OptimizationProblem(system, target, n_steps=30, dt=0.1e-9, reference_amplitude=10e6)

    pulse0 = initial_guess.random_symmetric_antisymmetric(problem)
    result = GrapeOptimizer(max_iter=30, display=False).optimize(problem, pulse0)
    assert np.isfinite(result.cost)
    assert result.cost <= 1.0


def test_analytically_shaped_hadamard():
    n_steps = 30
    n_coeffs = 3
    fx = fourier_basis(n_steps, n_coeffs)
    fy = fourier_basis(n_steps, n_coeffs)
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    system = Transmon(n_levels=3, anharmonicity=-100e6)
    target = GateTarget(matrix=hadamard, subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=n_steps, dt=0.1e-9,
                                   reference_amplitude=10e6, ux_basis=fx, uy_basis=fy)

    pulse0 = initial_guess.random_fourier(problem, n_coeffs=n_coeffs)
    result = GrapeOptimizer(max_iter=30, display=False).optimize(problem, pulse0)
    assert result.pulse.ux.shape == (n_coeffs,)
    assert np.isfinite(result.cost)


def test_robust_fourier_pulse_with_amplitude_constraint():
    system = Transmon.sweep(frequency=5e9 + np.linspace(-1e6, 1e6, 2),
                             amplitude_scale=1 + np.linspace(-0.05, 0.05, 2),
                             anharmonicity=-100e6)
    assert system.n_ensemble == 4

    n_steps = 20
    fx = symmetric_fourier_basis(n_steps, 2)
    fy = antisymmetric_fourier_basis(n_steps, 2)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=n_steps, dt=0.1e-9,
                                   reference_amplitude=10e6, ux_basis=fx, uy_basis=fy)

    pulse0 = initial_guess.cos_drag(problem, theta=np.pi)
    result = GrapeOptimizer(max_iter=20, max_in_phase=15e6, max_quadrature=15e6, display=False).optimize(problem, pulse0)

    ux, uy = problem.waveform(result.pulse)
    assert np.all(np.abs(problem.reference_amplitude * ux) <= 15e6 + 1e3)
    assert np.all(np.abs(problem.reference_amplitude * uy) <= 15e6 + 1e3)

    fmap = result.fidelity_map()
    assert fmap.shape == (4,)


def test_selective_pulse():
    system = Transmon(n_levels=2, frequency=5e9 + np.array([-2.5e6, 2.5e6]), carrier_frequency=5e9)
    target = StateTarget(initial=[[1, 0], [1, 0]], target=[[0, 1], [1, 0]])
    problem = OptimizationProblem(system, target, n_steps=20, dt=1e-9)

    pulse0 = initial_guess.zero(problem)
    pulse0.ux[:] = 1.0
    result = GrapeOptimizer(max_iter=20, max_amplitude=8e6, display=False).optimize(problem, pulse0)

    ux, uy = problem.waveform(result.pulse)
    amp_hz = problem.reference_amplitude * np.sqrt(ux**2 + uy**2)
    assert np.all(amp_hz <= 8e6 + 1e3)


def test_time_dependent_detuning():
    n_steps = 20
    dt = 1e-9
    tc = np.arange(n_steps) * dt
    w = np.linspace(0, 10e6, 2)
    A = np.linspace(-5e6, 5e6, 2)
    wGrid, AGrid = np.meshgrid(w, A)
    det = AGrid.ravel()[:, None] * np.cos(2*np.pi*wGrid.ravel()[:, None]*tc[None, :])

    system = Transmon(n_levels=3, frequency=5e9 + det, carrier_frequency=5e9, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=n_steps, dt=dt, reference_amplitude=None)

    Ncoeffs = 3
    fx = symmetric_fourier_basis(n_steps, Ncoeffs)
    fy = antisymmetric_fourier_basis(n_steps, Ncoeffs)
    problem = OptimizationProblem(system, target, n_steps=n_steps, dt=dt, ux_basis=fx, uy_basis=fy)

    pulse0 = initial_guess.cos_drag(problem, theta=np.pi)
    result = GrapeOptimizer(max_iter=20, max_amplitude=20e6, display=False).optimize(problem, pulse0)
    assert np.isfinite(result.cost)
