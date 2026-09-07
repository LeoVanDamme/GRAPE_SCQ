import numpy as np
import pytest

from grape_scq import Transmon, GateTarget, StateTarget, OptimizationProblem


def test_gate_target_rejects_non_square():
    with pytest.raises(ValueError):
        GateTarget(matrix=np.ones((2, 3)))


def test_gate_target_subspace_size_must_match():
    with pytest.raises(ValueError):
        GateTarget(matrix=np.eye(2), subspace=[0, 1, 2])


def test_multi_gate_target_leading_axis_convention():
    I2, X, Y = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
    gates = np.stack([I2, X, Y])  # (3, 2, 2): natural leading-axis stacking

    target = GateTarget(matrix=gates, subspace=[0, 1])
    system = Transmon(frequency=[4.9e9, 5.0e9, 5.1e9])
    problem = OptimizationProblem(system, target, n_steps=5, dt=0.1e-9)

    for k, gate in enumerate(gates):
        assert np.allclose(problem.Tardagg[:, :, k], gate.conj().T)


def test_multi_gate_target_wrong_ensemble_size_raises():
    gates = np.stack([np.eye(2), np.eye(2)])  # 2 gates
    target = GateTarget(matrix=gates, subspace=[0, 1])
    system = Transmon(frequency=[4.9e9, 5.0e9, 5.1e9])  # 3-member ensemble
    with pytest.raises(ValueError):
        OptimizationProblem(system, target, n_steps=5, dt=0.1e-9)


def test_multi_state_target_leading_axis_convention():
    states = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
    target = StateTarget(initial=np.array([1, 0, 0], dtype=complex), target=states)
    system = Transmon(n_levels=3, frequency=[4.9e9, 5.0e9, 5.1e9])
    problem = OptimizationProblem(system, target, n_steps=5, dt=0.1e-9)

    for k, state in enumerate(states):
        assert np.allclose(problem.Tardagg[0, :, k], state.conj())


def test_state_target_level_mismatch_raises():
    target = StateTarget(initial=[1, 0], target=[0, 1])
    system = Transmon(n_levels=3)
    with pytest.raises(ValueError):
        OptimizationProblem(system, target, n_steps=5, dt=0.1e-9)


def test_analytic_basis_shapes():
    system = Transmon()
    target = GateTarget(matrix=np.eye(2), subspace=[0, 1])
    basis = np.random.randn(3, 20)
    problem = OptimizationProblem(system, target, n_steps=20, dt=0.1e-9, ux_basis=basis, uy_basis=basis)
    assert problem.n_x == 3
    assert problem.n_y == 3
    assert problem.is_shaped_x and problem.is_shaped_y


def test_basis_wrong_n_steps_raises():
    system = Transmon()
    target = GateTarget(matrix=np.eye(2), subspace=[0, 1])
    basis = np.random.randn(3, 19)  # n_steps=20 expected
    with pytest.raises(ValueError):
        OptimizationProblem(system, target, n_steps=20, dt=0.1e-9, ux_basis=basis)


def test_with_system_rebuilds_hamiltonians():
    system = Transmon(frequency=5e9)
    target = GateTarget(matrix=np.eye(2), subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=5, dt=0.1e-9)
    new_system = system.with_(frequency=5.5e9)
    problem2 = problem.with_system(new_system)
    assert not np.allclose(problem.H0dt, problem2.H0dt)
