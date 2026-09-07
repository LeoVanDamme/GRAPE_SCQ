from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from .targets import GateTarget

"""Post-hoc analysis: forward-simulating a solved pulse to compute
populations/fidelities for plotting. Uses plain scipy matrix exponentials
(not the Numba engine) since these run only a handful of times per notebook,
not once per optimizer iteration.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
"""


def eigenstate(index: int, n_levels: int) -> np.ndarray:
    """The basis state |index> in an n_levels-dimensional Hilbert space."""
    if not (0 <= index < n_levels):
        raise ValueError(f"index must be in [0, {n_levels}), got {index}")
    psi = np.zeros(n_levels, dtype=complex)
    psi[index] = 1.0
    return psi


def propagate_state(problem, pulse, initial_state) -> np.ndarray:
    """Propagate `initial_state` forward under `pulse` for every ensemble
    member. Returns an array of shape (n_levels, n_steps+1, n_ensemble)."""
    ux, uy = problem.waveform(pulse)
    u = ux + 1j * uy

    initial_state = np.asarray(initial_state, dtype=complex)
    if initial_state.ndim == 1:
        initial_state = np.broadcast_to(initial_state, (problem.n_ensemble, initial_state.shape[0])).T
    elif initial_state.ndim == 2 and initial_state.shape[0] != problem.system.n_levels:
        initial_state = initial_state.T

    n_levels = problem.system.n_levels
    psi = np.zeros((n_levels, problem.n_steps + 1, problem.n_ensemble), dtype=complex)
    psi[:, 0, :] = initial_state
    for k in range(problem.n_ensemble):
        for n in range(problem.n_steps):
            drift = problem.H0dt[:, :, problem.itH[n], k]
            drive = u[n] * problem.HDdt[:, :, problem.itH[n], k]
            Hdt = drift + drive + drive.conj().T
            psi[:, n + 1, k] = expm_multiply(-1j * Hdt, psi[:, n, k])
    return psi


def population(problem, pulse, initial_state, measured_state) -> np.ndarray:
    """Probability |<measured_state|psi(t)>|^2 as the system evolves under
    `pulse`, for every ensemble member. Returns shape (n_steps+1, n_ensemble)
    (squeezed to (n_steps+1,) if there's a single ensemble member)."""
    measured_state = np.asarray(measured_state, dtype=complex)
    psi = propagate_state(problem, pulse, initial_state)
    prob = np.zeros((problem.n_steps + 1, problem.n_ensemble))
    for k in range(problem.n_ensemble):
        for n in range(problem.n_steps + 1):
            prob[n, k] = np.abs(psi[:, n, k].conj() @ measured_state) ** 2
    return np.squeeze(prob)


def fidelity_map(problem, pulse) -> np.ndarray:
    """Fidelity for every ensemble member under `pulse`:
      - for a GateTarget: |Tr(U_target^dagger . U_F)|^2 / Nc^2 on the computational subspace
      - for a StateTarget: |<target|psi_F>|^2
    Returns shape (n_ensemble,) (a 0-d/scalar-like array if n_ensemble == 1)."""
    ux, uy = problem.waveform(pulse)
    u = ux + 1j * uy
    n_levels = problem.system.n_levels

    fidelities = np.zeros(problem.n_ensemble)
    for k in range(problem.n_ensemble):
        U = np.eye(n_levels, dtype=complex)
        for n in range(problem.n_steps):
            drift = problem.H0dt[:, :, problem.itH[n], k]
            drive = u[n] * problem.HDdt[:, :, problem.itH[n], k]
            Hdt = drift + drive + drive.conj().T
            U = expm(-1j * Hdt) @ U

        if isinstance(problem.target, GateTarget):
            subspace = problem.CompSpace
            Uc = U[np.ix_(subspace, subspace)]
            trace = np.trace(problem.Tardagg[:, :, k] @ Uc)
            fidelities[k] = (np.abs(trace) / problem.Nc) ** 2
        else:
            psi0 = problem.psi0[:, k]
            psiF = U @ psi0
            overlap = (problem.Tardagg[:, :, k] @ psiF).item()
            fidelities[k] = np.abs(overlap) ** 2

    return np.squeeze(fidelities)
