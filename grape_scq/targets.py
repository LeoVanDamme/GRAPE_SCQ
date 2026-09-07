from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class GateTarget:
    """A target unitary gate on a computational subspace of the full Hilbert
    space. `matrix` is either a single (Nc, Nc) gate (used for every ensemble
    member) or a per-ensemble-member stack of shape (n_ensemble, Nc, Nc)
    (leading axis = ensemble member -- no reshaping tricks required).
    `subspace` lists which of the full-space levels are the Nc computational
    states, e.g. [0, 1] for a qubit embedded in a 3-level transmon.

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    matrix: np.ndarray
    subspace: Sequence[int] = (0, 1)

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=complex)
        self.subspace = np.asarray(self.subspace, dtype=np.int64)

        if self.matrix.ndim == 2:
            if self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError(f"GateTarget.matrix must be square, got shape {self.matrix.shape}")
        elif self.matrix.ndim == 3:
            if self.matrix.shape[1] != self.matrix.shape[2]:
                raise ValueError(
                    "GateTarget.matrix with a per-ensemble-member leading axis must have "
                    f"shape (n_ensemble, Nc, Nc), got {self.matrix.shape}"
                )
        else:
            raise ValueError(
                "GateTarget.matrix must have shape (Nc, Nc) (same gate for every ensemble "
                f"member) or (n_ensemble, Nc, Nc) (one gate per member), got shape {self.matrix.shape}"
            )

        if self.subspace.ndim != 1 or len(self.subspace) != self.n_computational_states:
            raise ValueError(
                f"GateTarget.subspace must list exactly Nc={self.n_computational_states} "
                f"level indices, got {self.subspace}"
            )
        if np.any(self.subspace < 0):
            raise ValueError("GateTarget.subspace indices must be non-negative")

    @property
    def n_computational_states(self) -> int:
        return self.matrix.shape[-1]

    def _per_member_matrix(self, n_ensemble: int) -> np.ndarray:
        """Broadcast to shape (n_ensemble, Nc, Nc)."""
        if self.matrix.ndim == 2:
            return np.broadcast_to(self.matrix, (n_ensemble,) + self.matrix.shape)
        if self.matrix.shape[0] != n_ensemble:
            raise ValueError(
                f"GateTarget provides {self.matrix.shape[0]} per-ensemble-member gates, "
                f"but the problem's ensemble has {n_ensemble} members."
            )
        return self.matrix

    def _dagger(self, n_ensemble: int) -> np.ndarray:
        """(Nc, Nc, n_ensemble) array of the conjugate-transposed target gate,
        as consumed by the optimization engine."""
        per_member = self._per_member_matrix(n_ensemble)
        return np.moveaxis(per_member.conj().transpose(0, 2, 1), 0, 2)


@dataclass
class StateTarget:
    """A target state-to-state transfer in the full n_levels-dimensional
    Hilbert space. `initial`/`target` are each either a single (n_levels,)
    state vector (used for every ensemble member) or a per-ensemble-member
    stack of shape (n_ensemble, n_levels).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    initial: np.ndarray
    target: np.ndarray

    def __post_init__(self):
        self.initial = np.asarray(self.initial, dtype=complex)
        self.target = np.asarray(self.target, dtype=complex)
        for name, arr in (("initial", self.initial), ("target", self.target)):
            if arr.ndim not in (1, 2):
                raise ValueError(
                    f"StateTarget.{name} must have shape (n_levels,) (same state for every "
                    f"ensemble member) or (n_ensemble, n_levels) (one per member), got shape {arr.shape}"
                )
        if self.initial.shape[-1] != self.target.shape[-1]:
            raise ValueError(
                f"StateTarget.initial and .target must both live in the same n_levels-dimensional "
                f"space, got {self.initial.shape[-1]} and {self.target.shape[-1]}"
            )

    @property
    def n_levels(self) -> int:
        return self.target.shape[-1]

    def _per_member(self, arr: np.ndarray, n_ensemble: int) -> np.ndarray:
        if arr.ndim == 1:
            return np.broadcast_to(arr, (n_ensemble, arr.shape[0]))
        if arr.shape[0] != n_ensemble:
            raise ValueError(
                f"StateTarget provides {arr.shape[0]} per-ensemble-member states, "
                f"but the problem's ensemble has {n_ensemble} members."
            )
        return arr

    def _psi0(self, n_ensemble: int) -> np.ndarray:
        """(n_levels, n_ensemble) array of initial states, as consumed by the engine."""
        return self._per_member(self.initial, n_ensemble).T

    def _dagger(self, n_ensemble: int) -> np.ndarray:
        """(1, n_levels, n_ensemble) array of the conjugated target state (bra), as
        consumed by the engine."""
        per_member = self._per_member(self.target, n_ensemble)
        return np.moveaxis(per_member.conj(), 0, 1)[None, :, :]
