from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .system import Transmon
from .targets import GateTarget, StateTarget
from ._engine.dtypes import dtype_float, dtype_complex, dtype_int

Target = Union[GateTarget, StateTarget]


@dataclass
class OptimizationProblem:
    """Everything needed to define a pulse-optimization problem: the physical
    system, the target (gate or state), the time grid, and (optionally) an
    analytic pulse parametrization and per-ensemble-member cost weights.

    Immutable by convention -- to re-evaluate on a different system (e.g. a
    finer robustness grid, as in the "extend the map for a nicer plot" step
    of the robust-pulse examples), use `problem.with_system(new_system)`
    rather than mutating fields in place.

    INPUTS:
        system                : Transmon (or ensemble of Transmons) defining the Hamiltonian
        target                : GateTarget or StateTarget
        n_steps                : number of timesteps
        dt                     : timestep duration in seconds (scalar, or (n_steps,) for
                                  unequal timesteps)
        reference_amplitude    : pulse amplitude normalization in Hz; defaults to
                                  1/(2*duration) if not given
        ux_basis, uy_basis     : optional (n_coeffs, n_steps) analytic pulse bases -- if
                                  given, the optimization variable is a set of coefficients
                                  rather than the direct per-timestep pulse
        weights                : optional per-ensemble-member cost weights, shape (n_ensemble,)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    system: Transmon
    target: "Target"
    n_steps: int
    dt: float = 0.1e-9
    reference_amplitude: float | None = None
    ux_basis: np.ndarray | None = None
    uy_basis: np.ndarray | None = None
    weights: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.target, (GateTarget, StateTarget)):
            raise TypeError("OptimizationProblem.target must be a GateTarget or a StateTarget")
        self.is_gate_problem = isinstance(self.target, GateTarget)

        dt_arr = np.atleast_1d(np.asarray(self.dt, dtype=float))
        if dt_arr.ndim != 1 or dt_arr.shape[0] not in (1, self.n_steps):
            raise ValueError(
                f"OptimizationProblem.dt must be a scalar (equal timesteps) or an array of "
                f"shape (n_steps,)={self.n_steps}, got shape {dt_arr.shape}"
            )
        self.t = np.concatenate(([0.0], np.cumsum(np.broadcast_to(dt_arr, (self.n_steps,)))))
        self.tc = self.t[:-1]
        self.duration = float(self.t[-1])

        if self.reference_amplitude is None:
            self.reference_amplitude = 1.0 / (2 * self.duration)

        n_time = max(self.system.n_time_samples, dt_arr.shape[0])
        dt_broadcast = np.broadcast_to(dt_arr, (n_time,))
        self.H0dt, self.HDdt, self.itH = self.system._hamiltonians(
            dt_broadcast, self.n_steps, self.reference_amplitude, n_time=n_time,
        )
        self.n_ensemble = self.system.n_ensemble

        self.ux_basis, self.TruxBasis, self.n_x, self.is_shaped_x = self._normalize_basis(self.ux_basis, "ux_basis")
        self.uy_basis, self.TruyBasis, self.n_y, self.is_shaped_y = self._normalize_basis(self.uy_basis, "uy_basis")

        if self.weights is None:
            self.Weight = np.ones((1, self.n_ensemble), dtype=dtype_float)
        else:
            w = np.squeeze(np.asarray(self.weights, dtype=float))
            if w.ndim == 0:
                w = np.full(self.n_ensemble, float(w))
            if w.shape != (self.n_ensemble,):
                raise ValueError(
                    f"OptimizationProblem.weights must have shape (n_ensemble,)=({self.n_ensemble},), "
                    f"got shape {np.asarray(self.weights).shape}"
                )
            self.Weight = w.reshape(1, -1)

        if self.is_gate_problem:
            self.Tardagg = self.target._dagger(self.n_ensemble)
            self.CompSpace = self.target.subspace
            self.Nc = self.target.n_computational_states
            self.psi0 = None
        else:
            if self.target.n_levels != self.system.n_levels:
                raise ValueError(
                    f"StateTarget lives in a {self.target.n_levels}-level space but the "
                    f"Transmon has n_levels={self.system.n_levels}"
                )
            self.Tardagg = self.target._dagger(self.n_ensemble)
            self.psi0 = self.target._psi0(self.n_ensemble)
            self.CompSpace = None
            self.Nc = None

    def _normalize_basis(self, basis, name):
        if basis is None:
            return np.eye(self.n_steps), np.eye(self.n_steps), self.n_steps, False
        basis = np.atleast_2d(np.asarray(basis, dtype=float))
        if basis.shape[1] != self.n_steps:
            raise ValueError(
                f"OptimizationProblem.{name} must have shape (n_coeffs, n_steps)=(*, {self.n_steps}), "
                f"got shape {basis.shape}"
            )
        return basis, basis.T, basis.shape[0], True

    def with_system(self, new_system: Transmon) -> "OptimizationProblem":
        """Return a copy of this problem re-evaluated against a different
        Transmon (e.g. a finer robustness grid), keeping everything else
        (target, time grid, pulse parametrization) the same."""
        return dataclasses.replace(self, system=new_system)

    def with_(self, **changes) -> "OptimizationProblem":
        """Return a copy of this problem with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    def waveform(self, pulse: "Pulse") -> tuple[np.ndarray, np.ndarray]:
        """Convert a Pulse (which holds coefficients if this problem uses an
        analytic basis, or the direct per-timestep pulse otherwise) into the
        actual (n_steps,) in-phase/quadrature waveform."""
        return self.TruxBasis @ pulse.ux, self.TruyBasis @ pulse.uy

    def validate_pulse(self, ux, uy):
        """Check that a candidate pulse (or coefficient vector, if this
        problem uses an analytic basis) has the right shape."""
        ux = np.atleast_1d(np.asarray(ux, dtype=float))
        uy = np.atleast_1d(np.asarray(uy, dtype=float))
        if ux.shape[0] != self.n_x:
            raise ValueError(f"ux must have shape ({self.n_x},), got {ux.shape}")
        if uy.shape[0] != self.n_y:
            raise ValueError(f"uy must have shape ({self.n_y},), got {uy.shape}")
        return ux, uy

    def _engine_inputs(self):
        """Flat, Numba-friendly arrays consumed by the cost-function engine."""
        common = dict(
            Nt=dtype_int(self.n_steps),
            NLevels=dtype_int(self.system.n_levels),
            Nhp=dtype_int(self.n_ensemble),
            Nx=dtype_int(self.n_x),
            Ny=dtype_int(self.n_y),
            H0dt=self.H0dt.astype(dtype_complex),
            HDdt=self.HDdt.astype(dtype_complex),
            Tardagg=self.Tardagg.astype(dtype_complex),
            uxBasis=self.ux_basis.astype(dtype_float),
            uyBasis=self.uy_basis.astype(dtype_float),
            TruxBasis=self.TruxBasis.astype(dtype_float),
            TruyBasis=self.TruyBasis.astype(dtype_float),
            itH=self.itH.astype(dtype_int),
            Weight=self.Weight.astype(dtype_float),
        )
        if self.is_gate_problem:
            common["CompSpace"] = self.CompSpace.astype(dtype_int)
            common["Nc"] = dtype_int(self.Nc)
        else:
            common["psi0"] = self.psi0.astype(dtype_complex)
        return common
