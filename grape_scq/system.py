from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


def _as_ensemble_array(value, name: str) -> np.ndarray:
    """Normalize a Hamiltonian parameter to shape (n_ensemble, n_time):
    a scalar broadcasts to every ensemble member and every timestep; a 1-D
    array of length n_ensemble is constant in time; a 2-D array is
    (n_ensemble, n_time) directly."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"Transmon.{name} must be a scalar, a 1-D array (one value per "
        f"ensemble member), or a 2-D array (ensemble member x time) -- got "
        f"an array of shape {arr.shape}."
    )


@dataclass
class Transmon:
    """A (possibly time-dependent, possibly an ensemble of) transmon-like
    superconducting qubit: n_levels energy levels, a frequency, an
    anharmonicity, and a drive amplitude scale factor.

    `frequency`, `anharmonicity` and `amplitude_scale` each accept:
      - a scalar (same value for every ensemble member, constant in time),
      - a 1-D array of length n_ensemble (one value per ensemble member,
        constant in time) -- this is how robust/multi-parameter-set
        optimization is expressed (see `Transmon.sweep`),
      - a 2-D array of shape (n_ensemble, n_steps) (time-dependent, e.g. a
        drifting detuning).
    Mixing shapes across the three fields is fine as long as each field's
    ensemble size and time-length are each either 1 (broadcast) or match
    the others' non-1 sizes.

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    n_levels: int = 3
    frequency: float = 5.0e9
    anharmonicity: float = -100.0e6
    amplitude_scale: float = 1.0
    carrier_frequency: float | None = None
    grid_shape: tuple[int, ...] | None = field(default=None, repr=False)
    grid_axes: Mapping[str, np.ndarray] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.frequency = _as_ensemble_array(self.frequency, "frequency")
        self.anharmonicity = _as_ensemble_array(self.anharmonicity, "anharmonicity")
        self.amplitude_scale = _as_ensemble_array(self.amplitude_scale, "amplitude_scale")

        ensemble_sizes = {a.shape[0] for a in (self.frequency, self.anharmonicity, self.amplitude_scale) if a.shape[0] != 1}
        if len(ensemble_sizes) > 1:
            raise ValueError(
                f"Transmon.frequency, .anharmonicity and .amplitude_scale disagree on "
                f"the ensemble size: got {ensemble_sizes} (each must be 1, to broadcast, "
                f"or match the others)."
            )
        time_sizes = {a.shape[1] for a in (self.frequency, self.anharmonicity, self.amplitude_scale) if a.shape[1] != 1}
        if len(time_sizes) > 1:
            raise ValueError(
                f"Transmon.frequency, .anharmonicity and .amplitude_scale disagree on "
                f"the number of time samples: got {time_sizes} (each must be 1, to "
                f"broadcast, or match the others)."
            )

        if self.carrier_frequency is None:
            self.carrier_frequency = float(np.mean(self.frequency))

    @property
    def n_ensemble(self) -> int:
        return max(self.frequency.shape[0], self.anharmonicity.shape[0], self.amplitude_scale.shape[0])

    @property
    def n_time_samples(self) -> int:
        """1 if the Hamiltonian is static (equal timesteps, time-independent
        parameters), else the number of distinct time samples provided."""
        return max(self.frequency.shape[1], self.anharmonicity.shape[1], self.amplitude_scale.shape[1])

    @classmethod
    def sweep(cls, n_levels: int = 3, anharmonicity: float = -100.0e6,
              carrier_frequency: float | None = None, **swept: np.ndarray) -> "Transmon":
        """Build an ensemble on a meshgrid of one or more swept parameters,
        e.g. `Transmon.sweep(frequency=5e9 + detunings, amplitude_scale=1 + errors)`.
        Remembers the grid shape and axis values so results can later be
        reshaped into a 2-D fidelity map automatically (see `Result.plot_fidelity_map`).
        Non-swept Hamiltonian fields (e.g. `anharmonicity`, unless also passed
        as a swept keyword) stay scalar."""
        if not swept:
            raise ValueError("Transmon.sweep() needs at least one swept keyword argument, "
                              "e.g. frequency=np.linspace(...).")

        axis_values = {name: np.asarray(values, dtype=float) for name, values in swept.items()}
        for name, values in axis_values.items():
            if values.ndim != 1:
                raise ValueError(f"Transmon.sweep(): '{name}' must be a 1-D array of values to sweep over.")

        names = list(axis_values)
        grids = np.meshgrid(*(axis_values[name] for name in names), indexing="ij")
        grid_shape = grids[0].shape

        fields = {"n_levels": n_levels, "anharmonicity": anharmonicity, "carrier_frequency": carrier_frequency}
        for name, grid in zip(names, grids):
            fields[name] = grid.ravel()

        instance = cls(**fields)
        instance.grid_shape = grid_shape
        instance.grid_axes = axis_values
        return instance

    def with_(self, **changes) -> "Transmon":
        """Return a copy of this Transmon with the given fields replaced,
        e.g. `system.with_(frequency=new_grid)`."""
        return dataclasses.replace(self, **changes)

    def _hamiltonians(self, dt: np.ndarray, n_steps: int, reference_amplitude: float, n_time: int | None = None):
        """Build the drift (H0dt) and drive (HDdt) Hamiltonians, each scaled
        by the timestep dt, and the per-timestep Hamiltonian-index array itH
        used to keep memory usage down when the Hamiltonian is static.

        `n_time` lets the caller (OptimizationProblem, which also knows how
        many distinct timesteps `dt` provides) force a time-length larger
        than this Transmon's own `n_time_samples`, e.g. when the Hamiltonian
        is static but the timestep duration varies. `dt` must already be
        broadcast to that same length.

        Returns:
            H0dt : (n_levels, n_levels, n_time, n_ensemble)
            HDdt : (n_levels, n_levels, n_time, n_ensemble)  (pre-Hermitization; the
                   engine adds the Hermitian-conjugate drive term itself)
            itH  : (n_steps,) int64 -- index into the time axis of H0dt/HDdt for each step
        """
        n_ensemble = self.n_ensemble
        n_time = self.n_time_samples if n_time is None else max(n_time, self.n_time_samples)

        freq = np.broadcast_to(self.frequency, (n_ensemble, n_time))
        anharm = np.broadcast_to(self.anharmonicity, (n_ensemble, n_time))
        amp = reference_amplitude * np.broadcast_to(self.amplitude_scale, (n_ensemble, n_time))

        raise_op = np.diag(np.sqrt(np.arange(1, self.n_levels)), k=-1)   # a-dagger
        lower_op = np.diag(np.sqrt(np.arange(1, self.n_levels)), k=1)    # a
        number_op = raise_op @ lower_op

        H0dt = np.zeros((self.n_levels, self.n_levels, n_time, n_ensemble), dtype=complex)
        HDdt = np.zeros((self.n_levels, self.n_levels, n_time, n_ensemble), dtype=complex)
        for k in range(n_ensemble):
            for t in range(n_time):
                H0 = (2*np.pi*anharm[k, t] * (raise_op@raise_op@lower_op@lower_op) / 2
                      + 2*np.pi*(freq[k, t] - self.carrier_frequency) * number_op)
                H0dt[:, :, t, k] = H0 * dt[t]
                HD = 0.5 * 2*np.pi*amp[k, t] * raise_op
                HDdt[:, :, t, k] = HD * dt[t]

        itH = np.zeros(n_steps, dtype=np.int64) if n_time == 1 else np.arange(n_steps, dtype=np.int64)
        return H0dt, HDdt, itH
