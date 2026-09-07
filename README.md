# grape_scq

A Python package for designing control pulses for superconducting (transmon)
qubits using GRAPE (GRadient Ascent Pulse Engineering). Given a physical
system, a target gate or state transfer, and a time grid, it optimizes the
in-phase/quadrature drive pulse — using an exact adjoint-state gradient
computed by a fast Numba engine — to reach the target while suppressing
leakage to non-computational levels, minimizing pulse energy, satisfying
amplitude constraints, and/or remaining robust to parameter uncertainty
(detuning, amplitude miscalibration, drift). See `doc/` for a full write-up
of the underlying physics and worked examples with figures.

## Installation

```bash
pip install -e .
# or: pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess

# A 12 ns NOT gate on a 3-level transmon (leakage-aware)
system = Transmon(n_levels=3, anharmonicity=-100e6)
target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
problem = OptimizationProblem(system, target, n_steps=120, dt=0.1e-9, reference_amplitude=10e6)

pulse0 = initial_guess.cos_drag(problem, theta=np.pi)
result = GrapeOptimizer(max_iter=500, minimize_energy=True).optimize(problem, pulse0)

print(result.summary())
result.plot_pulse()
result.plot_population(initial_state=0, measured_state=2)   # leakage to |2>
```

## Object model

- **`Transmon`** — the physical system: `n_levels`, `frequency`, `anharmonicity`,
  `amplitude_scale`, `carrier_frequency`. Any of the first four can be an array to
  describe an ensemble (robust optimization) or a time-dependent Hamiltonian.
  `Transmon.sweep(frequency=..., amplitude_scale=...)` builds a 2-D parameter
  grid and remembers its shape/axis labels for later fidelity-map plotting.
- **`GateTarget(matrix, subspace)`** / **`StateTarget(initial, target)`** — explicit
  target types (no shape-based gate-vs-state inference).
- **`OptimizationProblem(system, target, n_steps, dt, ...)`** — composes the above
  plus the time grid and (optionally) an analytic pulse basis
  (`ux_basis`/`uy_basis`, e.g. from `bases.fourier_basis`). Immutable by
  convention; use `problem.with_system(new_system)` to re-evaluate on a
  different system.
- **`GrapeOptimizer(max_iter=..., max_amplitude=..., minimize_energy=..., ...)`** —
  `optimizer.optimize(problem, initial_pulse) -> Result`.
- **`initial_guess`** — `cos_drag`, `random_fourier`, `random_symmetric_antisymmetric`,
  `constant`, `zero`; each returns a `Pulse`.
- **`Result`** — `.pulse`, `.average_fidelity`, `.fidelity_map()`, `.population(...)`,
  `.plot_pulse()`, `.plot_population(...)`, `.plot_fidelity_map()`, `.summary()`.

## Examples

See `examples/` — the same six scenarios as the original package's notebooks
(leakage-free gate, two-photon transfer, analytic pulse shaping, robust
Fourier pulse with constraints, selective pulse, time-dependent-detuning
robustness), rebuilt against this API.

## Testing

```bash
pip install -e ".[test]"
pytest
```

## Relationship to GRAPE_SCQ

This is a from-scratch object-model redesign built on the same underlying
math as [GRAPE_SCQ](https://github.com/LeoVanDamme/GRAPE_SCQ) (exact
adjoint-state gradients, single eigendecomposition per timestep). See that
repository for the citable, published version and `CITATION.cff`.
