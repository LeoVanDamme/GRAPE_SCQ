"""
grape_scq: GRAPE pulse optimization for superconducting qubits,
with an object model designed around ease of setup and visualization.

Quick start:
    from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess

    system = Transmon(n_levels=3, frequency=5e9, anharmonicity=-100e6)
    target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])
    problem = OptimizationProblem(system, target, n_steps=120, dt=0.1e-9, reference_amplitude=10e6)

    pulse0 = initial_guess.cos_drag(problem, theta=np.pi)
    result = GrapeOptimizer(max_iter=500, minimize_energy=True).optimize(problem, pulse0)

    result.plot_pulse()
    result.plot_population(initial_state=0, measured_state=2)
    print(result.summary())

See examples/ and the README for full walkthroughs, including state-to-state
targets, analytic pulse shaping, amplitude constraints, and robust
multi-parameter (ensemble) optimization with fidelity-map visualization.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
"""

from .system import Transmon
from .targets import GateTarget, StateTarget
from .problem import OptimizationProblem
from .optimizer import GrapeOptimizer
from .result import Pulse, Result
from . import initial_guess
from . import analysis
from .bases import fourier_basis, symmetric_fourier_basis, antisymmetric_fourier_basis
from .analysis import eigenstate

__all__ = [
    "Transmon",
    "GateTarget",
    "StateTarget",
    "OptimizationProblem",
    "GrapeOptimizer",
    "Pulse",
    "Result",
    "initial_guess",
    "analysis",
    "fourier_basis",
    "symmetric_fourier_basis",
    "antisymmetric_fourier_basis",
    "eigenstate",
]
