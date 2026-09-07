"""One-off script to generate the example notebooks. Not part of the package
-- run once to (re)build examples/*.ipynb, then execute them with
`jupyter nbconvert --to notebook --execute --inplace`."""
import nbformat as nbf
import os

HERE = os.path.dirname(__file__)


def build(filename, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_code_cell(src) if kind == "code" else nbf.v4.new_markdown_cell(src)
        for kind, src in cells
    ]
    with open(os.path.join(HERE, filename), "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# ---------------------------------------------------------------------------
build("01_leakage_free_gate.ipynb", [
("code", '''# Design a 12 ns NOT gate (pi rotation) on the computational subspace {|0>, |1>}
# of a transmon qubit.
# The control pulse is optimized to:
#   - Suppress leakage into the non-computational level |2>,
#   - Minimize the total pulse energy, given by the integral (Ωx^2 + Ωy^2) dt.
# The system is modeled with an anharmonicity of -100 MHz.
# Estimated script runtime: ~5 seconds.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess'''),
("code", '''# Physical system and target
system = Transmon(
    n_levels=3,          # Number of energy levels
    anharmonicity=-100e6,  # Anharmonicity in Hz
)
target = GateTarget(
    matrix=[[0, 1], [1, 0]],  # Target gate (NOT gate)
    subspace=[0, 1],          # Computational states indices
)
problem = OptimizationProblem(
    system, target,
    n_steps=120,             # Number of time steps
    dt=0.1e-9,                # Timestep in s.
    reference_amplitude=10e6, # Reference pulse amplitude
)'''),
("code", '''# Initial guess: Drag-like pulse
pulse0 = initial_guess.cos_drag(problem, theta=np.pi)'''),
("code", '''# Optimize the pulse
optimizer = GrapeOptimizer(
    max_iter=500,        # Maximum number of iterations
    minimize_energy=True,  # Find minimum-energy solution
)
result = optimizer.optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Compare fidelity
initial_result_fidelity = 1 - result.initial_cost
print(f"Fidelity of DRAG pulse = {initial_result_fidelity*100:.3f} %")
print(f"Fidelity of GRAPE pulse = {result.average_fidelity*100:.3f} %")'''),
("code", '''# Display in-phase and quadrature pulse components (GRAPE vs. initial DRAG guess)
result.plot_pulse(compare_to_initial=True)
plt.title("GRAPE versus DRAG pulse shapes")
plt.show()'''),
("code", '''# Compute and display leakage as a function of time
plt.figure()
plt.plot(problem.t, result.population(0, 2) * 100, label=r'$P_{|0\\rangle\\rightarrow|2\\rangle}$')
plt.plot(problem.t, result.population(1, 2) * 100, label=r'$P_{|1\\rangle\\rightarrow|2\\rangle}$')
plt.xlabel('Time (s)')
plt.ylabel('Probability (%)')
plt.legend()
plt.title("Leakage probability")
plt.show()'''),
])

# ---------------------------------------------------------------------------
build("02_two_photon_transition.ipynb", [
("code", '''# Design a 20 ns two-photon transition from |0> to |2> in a transmon qubit.
# The optimization aims to:
#   - Maximize population transfer from |0> to |2> via a two-photon process,
#   - Minimize leakage to level |3>,
#   - Minimize the total pulse energy.
# The system is modeled with an anharmonicity of -200 MHz.
# Estimated script runtime: ~15 seconds.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, StateTarget, OptimizationProblem, GrapeOptimizer, initial_guess'''),
("code", '''# Physical system and target: drive at the two-photon carrier nu_q + alpha/2
system = Transmon(
    n_levels=4,
    anharmonicity=-200e6,
    frequency=5e9,
    carrier_frequency=5e9 - 200e6/2,
)
target = StateTarget(initial=[1, 0, 0, 0], target=[0, 0, 1, 0])
problem = OptimizationProblem(system, target, n_steps=200, dt=0.1e-9, reference_amplitude=100e6)'''),
("code", '''# Initial pulse guess
pulse0 = initial_guess.random_symmetric_antisymmetric(problem)'''),
("code", '''# Optimize the pulse
optimizer = GrapeOptimizer(max_iter=1000, minimize_energy=True)
result = optimizer.optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Display in-phase and quadrature pulse components
result.plot_pulse(compare_to_initial=False)
plt.title("Optimized pulse shapes")
plt.show()'''),
("code", '''# Display transition probabilities
plt.figure()
plt.plot(problem.t, result.population(0, 1)*100, label=r'$P_{|0\\rangle\\rightarrow|1\\rangle}$')
plt.plot(problem.t, result.population(0, 2)*100, label=r'$P_{|0\\rangle\\rightarrow|2\\rangle}$')
plt.plot(problem.t, result.population(0, 3)*100, label=r'$P_{|0\\rangle\\rightarrow|3\\rangle}$ (leakage)')
plt.xlabel('Time (s)')
plt.ylabel('Probability (%)')
plt.legend()
plt.title("Transition probabilities")
plt.show()'''),
])

# ---------------------------------------------------------------------------
build("03_analytic_pulse_shaping.ipynb", [
("code", '''# Design a 16 ns parametrized control pulse to implement a Hadamard gate in
# a transmon qubit while suppressing leakage to the |2> level.
#
# The pulse is parametrized as a Fourier series, so the optimizer works on a
# handful of coefficients (ux_basis/uy_basis) rather than the full
# per-timestep pulse.
# Estimated script runtime: ~2 seconds.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess'''),
("code", '''# Physical system and target
n_steps = 160
n_coeffs = 3
# Analytic basis: ux(t) = sum ax(n)*fx(n,t), uy(t) = sum ay(n)*fy(n,t),
# using odd harmonics only -- fx/fy(n,t) = sin/cos((2n+1)*pi*t/T)
tau = np.linspace(0, 1, n_steps)
fx = np.array([np.sin((2*n+1)*np.pi*tau) for n in range(n_coeffs)])
fy = np.array([np.cos((2*n+1)*np.pi*tau) for n in range(n_coeffs)])

system = Transmon(n_levels=3, anharmonicity=-200e6)
target = GateTarget(matrix=np.array([[1, 1], [1, -1]])/np.sqrt(2), subspace=[0, 1])  # Hadamard
problem = OptimizationProblem(system, target, n_steps=n_steps, dt=0.1e-9,
                               ux_basis=fx, uy_basis=fy)'''),
("code", '''# Initial guess: random low-order Fourier coefficients
pulse0 = initial_guess.random_fourier(problem, n_coeffs=n_coeffs)'''),
("code", '''# Optimize the pulse
result = GrapeOptimizer(max_iter=500).optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Display in-phase and quadrature pulse components
result.plot_pulse(compare_to_initial=False)
plt.title("Analytically shaped Hadamard gate")
plt.show()'''),
("code", '''# Display transition probabilities
plt.figure()
plt.plot(problem.t, result.population(0, 1)*100, label=r'$P_{|0\\rangle\\rightarrow|1\\rangle}$')
plt.plot(problem.t, result.population(1, 0)*100, label=r'$P_{|1\\rangle\\rightarrow|0\\rangle}$')
plt.plot(problem.t, result.population(0, 2)*100, label=r'$P_{|0\\rangle\\rightarrow|2\\rangle}$ (leakage)')
plt.plot(problem.t, result.population(1, 2)*100, label=r'$P_{|1\\rangle\\rightarrow|2\\rangle}$ (leakage)')
plt.xlabel('Time (s)')
plt.ylabel('Probability (%)')
plt.legend()
plt.title("Transition probabilities")
plt.show()'''),
])

# ---------------------------------------------------------------------------
build("04_robust_fourier_pulse.ipynb", [
("code", '''# Design of a 100 ns control pulse, parameterized as a Fourier series, to
# implement a pi/2 gate that is robust to amplitude miscalibration and
# frequency detuning.
#
# Robustness is ensured by optimizing over a Transmon.sweep(...) ensemble --
# a grid of (detuning, amplitude-scale) pairs -- to maximize the average gate
# fidelity across the grid, subject to an amplitude constraint.
# Estimated script runtime: ~15 minutes.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess
from grape_scq.bases import symmetric_fourier_basis, antisymmetric_fourier_basis'''),
("code", '''# Physical system: a robustness grid over amplitude error and detuning
Na, Nd = 5, 5
amp_error = np.linspace(-0.1, 0.1, Na)          # Amplitude error [-10%, 10%]
detuning = 0.2e6 * np.linspace(-1, 1, Nd)       # Detuning [-0.2 MHz, 0.2 MHz]

system = Transmon.sweep(
    frequency=5e9 + detuning,
    amplitude_scale=1 + amp_error,
    anharmonicity=-200e6,
)

target = GateTarget(matrix=np.array([[1, -1j], [-1j, 1]])/np.sqrt(2), subspace=[0, 1])  # pi/2 gate'''),
("code", '''# Analytic pulse basis (Fourier series)
n_steps = 200
n_coeffs = 5
fx = symmetric_fourier_basis(n_steps, n_coeffs)
fy = antisymmetric_fourier_basis(n_steps, n_coeffs)

problem = OptimizationProblem(system, target, n_steps=n_steps, dt=0.5e-9,
                               reference_amplitude=5e6, ux_basis=fx, uy_basis=fy)'''),
("code", '''# Initial guess: DRAG-like pulse
pulse0 = initial_guess.cos_drag(problem, theta=np.pi/2)'''),
("code", '''# Optimize with amplitude constraints
optimizer = GrapeOptimizer(max_iter=2000, max_in_phase=15e6, max_quadrature=15e6)
result = optimizer.optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Display pulse shapes
result.plot_pulse(compare_to_initial=False)
plt.show()'''),
("code", '''# Extend the grid for a finer fidelity map, and re-evaluate the SAME pulse on it
fine_system = Transmon.sweep(
    frequency=5e9 + 0.65e6*np.linspace(-1, 1, 35),
    amplitude_scale=1 + 0.2*np.linspace(-1, 1, 35),
    anharmonicity=-200e6,
)
fine_problem = problem.with_system(fine_system)'''),
("code", '''# Display the fidelity map on the fine grid
from grape_scq import analysis
fmap = analysis.fidelity_map(fine_problem, result.pulse).reshape(fine_system.grid_shape)
plt.pcolormesh(fine_system.grid_axes["frequency"]/1e6 - 5e3,
               (fine_system.grid_axes["amplitude_scale"]-1)*100,
               fmap.T*100, shading='auto')
plt.colorbar(label='Gate fidelity (%)')
plt.xlabel('Detuning (MHz)')
plt.ylabel('Amplitude deviation (%)')
plt.show()'''),
])

# ---------------------------------------------------------------------------
build("05_selective_pulse.ipynb", [
("code", '''# Design of a 120 ns selective pulse that maps |0> to a DIFFERENT computational
# outcome depending on the qubit's detuning: |0> -> |0> when detuned by -2.5 MHz,
# and |0> -> |1> when detuned by +2.5 MHz.
# The qubits are modeled as ideal two-level systems (n_levels = 2).
# Selectivity is achieved by simultaneously optimizing over two ensemble
# members corresponding to the two detunings, each with its own StateTarget.
# The pulse amplitude is constrained such that
#                 2*pi*nu_ref*(ux^2+uy^2)^(1/2) < 8 MHz.
#
# The resulting sequence resembles a Ramsey experiment.
# Note: This solution was shown to be time-optimal (Phys. Rev. A 98, 043421)
# for the equivalent (swapped-target) problem in the original GRAPE_SCQ example.
# Estimated script runtime: ~5 minutes.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, StateTarget, OptimizationProblem, GrapeOptimizer, initial_guess'''),
("code", '''# Physical system: two ensemble members, at +/- 2.5 MHz detuning
system = Transmon(
    n_levels=2,
    frequency=5e9 + np.array([-2.5e6, 2.5e6]),
    carrier_frequency=5e9,
)
# Both members start in |0>; member 0 (detuning -2.5 MHz) targets |0>, member 1 (+2.5 MHz) targets |1>
target = StateTarget(initial=[[1, 0], [1, 0]], target=[[1, 0], [0, 1]])
problem = OptimizationProblem(system, target, n_steps=120, dt=1e-9)'''),
("code", '''# Initial guess: constant drive
pulse0 = initial_guess.zero(problem)
pulse0.ux[:] = 1.0'''),
("code", '''# Optimize the pulse under an amplitude constraint
optimizer = GrapeOptimizer(max_iter=2000, max_amplitude=8e6)
result = optimizer.optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Display in-phase and quadrature pulse components
result.plot_pulse(compare_to_initial=False)
plt.title("Selective pulse")
plt.show()'''),
("code", '''# Display transition probabilities |0> -> |1> for both ensemble members
plt.figure()
plt.plot(problem.t, result.population(0, 1)*100)
plt.xlabel('Time (s)')
plt.ylabel(r'$P_{|0\\rangle\\rightarrow|1\\rangle}$ (%)')
plt.legend([r'$\\delta = -2.5$ MHz', r'$\\delta = +2.5$ MHz'])
plt.title("Transition probabilities")
plt.show()'''),
])

# ---------------------------------------------------------------------------
build("06_time_dependent_detuning.ipynb", [
("code", '''# Designs a 150 ns NOT gate (pi rotation) that is robust against
# time-dependent detuning and minimizes leakage into the |2> state.
#
# Robustness is achieved by assuming a detuning of the form:
#             delta(t) = A*cos(2*pi*w*t)
# and optimizing the gate over a grid of w in [0, 10 MHz] and
# A in [-5 MHz, 5 MHz] to ensure high fidelity across this range.
#
# The control pulse is decomposed into a Fourier series with n in {1,...,10}.
# The pulse amplitude is constrained such that (Ωx^2+Ωy^2)^(1/2) < 20 MHz.
# Estimated script runtime: ~15 minutes.

# AUTHOR:
#     Leo Van Damme / Technical University of Munich, 2025'''),
("code", '''import numpy as np
import matplotlib.pyplot as plt
from grape_scq import Transmon, GateTarget, OptimizationProblem, GrapeOptimizer, initial_guess, analysis
from grape_scq.bases import symmetric_fourier_basis, antisymmetric_fourier_basis'''),
("code", '''# Build the time-dependent detuning ensemble: a grid of (w, A) pairs, each
# giving its own (n_steps,)-shaped detuning trace delta(t) = A*cos(2*pi*w*t)
n_steps = 150
dt = 1e-9
tc = np.arange(n_steps) * dt
Nw, NA = 5, 5
w = np.linspace(0, 10e6, Nw)
A = np.linspace(-5e6, 5e6, NA)
wGrid, AGrid = np.meshgrid(w, A)
detuning = AGrid.ravel()[:, None] * np.cos(2*np.pi*wGrid.ravel()[:, None] * tc[None, :])

system = Transmon(n_levels=3, frequency=5e9 + detuning, carrier_frequency=5e9, anharmonicity=-100e6)
system.grid_shape = wGrid.shape
system.grid_axes = {"w": w, "A": A}

target = GateTarget(matrix=[[0, 1], [1, 0]], subspace=[0, 1])'''),
("code", '''# Analytic pulse basis (Fourier series)
Ncoeffs = 10
fx = symmetric_fourier_basis(n_steps, Ncoeffs)
fy = antisymmetric_fourier_basis(n_steps, Ncoeffs)
problem = OptimizationProblem(system, target, n_steps=n_steps, dt=dt, ux_basis=fx, uy_basis=fy)'''),
("code", '''# Initial guess: DRAG-like pulse
pulse0 = initial_guess.cos_drag(problem, theta=np.pi)'''),
("code", '''# Optimize with an amplitude constraint
optimizer = GrapeOptimizer(max_iter=2000, max_amplitude=20e6)
result = optimizer.optimize(problem, pulse0)
print(result.summary())'''),
("code", '''# Display in-phase and quadrature pulse components
result.plot_pulse(compare_to_initial=False)
plt.title("Optimized pulse")
plt.show()'''),
("code", '''# Extend the grid for a finer fidelity map, and compare against the initial guess
Nw2, NA2 = 25, 25
w2 = np.linspace(0, 12e6, Nw2)
A2 = np.linspace(-7e6, 7e6, NA2)
wGrid2, AGrid2 = np.meshgrid(w2, A2)
detuning2 = AGrid2.ravel()[:, None] * np.cos(2*np.pi*wGrid2.ravel()[:, None] * tc[None, :])

fine_system = Transmon(n_levels=3, frequency=5e9 + detuning2, carrier_frequency=5e9, anharmonicity=-100e6)
fine_system.grid_shape = wGrid2.shape
fine_system.grid_axes = {"w": w2, "A": A2}
fine_problem = problem.with_system(fine_system)

F = analysis.fidelity_map(fine_problem, result.pulse).reshape(fine_system.grid_shape)
F0 = analysis.fidelity_map(fine_problem, pulse0).reshape(fine_system.grid_shape)'''),
("code", '''plt.pcolormesh(w2/1e6, A2/1e6, F.T*100, shading='auto', vmin=50, vmax=100)
plt.colorbar(label='Gate fidelity (%)')
plt.xlabel(r'$\\omega$ (MHz)')
plt.ylabel('A (MHz)')
plt.title("Optimized pulse")
plt.show()

plt.pcolormesh(w2/1e6, A2/1e6, F0.T*100, shading='auto', vmin=50, vmax=100)
plt.colorbar(label='Gate fidelity (%)')
plt.xlabel(r'$\\omega$ (MHz)')
plt.ylabel('A (MHz)')
plt.title("DRAG pulse")
plt.show()'''),
])

print("Notebooks written.")
