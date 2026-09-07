from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import analysis


@dataclass
class Pulse:
    """An in-phase/quadrature control pulse. Holds direct per-timestep
    values (shape (n_steps,)) unless the OptimizationProblem it belongs to
    uses an analytic pulse basis, in which case it holds basis coefficients
    (shape (n_coeffs,)) -- use `problem.waveform(pulse)` to get the actual
    time-domain waveform either way.

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    ux: np.ndarray
    uy: np.ndarray

    def __post_init__(self):
        self.ux = np.atleast_1d(np.asarray(self.ux, dtype=float))
        self.uy = np.atleast_1d(np.asarray(self.uy, dtype=float))


@dataclass
class Result:
    """The outcome of a GrapeOptimizer.optimize(...) call: the optimized
    pulse, the problem it was solved for, and convenience methods for
    inspecting and plotting it -- this is the main place to look for
    "how good is my pulse and what does it look like".

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    problem: "OptimizationProblem"
    pulse: Pulse
    initial_pulse: Pulse
    cost: float
    initial_cost: float
    n_iter: int
    message: str
    n_preoptim_iter: int = 0

    @property
    def average_fidelity(self) -> float:
        """1 - cost: the ensemble-averaged, weighted fidelity the optimizer
        actually minimized against."""
        return 1.0 - self.cost

    def fidelity_map(self) -> np.ndarray:
        """Per-ensemble-member fidelity of the optimized pulse (shape
        (n_ensemble,), or a scalar if there's a single ensemble member) --
        recomputed by direct forward simulation (independent of the
        optimizer's own cost bookkeeping)."""
        return analysis.fidelity_map(self.problem, self.pulse)

    def population(self, initial_state, measured_state) -> np.ndarray:
        """Transition probability |<measured_state|psi(t)>|^2 over time under
        the optimized pulse. `initial_state`/`measured_state` may be state
        vectors or level indices (int), e.g. `result.population(0, 2)` for
        the |0>-to-|2> leakage probability."""
        n_levels = self.problem.system.n_levels
        if isinstance(initial_state, (int, np.integer)):
            initial_state = analysis.eigenstate(initial_state, n_levels)
        if isinstance(measured_state, (int, np.integer)):
            measured_state = analysis.eigenstate(measured_state, n_levels)
        return analysis.population(self.problem, self.pulse, initial_state, measured_state)

    def summary(self) -> str:
        lines = [
            "GRAPE optimization result",
            f"  cost:      {self.initial_cost:.6f} -> {self.cost:.6f}",
            f"  fidelity:  {1-self.initial_cost:.6f} -> {self.average_fidelity:.6f}",
            f"  iterations: {self.n_iter}" + (f" (+ {self.n_preoptim_iter} pre-optimization)" if self.n_preoptim_iter else ""),
        ]
        if self.message:
            lines.append(f"  message:   {self.message}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    # -- plotting -----------------------------------------------------
    def plot_pulse(self, compare_to_initial: bool = True, ax=None, in_hz: bool = True):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        scale = self.problem.reference_amplitude if in_hz else 1.0
        ux, uy = self.problem.waveform(self.pulse)
        t = self.problem.tc
        ax.plot(t, scale * ux, label=r"$\Omega_x/(2\pi)$")
        ax.plot(t, scale * uy, label=r"$\Omega_y/(2\pi)$")
        if compare_to_initial:
            ux0, uy0 = self.problem.waveform(self.initial_pulse)
            ax.plot(t, scale * ux0, linestyle="--", color="#1f77b4", label=r"$\Omega_x/(2\pi)$ (initial)")
            ax.plot(t, scale * uy0, linestyle="--", color="#ff7f0e", label=r"$\Omega_y/(2\pi)$ (initial)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (Hz)" if in_hz else "Amplitude (normalized)")
        ax.legend(loc="upper right")
        ax.set_title("Optimized pulse")
        return ax

    def plot_population(self, initial_state, measured_state, ax=None, label=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        prob = self.population(initial_state, measured_state) * 100
        t = self.problem.t
        if prob.ndim == 1:
            ax.plot(t, prob, label=label)
        else:
            for k in range(prob.shape[1]):
                ax.plot(t, prob[:, k], label=(label or f"member {k}"))
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability (%)")
        return ax

    def plot_fidelity_map(self, ax=None):
        """2-D pcolormesh of per-ensemble-member fidelity, using the axis
        values/labels remembered by `Transmon.sweep(...)`. Raises if the
        problem's system wasn't built with `Transmon.sweep`."""
        import matplotlib.pyplot as plt

        system = self.problem.system
        if system.grid_shape is None:
            raise ValueError(
                "plot_fidelity_map() needs a system built with Transmon.sweep(...) "
                "so the ensemble can be reshaped into a 2-D grid."
            )
        names = list(system.grid_axes)
        if len(names) != 2:
            raise ValueError(
                f"plot_fidelity_map() only supports a 2-D sweep, got a sweep over {names}"
            )
        x_name, y_name = names
        x_vals, y_vals = system.grid_axes[x_name], system.grid_axes[y_name]
        fmap = self.fidelity_map().reshape(system.grid_shape)

        ax = ax or plt.gca()
        mesh = ax.pcolormesh(x_vals, y_vals, fmap.T * 100, shading="auto", vmin=0, vmax=100)
        plt.colorbar(mesh, ax=ax, label="Fidelity (%)")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("Fidelity map")
        return ax
