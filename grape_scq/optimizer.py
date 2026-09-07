from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .problem import OptimizationProblem
from .result import Pulse, Result
from ._engine.dtypes import dtype_float
from ._engine.cost_functions import CostFunctionGate, CostFunctionState
from ._engine import constraints as _c


@dataclass
class GrapeOptimizer:
    """Optimizes a pulse against an OptimizationProblem by minimizing its
    infidelity with `scipy.optimize.minimize` (L-BFGS-B, or trust-constr
    when an amplitude constraint is set), using the exact adjoint-state
    gradient computed by the engine.

    INPUTS:
        max_iter                : maximum number of solver iterations
        display                 : print progress/results
        max_in_phase             : maximum |Omega_x| in Hz (adds a constraint)
        max_quadrature           : maximum |Omega_y| in Hz (adds a constraint)
        max_amplitude            : maximum sqrt(Omega_x^2+Omega_y^2) in Hz (adds a constraint)
        energy_penalty_weight    : penalty weight lambda on pulse energy: J += lambda * integral(ux^2+uy^2)
        minimize_energy          : pre-optimize with a decaying energy penalty to find a
                                    minimum-energy solution (see `n_suboptim`)
        n_suboptim               : number of pre-optimizations when minimize_energy is set
        energy_min_weight_max    : initial penalty weight for the first pre-optimization;
                                    defaults to reference_amplitude^2 * duration^2

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    max_iter: int = 1000
    display: bool = True
    max_in_phase: float | None = None
    max_quadrature: float | None = None
    max_amplitude: float | None = None
    energy_penalty_weight: float = 0.0
    minimize_energy: bool = False
    n_suboptim: int = 10
    energy_min_weight_max: float | None = None
    xtol: float = 1e-12
    ftol: float = 1e-12
    gtol: float = 1e-12
    max_fun: float = 1e18

    _iter: int = field(default=0, init=False, repr=False)

    def optimize(self, problem: OptimizationProblem, initial_pulse: Pulse) -> Result:
        ux0, uy0 = problem.validate_pulse(initial_pulse.ux, initial_pulse.uy)
        optimvar = np.concatenate([ux0, uy0])

        cost_fn = self._cost_function(problem, self.energy_penalty_weight)
        constraints = self._constraints(problem)

        if self.display:
            print("Compiling cost function...", end=" ", flush=True)
        initial_cost = float(np.asarray(cost_fn(optimvar)[0]).ravel()[0])
        if self.display:
            print("Done.")

        if self.minimize_energy:
            optimvar, n_preoptim_iter = self._preoptimize_energy_min(problem, optimvar)
        else:
            n_preoptim_iter = 0

        self._iter = 0
        res = self._call_solver(problem, optimvar, cost_fn, constraints)

        ux = res.x[:problem.n_x]
        uy = res.x[problem.n_x:problem.n_x + problem.n_y]

        result = Result(
            problem=problem,
            pulse=Pulse(ux, uy),
            initial_pulse=Pulse(ux0, uy0),
            cost=float(res.fun),
            initial_cost=initial_cost,
            n_iter=self._iter,
            n_preoptim_iter=n_preoptim_iter,
            message=str(getattr(res, "message", "")),
        )

        if self.display:
            print(result.summary())

        return result

    # -- internals ------------------------------------------------------
    def _cost_function(self, problem: OptimizationProblem, energy_penalty_weight: float):
        engine_inputs = problem._engine_inputs()
        fn = CostFunctionGate if problem.is_gate_problem else CostFunctionState
        penalty = dtype_float(energy_penalty_weight)
        return lambda x: fn(x, EnergyPenaltyWeight=penalty, **engine_inputs)

    def _constraints(self, problem: OptimizationProblem):
        constraints = []
        Trux, Truy, Nx, Ny, Nt = problem.TruxBasis, problem.TruyBasis, problem.n_x, problem.n_y, problem.n_steps
        ref = problem.reference_amplitude

        if self.max_amplitude is not None:
            max_amp = self.max_amplitude
            constraints.append({
                "type": "ineq",
                "fun": lambda x: _c.amplitude_constraint(x, Trux, Truy, Nx, Ny, max_amp, ref),
                "jac": lambda x: _c.amplitude_constraint_jac(x, Trux, Truy, Nx, Ny),
            })
        if self.max_in_phase is not None:
            max_x = self.max_in_phase
            constraints.append({
                "type": "ineq",
                "fun": lambda x: _c.in_phase_constraint(x, Trux, Truy, Nx, Ny, max_x, ref),
                "jac": lambda x: _c.in_phase_constraint_jac(x, Trux, Truy, Nx, Ny, Nt),
            })
        if self.max_quadrature is not None:
            max_y = self.max_quadrature
            constraints.append({
                "type": "ineq",
                "fun": lambda x: _c.quadrature_constraint(x, Trux, Truy, Nx, Ny, max_y, ref),
                "jac": lambda x: _c.quadrature_constraint_jac(x, Trux, Truy, Nx, Ny, Nt),
            })
        return constraints

    def _callback(self, xk, state=None):
        self._iter += 1
        if self.display:
            print(f"Iteration = {self._iter}/{self.max_iter}", end="\r", flush=True)

    def _call_solver(self, problem, optimvar, cost_fn, constraints):
        if all(x is None for x in (self.max_in_phase, self.max_quadrature, self.max_amplitude)):
            return minimize(
                cost_fn, optimvar, jac=True, method="L-BFGS-B", callback=self._callback,
                options={"maxiter": self.max_iter, "gtol": self.gtol, "ftol": self.ftol, "maxfun": self.max_fun},
            )
        return minimize(
            cost_fn, optimvar, jac=True, method="trust-constr", constraints=constraints,
            callback=self._callback,
            options={"maxiter": self.max_iter, "xtol": self.xtol, "gtol": self.gtol},
        )

    def _preoptimize_energy_min(self, problem: OptimizationProblem, optimvar):
        weight_max = self.energy_min_weight_max
        if weight_max is None:
            weight_max = problem.reference_amplitude ** 2 * problem.duration ** 2

        saved_max_iter = self.max_iter
        self.max_iter = round(self.max_iter / self.n_suboptim)
        n_preoptim_iter = 0
        try:
            for k in range(self.n_suboptim):
                weight = weight_max * (self.n_suboptim - k - 1) / (self.n_suboptim - 1)
                cost_fn = self._cost_function(problem, weight)
                constraints = self._constraints(problem)
                res = self._call_solver(problem, optimvar, cost_fn, constraints)
                optimvar = res.x
                n_preoptim_iter += res.nit
                if self.display:
                    print(f"Pre-optimization = {k+1}/{self.n_suboptim}, "
                          f"cost (+ penalty) = {res.fun:.6f}", end="\r")
        finally:
            self.max_iter = saved_max_iter
        if self.display:
            print("")
        return optimvar, n_preoptim_iter
