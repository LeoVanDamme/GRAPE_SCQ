from __future__ import annotations

import numpy as np

"""Analytic pulse bases: pass the returned (n_coeffs, n_steps) arrays as
`ux_basis`/`uy_basis` to OptimizationProblem to optimize a handful of Fourier
coefficients instead of the full per-timestep pulse.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
"""


def fourier_basis(n_steps: int, n_coeffs: int) -> np.ndarray:
    """f_0(t)=1, then alternating cos(k*pi*t/T)/sin(k*pi*t/T) harmonics of
    increasing k. Shape (n_coeffs, n_steps)."""
    f = np.zeros((n_coeffs, n_steps))
    if n_coeffs == 0:
        return f
    tau = np.linspace(0, 1, n_steps)
    f[0, :] = 1.0
    k, n = 1, 1
    while n < n_coeffs:
        f[n, :] = np.cos(k * np.pi * tau)
        n += 1
        if n < n_coeffs:
            f[n, :] = np.sin(k * np.pi * tau)
            n += 1
        k += 1
    return f


def symmetric_fourier_basis(n_steps: int, n_coeffs: int) -> np.ndarray:
    """f_n(t) = sin((2n+1)*pi*t/T). Shape (n_coeffs, n_steps)."""
    tau = np.linspace(0, 1, n_steps)
    return np.array([np.sin((2*n+1)*np.pi*tau) for n in range(n_coeffs)])


def antisymmetric_fourier_basis(n_steps: int, n_coeffs: int) -> np.ndarray:
    """f_0(t)=1, f_n(t) = sin(2n*pi*t/T) for n>=1. Shape (n_coeffs, n_steps)."""
    tau = np.linspace(0, 1, n_steps)
    f = np.ones((n_coeffs, n_steps))
    for n in range(n_coeffs):
        f[n, :] = np.sin(2*(n+1)*np.pi*tau)
    return f
