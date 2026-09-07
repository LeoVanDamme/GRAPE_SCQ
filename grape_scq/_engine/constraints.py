import numpy as np

"""
Amplitude inequality constraints (c >= 0 must hold) and their gradients w.r.t.
the concatenated optimization variable, in normalized units (the pulse is
normalized by the problem's reference amplitude before these are evaluated,
same convention as the cost functions in cost_functions.py).

Unlike the rest of the engine, these are plain (non-jitted) functions: scipy
calls them a handful of times per iteration, not once per timestep, so the
Numba compilation overhead isn't worth paying here.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
"""

def _split(optimvar, TruxBasis, TruyBasis, Nx, Ny):
    ux = TruxBasis @ optimvar[:Nx]
    uy = TruyBasis @ optimvar[Nx:Nx+Ny]
    return ux, uy

def amplitude_constraint(optimvar, TruxBasis, TruyBasis, Nx, Ny, max_amplitude, reference_amplitude):
    ux, uy = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return (max_amplitude/reference_amplitude)**2 - ux**2 - uy**2

def amplitude_constraint_jac(optimvar, TruxBasis, TruyBasis, Nx, Ny):
    ux, uy = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return -2 * np.hstack((np.diag(ux) @ TruxBasis, np.diag(uy) @ TruyBasis))

def in_phase_constraint(optimvar, TruxBasis, TruyBasis, Nx, Ny, max_in_phase, reference_amplitude):
    ux, _ = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return (max_in_phase/reference_amplitude)**2 - ux**2

def in_phase_constraint_jac(optimvar, TruxBasis, TruyBasis, Nx, Ny, Nt):
    ux, _ = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return -2 * np.hstack((np.diag(ux) @ TruxBasis, np.zeros((Nt, Ny))))

def quadrature_constraint(optimvar, TruxBasis, TruyBasis, Nx, Ny, max_quadrature, reference_amplitude):
    _, uy = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return (max_quadrature/reference_amplitude)**2 - uy**2

def quadrature_constraint_jac(optimvar, TruxBasis, TruyBasis, Nx, Ny, Nt):
    _, uy = _split(optimvar, TruxBasis, TruyBasis, Nx, Ny)
    return -2 * np.hstack((np.zeros((Nt, Nx)), np.diag(uy) @ TruyBasis))
