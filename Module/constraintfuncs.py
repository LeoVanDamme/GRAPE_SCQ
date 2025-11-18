import numpy as np

'''
DESCRIPTION:
    Computes amplitude constraints.

INPUTS:
    optimvar : array containing the concatenated control parameters
    p        : Structure containing parameters from ProblemParameters
    gopt     : Optimization options. May include:
                 opt.MaxAmplitude  → Maximum amplitude sqrt(Ωx^2+Ωy^2) in Hz
                 opt.MaxInPhase    → Maximum |Ωx| pulse component in Hz
                 opt.MaxQuadrature → Maximum |Ωy| pulse component in Hz
 
OUTPUTS:
    c  : Inequality constraint (c ≥ 0 must hold).
    gc : Gradients of inequality constraints w.r.t. optimvar.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
'''
#############################################################
############# Maximum amplitude sqrt(Ωx^2+Ωy^2) #############
def CAmplitude(p,gopt,optimvar):
    ux = p.TruxBasis @ optimvar[:p.Nx]
    uy = p.TruyBasis @ optimvar[p.Nx:p.Nx+p.Ny]
    c = (gopt.MaxAmplitude/p.nuRef)**2-ux**2-uy**2
    return c

def GCAmplitude(p,optimvar):
    ux = p.TruxBasis @ optimvar[:p.Nx]
    uy = p.TruyBasis @ optimvar[p.Nx:p.Nx+p.Ny]
    gc = -2 * np.hstack( (np.diag(ux)@p.TruxBasis, np.diag(uy)@p.TruyBasis) )
    return gc

#############################################################
############## Maximum in-phase component |Ωx| ##############
def CInPhase(p,gopt,optimvar):
    ux = p.TruxBasis @ optimvar[:p.Nx]
    c=(gopt.MaxInPhase/p.nuRef)**2-ux**2
    return c

def GCInPhase(p,optimvar):
    ux = p.TruxBasis @ optimvar[:p.Nx]
    gc = -2 * np.hstack( (np.diag(ux)@p.TruxBasis, np.zeros((p.Nt,p.Ny))) )
    return gc

#############################################################
############# Maximum quadrature component |Ωy| #############
def CQuadrature(p,gopt,optimvar):
    uy = p.TruyBasis @ optimvar[p.Nx:p.Nx+p.Ny]
    c=(gopt.MaxQuadrature/p.nuRef)**2-uy**2
    return c

def GCQuadrature(p,optimvar):
    uy = p.TruyBasis @ optimvar[p.Nx:p.Nx+p.Ny]
    gc = -2 * np.hstack( (np.zeros((p.Nt,p.Nx)), np.diag(uy)@p.TruyBasis) )
    return gc
