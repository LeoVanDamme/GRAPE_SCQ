import numpy as np
import matplotlib.pyplot as plt
from .utils import FormatDataType, FormatControlInput
from .constraintfuncs import *
from .costfunctions import CostFunctionGate, CostFunctionState

##########################################################################
################## Assign cost function and constraints ##################
def GetCostAndConstr(p,gopt):
    '''
    DESCRIPTION:
        Select and return the appropriate cost function and constraints.

     The function:
        1. Determines whether to use a gate-based or state-based cost function
           depending on the size of p.Target.
        2. Checks if amplitude constraints are enabled.
        3. Returns function handles for cost and constraints.

    INPUTS:
        p    : Structure containing parameters from ProblemParameters, including:
                    p.Target             → Target state or gate
        gopt : Struct containing GRAPE options from Grape, including:
                    gopt.MaxAmplitude    → Maximum amplitude sqrt(Ωx^2+Ωy^2) in Hz
                    gopt.MaxInPhase      → Maximum Ωx pulse component in Hz
                    gopt.MaxQuadrature   → Maximum Ωy pulse component in Hz

    OUTPUTS:
        CostFun   : Function handle to the selected cost function
        ConstrFun : Function handle to constraint function (or [] if no constraint)

     NOTES:
        - Gate fidelity cost is used if p.Target is 2D.
        - State fidelity cost is used if p.Target is a 1D.
        - If no amplitude constraints are set, Constr is returned as [].

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    # Assign type to parameters used in numba
    Nt, NLevels, CompSpace, psi0, \
    Nhp, Nx, Ny, Nc, H0dt, HDdt, Tardagg,\
    uxBasis, uyBasis, TruxBasis, TruyBasis,\
    itH, Weight, nGrad, EnergyPenaltyWeight = FormatDataType(p,gopt)
    
    # Assign Cost function
    if p.IsProblemGate: # Target gate
        CostFun = lambda x: CostFunctionGate(x,Nt, NLevels, CompSpace,
                                Nhp, Nx, Ny, Nc, H0dt, HDdt, Tardagg,
                                uxBasis, uyBasis, TruxBasis, TruyBasis, itH,
                                Weight, nGrad, EnergyPenaltyWeight)
    else: # Target state
        CostFun = lambda x: CostFunctionState(x, Nt, NLevels, psi0,
                                Nhp, Nx, Ny, H0dt, HDdt, Tardagg,
                                uxBasis, uyBasis, TruxBasis, TruyBasis, itH,
                                Weight, nGrad, EnergyPenaltyWeight)

    # Assign constraints    
    ConstrFun = []
    if gopt.MaxAmplitude is not None:
        CAmp = lambda x: CAmplitude(p, gopt, x) 
        GCAmp = lambda x: GCAmplitude(p, x)
        ConstrFun.append({'type': 'ineq', 'fun': CAmp, 'jac': GCAmp})
    if gopt.MaxInPhase is not None:
        CinP = lambda x: CInPhase(p, gopt, x) 
        GCInP = lambda x: GCInPhase(p, x)
        ConstrFun.append({'type': 'ineq', 'fun': CinP, 'jac': GCInP})
    if gopt.MaxQuadrature is not None:
        CQuad = lambda x: CQuadrature(p, gopt, x) 
        GCQuad = lambda x: GCQuadrature(p, x)
        ConstrFun.append({'type': 'ineq', 'fun': CQuad, 'jac': GCQuad})
 
    return CostFun, ConstrFun 

##############################################################
################## Check gradients accuracy ##################
def VerifJacobian(p, gopt, XCtrl,YCtrl):
    '''
    DESCRIPTION:
        Display analytic gradients versus finite-differences

    INPUTS:
        Xctrl  : In-phase component of the pulse. Can be either:
                    - The direct pulse shape ux as a function of time (Nt,)
                    - A row of coefficients a_n if p.uxBasis is provided (Nx,).
        Yctrl  : Quadrature pulse component. Can be either:
                    - The direct pulse shape uy as a function of time (Nt,).
                    - A row of coefficients b_n if p.uyBasis is provided (Ny,).
        p       : Structure containing parameters from ProblemParameters.
        gopt    : Struct containing GRAPE options from Grape.

    OUTPUTS: 
        Figure : plot of analytic gradients versus finite-differences

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    # Validate inputs
    XCtrl, YCtrl = FormatControlInput(p,XCtrl,YCtrl)

    # Concatenate controls
    optimvar=np.concatenate([XCtrl,YCtrl])
    CostFun, _ = GetCostAndConstr(p,gopt)

    # Tell the user about compilation
    print("Compiling cost function...")

    # Analytical gradients
    J, gth = CostFun(optimvar)
    gnum = np.zeros_like(gth)

    # Finite difference gradients
    epsi = 1e-8
    for n in range(len(optimvar)):
        optimvar_tmp = np.copy(optimvar)
        optimvar_tmp[n] = optimvar[n] + epsi
        Jtmp, _ = CostFun(optimvar_tmp)
        gnum[n] = (Jtmp - J) / epsi
        print(f'n={n+1}/{len(optimvar)}')

    # Compute relative error
    absTol = 1e-12
    relErr = np.abs(gth - gnum) / (np.abs(gth) + np.abs(gnum) + absTol)
    maxErr = max(relErr)

    # Plot both the analytical and numerical gradients
    plt.figure()
    plt.plot(gth, label="Analytical Gradients")
    plt.plot(gnum, label="Finite-diff Gradients", linestyle='dashed')
    plt.xlabel('Control parameter')
    plt.ylabel('Gradient')
    plt.title(f'Max ( |g_th-g_num|/(|g_th|+|g_num|) )={maxErr:.6f}')
    plt.legend()
    plt.show()

    return gth, gnum

