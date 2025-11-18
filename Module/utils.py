import numpy as np
from numba import njit
from .config import UseNumba, UseParallel, UseFastMath, UseCache,\
                    dtype_int,dtype_float,dtype_complex

###############################################################
#################### Build numba decorator ####################
def NumbaDecorator(UseNumba = True, UseParallel = False, 
                    UseFastMath = False, UseCache = True):
    '''
    DESCRIPTION:
        Build a decorator for cost functions (and cost functions_utils) based
        on the settings provided in config.py.

    INPUTS:
        UseNumba    : optimize using numba
        Useparallel : use parallel computation (only if UseNumba is True)
        FastMath    : njit option to boost the computation
        UseCache    : Use the cache to avoid re-compilation when optimizing twice

    OUTPUTS:
        usejit      : decorator for compiled functions
        usejitpar   : decorator for parallel compiled function

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    
    if UseNumba and not UseParallel:
        usejit = njit(fastmath=UseFastMath, cache=UseCache)
        usejitpar = usejit
    elif UseNumba and UseParallel:
        usejit = njit(fastmath=UseFastMath, cache=UseCache)
        usejitpar = njit(fastmath=UseFastMath, cache=UseCache, parallel=True)
    elif not UseNumba:
        usejit = lambda f: f
        usejitpar = lambda g: g

    return usejit, usejitpar

# Call the function here to make e decorators directly importable
usejit, usejitpar = NumbaDecorator(UseNumba, UseParallel, UseFastMath, UseCache)

###############################################################
##################### Format pulse inputs #####################
def FormatControlInput(p,Ctrlx,Ctrly):
    '''
    Check if the dimension of the provided control parameters is OK
    and shape it.
    '''
    
    # Make control as array in case of single control parameter
    if np.asarray( Ctrlx ).ndim == 0:
        Ctrlx = np.atleast_1d( Ctrlx )
    if np.asarray( Ctrly ).ndim == 0:
        Ctrly = np.atleast_1d( Ctrly )

    if p.uxBasis.size == 0 and Ctrlx.shape[0] != p.Nt:
        raise TypeError(f"Initialization of Ctrlx must be of shape ({p.Nt},)")
    elif p.uxBasis.size != 0 and Ctrlx.shape[0] != p.Nx:
        raise TypeError(f"Initialization of Ctrlx must be of shape ({p.Nx},)"
                        f" to be consistent with p.uxBasis")
    if p.uyBasis.size==0 and Ctrly.shape[0] != p.Nt:
        raise TypeError(f"Initialization of Ctrly must be of shape ({p.Nt},)")
    elif p.uyBasis.size != 0 and Ctrly.shape[0] != p.Ny:
        raise TypeError(f"Initialization of Ctrly must be of shape ({p.Ny},)"
                        f" to be consistent with p.uyBasis")
    
    return Ctrlx, Ctrly

###############################################################
################## Format datatype for numba ##################
def FormatDataType(p,gopt):
    '''
    DESCRIPTION:
        Assign a data type to parameters passed in the cost functions to avoid 
        ambiguities in Numba.

    INPUTS:
        p    : Struct containing system parameters from ProblemParameters
        gopt : Struct containing optimization options from Grape

    OUTPUTS:
        Nt                  : Number of time steps
        NLevels             : Number of energy levels
        CompSpace           : Computational subspace indices, e.g.: np.array([0,1])
        Psi0                : Initial state (used only for state-to-state problems)
        Nhp                 : Number of Hamiltonian parameter sets
        Nx                  : Number of control parameters for the in-phase pulse component
        Ny                  : Number of control parameters for the quadrature pulse component
        Nc                  : Number of computational states
        H0dt                : Drift Hamiltonian multiplied by dt
        HDdt                : Drive Hamiltonian  multiplied by dt
        Tardagg             : Conjugate-transposed target gate/state for fidelity calculation
        uxBasis/y           : Multiplication matrix for the case of analytically shaped pulses
        TruxBasis/y         : Transpose of uxBasis/y
        itH                 : Array used to reduce memory usage when Nht = 1
        Weight              : Reshaped weights on the cost function
        nGrad               : Number increasing gradients accurracy
        EnergyPenaltyWeight : Penalty weigth on the pulse energy

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    Nt = dtype_int(p.Nt)
    NLevels = dtype_int(p.NLevels)
    CompSpace = p.CompSpace.astype(dtype_int)
    Psi0 = p.Psi0.astype(dtype_complex)
    Nhp = dtype_int(p.Nhp)
    Nx = dtype_int(p.Nx)
    Ny = dtype_int(p.Ny)
    Nc = dtype_int(p.Nc)
    H0dt = p.H0dt.astype(dtype_complex)
    HDdt = p.HDdt.astype(dtype_complex)
    Tardagg = p.Tardagg.astype(dtype_complex)
    uxBasis = p.uxBasis.astype(dtype_float)
    uyBasis = p.uyBasis.astype(dtype_float)
    TruxBasis = p.TruxBasis.astype(dtype_float)
    TruyBasis = p.TruyBasis.astype(dtype_float)
    itH = p.itH.astype(dtype_int)
    Weight = p.Weight.astype(dtype_float)
    nGrad = dtype_int(gopt.nGrad)
    EnergyPenaltyWeight = dtype_float(gopt.EnergyPenaltyWeight)

    return Nt,NLevels,CompSpace,Psi0,Nhp,Nx,Ny,Nc,H0dt,\
        HDdt,Tardagg,uxBasis,uyBasis,TruxBasis,TruyBasis,itH,Weight,\
            nGrad, EnergyPenaltyWeight
