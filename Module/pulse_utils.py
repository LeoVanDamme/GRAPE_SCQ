import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from .parameters import ProblemParameters
from .grape import Grape

##################################################################################
###################### Initialize control parameters #############################
def InitPulse(p, PulseType = "Random", NCoeffs = 3, ThetaTar = np.pi):
    '''
DESCRIPTION:
    Initialize the in-phase and quadrature control parameters based on a chosen 
    initialization scheme. Several pulse types are supported, including:
        - Random     : Random Fourier series.
        - RandSymAnt : Random Fourier series: symmetric ux and antisymmetric uy.
        - CosDrag    : DRAG-like pulse optimized for minimal leakage.
        - Constant   : Constant pulse.
    If analytic pulse shaping is active (p.uxBasis and p.uyBasis not empty),
    the output is given in terms of coefficients for the basis functions.

INPUTS:
    p         : Struct with pulse/system parameters, including:
                    p.Nt        - Number of time steps
                    p.uxBasis   - Basis functions for analytic shaping of ux
                    p.uyBasis   - Basis functions for analytic shaping of uy
                    p.nuRef     - Reference pulse amplitude
                    p.dt        - Time step duration
                    p.Tp        - Pulse duration
    PulseType : Can be "Random", "RandSymAnt",  "CosDrag" or "Constant" 
                (default: 'Random')
    ThetaTar  : Target flip angle (radians) for CosDrag/Constant
                (default: pi)
    NCoeffs   : Number of Fourier coefficients for fourier series
                (default: 3)

OUTPUTS:
    Xctrl : In-phase pulse shape (ux) or parameters: (Nt,) or (Nx,)
    Yctrl : Quadrature pulse shape (uy) or parameters: (Nt,) or (Ny,)
        where Nx and Ny are the number of coefficients when the
        pulse is analytically shaped

NOTES:
    - If PulseType = 'CosDRAG', an internal GRAPE optimization is run to 
    refine the DRAG parameters A and B for better fidelity.
    - If analytic pulse shaping is enabled, Xctrl and Yctrl contain 
    optimized coefficients rather than direct functions of time.

AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
    '''

    print("Initializing pulse...")

    # Initialize matrices
    ux = np.zeros(p.Nt)
    uy = np.zeros(p.Nt)

    # Define direct pulse shapes
    if PulseType == "Random":
        f = FourierBasis(p.Nt,NCoeffs)
        ux = f.T @ (2 * np.random.rand(NCoeffs) - 1) / NCoeffs
        uy = f.T @ (2 * np.random.rand(NCoeffs) - 1) / NCoeffs
    elif PulseType == "RandSymAnt":
        fx = FourierSym(p.Nt,NCoeffs)
        fy = FourierAnt(p.Nt,NCoeffs)
        ux = fx.T @ (2 * np.random.rand(NCoeffs) - 1) / NCoeffs
        uy = fy.T @ (2 * np.random.rand(NCoeffs) - 1) / NCoeffs
    elif PulseType == "Constant":
        ux=ThetaTar*np.ones(p.Nt)/(2*np.pi*p.nuRef*p.Nt*np.mean(p.dt))
        uy=np.zeros(p.Nt)
    elif PulseType == "CosDrag":
        ux, uy, A, B = CosineDRAG(p,ThetaTar=ThetaTar)
    else:
        raise ValueError(f"{PulseType} is not recognized as a pulse initialization"\
                         "method. Choose 'Random', 'RandSymAnt',  'CosDrag' or 'Constant'")
                
    # Convert in the analytical basis if any
    if p.IsShapedX:
        Cfs_guess = 1e-3 * np.ones(p.uxBasis.shape[0])
        Xctrl = FitFunction(ux, p.uxBasis, Cfs_guess )
    else:
        Xctrl = ux
    if p.IsShapedY:
        Cfs_guess = 1e-3 * np.ones(p.uyBasis.shape[0])
        Yctrl = FitFunction(uy, p.uyBasis, Cfs_guess)
    else:
        Yctrl = uy

    print("Pulse initialized.")

    return Xctrl, Yctrl

##################################################################################
####################### Derive a simple cosine DRAGE pulse #######################
def CosineDRAG(p,ThetaTar):
    '''
    DESCRIPTION:
        Derive a Drag-like pulse of the form:
            ux = A*sin( π*t/T )
            uy = B*cos( π*t/T )
        realizing a leakage-free gate of rotation θ_Tar about x in a 3 level
        system driven on-resonance, with duration and anharmonicity
        defined in p.

    INPUT ARGUMENTS:
        p        : Problem parameters defined in ProblemParameters
        ThetaTar : Desired rotation angle

    OUTPUT ARGUMENTS:
        ux, uy, A, B

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    # Time grid (normalized)
    tau = np.linspace(0, 1, p.Nt)
    fx = np.sin(np.pi*tau)
    fy = np.cos(np.pi*tau)
    p_local = ProblemParameters(
                nuRef = p.nuRef, 
                alpha = np.mean(p.alpha),
                dt = np.mean(p.dt),
                Nt = p.Nt,
                uxBasis = fx,
                uyBasis = fy,
                Target = expm(-1j*ThetaTar*np.array([[0,1],[1,0]])/2)
                )
    A0=2*ThetaTar/(p_local.nuRef*p_local.Tp*2*np.pi)
    B0=A0/10
    gopt_local = Grape(Display = False, Maxiter = 500)
    A, B, J = gopt_local.Optimize(p_local,A0,B0)
    ux = p_local.TruxBasis @ A
    uy = p_local.TruyBasis @ B

    return ux, uy, A, B


##################################################################################
############## Fit a function of time using analytical basis #####################
def FitFunction(FTar, FunBasis, Coeffs_ini):
    '''
    DESCRIPTION:
        Fits a target function using a linear combination of user-defined 
        analytical functions. It finds the coefficients that minimize the following 
        cost function:
            C = ∫ |F(t) - ∑ C(n) * h(n,t)|^2 dt
        where:
        - F(t) is the target function to be approximated (FTar)
        - h(n,t) is a set of analytical basis functions (FunBasis)
        - C(n) are the coefficients to be optimized (Coeffs)

    INPUT ARGUMENTS:
        Coeffs_ini  : Initial guess for the coefficients (NCoeffs,)
        FTar        : Target function to fit (Nt,)
        FunBasis    : Analytical set of functions h(n,t) (NCoeffs,Nt)

    OUTPUT ARGUMENTS:
        Coeffs : optimized coefficients (1 x NCoeffs)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    Coeffs_ini = np.atleast_1d(Coeffs_ini)
    J = lambda x: L2norm(FTar, FunBasis, x)
    res = minimize(J,Coeffs_ini,jac=True,method='L-BFGS-B',
                options={ 'maxiter': 500,'gtol': 1e-12,'ftol': 1e-12,
                          'maxfun': 1e12,'disp': False}
                )
    Coeffs = res.x
    return Coeffs

# Helper for FitFunction
def L2norm(FTar, FunBasis, Coeffs):
    F = FunBasis.T @ Coeffs
    J=np.sum((F-FTar)**2)/FTar.size
    GJ = 2 * FunBasis @ (F - FTar) / FTar.size
    return J, GJ

##################################################################################
########################## Examples of analytical bases ##########################

#### Fourier symmetric
def FourierSym(Nt,Ncoeffs):
    '''
    DESCRIPTION:
        Harmonics for analytical pulse shaping. These are symmetric fourier series
        of the form:
            f_n(t) = sin( (2n+1)*π*t/T )

    INPUTS:
        Nt : Number of time steps
        NCoeffs: Number of harmonics

    OUTPUTS:
        2D- array of size (NCoeffs, Nt) representing the harmonics as a 
        function of time

    AUTHOR:
    Leo Van Damme / Technical University of Munich, 2025
    '''

    f=np.zeros((Ncoeffs,Nt))
    tau=np.linspace(0,1,Nt)
    for n in range(Ncoeffs):
        f[n,:] = np.sin((2*n+1)*np.pi*tau)

    return f

#### Fourier antisymmetric
def FourierAnt(Nt,Ncoeffs):
    '''
    DESCRIPTION:
        Harmonics for analytical pulse shaping. These are antisymmetric fourier 
        series of the form:
            f_n(t) = sin( 2n*π*t/T )

    INPUTS:
        Nt : Number of time steps
        NCoeffs: Number of harmonics

    OUTPUTS:
        2D- array of size (NCoeffs, Nt) representing the harmonics as a 
        function of time

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    f=np.zeros((Ncoeffs,Nt))
    tau=np.linspace(0,1,Nt)
    f[0,:] = np.ones((1,Nt))
    for n in range(Ncoeffs):
        f[n,:] = np.sin(2*(n+1)*np.pi*tau)

    return f

#### Fourier
def FourierBasis(Nt,Ncoeffs):
    '''
    DESCRIPTION:
        Harmonics for analytical pulse shaping of the form:
        f_n(t) = cos( k*π*t/T )
        f_n+1(t) = sin( k*π*t/T )

    INPUTS:
        Nt : Number of time steps
        NCoeffs: Number of harmonics

    OUTPUTS:
        2D- array of size (NCoeffs, Nt) representing the harmonics as a 
        function of time

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    f=np.zeros((Ncoeffs,Nt))
    tau=np.linspace(0,1,Nt)
  
    f[0,:] = np.ones((1,Nt))
    k = 1
    for n in range(1,Ncoeffs-1,2):
        f[n,:] = np.cos(k*np.pi*tau)
        f[n+1,:] = np.sin(k*np.pi*tau)
        k=k+1

    f[Ncoeffs-1,:] = np.cos(k*np.pi*tau)

    return f
    


    


