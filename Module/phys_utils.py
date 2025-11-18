import numpy as np
from .config import dtype_complex
import copy as cp
from scipy.sparse.linalg import expm, expm_multiply

#######################################################################
########################## Propagate state  ###########################
def PropState(p,Psi0,ux,uy):
    '''
    DESCRIPTION:
        Propagate a state forward under the Hamiltonians defined in p and the
        pulse defined by ux and uy

    INPUTS:
        p    : Struct with pulse/system parameters, including:
                Nhp  : Number of parameter sets
                Nt   : Number of time steps
                H0dt : Drift Hamiltonian multiplied by dt
                        - Shape (NLevels, NLevels, Nht, Nhp)
                HDdt : Drive Hamiltonian  multiplied by dt
                        - Shape (NLevels, NLevels, Nht, Nhp)
        ux   : Normalized in-phase pulse shape: (Nt,)
        uy   : Normalized quadrature pulse shape: (Nt,)
        Psi0 : Initial state (Nlevels,) or (NLevels*Nhp,)

    OUPUTS:
        Psi : State as a function of time (NLevels, Nt+1, Nhp)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025      
    '''

    pl = cp.deepcopy(p)
    pl.Set(Psi0 = Psi0)
    Psi = np.zeros((pl.NLevels,pl.Nt+1,pl.Nhp),dtype=dtype_complex)
    Psi[:,0,:] = pl.Psi0

    u = ux + 1j*uy
    for k in range(pl.Nhp):
        for n in range(pl.Nt):
            Hdrift = p.H0dt[:,:,pl.itH[n],k]
            HDrive = u[n]*p.HDdt[:,:,pl.itH[n],k] 
            Hdt = Hdrift + HDrive + HDrive.conj().T
            Psi[:,n+1,k] = expm_multiply(-1j*Hdt,Psi[:,n,k])

    return Psi

#######################################################################
##################### Fidelity per parameter set  #####################
def GateFidelityMap(p,ux,uy):
    '''
    DESCRIPTION:
        Compute the gate fidelity map under a pulse defined by ux and uy. 

    INPUTS:
        p   : Struct with pulse/system parameters, including:
                Nt        - Number of timesteps
                NLevels   - Dimension of the Hilbert space
                CompSpace - Indices of the computational states

        ux  : In phase pulse shape: (Nt,)
        uy  : Quadrature pulse shape: (Nt,)

    OUTPUTS:
        F   : Fidelity values for each parameter set (Nhp,)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    if not p.IsProblemGate:
        raise ValueError("GateFidelityMap applies only when the Target " \
        "defined in ProblemParameters is a gate.")
    
    F = np.zeros(p.Nhp)+10
    u = ux + 1j*uy
    for k in range(p.Nhp):
        print(f"Computation of the gate fidelity map {k+1}/{p.Nhp}", end = '\r',flush = True)
        U = np.eye(p.NLevels,dtype=dtype_complex)
        for n in range(p.Nt):
            Hdrift = p.H0dt[:,:,p.itH[n],k]
            HDrive = u[n]*p.HDdt[:,:,p.itH[n],k] 
            Hdt = Hdrift + HDrive + HDrive.conj().T
            U = expm(-1j*Hdt) @ U
        Uc = U[np.ix_(p.CompSpace,p.CompSpace)]
        TrUTarUF = np.trace(p.Tardagg[:,:,k]@Uc)
        F[k] = (np.abs(TrUTarUF)/p.Nc)**2
    
    print("")

    return F

#######################################################################
###################### Measurement probability  #######################
def StateProbability(p,ux,uy,InitState,MeasState):
    '''
    DESCRIPTION:
        Computes the population (probability) of a specific quantum state 
        as the system evolves under the control pulse. The state is 
        propagated using the system Hamiltonian, and the overlap with the 
        measured state is computed at each time step.
        The output is the squared magnitude of the inner product:
                                |⟨Ψ_m | Ψ_t⟩|²
        where:
        - |Ψ_t> is the evolved state starting from |Ψ_0>='InitState'
        - |Ψ_m> is the measured state vector 'MeasState'

    INPUTS
        p       : Struct with pulse/system parameters, including:
                    Nt        - Number of timesteps
                    Nhp       - Number of parameter sets
                    NLevels   - Dimension of the Hilbert space
                    CompSpace - Indices of the computational states

        ux      : In phase pulse shape: (Nt,)
        uy      : Quadrature pulse shape: (Nt,)
    InitState   : Initial state vector (default: ground state |0>)
                  Vector of shape (NLevels,)
    MeasState   : Measured state vector (default: ground state |0>)
                  Vector of shape (NLevels,)

    OUTPUTS:
        Pop : Population (probability) of the measured state as a function of time
            - Size: (Nt+1, Nhp) where Nhp is the number of parameter sets
                    and Nt + 1 is the number of sampled time points 
                    (including t=0 and t=final)
              Pop(t, k) = probability at time step t for parameter set k     

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025 
    '''
    MeasState = np.asarray(MeasState)
    ProbaIM = np.zeros((p.Nt+1, p.Nhp))
    Psi = PropState(p,InitState,ux,uy)
    for k in range (p.Nhp):
        for n in range(p.Nt+1):
            ProbaIM[n,k] = abs(Psi[:,n,k].conj().T @ MeasState)**2

    ProbaIM = np.squeeze(ProbaIM)
    return ProbaIM

#######################################################################
###################### Construct an eigenstate  #######################
def EigState(StateIndice,NLevels):
    '''
    DESCRIPTION:
        Generates a basis state vector |n⟩ in an NLevels-dimensional Hilbert space,
        where the specified state index is given as a string (e.g., '0', '1', ...).
    
    INPUT ARGUMENTS:
        State       : String specifying the desired state ('0' → |0⟩, '1' → |1⟩, etc.)
        NLevels     : Number of levels in the full Hilbert space

    OUTPUT ARGUMENT:
        psi         : Vector (NLevels,) representing the chosen basis state.
    
    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    Psi = np.zeros(NLevels,dtype=dtype_complex)
    AssignedPsi = False

    for n in range(NLevels):
        if StateIndice == f"{n}" or StateIndice == f"|{n}>":
            Psi[n] = 1
            AssignedPsi = True

    if not AssignedPsi:
        raise ValueError("First argument must be a string of the form " \
        "'0' (or '|0>'), '1' (or '|1>') etc...")
    
    return Psi
