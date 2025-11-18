import numpy as np
from .config import dtype_float, dtype_complex
from .utils import usejit

###############################################################################
####################### Gradients for gate optimization #######################
@usejit
def ComputeJacobianGate(u,UF,H0dt_k,HDdt_k,Vd,Nt,itH,nGrad):
    """
    DESCRIPTION:
        Compute the gradient (Jacobian) of the gate cost function with respect 
        to control amplitudes ux and uy at each time step. The Jacobian provides 
        dJ/dux and dJ/duy.
        The helper function ComputeDexp is used to approximate the derivative of
        matrix exponentials.

    INPUTS:
        u         : Control pulse in complex form u = ux + i·uy (Nt,)
        UF        : Forward-propagated unitary at final time (NLevels, NLevels)
        H0dt_k    : Drift part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        HDdt_k    : Drive part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        Vd        : Adjoin evolution operator (NLevels, NLevels)
        Nt        : Number of timesteps
        itH       : Allows for adapting to the size of H0dt_k and HDdt_k (Nt,)
        nGrad     : Number increasing the accuracy of matrix exponential derivatives 

    OUTPUTS:
        dJdux     : Gradient of cost function w.r.t. ux at each timestep (Nt,)
        dJduy     : Gradient of cost function w.r.t. uy at each timestep (Nt,)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    dJdux = np.zeros(Nt,dtype=dtype_float)
    dJduy = np.zeros(Nt,dtype=dtype_float)
    U = UF
    for n in range(Nt-1, -1, -1):
        H0dt_kn = np.ascontiguousarray( H0dt_k[:,:,itH[n]] )
        HDdt_kn = np.ascontiguousarray( HDdt_k[:,:,itH[n]] )
        Hdt_kn= H0dt_kn + u[n]*HDdt_kn+ (u[n]*HDdt_kn).conj().T 
        Pn = ExpmH(-1j*Hdt_kn)
        cPn = np.conj(Pn).T
        Dx, Dy = ComputeDExp(Hdt_kn,HDdt_kn,nGrad)
        U = np.ascontiguousarray(U)

        dJdux[n]=-np.real( np.trace( Vd@Dx@U ) )
        dJduy[n]=-np.real( np.trace( Vd@Dy@U ) )

        Vd = Vd @ Pn
        U = cPn @ U

    return dJdux, dJduy

###############################################################################
################# Gradients for state-to-state optimization ###################
@usejit
def ComputeJacobianState(u,psiF,H0dt_k,HDdt_k,chid,Nt,itH,nGrad):
    """
    DESCRIPTION:
        Compute the gradient (Jacobian) of the state-to-state cost function with
        respect to control amplitudes ux and uy at each time step. The Jacobian 
        provides dJ/dux and dJ/duy.
        The helper function ComputeDexp is used to approximate the derivative of
        matrix exponentials.

    INPUTS:
        u         : Control pulse in complex form u = ux + i·uy (Nt,)
        psiF      : Forward-propagated state at final time (NLevels,)
        H0dt_k    : Static part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        HDdt_k    : Driven part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        chid      : Adjoin state (1, NLevels)
        Nt        : Number of timesteps
        itH       : Allows for adapting to the size of H0dt_k and HDdt_k (Nt,)
        nGrad     : Number increasing the accuracy of matrix exponential derivatives 

    OUTPUTS:
        dJdux     : Gradient of cost function w.r.t. ux at each timestep (Nt,)
        dJduy     : Gradient of cost function w.r.t. uy at each timestep (Nt,)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    dJdux = np.zeros(Nt,dtype=dtype_float)
    dJduy = np.zeros(Nt,dtype=dtype_float)
    psi = psiF
    for n in range(Nt-1, -1, -1):
        H0dt_kn = np.ascontiguousarray( H0dt_k[:,:,itH[n]] )
        HDdt_kn = np.ascontiguousarray( HDdt_k[:,:,itH[n]] )
        Hdt_kn= H0dt_kn + u[n]*HDdt_kn+ (u[n]*HDdt_kn).conj().T 
        Pn = ExpmH(-1j*Hdt_kn)
        cPn = np.conj(Pn).T
        Dx, Dy = ComputeDExp(Hdt_kn,HDdt_kn,nGrad)

        dJdux[n]=-np.real( chid@Dx@psi ).item()
        dJduy[n]=-np.real( chid@Dy@psi ).item()

        chid = chid @ Pn
        psi = cPn @ psi

    return dJdux, dJduy

###############################################################################
################ Forward propagation of evolution operator ####################
@usejit
def PropForward(u,H0dt_k,HDdt_k,Nt,NLevels,itH):
    '''
    DESCRIPTION:
        Propagate the evolution operator U forward under the pulse u = ux + iuy.

    INPUTS:
        u       : Control pulse in complex form u = ux + i·uy (Nt,)
        H0dt_k  : Static part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        HDdt_k  : Driven part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        Nt      : Number of timesteps.
        NLevels : Dimension of the Hilbert space.
        itH     : Array used to reduce memory usage when hamiltonian parameters are 
                  not time-dependent
    OUTPUTS:
        U       : Evolution operator at final time (NLevels, NLevels).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    U=np.eye(NLevels, dtype=dtype_complex)
    for n in range(Nt):
        Hdt_kn=H0dt_k[:,:,itH[n]] + u[n]*HDdt_k[:,:,itH[n]] + (u[n]*HDdt_k[:,:,itH[n]]).conj().T
        P = ExpmH(-1j*Hdt_kn)
        U = P  @ U  

    U = U
    return U

###############################################################################
##################### Derivative of matrix exponential ########################
@usejit
def ComputeDExp(Hdt_kn,HDdt_kn,nGrad):
    '''
    DESCRIPTION:
        Compute approximate derivatives of a unitary propagator P with respect to
        control amplitudes ux and uy. Specifically, this function evaluates 
        (∂P/∂ux)·P† and (∂P/∂uy)·P†, where:

            P    = exp(-iHdt)
            H    = system Hamiltonian
            dt   = timestep duration

        The analytical expressions for these derivatives are given by:

            (∂P/∂ux)·P† = ∫ exp(-iHdt·s)·(-i∂H/∂ux)·exp(iHdt·s)·ds
            (∂P/∂uy)·P† = ∫ exp(-iHdt·s)·(-i∂H/∂uy)·exp(iHdt·s)·ds

        evaluated on s ∈ [0,1]. The computation uses a numerical integration method 
        with nGrad subdivisions to improve accuracy.

    INPUTS:
        Hdt_kn  : Effective Hamiltonian Hn * dt at timestep n (NLevels, NLevels)
        HDdt_kn : Drive Hamiltonian H_drive * dt (NLevels, NLevels)
        nGrad   : Number of integration steps used to approximate the derivative

    OUTPUTS:
        Dx     : Approximation of (∂P/∂ux)·P† (NLevels, NLevels)
        Dy     : Approximation of (∂P/∂uy)·P† (NLevels, NLevels)

    NOTES:
        - midHdux represents -i∂H/∂ux 
        - midHduy represents -i∂H/∂uy

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    midHdux = -1j*( HDdt_kn+HDdt_kn.conj().T )
    midHduy = HDdt_kn-HDdt_kn.conj().T
    Dx = np.zeros_like(Hdt_kn)
    Dy = np.zeros_like(Hdt_kn)

    for l in range(nGrad+1):
        s = (l+0.5)/(nGrad+1)
        P = ExpmH(-1j * Hdt_kn * s)
        cP = P.conj().T
        dUx = P @ (midHdux @ cP)
        dUy = P @ (midHduy @ cP)
        Dx = Dx + 1/(nGrad+1) * dUx
        Dy = Dy + 1/(nGrad+1) * dUy

    return Dx, Dy

###############################################################################
############################# Matrix exponential ##############################
@usejit
def ExpmH(miHdt):
    """
    DESCRIPTION:
        Computes the matrix exponential of a skew-Hermitian matrix -i·H·dt, 
        where H is typically a hermitian Hamiltonian.

    INPUTS:
        miHdt : Skew-hermitian matrix -i·H·dt (NLevels, NLevels).

    OUTPUTS:
        expK : matrix exponential of miHdt (NLevels, NLevels).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """
    H = 1j * miHdt
    w, V = np.linalg.eigh(H)
    expK = V @ np.diag(np.exp(-1j * w)) @ V.conj().T

    return expK

###############################################################################
#################### Extract computational space operator #####################
@usejit
def ExtractUFComp(UF, CompSpace):
    """
    DESCRIPTION:
        Extracts a submatrix from the evolution operator UF based on the indices 
        provided in CompSpace. Note that the extracted operator is not necessarilly 
        unitary.

    INPUTS:
        UF        : Matrix from which elements will be extracted (NLevels, NLevels).
        CompSpace : Computational subspace indices. Shape (Nc,) where Nc is the 
                    number of computational states
    
    OUTPUTS:
        UFc : Extrated matrix (Nc, Nc).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """
    n = len(CompSpace)
    UFc = np.empty((n, n), dtype=dtype_complex)
    for i in range(n):
        for j in range(n):
            UFc[i, j] = UF[CompSpace[i], CompSpace[j]]
    return UFc

###############################################################################
#################### Extract computational ajoin operator #####################
@usejit
def AssignVdCom(MFull, CompSpace, MComp):
    """
    DESCRIPTION:
        Modifies a subspace of a bigger matrix MFull by assigning a submatrix MComp
        according to the subspace indices defined in CompSpace.

    INPUTS:
        MFull     : Matrix in which a submatrix is assigned (NLevels, NLevels).
        CompSpace : Computational subspace indices. Shape (Nc,) where Nc is the 
                    number of computational states.
        MComp     : Sub-matrix to be assigned. Shape (Nc, Nc).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """
    n = len(CompSpace)
    for i in range(n):
        for j in range(n):
            MFull[CompSpace[i], CompSpace[j]] = MComp[i, j]
