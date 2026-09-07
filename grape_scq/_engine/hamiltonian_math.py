import numpy as np
from .dtypes import dtype_float, dtype_complex
from .jit import jit

###############################################################################
################ Forward propagation of evolution operator ####################
@jit
def PropForward(u,H0dt_k,HDdt_k,Nt,NLevels,itH):
    '''
    DESCRIPTION:
        Propagate the evolution operator U forward under the pulse u = ux + iuy.
        Alongside the final propagator, this also caches the eigendecomposition
        (V_n, w_n) of the effective Hamiltonian and the propagator P_n at every
        timestep n. Both the backward adjoint sweep (ComputeJacobianGate/State)
        and the exact matrix-exponential derivative (ComputeDExp) need this same
        eigendecomposition, so computing it once here and reusing it avoids
        redoing the dominant O(NLevels^3) eigh cost up to three times per
        timestep.

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
        Pall    : Propagator P_n = exp(-i*Hdt_n) at every timestep (NLevels, NLevels, Nt).
        Vall    : Eigenvectors of Hdt_n at every timestep (NLevels, NLevels, Nt).
        Wall    : Eigenvalues of Hdt_n at every timestep (NLevels, Nt).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    U = np.eye(NLevels, dtype=dtype_complex)
    Pall = np.zeros((NLevels,NLevels,Nt), dtype=dtype_complex)
    Vall = np.zeros((NLevels,NLevels,Nt), dtype=dtype_complex)
    Wall = np.zeros((NLevels,Nt), dtype=dtype_float)
    for n in range(Nt):
        Hdt_kn = H0dt_k[:,:,itH[n]] + u[n]*HDdt_k[:,:,itH[n]] + (u[n]*HDdt_k[:,:,itH[n]]).conj().T
        w, V = np.linalg.eigh(Hdt_kn)
        P = V @ np.diag(np.exp(-1j*w)) @ V.conj().T
        Pall[:,:,n] = P
        Vall[:,:,n] = V
        Wall[:,n] = w
        U = P @ U

    return U, Pall, Vall, Wall

###############################################################################
####################### Gradients for gate optimization #######################
@jit
def ComputeJacobianGate(UF,HDdt_k,Vd,Nt,itH,Pall,Vall,Wall):
    """
    DESCRIPTION:
        Compute the gradient (Jacobian) of the gate cost function with respect
        to control amplitudes ux and uy at each time step. The Jacobian provides
        dJ/dux and dJ/duy.
        The helper function ComputeDExp is used to compute the exact derivative
        of the matrix exponential at each timestep, from the eigendecomposition
        cached by PropForward.

    INPUTS:
        UF        : Forward-propagated unitary at final time (NLevels, NLevels)
        HDdt_k    : Drive part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        Vd        : Adjoin evolution operator (NLevels, NLevels)
        Nt        : Number of timesteps
        itH       : Allows for adapting to the size of HDdt_k (Nt,)
        Pall      : Cached propagators from PropForward (NLevels, NLevels, Nt)
        Vall      : Cached eigenvectors from PropForward (NLevels, NLevels, Nt)
        Wall      : Cached eigenvalues from PropForward (NLevels, Nt)

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
        HDdt_kn = np.ascontiguousarray( HDdt_k[:,:,itH[n]] )
        Pn = np.ascontiguousarray( Pall[:,:,n] )
        V = np.ascontiguousarray( Vall[:,:,n] )
        w = Wall[:,n]
        cPn = np.conj(Pn).T
        Dx, Dy = ComputeDExp(w,V,HDdt_kn)
        U = np.ascontiguousarray(U)

        dJdux[n]=-np.real( np.trace( Vd@Dx@U ) )
        dJduy[n]=-np.real( np.trace( Vd@Dy@U ) )

        Vd = Vd @ Pn
        U = cPn @ U

    return dJdux, dJduy

###############################################################################
################# Gradients for state-to-state optimization ###################
@jit
def ComputeJacobianState(psiF,HDdt_k,chid,Nt,itH,Pall,Vall,Wall):
    """
    DESCRIPTION:
        Compute the gradient (Jacobian) of the state-to-state cost function with
        respect to control amplitudes ux and uy at each time step. The Jacobian
        provides dJ/dux and dJ/duy.
        The helper function ComputeDExp is used to compute the exact derivative
        of the matrix exponential at each timestep, from the eigendecomposition
        cached by PropForward.

    INPUTS:
        psiF      : Forward-propagated state at final time (NLevels,)
        HDdt_k    : Driven part of Hamiltonian (NLevels, NLevels, 1 or Nt)
        chid      : Adjoin state (1, NLevels)
        Nt        : Number of timesteps
        itH       : Allows for adapting to the size of HDdt_k (Nt,)
        Pall      : Cached propagators from PropForward (NLevels, NLevels, Nt)
        Vall      : Cached eigenvectors from PropForward (NLevels, NLevels, Nt)
        Wall      : Cached eigenvalues from PropForward (NLevels, Nt)

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
        HDdt_kn = np.ascontiguousarray( HDdt_k[:,:,itH[n]] )
        Pn = np.ascontiguousarray( Pall[:,:,n] )
        V = np.ascontiguousarray( Vall[:,:,n] )
        w = Wall[:,n]
        cPn = np.conj(Pn).T
        Dx, Dy = ComputeDExp(w,V,HDdt_kn)

        dJdux[n]=-np.real( chid@Dx@psi ).item()
        dJduy[n]=-np.real( chid@Dy@psi ).item()

        chid = chid @ Pn
        psi = cPn @ psi

    return dJdux, dJduy

###############################################################################
##################### Derivative of matrix exponential ########################
@jit
def ExpSincKernel(Wmat):
    '''
    DESCRIPTION:
        Elementwise kernel K(x) = i*(exp(-i*x)-1)/x, with K(0) = 1, used to build
        the exact Frechet derivative of a matrix exponential from the eigenvalue
        gaps of the underlying Hamiltonian. It corresponds to the closed form of
        the integral  K(w_i-w_j) = ∫_0^1 exp(-i(w_i-w_j)s) ds .

    INPUTS:
        Wmat : Matrix of eigenvalue differences w_i - w_j (NLevels, NLevels)

    OUTPUTS:
        K    : Elementwise kernel matrix (NLevels, NLevels)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    n = Wmat.shape[0]
    K = np.zeros((n,n), dtype=dtype_complex)
    for i in range(n):
        for j in range(n):
            x = Wmat[i,j]
            if x == 0:
                K[i,j] = 1.0
            else:
                K[i,j] = 1j*(np.exp(-1j*x)-1)/x
    return K

@jit
def ComputeDExp(w,V,HDdt_kn):
    '''
    DESCRIPTION:
        Compute the exact derivatives of a unitary propagator P with respect to
        control amplitudes ux and uy. Specifically, this function evaluates
        (∂P/∂ux)·P† and (∂P/∂uy)·P†, where:

            P    = exp(-iHdt)
            H    = system Hamiltonian
            dt   = timestep duration

        The analytical expressions for these derivatives are given by:

            (∂P/∂ux)·P† = ∫ exp(-iHdt·s)·(-i∂H/∂ux)·exp(iHdt·s)·ds
            (∂P/∂uy)·P† = ∫ exp(-iHdt·s)·(-i∂H/∂uy)·exp(iHdt·s)·ds

        evaluated on s ∈ [0,1]. Rather than approximating this integral by
        quadrature, it is evaluated in closed form using the eigendecomposition
        Hdt = V·diag(w)·V† (Daleckii-Krein / divided-difference formula for the
        Frechet derivative of the matrix exponential):

            (∂P/∂ux)·P† = V·[ (V†·(-i∂H/∂ux)·V) ⊙ K(W) ]·V†
            (∂P/∂uy)·P† = V·[ (V†·( ∂H/∂uy)·V) ⊙ K(W) ]·V†

        where W_ij = w_i - w_j, ⊙ is the elementwise (Hadamard) product, and
        K is the ExpSincKernel.

    INPUTS:
        w       : Eigenvalues of Hdt_kn, from PropForward (NLevels,)
        V       : Eigenvectors of Hdt_kn, from PropForward (NLevels, NLevels)
        HDdt_kn : Drive Hamiltonian H_drive * dt (NLevels, NLevels)

    OUTPUTS:
        Dx     : Exact (∂P/∂ux)·P† (NLevels, NLevels)
        Dy     : Exact (∂P/∂uy)·P† (NLevels, NLevels)

    NOTES:
        - midHdux represents -i∂H/∂ux
        - midHduy represents -i∂H/∂uy
        - Hdt_kn is Hermitian by construction, so w and V come from an eigh
          decomposition (real eigenvalues, unitary eigenvector matrix).

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''

    cV = V.conj().T

    n = V.shape[0]
    Wmat = np.zeros((n,n), dtype=dtype_float)
    for i in range(n):
        for j in range(n):
            Wmat[i,j] = w[i]-w[j]
    K = ExpSincKernel(Wmat)

    midHdux = -1j*( HDdt_kn+HDdt_kn.conj().T )
    midHduy = HDdt_kn-HDdt_kn.conj().T

    Bx = cV @ midHdux @ V
    By = cV @ midHduy @ V

    Dx = V @ (Bx*K) @ cV
    Dy = V @ (By*K) @ cV

    return Dx, Dy

###############################################################################
############################# Matrix exponential ##############################
@jit
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
@jit
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
@jit
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
