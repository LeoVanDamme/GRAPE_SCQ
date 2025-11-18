import numpy as np
from numba import prange
from .config import dtype_float, dtype_complex
from .utils import usejitpar
from .costfunctions_utils import *

####################################################################################
############################# Gate cost and gradients ##############################
@usejitpar
def CostFunctionGate(optimvar,Nt,NLevels,CompSpace,Nhp,Nx,Ny,Nc,H0dt,HDdt,
                     Tardagg,uxBasis,uyBasis,TruxBasis,TruyBasis,itH,Weight,nGrad,
                     EnergyPenaltyWeight):
    '''
    DESCRIPTION:
        Computes the cost and gradient for minimizing the infidelity of the 
        implemented  gate as compared to a target gate in the computational
        space. If there are Nc computational states, it is defined as 
                       J = 1 - |Tr(U_Tar'*U_F)|^2 / Nc 
        where U_Tar is a Nc x Nc Target gate and U_F is the evolution operator on 
        the computational space.

    INTPUTS:
        optimvar            : array containing the concatenated control parameters
        Nt                  : Number of time steps
        NLevels             : Number of energy levels
        CompSpace           : Computational subspace indices, e.g.: [0,1]
        Nhp                 : Number of parameter sets
        Nx                  : Number of control parameters for Ωx 
        Ny                  : Number of control parameters for Ωy
        Nc                  : Number of computational states
        H0dt                : Drift Hamiltonian multiplied by dt
        HDdt                : Drive Hamiltonian multiplied by dt
        Tardagg             : Conjugate-transposed target gate
        uxBasis/y           : Analytical basis for shaped pulses
        TruxBasis/y         : Transpose of uxBasis/y
        itH                 : Array used to reduce memory usage when Nht = 1
        Weight              : Reshaped weights on the cost function
        nGrad               : Number increasing gradients accurracy
        EnergyPenaltyWeight : Penalty weigth on the pulse energy

    OUTPUTS:
        J  : Cost value (average infidelity)
        GJ : Gradients: shape (Nx+Ny,:)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    
    ######## Assign right type to optimvar
    optimvar = optimvar.astype(dtype_float)

    ######## Normalized pulse in complex form
    ux = TruxBasis @ optimvar[:Nx]
    uy = TruyBasis @ optimvar[Nx:Nx+Ny]
    u = ux + 1j*uy

    ######## Forward propagation and cost computation
    Jk = np.zeros(Nhp, dtype=dtype_float)
    dJdxvar = np.zeros((Nhp,Nx), dtype=dtype_float)
    dJdyvar = np.zeros((Nhp,Ny), dtype=dtype_float) 

    for k in prange(Nhp):

        H0dt_k=np.ascontiguousarray( H0dt[:,:,:,k] )
        HDdt_k=np.ascontiguousarray( HDdt[:,:,:,k] )

        # Forward propagation and cost per parameter set
        UF=PropForward(u,H0dt_k,HDdt_k,Nt,NLevels,itH)          # Ev. operator in full Hilbert space
        UFc=np.ascontiguousarray( ExtractUFComp(UF,CompSpace) ) # Ev. operator in computational space
        UTarDagg=np.ascontiguousarray(Tardagg[:,:,k])           # U_Tar† (in computational space)
        TrUTUF=np.trace(UTarDagg @ UFc)                         # Tr(U_Tar†*U_Fc) in computational space
        Jk[k]=1-(np.abs(TrUTUF)/Nc)**2                          # Cost for parameter set k
    
        # Computation of the gradients using GRAPE
        Vd = np.zeros((NLevels,NLevels), dtype=dtype_complex)
        cTrUTUF = np.conj(TrUTUF)
        Vdc = 2*UTarDagg*cTrUTUF/(Nc**2)
        AssignVdCom(Vd,CompSpace,Vdc)
        Vd = np.ascontiguousarray(Vd)
        dJdux, dJduy = ComputeJacobianGate(u,UF,H0dt_k,HDdt_k,Vd,Nt,itH,nGrad)

        # Adapt gradients to parametrized pulses
        dJdxvar[k,:] = uxBasis @ dJdux
        dJdyvar[k,:] = uyBasis @ dJduy


    J = Weight @ Jk / Nhp
    Gx = Weight @ dJdxvar / Nhp
    Gy = Weight @ dJdyvar / Nhp


    if EnergyPenaltyWeight != 0:
        # Add penalty on pulse energy
        J=J+EnergyPenaltyWeight*np.sum(ux**2+uy**2)/Nt
        Gx=Gx+2*EnergyPenaltyWeight*uxBasis@ux/Nt
        Gy=Gy+2*EnergyPenaltyWeight*uyBasis@uy/Nt

    G=np.concatenate((Gx.ravel(),Gy.ravel()))

    return J, G

####################################################################################
######################## State-to-state cost and gradients #########################
@usejitpar
def CostFunctionState(optimvar,Nt,NLevels,psi0,Nhp,Nx,Ny,H0dt,HDdt,
                     Tardagg,uxBasis,uyBasis,TruxBasis,TruyBasis,itH,Weight,nGrad,
                     EnergyPenaltyWeight):
    '''
    DESCRIPTION:
        Computes the cost and gradient for minimizing a state-to-state infidelity 
        defined as 1 - |<psi_Tar | psi_F>|^2.

    INTPUTS:
        optimvar            : array containing the concatenated control parameters
        Nt                  : Number of time steps
        NLevels             : Number of energy levels
        Psi0                : Initial state
        Nhp                 : Number of parameter sets
        Nx                  : Number of control parameters for Ωx 
        Ny                  : Number of control parameter for Ωy
        H0dt                : Drift Hamiltonian multiplied by dt
        HDdt                : Drive Hamiltonian  multiplied by dt
        Tardagg             : Conjugate-transposed target gate
        uxBasis/y           : Analytical basis for shaped pulses
        TruxBasis/y         : Transpose of uxBasis/y
        itH                 : Array used to reduce memory usage when Nht = 1
        Weight              : Reshaped weights on the cost function
        nGrad               : Number increasing gradients accurracy
        EnergyPenaltyWeight : Penalty weigth on the pulse energy

    OUTPUTS:
        J  : Cost value (average infidelity)
        GJ : Gradients: shape (Nx+Ny,:)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    '''
    
    ######## Assign right type to optimvar ########
    optimvar = optimvar.astype(dtype_float)

    ######## Normalized pulse in complex form ########
    ux = TruxBasis @ optimvar[:Nx]
    uy = TruyBasis @ optimvar[Nx:Nx+Ny]
    u = ux + 1j*uy

    ######## Forward propagation and cost computation ########
    Jk = np.zeros(Nhp, dtype=dtype_float)
    dJdxvar = np.zeros((Nhp,Nx), dtype=dtype_float)
    dJdyvar = np.zeros((Nhp,Ny), dtype=dtype_float) 

    for k in prange(Nhp):

        H0dt_k=np.ascontiguousarray( H0dt[:,:,:,k] )
        HDdt_k=np.ascontiguousarray( HDdt[:,:,:,k] )

        # Forward propagation and cost per parameter set
        UF=PropForward(u,H0dt_k,HDdt_k,Nt,NLevels,itH)   # Ev. operator in full Hilbert space
        psik0 = np.ascontiguousarray(psi0[:,k])          # Initial state |psi_0> for parameter set k
        psiF = UF @ psik0                                # Final state |psi_F>=U_F|psi_0>
        Tardaggk = np.ascontiguousarray(Tardagg[:,:,k])  # Target state in bra <psi_Tar|
        PsiTPsiF = np.dot(Tardaggk,psiF).item();         # Scalar product <psi_Tar|psi_F>
        Jk[k]=1.0-(np.abs(PsiTPsiF))**2                  # Cost for parameter set k

        # Computation of the gradients using GRAPE
        cPsiTPsiF = np.conj(PsiTPsiF)                    # Conjugate of <psi_Tar|psi_F>
        Chid=2*Tardaggk*cPsiTPsiF                        # Adjoin state in bra <chi|
        dJdux, dJduy = ComputeJacobianState(u,psiF,H0dt_k,HDdt_k,Chid,Nt,itH,nGrad)
        
        # Adapt gradients to parametrized pulses
        dJdxvar[k,:] = uxBasis @ dJdux
        dJdyvar[k,:] = uyBasis @ dJduy

    J =  Weight @ Jk / Nhp
    Gx = Weight @ dJdxvar / Nhp
    Gy = Weight @ dJdyvar / Nhp

    if EnergyPenaltyWeight != 0:
        # Add penalty on pulse energy
        J=J+EnergyPenaltyWeight*np.sum(ux**2+uy**2)/Nt
        Gx=Gx+2*EnergyPenaltyWeight*uxBasis@ux/Nt
        Gy=Gy+2*EnergyPenaltyWeight*uyBasis@uy/Nt

    G=np.concatenate((Gx.ravel(),Gy.ravel()))

    return J, G