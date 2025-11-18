import numpy as np
from scipy.optimize import minimize
from .utils import FormatControlInput
from .grape_utils import GetCostAndConstr

class Grape:
    """
    DESCRIPTIONS:
        This class allows to optimize pulse parameters by minimizing a cost function.
        The cost function can be, in the simplest case (for a single 2-level system):
            - The gate infidelity, defined as 
                            J = 1 - |Tr(U_Tar† · U_F)|² / 2
            - The state-to-state infidelity, defined as 
                                J = 1 - |<ψ_Tar | ψ_F>|²
        The choice of the cost function depends on the type of target provided in ProblemParameters.
    
    INPUTS
        xtol, ftol, gtol, maxfun:  See scipy.optimize.minimize for details
        Display             : Display optimization results
        Maxiter             : Maximum number of iterations
        MaxInPhase          : Maximum amplitude of in-phase (Ωx) component in Hz
        MaxQuadrature       : Maximum amplitude of quadrature (Ωy) component in Hz
        MaxAmplitude        : Maximum amplitude of  (Ωx²+Ωy²)¹⸍² in Hz
        EnergyPenaltyWeight : Penalty weigth on the pulse energy: J=J+λ·∫(Ωx²+Ωy²)dt
        nGrad               : Number increasing gradients accurracy
        EnergyMinimum       : Preoptimize the pulse to solve the problem with minimum energy
        NSubOptim           : Number of preoptimizations when EnergyMinim is on
        EnergyMinWeightMax  : Initial pulse energy penalty when EnergyMinim is on

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """

    def __init__(self, Display=True, Maxiter=1000, MaxInPhase=None,MaxQuadrature=None, 
                 MaxAmplitude=None, EnergyPenaltyWeight=0.0,EnergyMinimum=False, 
                 NSubOptim=10, EnergyMinWeightMax=None, xtol=1e-12,ftol=1e-12, 
                 gtol=1e-12,maxfun=1e18,nGrad=3):
        self.Display = Display
        self.Maxiter=Maxiter
        self.MaxInPhase = MaxInPhase
        self.MaxQuadrature = MaxQuadrature
        self.MaxAmplitude = MaxAmplitude
        self.gtol = gtol
        self.ftol = ftol
        self.xtol = xtol
        self.maxfun = maxfun
        self.EnergyPenaltyWeight = EnergyPenaltyWeight
        self.EnergyMinimum = EnergyMinimum
        self.NSubOptim = NSubOptim
        self.EnergyMinWeightMax = EnergyMinWeightMax
        self.nGrad=nGrad
        self.iter = 0
        self.Preiter = 0
        self.DispCompile = True


    ##############################################################################
    ################################ Optimization ################################
    def Optimize(self,p,Xctrl0,Yctrl0):
        '''
        DESCRIPTION:
            Performs pulse optimization by minimizing a cost function. The cost 
            function can be, in the simplest case (for a single 2-level system):
                - The gate infidelity, defined as 
                                J = 1 - |Tr(U_Tar† · U_F)|² / 2
                - The state-to-state infidelity, defined as 
                                    J = 1 - |<ψ_Tar | ψ_F>|²
            The choice of cost function depends on the type of target provided in 
            ProblemParameters.
        
        INPUTS:
            Xctrl0  : Initial guess for the in-phase pulse parameters. 
                    Can be either:
                        - The direct pulse shape ux as a function of time (Nt,).
                        - A row of coefficients a_n if p.uxBasis is provided (Nx,).
            Yctrl0  :  Initial guess for the quadrature pulse parameters.
                    Can be either:
                        - The direct pulse shape uy as a function of time (Nt,).
                        - A row of coefficients b_n if p.uyBasis is provided (Ny,).
            p       : Problem parameters defined in ProblemParameters

        OUTPUTS:
            Xctrl   : Optimized in-phase pulse component. Can be either:
                        - The direct in-phase pulse shape (Nt,).
                        - A row of coefficients a_n if p.uxBasis is provided (Nx,).
            Yctrl   : Optimized quadrature pulse component. Can be either:
                        - The direct quadrature pulse shape (Nt,).
                        - A row of coefficients b_n if p.uyBasis is provided (Ny,).
            J       : Cost at the end of the optimization

        AUTHOR:
            Leo Van Damme / Technical University of Munich, 2025
        '''

        # Validate inputs
        Xctrl0, Yctrl0 = FormatControlInput(p,Xctrl0,Yctrl0)

        # Concatenate optimized variable for the solver
        optimvar=np.concatenate([Xctrl0,Yctrl0])
        
        # Determine the cost function and constraints to be used
        CostFun, Constraints = GetCostAndConstr(p,self)

        # Compile cost function and get initial cost
        print("Compiling cost function...",end=" ",flush=True)
        J_ini, _ = CostFun(optimvar)
        J_ini = J_ini.item()
        print("Done.")

        # Pre-optimizations when EnergyMinimum is true
        if self.EnergyMinimum:
            optimvar = self.PreOptimizeEnergyMin(p, optimvar)

        # Optimize
        res = self.CallSolver(optimvar, CostFun, Constraints)
        
        # Termination info
        if self.Display:
            print(f"########### OPTIMIZATION RESULTS ###########")
            print(f"Cost before optimization: {J_ini:.12f}")        
            print(f"Cost after optimization: {res.fun:.12f}")
            print(f"Number of Iterations: {self.iter}")
            if self.EnergyMinimum:
                print(f"Number of iteration during preoptimization: {self.Preiter}")           
            if self.iter == self.Maxiter:
                print(f"Message: The maximum number of iterations has been reached")
        

        # Return control parameters and cost
        Xctrl = res.x[:p.Nx]
        Yctrl = res.x[p.Nx:p.Nx+p.Ny]
        J = res.fun

        return Xctrl, Yctrl, J
    

    ##############################################################################
    ######################### Optimization live display ##########################
    def callback(self,xk, state=None):
        '''
        DESCRIPTION:
            Shows the iteration number during optimization.
        '''
        self.iter += 1
        print(f"Iteration = {self.iter}/{self.Maxiter}", end='\r', flush=True)

    ##############################################################################
    ######################## Energy-minimum optimization #########################
    def PreOptimizeEnergyMin(self,p, optimvar):
        '''
        DESCRIPTION:
            Performs pulse optimization while minimizing the pulse energy given by
                    ∫(Ωx²+Ωy²)dt,
            where Ωx and Ωy are the in-phase and quadrature pulse components. 
            The energy is minimized by applying several individual optimizations 
            using a penalty in the cost function as
                    J=J0+λ·∫(Ωx²+Ωy²)dt.
            The penalty weight λ is made smaller and smaller until it reaches 0 in 
            the last optimization.
            
        INPUTS:
            optimvar : array containing the concatenated control parameters
            p        : Structure containing parameters from ProblemParameters
            gopt     : Optimization options, including:
                            opt.EnergyMinWeightMax  → Initial penalty weight 
                                                      (∝ λ of first optim.)
                            opt.NSubOptim           → Number of optimizations

        OUTPUTS:
            optimvar : Concatenated control parameters after optimization.

        AUTHOR:
            Leo Van Damme / Technical University of Munich, 2025
        '''
        
        SavedMaxiter = self.Maxiter
        SavedPenalty = self.EnergyPenaltyWeight
        savedCallBack = self.callback

        self.Maxiter=round(self.Maxiter/self.NSubOptim)
        self.callback=None

        if self.EnergyMinWeightMax is None:
            self.EnergyMinWeightMax=p.nuRef**2*p.Tp**2

        for k in range(self.NSubOptim):
            self.EnergyPenaltyWeight=self.EnergyMinWeightMax*(self.NSubOptim-k-1)/(self.NSubOptim-1)
            CostFun, Constraints =GetCostAndConstr(p,self)
            res = self.CallSolver(optimvar, CostFun, Constraints)
            optimvar=res.x
            print(f"Preoptimization = {k+1}/{self.NSubOptim}, Cost (+ penalty) = {res.fun:.6f}",end='\r')
            self.Preiter += res.nit

        self.EnergyPenaltyWeight=SavedPenalty
        self.Maxiter = SavedMaxiter
        self.callback = savedCallBack
        print('')

        return optimvar

    def CallSolver(self,optimvar, CostFun, Constraints):

        # Optimize: unconstrained case
        if all(x is None for x in [self.MaxInPhase, self.MaxQuadrature, self.MaxAmplitude]):
            res = minimize(
                    CostFun,
                    optimvar,
                    jac=True,
                    method='L-BFGS-B',
                    callback=self.callback,
                    options={
                        'maxiter': self.Maxiter,'gtol': self.gtol,
                        'ftol': self.ftol,'maxfun': self.maxfun,'disp': False
                    }
                )
        else: # Contrained case
            res = minimize(
                    CostFun,
                    optimvar,
                    jac=True,
                    method='trust-constr',
                    constraints=Constraints,
                    callback=self.callback,
                    options={
                        'maxiter': self.Maxiter,'xtol': self.xtol,
                        'gtol':self.gtol,'disp': False
                    }
                )
            
        return res