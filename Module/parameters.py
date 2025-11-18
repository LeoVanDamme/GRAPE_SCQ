import numpy as np

class ProblemParameters:
    """
    DESCRIPTION:
        Defines the physical parameters of the problem, including qubit frequency,
        anharmonicity, target gate or state, computational subspace, and other
        relevant settings.

        All Hamiltonian parameters are specified in Hz, and later converted to rad/s
        (multiplied by 2π) automatically.

    INPUT PARAMETERS:
        QubitFreq       : Qubit frequency in lab frame (Hz), default = 5 GHz
        CarrFreq        : Carrier frequency, default = mean(QubitFreq)        
        alpha           : Anharmonicity parameter (Hz), default = -100 MHz
        AmpScale        : Scale factor on pulse amplitude (unitless), default = 1
        nuRef           : Reference pulse amplitude (Hz), default = 1/(2*Tp) with Tp: pulse duration
        dt              : Time step (seconds), default = 0.1 ns
        Nt              : Number of time steps, default = 200
        NLevels         : Number of energy levels, default = 3
        CompSpace       : Computational subspace indices, default = [0, 1]
        Target          : Target gate or state, default = [[0,-1j],[-1j,0]]
        Psi0            : Initial state, default: |0>
        Weight          : Cost function weight, default: no weight
        uxBasis         : Analytical basis for in-phase component, default: [] (standard grape)
        uyBasis         : Analytical basis for quadrature component, default: [] (standard grape)

    DIMENSIONS OF INPUT PARAMETERS:
        QubitFreq, alpha, AmpScale:
            - scalar      → Single value, constant in time
            - (Nhp,)      → Multiple values, constant in time
            - (1,Nt)      → Single value, time-varying
            - (Nhp, Nt)   → Multiple values, time-varying
        CarrFreq, nuRef:
            - Scalar
        dt:
            - scalar   → Equal time steps
            - (Nt,)    → Unequal time steps
        CompSpace:
            - (Nc,)    → Nc: number of computational states (Nc ≤ NLevels)
        Target:
            - If target gate:
                - (Nc, Nc)          → Same gate for all parameter sets
                - (Nhp * Nc, Nc)    → One gate per parameter set
            - If target state:
                - (NLevels,)        → Same state for all parameter sets
                - (Nhp * NLevels,)  → One state per parameter set
        Weight:
            - (Nhp,)
        Psi0:
            - (NLevels,)          → Same initial state for all sets
            - (Nhp * NLevels,)    → One initial state per set
        uxBasis:
            - (Nx, Nt)  → Nx is the number of functions used to shape ux
        uyBasis:
            - (Ny, Nt)  → Ny is the number of functions used to shape uy

    OUTPUT PARAMETERS:
        Input parameters + 
            Tp          : Total pulse duration in seconds
            t           : Time grid (interval boundaries) 
                            - Shape (Nt+1,)
            tc          : Start of each time step
                            - Shape (Nt,)
            Nhp         : Number of parameter sets
            Nc          : Number of computational states
            Nht         :   - 1 if static Hamiltonian parameters and equal timesteps
                            - Nt if time-varying parameters and/or unequal timesteps
            uxBasis/y   : Analytical basis for the case of analytically shaped pulses
            TruxBasis/y : Transpose of uxBasis/y
            Tardagg     : Conjugate-transposed target gate/state for fidelity calculation
                            - If the target is a gate:  (Nc, Nc, Nhp)
                            - If the target is a state: (1,  Nc, Nhp)
            Psi0        : Initial state (used only for state-to-state problems)
                            - Shape: (NLevels, 1, Nhp)
            H0dt        : Drift Hamiltonian multiplied by dt
                            - Shape (NLevels, NLevels, Nht, Nhp)
            HDdt        : Drive Hamiltonian  multiplied by dt
                            - Shape (NLevels, NLevels, Nht, Nhp)
            Weight      : Reshaped weights for the cost function
                            - Shape: (1, Nhp)
            itH         : Array used to reduce memory usage when Nht = 1
                            - Shape: (Nt,)

    AUTHOR:
        Leo Van Damme / Technical University of Munich, 2025
    """
    def __init__(self, 
                QubitFreq=5.0e9,
                CarrFreq=[],
                alpha=-100.0e6,
                AmpScale=1.0,
                nuRef=[],
                NLevels=3,
                dt=0.1e-9,
                Nt=200,
                CompSpace=np.array([0,1]),
                Target=np.array([[0,-1j],[-1j,0]]),
                Psi0=[],
                Weight=[],
                uxBasis=[],
                uyBasis=[]
                ):
        
        # Shape inputs and derive matrices
        self.Set(QubitFreq=QubitFreq, CarrFreq=CarrFreq, alpha=alpha, 
                 AmpScale=AmpScale, nuRef=nuRef, NLevels=NLevels,dt=dt,
                 Nt=Nt, CompSpace=CompSpace, Target=Target, Psi0=Psi0, 
                 Weight=Weight, uxBasis=uxBasis, uyBasis=uyBasis
            ) 
        
    #######################################################################
    ################## Shape inputs and derive matrices ###################
    def Set(self, **kwargs):

        # Assign raw inputs
        allowed_keys = {'QubitFreq', 'CarrFreq', 'alpha', 'AmpScale',
                        'nuRef', 'NLevels', 'dt', 'Nt', 'CompSpace',
                        'Target', 'Psi0', 'Weight', 'uxBasis', 'uyBasis'}
        for key, value in kwargs.items():
            if key in allowed_keys:
                setattr(self, key+"_in", value)
            else:
                raise AttributeError(f"Cannot set unknown attribute '{key}'")

        # Return cleaned inputs and derived matrices
        self.TimeGrids()
        self.ParameterDimensions()
        self.AssignCarrFreq()
        self.AssignRefReq()
        self.SetBasisMatrices()
        self.ShapeTargetDagger()
        self.ShapePsi0()
        self.ComputeHamiltonians()
        self.SetWeight()

    #######################################################################
    ######################### Time discretization #########################
    def TimeGrids(self):
        self.FormatTimeSteps()
        dtarr = np.broadcast_to(self.dt,(self.Nt,))
        self.t = np.concatenate(([0], np.cumsum(dtarr))) # Time grid (interval boundaries)
        self.tc = self.t[:-1] # Start of each time step (exclude last point) 
        self.Tp = self.t[-1]   # Total pulse duration (last time point)
         
    def FormatTimeSteps(self):
        self.Nt = self.Nt_in
        self.dt = np.squeeze(np.asarray(self.dt_in))
        if self.dt.ndim == 0:
            self.dt = self.dt.reshape(1,)
        elif self.dt.ndim == 1 and self.dt.shape[0] == self.Nt:
            pass
        else:
            raise TypeError("dt must be of shape:\n"
                            "   - scalar    → Equal time steps\n"
                            "   - (Nt,)     → Unequal time steps"
                            ) 

    #######################################################################
    ##################### Hamiltonian parameter space #####################
    def ParameterDimensions(self):

        # Format parameters
        self.FormatHP()

        # Number of parameter sets
        self.Nhp =  max( self.QubitFreq.shape[0], self.AmpScale.shape[0], 
                         self.alpha.shape[0] )

        # Time variation: Nht=1 (static) or Nht=Nt (time-varying)
        self.Nht = max( self.QubitFreq.shape[1], self.AmpScale.shape[1], 
                        self.alpha.shape[1], self.dt.shape[0] )

        # Computational subspace
        self.FormatSpace()
        self.Nc = self.CompSpace.shape[0]

        # Check dimensions errors
        self.CheckHPDim()

    def FormatHP(self):
        self.AmpScale = np.asarray(self.AmpScale_in)
        self.QubitFreq = np.asarray(self.QubitFreq_in)
        self.alpha = np.asarray(self.alpha_in)
        for field in ["AmpScale", "QubitFreq", "alpha"]:
            arr = np.asarray(getattr(self, field))
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr[:,np.newaxis]
            elif arr.ndim == 2:
                pass
            else:
                raise TypeError("Hamiltonian parameters QubitFreq, AmpScale and alpha must be consistent.\n"
                "They must have shape:\n"
                "    - scalar       → Single value, constant in time\n"
                "    - (Nhp,)       → Multiple values, constant in time\n"
                "    - (1,Nt)       → Single value, time-varying\n"
                "    - (Nhp, Nt)    → Multiple values, time-varying\n"
                "where Nhp is unique and represents the number of parameter sets."
                )
            setattr(self, field, arr)

    def CheckHPDim(self):
        for name, val in zip(["QubitFreq", "AmpScale", "alpha"], [self.QubitFreq, self.AmpScale, self.alpha]):
            nrows = val.shape[0]
            ncols = val.shape[1]
            if (nrows !=1 and nrows != self.Nhp) or (ncols != 1 and ncols != self.Nt):
                raise TypeError("Hamiltonian parameters QubitFreq, AmpScale and alpha must be consistent.\n"
                "They must have shape:\n"
                "    - scalar       → Single value, constant in time\n"
                "    - (Nhp,)       → Multiple values, constant in time\n"
                "    - (1,Nt)       → Single value, time-varying\n"
                "    - (Nhp, Nt)    → Multiple values, time-varying\n"
                "where Nhp is unique and represents the number of parameter sets."
                )
        
    #######################################################################
    ########################## Carrier frequency ##########################
    # Check errors and assign Carrfreq if not provided
    def AssignCarrFreq(self):
        # Carrier frequency 
        self.CarrFreq = np.squeeze(np.asarray(self.CarrFreq_in))
        if self.CarrFreq.size != 0 and self.CarrFreq.ndim != 0:
            raise TypeError("CarrFreq must be a scalar")
        
        if self.CarrFreq.size == 0:
            self.CarrFreq = np.mean(self.QubitFreq)

    #######################################################################
    ######################### Reference frequency #########################
    # Check errors and assign nuRef if not provided
    def AssignRefReq(self):
        self.nuRef = np.squeeze(np.asarray(self.nuRef_in))
        if self.nuRef.size !=0 and self.nuRef.ndim != 0:
            raise TypeError("nuRef must be a scalar")
        
        if self.nuRef.size == 0:
            dtarr = np.broadcast_to(self.dt,(self.Nt,))
            Tp = np.sum(dtarr)
            self.nuRef = 1/(2*Tp)
    
    #######################################################################
    ############### Hilbert space and computational subspace ##############
    def FormatSpace(self):
        self.NLevels = self.NLevels_in
        self.CompSpace = np.squeeze(np.asarray(self.CompSpace_in))
        if self.CompSpace.size > self.NLevels:
            raise TypeError ("The size of CompSpace must be smaller "\
            "than NLevels")
        if np.max(self.CompSpace) > self.NLevels:
            raise TypeError ("Problem with CompSpace: the computational states " \
            "must be embeded in the full Hilbert space (indices <= NLevels)")


    #######################################################################
    ########################## Analytical basis ###########################
    def SetBasisMatrices(self):

        # Check and formate the input bases
        self.FormatBases()

        # Get dimensions and transpose
        if self.uxBasis.size !=0:
            self.IsShapedX = True
            self.Nx=self.uxBasis.shape[0]
            self.TruxBasis = np.transpose(self.uxBasis)
        else:
            self.IsShapedX = False
            self.Nx = self.Nt
            self.uxBasis = np.eye(self.Nt)
            self.TruxBasis = self.uxBasis.T
        if self.uyBasis.size !=0:
            self.IsShapedY = True
            self.Ny=self.uyBasis.shape[0]
            self.uyBasis = self.uyBasis
            self.TruyBasis = np.transpose(self.uyBasis)
        else:
            self.IsShapedY = False
            self.Ny = self.Nt
            self.uyBasis = np.eye(self.Nt)
            self.TruyBasis = self.uyBasis.T

    def FormatBases(self):
        self.uxBasis = np.squeeze(np.asarray(self.uxBasis_in))
        self.uyBasis = np.squeeze(np.asarray(self.uyBasis_in))
        for name, field in zip(["x", "y"],["uxBasis", "uyBasis"]):
            value = getattr(self, field)
            if value.ndim == 1 and value.shape[0] == self.Nt:
                value = value.reshape(1,self.Nt)
                setattr(self, field, value)
            elif value.ndim == 2 and value.shape[1] == self.Nt:
                pass
            elif value.size ==0:
                pass
            else:
                raise TypeError(f"{field} must be [] or NumPy array with shape (N{name}, Nt)\n"
                                f"    where N{name} is the number of functions used to shape u{name}")

    #######################################################################
    ######################### Get Target dagger  ##########################
    def ShapeTargetDagger(self):

        # Verify the inputs
        self.FormatTarget()

        # Compute the target dagger
        Targ = np.asarray(self.Target)

        if Targ.ndim == 2 and Targ.shape[0] == self.Nc and Targ.shape[1] == self.Nc: 
            # Single target gate
            Tardagg=np.reshape(self.Target.conj().T, (self.Nc,self.Nc,1), order='F')
            self.Tardagg=np.broadcast_to(Tardagg, (self.Nc,self.Nc,self.Nhp))
            self.Target = np.reshape(self.Target, (self.Nc,self.Nc), order='F') #Clean input
            self.IsProblemGate = True
        elif Targ.ndim == 2 and Targ.shape[0] == self.Nhp*self.Nc and Targ.shape[1] == self.Nc:
            # Multiple target gates
            self.Tardagg=np.reshape(self.Target.conj().T, (self.Nc,self.Nc,self.Nhp), order='F')
            self.Target = np.reshape(self.Target, (self.Nc,self.Nc,self.Nhp), order='F') #Clean input
            self.IsProblemGate = True
        elif Targ.ndim == 1 and Targ.shape[0] == self.NLevels:    
            # Single Target state
            Tardagg=np.reshape(self.Target.conj().T,(1,self.NLevels,1), order='F')
            self.Tardagg=np.broadcast_to(Tardagg, (1,self.NLevels,self.Nhp))
            self.Target = np.reshape(self.Target,self.NLevels, order='F') #Clean input
            self.IsProblemGate = False
        elif Targ.ndim == 1 and Targ.shape[0] == self.Nhp*self.NLevels:
            # Multiple Target states
            self.Tardagg=np.reshape(self.Target.conj().T,(1,self.NLevels,self.Nhp), order='F')
            self.Target = np.reshape(self.Target,(self.NLevels,self.Nhp), order='F') #Clean input
            self.IsProblemGate = False

    def FormatTarget(self): 
        self.Target = np.squeeze(np.asarray(self.Target_in))
        if self.Target.ndim == 1 and (self.Target.shape[0] == self.NLevels or self.Target.shape[0] == self.Nhp*self.NLevels):
            pass
        elif self.Target.ndim == 2 and self.Target.shape[1] == self.Nc and (self.Target.shape[0] == self.Nc or self.Target.shape[0] == self.Nhp*self.Nc):
            pass
        else:
            raise ValueError("The dimension of Target is incorrect. The shape must be:\n"
                "If target gate:\n"
                "    - (Nc, Nc)         → Same gate for all parameter sets\n"
                "    - (Nhp * Nc, Nc)   → One gate per parameter set\n"
                "If target state:\n"
                "    - (NLevels,)       → Same state for all parameter sets\n"
                "    - (Nhp * NLevels,) → One state per parameter set\n"
                "where Nhp is the number of Hamiltonian parameter sets and \n"
                "Nc the number of computational states")
        
    #######################################################################
    ###################### Assign and reshape Psi0  #######################
    def ShapePsi0(self):

        # Check input Psi0
        self.FormatPsi0()

        # Shape it as (NLevels, Nhp)
        if self.Psi0.size ==0:
            self.Psi0=np.zeros((self.NLevels,1))
            self.Psi0[0]=1
        elif self.Psi0.ndim == 1 and self.Psi0.shape[0]==self.NLevels:
            self.Psi0=self.Psi0.reshape((self.NLevels,1), order='F')
        elif self.Psi0.ndim == 1 and self.Psi0.shape[0]==self.NLevels*self.Nhp:
            self.Psi0 = np.reshape(self.Psi0, (self.NLevels, self.Nhp), order='F')
        self.Psi0 = np.broadcast_to(self.Psi0,(self.NLevels,self.Nhp))

    def FormatPsi0(self):
        self.Psi0 = np.squeeze(self.Psi0_in)
        if self.Psi0.size !=0 and self.Psi0.shape[0] != self.NLevels \
            and self.Psi0.shape[0] != self.NLevels*self.Nhp:
            raise ValueError("Psi0 must have dimensions:\n"
            "  - (NLevels,) if the initial state is the same for all sets of parameters\n"
            "  - (Nhp*NLevels,) if you want specic initial state for each set of parameters\n"
            "  - Psi0 = [] if you don't need it.\n"
            "where Nhp is the number of Hamiltonian parameter sets."
            )
        
    #######################################################################
    ####################### Compute Hamiltonians  #########################
    def ComputeHamiltonians(self):

        # Harmonize hamiltonian parameters
        Frq = np.broadcast_to(self.QubitFreq, (self.Nhp,self.Nht))
        Amp = self.nuRef*np.broadcast_to(self.AmpScale, (self.Nhp,self.Nht))
        Anh = np.broadcast_to(self.alpha, (self.Nhp,self.Nht))
        dt = np.broadcast_to(self.dt, (self.Nht,))

        # Creation and destruction operators  a† and a
        ap = np.diag(np.sqrt(np.arange(1, self.NLevels)), k=-1)
        am = np.diag(np.sqrt(np.arange(1, self.NLevels)), k=1)

        # Compute drif and drive Hamiltonian matrices
        self.H0dt=np.zeros((self.NLevels,self.NLevels,self.Nht,self.Nhp))
        self.HDdt=np.zeros((self.NLevels,self.NLevels,self.Nht,self.Nhp))
        for n in range(self.Nhp):
            for nt in range(self.Nht):
                # Natural Hamiltonian
                H0=2*np.pi*Anh[n,nt]*ap@ap@am@am/2+2*np.pi*(Frq[n,nt]-self.CarrFreq)*ap@am
                self.H0dt[:,:,nt,n]=H0*dt[nt]
                
                # Drive Hamiltonian (will become hermitian later)
                HD=0.5*2*np.pi*Amp[n,nt]*ap
                self.HDdt[:,:,nt,n]=HD*dt[nt]

        # Hamiltonian time indices 
        # Adapt the indices of the Hamiltonian to use at each time step.
        if self.Nht == 1:
            self.itH = np.zeros(self.Nt).astype(int)
        else:
            self.itH = np.arange(self.Nt).astype(int)

    #######################################################################
    #################### Weight on the cost function ######################
    def SetWeight(self):
        self.FormatWeight()
        if self.Weight.ndim == 1 and self.Weight.shape[0] == self.Nhp:
            self.Weight=self.Weight.reshape(1,-1)
        elif self.Weight.ndim == 1 and self.Weight.shape[0] == 1:
            self.Weight=self.Weight*np.ones((1,self.Nhp))
        elif self.Weight.size == 0:
            self.Weight=np.ones((1,self.Nhp))

    def FormatWeight(self):
        self.Weight = np.squeeze(np.asarray(self.Weight_in))
        if self.Weight.size == 0:
            pass
        elif self.Weight.ndim == 1 and self.Weight.shape[0] == self.Nhp:
            pass
        elif self.Weight.ndim == 1 and self.Weight.shape[0] == 1:
            pass
        else:
            raise TypeError("Weight must be [] or aNumpy array of shape (Nhp,)," \
                            " where Nhp is the number of Hamiltonian parameter sets")
        
        
            
       
        
        