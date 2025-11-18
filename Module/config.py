import numpy as np

##################################################################################
########################### Optimization configuration ###########################

# Use numba
UseNumba = True

# Numba options
UseFastMath = False
UseParallel = False
UseCache = True

# Data types
dtype_float = np.float64
dtype_complex = np.complex128 # problem with complex64 and UseNumba
dtype_int = np.int64
