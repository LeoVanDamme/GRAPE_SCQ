import numpy as np

"""Fixed dtypes used throughout the Numba-jitted engine. Numba needs
consistent, unambiguous dtypes -- complex64 in particular does not play well
with Numba, hence complex128 throughout."""

dtype_float = np.float64
dtype_complex = np.complex128
dtype_int = np.int64
