from numba import njit

"""Numba JIT configuration for the engine. Kept as simple module-level flags
(rather than a user-facing setting) since they control an implementation
detail -- whether the hot inner loop is JIT-compiled -- not the physics."""

USE_NUMBA = True
USE_PARALLEL = False
USE_FASTMATH = False
USE_CACHE = True

if USE_NUMBA and not USE_PARALLEL:
    jit = njit(fastmath=USE_FASTMATH, cache=USE_CACHE)
    jit_parallel = jit
elif USE_NUMBA and USE_PARALLEL:
    jit = njit(fastmath=USE_FASTMATH, cache=USE_CACHE)
    jit_parallel = njit(fastmath=USE_FASTMATH, cache=USE_CACHE, parallel=True)
else:
    jit = lambda f: f
    jit_parallel = lambda f: f
