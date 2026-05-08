# Cython declaration file for ranvar.digest.
# cimport this file from another Cython module to access Digest (and the
# helper C functions) directly at C speed without touching Python space.
#
# Example usage in another .pyx / .py (pure-Python-mode) file:
#
#   # Traditional .pyx:
#   from ranvar.digest cimport Digest, _rand, _randint, _randnorm
#
#   # Pure-Python-mode .py:
#   from cython.cimports.ranvar.digest import Digest, _rand, _randint, _randnorm

import numpy as np
cimport numpy as np

cdef class Digest:
    # Typed memory views — fast C-level array access
    cdef double[:] _bins
    cdef double[:] _cnts

    # Public numpy arrays (accessible from Python)
    cdef public object bins   # np.ndarray[double]
    cdef public object cnts   # np.ndarray[double]

    # Scalar attributes
    cdef public int maxBins
    cdef public int nActive

    # ---- cpdef methods (callable from both Python and C) ----------------

    cpdef int    _findLastLesserOrEqualIndex(self, double point)
    cpdef void   _shiftRightAndInsert(self, int idx, double point, double count)
    cpdef void   _shiftLeftAndOverride(self, int idx)
    cpdef void   _add(self, double point, double count)

    cpdef object mean(self)
    cpdef double cdf(self, double k)
    cpdef double ccdf(self, double k)
    cpdef double dcdf(self, double k)
    cpdef double dccdf(self, double k)
    cpdef double pmf(self, int kk)
    cpdef int    lower(self)
    cpdef int    upper(self)

    # ---- cdef methods (C-only, not visible from Python) -----------------

    cdef int    _findMinimumDifference(self)
    cdef double _sumWeights(self)


# Module-level C functions (cdef — not importable from Python)
cdef double _rand()
cdef double _randint(double l, double h)
cdef double _randnorm(double mu, double stdev)
