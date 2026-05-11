# Cython declaration file for ranvar.random.
# cimport this file from another Cython module to access the random sampling
# functions directly at C speed without touching Python space.
#
# Example usage in a pure-Python-mode .py file:
#
#   from cython.cimports.ranvar.random import rand, randint, randnorm

cdef double rand()
cdef double randint(double l, double h)
cdef double randnorm(double mu, double stdev)
cdef double randexp(double rate)
cdef double randgamma(double shape, double scale)
cdef double randpoisson(double lam)
cdef double randnegbinom(double n, double p)
cdef double randpert(double low, double mode, double high)
