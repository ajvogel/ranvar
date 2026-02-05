import cython as pyx

from cython.cimports.libc.math import round as c_round
from cython.cimports.libc.math import ceil as ceil
from cython.cimports.libc.math import floor as floor
from cython.cimports.libc.time import time as c_time
from cython.cimports.libc.stdlib import rand as c_rand
from cython.cimports.libc.stdlib import srand as c_srand
from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX



c_srand(c_time(pyx.NULL))

# Rand
@pyx.cfunc
def rand() -> pyx.double:
    out:pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
    return out

# RandInt
@pyx.ccall
def randint(low: pyx.double, high: pyx.double) -> pyx.double:
    return c_round((high - low) * rand() + low)

# @pyx.ccall
# def normal(mu: pyx.float , std: pyx.float) -> pyx.float:
#     return snorm()*std + mu
