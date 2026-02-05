"""
Contains Digest classes that approximates quantiles of data streams using
the t-digest algorithm. The t-digest algorithm is a probabilistic data structure
that can be used to estimate the quantiles of a data stream with a small memory footprint.
"""


from types import prepare_class

import numpy as np
import cython as pyx

if pyx.compiled:
    from cython.cimports.libc.math import round as c_round
    from cython.cimports.libc.math import ceil as ceil
    from cython.cimports.libc.math import floor as floor
    from cython.cimports.libc.time import time as c_time
    from cython.cimports.libc.stdlib import rand as c_rand
    from cython.cimports.libc.stdlib import srand as c_srand
    from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX
    # from cython.cimports.random import normal
else:
    ceil = np.ceil
    floor = np.floor



if pyx.compiled:
    print('Running through Cython!')
else:
    print('WARNING: Not Compiled.')


# c_srand(c_time(pyx.NULL))

#---[ Probability Distributions ]-----------------------------------------------

# # Rand
# @pyx.cfunc
# def _rand() -> pyx.double:
#     out:pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
#     return out

# # RandInt
# @pyx.cfunc
# def _randint(l: pyx.double, h: pyx.double) -> pyx.double:
#     return c_round((h - l) * _rand() + l)



# TDigest


LOWER = (1 - 0.9999) / 2
UPPER = 1 - LOWER

__ADD__:pyx.int = 0
__MUL__:pyx.int = 1
__MAX__:pyx.int = 2
__MIN__:pyx.int = 3
__SUB__:pyx.int = 4
__POW__:pyx.int = 5

DEFAULTS = {
    'maxBins':32,
    'compress':True,
    'tolerance':1e-3
}




@pyx.cclass
class TDigest():
    _bins: pyx.double[:]
    _cnts: pyx.double[:]

    bins: np.ndarray
    cnts: np.ndarray

    maxBins: pyx.int
    nActive: pyx.int

    def __init__(self, maxBins=32):
        self.maxBins = maxBins
        self.nActive = 0

        self.bins = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + 1, dtype=np.float64)

        self._bins = self.bins
        self._cnts = self.cnts

    def setBins(self, bins):

        self.bins = bins

        # If we don't set self._bins as well it will still refer to the previous
        # array and not the new one. In essence self._bins points to the memory
        # block and not the name self.bins

        self._bins = self.bins

    def getBins(self):
        return self.bins

    def setWeights(self, cnts):
        self.cnts = cnts
        self._cnts = self.cnts

    def getWeights(self):
        return self.cnts

    def setActiveBinCount(self, nActive):
        self.nActive = nActive

    def getActiveBinCount(self):
        return self.nActive

    def _assertCompiled(self):
        assert pyx.compiled

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _findLastLesserOrEqualIndex(self, point:pyx.double) -> pyx.int:

        idx:pyx.int = -1
        while True:
            if (self._bins[idx + 1] > point) or (idx + 1 == self.nActive):
                break
            else:
                idx += 1

        return idx

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _shiftRightAndInsert(self, idx:pyx.int, point:pyx.double, count:pyx.double) -> pyx.void:
        j: pyx.int

        for j in range(self.nActive - 1, idx, -1):
            self._bins[j+1] = self._bins[j]
            self._cnts[j+1] = self._cnts[j]

        self._bins[idx+1] = point
        self._cnts[idx+1] = count
        self.nActive += 1

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _findMinimumDifference(self) -> pyx.int:
        k: pyx.int
        dB: pyx.double
        minK: pyx.int
        minDiff: pyx.double

        minK    = -1
        minDiff = 9e9
        # We don't want to merge the first or last bin because we want to maintain the
        # tails. It also solves the problem where we try and sample a point that is before
        # the first centroid.
        for k in range(1, self.nActive - 2):
            dB = self._bins[k+1] - self._bins[k]
            if dB < minDiff:
                minDiff = dB
                minK    = k

        return minK

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _shiftLeftAndOverride(self, idx: pyx.int) -> pyx.void:
        j: pyx.int
        for j in range(idx, self.nActive-1):
            self._bins[j] = self._bins[j+1]
            self._cnts[j] = self._cnts[j+1]

        self._bins[self.nActive-1] = 0
        self._cnts[self.nActive-1] = 0

        self.nActive -= 1

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _sumWeights(self) -> pyx.double:
        som: pyx.double = 0
        i: pyx.int

        for i in range(self.nActive):
            som = som + self._cnts[i]

        return som

    def fit(self, x):
        for xx in x:
            self.add(xx)

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    @pyx.initializedcheck(False)
    def _add(self, point:pyx.double, count:pyx.double) -> pyx.void:

        idx:pyx.int = self._findLastLesserOrEqualIndex(point)

        if (idx >= 0) and self._bins[idx] == point:
            self._cnts[idx] += count
        else:
            self._shiftRightAndInsert(idx, point, count)

        if self.nActive > self.maxBins:
            k:pyx.int = self._findMinimumDifference()

            sumC:pyx.double = self._cnts[k+1] + self._cnts[k]

            self._bins[k] = (self._bins[k]*self._cnts[k] + self._bins[k+1]*self._cnts[k+1])
            self._bins[k] = self._bins[k] / sumC
            self._cnts[k] = sumC

            self._shiftLeftAndOverride(k+1)

    def centroids(self):
        return self.bins[:-1]

    def weights(self):
        return self.cnts[:-1]

    def add(self, point, count=1):
        self._add(point, count)

    def cdf(self, k):
        """

        Ted Dunning, Computing Extremely Accurate Quantiles Usings t-Digests
        """
        print('self.cdf(k)',k)
        som = 0
        c = self.bins
        m = self.cnts

        if k < self.lower():
            print('return 0.')
            return 0.
        elif k >= self.upper():
            print('return 1.')
            return 1.
        else:
            for i in range(self.nActive):
                if c[i] <= k < c[i+1]:
                    # We use the approach of Dunning here to improve interpolation when
                    # we have single weighted points.
                    if (m[i] > 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i+1] + m[i]) / 2

                    elif (m[i] == 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som
                        yi_n = yi + (m[i+1]) / 2

                    elif (m[i] > 1) & (m[i+1] == 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i]) / 2
                    else:
                        yi   = som
                        yi_n = yi

                    g    = (yi_n - yi) / (c[i+1] - c[i])
                    yk   = g*(k - c[i]) + yi

                    return yk / self._sumWeights()

                else:
                    som += m[i]

            print(k)
            print(self.getBins())
            print(self.getWeights())


    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def pmf(self, kk: pyx.int) -> pyx.double:
        return self.cdf(kk + 0.5) - self.cdf(kk - 0.5)




    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def lower(self) -> pyx.int:
        return int(self._bins[0])

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def upper(self) -> pyx.int:
        return int(self._bins[self.nActive - 1])

    def sample(self, size=1):
        p = np.random.rand()
        return int(round(self.quantile(p)))


    def quantile(self, p):
        # Using Regula Falsi
        a = self.lower()
        b = self.upper()

        fa = 0 - p
        fb = 1 - p

        iter = 0

        while ((b - a) > 0.1) and (iter < 100):
            c = (fb*a - fa*b) / (fb - fa)
            fc = self.cdf(c) - p

            if fc < 0:
                a  = c
                fa = fc
            elif fc > 0:
                b  = c
                fb = fc
            else:
                return c

            iter += 1

        return (b+a)/2

    def median(self):
        return self.quantile(0.5)
