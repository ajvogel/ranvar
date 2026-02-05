import numpy as np
import cython as pyx

import random

if pyx.compiled:
    from cython.cimports.libc.math import round as c_round
    from cython.cimports.libc.math import ceil as c_ceil
    from cython.cimports.libc.math import ceil as ceil
    from cython.cimports.libc.math import floor as c_floor
    from cython.cimports.libc.time import time as c_time
    from cython.cimports.libc.math import log as c_log
    from cython.cimports.libc.math import sqrt as c_sqrt
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


c_srand(c_time(pyx.NULL))

#======================================[ Random Variable ]=========================================

# @pyx.initializedcheck(False)
# @pyx.boundscheck(False)
# @pyx.wraparound(False)


def _reconstructDigest(maxBins, nActive, centroids, weights):
    digest = Digest(maxBins)
    digest.setBins(centroids)
    digest.setWeights(weights)
    digest.setActiveBinCount(nActive)
    return digest



@pyx.cclass
class Digest():
    """
    Implementation of the t-digest algorithm for computing quantiles.

    The t-digest algorithm is described in 'Computing Extremely Accurate Quantiles Using t-Digests'
    by Ted Dunning. This data structure provides approximate quantile computation with bounded
    memory usage and high accuracy, particularly at the tails of the distribution.

    The digest maintains a set of centroids (weighted points) that summarize the distribution.
    When the number of centroids exceeds maxBins, nearby centroids are merged to maintain
    the memory bound while preserving accuracy.

    Attributes:
        bins (np.ndarray): Array storing centroid values (x-coordinates)
        cnts (np.ndarray): Array storing centroid weights (counts)
        maxBins (int): Maximum number of centroids to maintain
        nActive (int): Current number of active centroids

    Example:
        >>> digest = Digest(maxBins=100)
        >>> digest.fit([1, 2, 3, 4, 5])
        >>> digest.quantile(0.5)  # Median
        3.0
    """
    _bins: pyx.double[:]
    _cnts: pyx.double[:]

    bins: np.ndarray
    cnts: np.ndarray

    maxBins: pyx.int
    nActive: pyx.int

    def __init__(self, maxBins=32):
        """Initialize a new t-digest.

        Args:
            maxBins (int, optional): Maximum number of centroids to maintain.
                                   Defaults to 32.
        """
        self.maxBins = maxBins
        self.nActive = 0

        self.bins = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + 1, dtype=np.float64)

        self._bins = self.bins
        self._cnts = self.cnts

    def __reduce__(self):
        return (_reconstructDigest, (self.maxBins, self.nActive, self.bins, self.cnts))


    def setBins(self, bins):
        """Set the bins (centroids) array.

        Args:
            bins (np.ndarray): Array of centroid values to set.

        Note:
            This method updates both the public bins array and the internal
            Cython memoryview _bins to ensure consistency.
        """

        self.bins = bins

        # If we don't set self._bins as well it will still refer to the previous
        # array and not the new one. In essence self._bins points to the memory
        # block and not the name self.bins

        self._bins = self.bins

    def getBins(self):
        """Get the bins (centroids) array.

        Returns:
            np.ndarray: Array containing the centroid values.
        """
        return self.bins

    def setWeights(self, cnts):
        """Set the weights (counts) array.

        Args:
            cnts (np.ndarray): Array of centroid weights to set.

        Note:
            This method updates both the public cnts array and the internal
            Cython memoryview _cnts to ensure consistency.
        """
        self.cnts = cnts
        self._cnts = self.cnts

    def getWeights(self):
        """Get the weights (counts) array.

        Returns:
            np.ndarray: Array containing the centroid weights.
        """
        return self.cnts

    def setActiveBinCount(self, nActive):
        """Set the number of active centroids.

        Args:
            nActive (int): Number of active centroids to set.
        """
        self.nActive = nActive

    def getActiveBinCount(self):
        """Get the number of active centroids.

        Returns:
            int: Current number of active centroids.
        """
        return self.nActive

    def _assertCompiled(self):
        """Assert that the code is running in compiled Cython mode.

        Raises:
            AssertionError: If not running in compiled Cython mode.
        """
        assert pyx.compiled

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _findLastLesserOrEqualIndex(self, point:pyx.double) -> pyx.int:
        """Find the index of the last centroid that is <= the given point.

        This method performs a linear search through the sorted centroids to find
        the insertion point for a new value.

        Args:
            point (float): The value to search for.

        Returns:
            int: Index of the last centroid <= point, or -1 if point is smaller
                than all centroids.
        """

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
        """Insert a new centroid at the specified position by shifting elements right.

        This method maintains the sorted order of centroids by shifting all centroids
        to the right of the insertion point and inserting the new centroid.

        Args:
            idx (int): Index after which to insert the new centroid.
            point (float): Value of the new centroid.
            count (float): Weight of the new centroid.
        """
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
        """Find the pair of adjacent centroids with minimum distance.

        This method is used when the digest exceeds maxBins to identify which
        centroids should be merged. It avoids merging the first and last centroids
        to preserve the tails of the distribution.

        Returns:
            int: Index of the first centroid in the pair with minimum distance,
                or -1 if no suitable pair is found.
        """
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
        """Remove a centroid by shifting all subsequent centroids left.

        This method maintains the sorted order and compactness of the centroid
        arrays by removing the centroid at the specified index.

        Args:
            idx (int): Index of the centroid to remove.
        """
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
        """Calculate the total weight of all active centroids.

        Returns:
            float: Sum of all centroid weights.
        """
        som: pyx.double = 0
        i: pyx.int

        for i in range(self.nActive):
            som = som + self._cnts[i]

        return som

    def fit(self, x):
        """Fit the digest to a collection of data points.

        This is a convenience method that adds all points in the collection
        to the digest with equal weight.

        Args:
            x (iterable): Collection of numeric values to add to the digest.
        """
        for xx in x:
            self.add(xx)

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    @pyx.initializedcheck(False)
    def _add(self, point:pyx.double, count:pyx.double) -> pyx.void:
        """Add a weighted point to the digest (internal implementation).

        This is the core method that implements the t-digest algorithm. It either
        updates an existing centroid if the point matches exactly, or inserts a new
        centroid. If the number of centroids exceeds maxBins, it merges the two
        closest centroids.

        Args:
            point (float): The value to add.
            count (float): The weight/count of the value.
        """

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
        """Get the centroid values.

        Returns:
            np.ndarray: Array of active centroid values (excluding unused slots).
        """
        return self.bins[:-1]

    def weights(self):
        """Get the centroid weights.

        Returns:
            np.ndarray: Array of active centroid weights (excluding unused slots).
        """
        return self.cnts[:-1]

    def add(self, point, count=1):
        """Add a weighted point to the digest.

        Args:
            point (float): The value to add.
            count (float, optional): The weight/count of the value. Defaults to 1.
        """
        self._add(point, count)

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    @pyx.initializedcheck(False)
    def mean(self):
        SxW: pyx.float = 0
        W: pyx.float = 0
        i: pyx.int

        for i in range(self.nActive):
            SxW += self._cnts[i]*self._bins[i]
            W   += self._cnts[i]

        return SxW/W

    def quantile(self, p):
        """Compute the quantile for a given probability.

        Uses linear interpolation between centroids to estimate the quantile.
        The interpolation accounts for the different weighting schemes at
        boundaries and interior points.

        Args:
            p (float): Probability value between 0 and 1.

        Returns:
            float: Estimated quantile value.

        Raises:
            No explicit validation, but p should be in [0, 1] for meaningful results.
        """
        if p <= 0:
            return self.lower()
        elif p >= 1:
            return self.upper()
        else:
            W  = self._sumWeights()
            m = self.cnts
            c = self.bins
            wi = 0
            w_ = p*W

            for i in range(self.nActive-1):
                if i == 0:
                    wGap = m[i] + m[i+1]/2
                elif i == self.nActive - 1:
                    wGap = m[i]/2 + m[i+1]
                else:
                    wGap = m[i]/2 + m[i+1]/2

                wi_n = wi + wGap

                if wi <= w_ < wi_n:
                    fraction = (w_ - wi) / wGap
                    c_ = fraction * (c[i+1] - c[i]) + c[i]
                    return c_

                wi = wi_n

            else:
                return self.upper()

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    @pyx.initializedcheck(False)
    def cdf(self, k:pyx.double) -> pyx.double:
        """Compute the cumulative distribution function at a given point.

        Implements the CDF estimation algorithm from Ted Dunning's paper
        'Computing Extremely Accurate Quantiles Using t-Digests'. Uses
        different interpolation strategies depending on whether centroids
        represent single points or aggregated ranges.

        Args:
            k (float): The point at which to evaluate the CDF.

        Returns:
            float: Estimated CDF value between 0 and 1.
        """
        som:pyx.double = 0
        i:pyx.int

        c = self._bins
        m = self._cnts


        if k <= self.lower():
            return 0.
        elif k >= self.upper():
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

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def pmf(self, kk: pyx.int) -> pyx.double:
        """Compute the probability mass function at an integer point.

        Calculates P(X = kk) by taking the difference CDF(kk + 0.5) - CDF(kk - 0.5).
        This approach provides a reasonable approximation for discrete distributions.

        Args:
            kk (int): Integer point at which to evaluate the PMF.

        Returns:
            float: Estimated probability mass at kk.
        """
        return self.cdf(kk + 0.5) - self.cdf(kk - 0.5)


    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def lower(self) -> pyx.int:
        """Get the minimum value in the digest.

        Returns:
            int: The smallest centroid value, cast to integer.
        """
        return int(self._bins[0])

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.cdivision(True)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def upper(self) -> pyx.int:
        """Get the maximum value in the digest.

        Returns:
            int: The largest centroid value, cast to integer.
        """
        return int(self._bins[self.nActive - 1])

    def sample(self, size=1):
        """Sample a single value from the distribution represented by the digest.

        Generates a random quantile and returns the corresponding value,
        rounded to the nearest integer.

        Args:
            size (int, optional): Currently unused, always returns a single sample.
                                Defaults to 1.

        Returns:
            int: A sampled integer value from the distribution.

        Note:
            Despite the size parameter, this method currently only returns
            a single sample.
        """
        p: pyx.double = _rand()
        #p = np.random.rand()
        return int(round(self.quantile(p)))


#==================================================================================================

#---[ Probability Distributions ]-----------------------------------------------

# Rand
@pyx.cfunc
@pyx.cdivision(True)
def _rand() -> pyx.double:
    out:pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
    return out

# RandInt
@pyx.cfunc
def _randint(l: pyx.double, h: pyx.double) -> pyx.double:
    l2: pyx.double  = l - 1
    out: pyx.double = c_ceil((h - l2) * _rand() + l2)

    if out < l:
        out = l

    return out

    #return c_round((h - l) * _rand() + l)

@pyx.cfunc
def _randnorm(mu: pyx.double, stdev: pyx.double) -> pyx.double:
    """
    https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    while True:
        x:pyx.double = 2*_rand() - 1
        y:pyx.double = 2*_rand() - 1

        s: pyx.double = x**2 + y**2

        if s < 1:
            break

    z: pyx.double = x * c_sqrt( -2*c_log(s) / s)
    return z * stdev + mu
