# cython: language_level=3
# distutils: language = c++
# distutils: include_dirs = ranvar/cpp

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from ranvar.digest_cpp cimport CppDigest


def _reconstruct_digest(int maxBins, int nActive, list bins_list, list cnts_list):
    d = Digest(maxBins)
    d._set_bins_list(bins_list)
    d._set_cnts_list(cnts_list)
    d._digest.setActiveBinCount(nActive)
    return d


cdef class Digest:
    """
    Cython extension type backed by the C++ ranvar::Digest class.

    Implements the t-digest algorithm for computing quantiles. The public API
    is intentionally identical to the pure-Python/Cython Digest so the two
    implementations can be used interchangeably.

    Attributes:
        maxBins (int): Maximum number of centroids to maintain.

    Example:
        >>> digest = Digest(maxBins=100)
        >>> digest.fit([1, 2, 3, 4, 5])
        >>> digest.quantile(0.5)
        3.0
    """

    def __cinit__(self, int maxBins=32):
        self._digest = new CppDigest(maxBins)

    def __dealloc__(self):
        del self._digest

    def __reduce__(self):
        return (
            _reconstruct_digest,
            (
                self._digest.getMaxBins(),
                self._digest.getActiveBinCount(),
                list(self._digest.getBins()),
                list(self._digest.getCnts()),
            ),
        )

    # --- Internal helpers used by pickling ---------------------------------

    cdef void _set_bins_list(self, list bins_list):
        cdef vector[double] v
        v.reserve(len(bins_list))
        for b in bins_list:
            v.push_back(<double>b)
        self._digest.setBins(v)

    cdef void _set_cnts_list(self, list cnts_list):
        cdef vector[double] v
        v.reserve(len(cnts_list))
        for c in cnts_list:
            v.push_back(<double>c)
        self._digest.setCnts(v)

    # --- Core mutating API -------------------------------------------------

    def fit(self, x):
        """Fit the digest to a collection of data points.

        Args:
            x (iterable): Collection of numeric values to add to the digest.
        """
        cdef vector[double] v
        for xx in x:
            v.push_back(<double>xx)
        self._digest.fit(v)

    def add(self, double point, double count=1.0):
        """Add a weighted point to the digest.

        Args:
            point (float): The value to add.
            count (float, optional): Weight of the value. Defaults to 1.
        """
        self._digest.add(point, count)

    # --- Read-only views ---------------------------------------------------

    def centroids(self):
        """Get the active centroid values.

        Returns:
            np.ndarray: Array of active centroid values.
        """
        cdef vector[double] v = self._digest.centroids()
        return np.array(v)

    def weights(self):
        """Get the active centroid weights.

        Returns:
            np.ndarray: Array of active centroid weights.
        """
        cdef vector[double] v = self._digest.weights()
        return np.array(v)

    # --- Statistics --------------------------------------------------------

    def lower(self):
        """Get the minimum value in the digest.

        Returns:
            int: The smallest centroid value.
        """
        return self._digest.lower()

    def upper(self):
        """Get the maximum value in the digest.

        Returns:
            int: The largest centroid value.
        """
        return self._digest.upper()

    def mean(self):
        """Compute the weighted mean.

        Returns:
            float: Weighted mean of all centroids.
        """
        return self._digest.mean()

    def quantile(self, double p):
        """Compute the quantile for a given probability.

        Args:
            p (float): Probability value in [0, 1].

        Returns:
            float: Estimated quantile value.
        """
        return self._digest.quantile(p)

    def cdf(self, double k):
        """Compute the cumulative distribution function at k.

        Args:
            k (float): Point at which to evaluate the CDF.

        Returns:
            float: Estimated CDF value in [0, 1].
        """
        return self._digest.cdf(k)

    def ccdf(self, double k):
        """Compute the complementary CDF (1 - CDF) at k.

        Args:
            k (float): Point at which to evaluate the CCDF.

        Returns:
            float: Estimated CCDF value in [0, 1].
        """
        return self._digest.ccdf(k)

    def dcdf(self, double k):
        """Compute the derivative of the CDF (PDF) at k.

        Args:
            k (float): Point at which to evaluate the PDF.

        Returns:
            float: Estimated PDF value.
        """
        return self._digest.dcdf(k)

    def dccdf(self, double k):
        """Compute the derivative of the CCDF at k.

        Args:
            k (float): Point at which to evaluate dCCDF/dk.

        Returns:
            float: Derivative of the CCDF (always <= 0).
        """
        return self._digest.dccdf(k)

    def pmf(self, int kk):
        """Compute the probability mass function at integer point kk.

        Args:
            kk (int): Integer point at which to evaluate the PMF.

        Returns:
            float: Estimated probability mass at kk.
        """
        return self._digest.pmf(kk)

    def sample(self, size=1):
        """Sample a value from the distribution.

        Args:
            size (int, optional): Currently unused; always returns one sample.

        Returns:
            int: A sampled integer value from the distribution.
        """
        return self._digest.sample()

    # --- Serialisation helpers (mirror Python Digest API) ------------------

    def setBins(self, bins):
        """Set the bins (centroids) array.

        Args:
            bins (array-like): Centroid values to set.
        """
        cdef vector[double] v
        v.reserve(len(bins))
        for b in bins:
            v.push_back(<double>b)
        self._digest.setBins(v)

    def getBins(self):
        """Get the bins (centroids) array.

        Returns:
            np.ndarray: Array of centroid values.
        """
        cdef vector[double] v = self._digest.getBins()
        return np.array(v)

    def setWeights(self, cnts):
        """Set the weights (counts) array.

        Args:
            cnts (array-like): Centroid weights to set.
        """
        cdef vector[double] v
        v.reserve(len(cnts))
        for c in cnts:
            v.push_back(<double>c)
        self._digest.setCnts(v)

    def getWeights(self):
        """Get the weights (counts) array.

        Returns:
            np.ndarray: Array of centroid weights.
        """
        cdef vector[double] v = self._digest.getCnts()
        return np.array(v)

    def setActiveBinCount(self, int nActive):
        """Set the number of active centroids.

        Args:
            nActive (int): Number of active centroids.
        """
        self._digest.setActiveBinCount(nActive)

    def getActiveBinCount(self):
        """Get the number of active centroids.

        Returns:
            int: Current number of active centroids.
        """
        return self._digest.getActiveBinCount()

    def _findLastLesserOrEqualIndex(self, double point):
        """Find the index of the last centroid <= point (mirrors Python Digest API)."""
        return self._digest.findLastLesserOrEqualIndex(point)

    def _shiftRightAndInsert(self, int idx, double point, double count):
        """Insert centroid at idx+1 shifting right (mirrors Python Digest API)."""
        self._digest.shiftRightAndInsert(idx, point, count)

    def _shiftLeftAndOverride(self, int idx):
        """Remove centroid at idx by shifting left (mirrors Python Digest API)."""
        self._digest.shiftLeftAndOverride(idx)

    def _assertCompiled(self):
        """Always passes — this is a compiled C++ extension."""
        pass

    @property
    def maxBins(self):
        """Maximum number of centroids (read-only)."""
        return self._digest.getMaxBins()

    @property
    def nActive(self):
        """Current number of active centroids (read-only)."""
        return self._digest.getActiveBinCount()
