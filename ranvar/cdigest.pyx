# cython: language_level=3
# distutils: language = c++
# distutils: include_dirs = ranvar/cpp

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from ranvar.cdigest cimport CppDigest, CppDigestArray


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
        # maxBins < 0 is a private sentinel used by from_ptr to skip allocation.
        if maxBins >= 0:
            self._digest = new CppDigest(maxBins)
            self._owned = True
        else:
            self._digest = NULL
            self._owned = False
        self._parent = None

    def __dealloc__(self):
        if self._owned and self._digest != NULL:
            del self._digest

    @staticmethod
    cdef Digest from_ptr(CppDigest* ptr, object parent=None):
        """Return a Digest that is a live view into an existing CppDigest.

        The view does *not* own the pointer and will never free it.  ``parent``
        should be the Python object (typically a DigestArray) that owns the
        underlying memory; holding this reference prevents premature GC.

        Warning: calling remove() or append() on the parent DigestArray may
        invalidate this pointer if the internal vector reallocates.
        """
        cdef Digest d = Digest.__new__(Digest, -1)
        d._digest = ptr
        d._owned = False
        d._parent = parent
        return d

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
        return self._digest.lower()

    def upper(self):
        return self._digest.upper()

    def mean(self):
        return self._digest.mean()

    def quantile(self, double p):
        return self._digest.quantile(p)

    def cdf(self, double k):
        return self._digest.cdf(k)

    def ccdf(self, double k):
        return self._digest.ccdf(k)

    def dcdf(self, double k):
        return self._digest.dcdf(k)

    def dccdf(self, double k):
        return self._digest.dccdf(k)

    def pmf(self, int kk):
        return self._digest.pmf(kk)

    def sample(self, size=1):
        return self._digest.sample()

    # --- Serialisation helpers (mirror Python Digest API) ------------------

    def setBins(self, bins):
        cdef vector[double] v
        v.reserve(len(bins))
        for b in bins:
            v.push_back(<double>b)
        self._digest.setBins(v)

    def getBins(self):
        cdef vector[double] v = self._digest.getBins()
        return np.array(v)

    def setWeights(self, cnts):
        cdef vector[double] v
        v.reserve(len(cnts))
        for c in cnts:
            v.push_back(<double>c)
        self._digest.setCnts(v)

    def getWeights(self):
        cdef vector[double] v = self._digest.getCnts()
        return np.array(v)

    def setActiveBinCount(self, int nActive):
        self._digest.setActiveBinCount(nActive)

    def getActiveBinCount(self):
        return self._digest.getActiveBinCount()

    def _findLastLesserOrEqualIndex(self, double point):
        return self._digest.findLastLesserOrEqualIndex(point)

    def _shiftRightAndInsert(self, int idx, double point, double count):
        self._digest.shiftRightAndInsert(idx, point, count)

    def _shiftLeftAndOverride(self, int idx):
        self._digest.shiftLeftAndOverride(idx)

    def _assertCompiled(self):
        pass

    @property
    def maxBins(self):
        return self._digest.getMaxBins()

    @property
    def nActive(self):
        return self._digest.getActiveBinCount()


# ---------------------------------------------------------------------------
# DigestArray
# ---------------------------------------------------------------------------

cdef class DigestArray:
    """
    A fixed-length array of Digest objects backed by C++ ranvar::DigestArray.

    Pre-populated with empty Digest instances on construction. Supports list-like
    access via dunder methods. __getitem__ returns a *live view* into the
    underlying element — mutations are reflected immediately in the array:

        da[i].fit(data)   # works — modifies the array element in-place

    Warning: the view's pointer is invalidated if the array is structurally
    modified (remove / append) after the view is obtained.

    Args:
        length (int): Number of Digest instances to pre-allocate.
        maxBins (int, optional): Maximum centroids per Digest. Defaults to 32.

    Example:
        >>> da = DigestArray(5)
        >>> len(da)
        5
        >>> da[0].fit([1, 2, 3])   # mutates the element directly
        >>> da.sample()   # numpy array of 5 sampled values
    """

    def __cinit__(self, int length, int maxBins=32):
        self._array = new CppDigestArray(length, maxBins)

    def __dealloc__(self):
        del self._array

    # --- List dunder methods -----------------------------------------------

    def __len__(self):
        return self._array.size()

    def __getitem__(self, idx):
        cdef int n = self._array.size()
        cdef int i
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            return [self[i] for i in range(start, stop, step)]
        i = idx
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"DigestArray index {idx} out of range")
        return Digest.from_ptr(self._array.at(i), self)

    def __setitem__(self, idx, value):
        cdef int n = self._array.size()
        cdef int i
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            indices = list(range(start, stop, step))
            if len(indices) != len(value):
                raise ValueError(
                    f"slice assignment mismatch: {len(indices)} vs {len(value)}"
                )
            for i, v in zip(indices, value):
                self[i] = v
            return
        i = idx
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"DigestArray index {idx} out of range")
        if not isinstance(value, Digest):
            raise TypeError("DigestArray elements must be Digest instances")
        self._array.set(i, (<Digest>value)._digest[0])

    def __delitem__(self, int idx):
        cdef int n = self._array.size()
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"DigestArray index {idx} out of range")
        self._array.remove(idx)

    def __iter__(self):
        cdef int i
        for i in range(self._array.size()):
            yield self[i]

    def __contains__(self, item):
        return NotImplemented

    def __repr__(self):
        return f"DigestArray(length={self._array.size()}, maxBins={self._array.getMaxBins()})"

    # --- Mutating helpers --------------------------------------------------

    def append(self, digest=None):
        """Append a Digest to the end of the array.

        Args:
            digest (Digest, optional): Digest to append. Appends a new empty
                Digest if omitted.
        """
        if digest is None:
            self._array.appendEmpty()
        else:
            if not isinstance(digest, Digest):
                raise TypeError("DigestArray elements must be Digest instances")
            self._array.append((<Digest>digest)._digest[0])

    # --- Sampling ----------------------------------------------------------

    def sample(self):
        """Sample one value from each Digest and return as a numpy array.

        Returns:
            np.ndarray: 1-D array of length len(self); entry i is a sample
                drawn from the i-th Digest (returns 0 for empty Digests).
        """
        cdef vector[double] v = self._array.sample()
        return np.array(v)

    # --- Properties --------------------------------------------------------

    @property
    def maxBins(self):
        """Maximum number of centroids per Digest (read-only)."""
        return self._array.getMaxBins()
