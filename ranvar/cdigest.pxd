# Cython declaration file for the C++ Digest / DigestArray wrappers.
#
# Other Cython modules can cimport this to access the C++ class declarations
# and the Python extension types at C speed:
#
#   from ranvar.cdigest cimport CppDigest, Digest, CppDigestArray, DigestArray

from libcpp.vector cimport vector

# ---------------------------------------------------------------------------
# Extern declarations for the C++ ranvar::Digest class
# ---------------------------------------------------------------------------

cdef extern from "digest.hpp" namespace "ranvar":

    cdef cppclass CppDigest "ranvar::Digest":
        CppDigest(int maxBins) except +

        void fit(const vector[double]& x) except +
        void add(double point, double count) except +

        vector[double] centroids() const
        vector[double] weights()   const

        int    lower() const
        int    upper() const
        double mean()  const
        double quantile(double p) const
        double cdf(double k)      const
        double ccdf(double k)     const
        double dcdf(double k)     const
        double dccdf(double k)    const
        double pmf(int kk)        const
        int    sample()

        int    getMaxBins()        const
        int    getActiveBinCount() const
        void   setActiveBinCount(int nActive)
        const vector[double]& getBins() const
        const vector[double]& getCnts() const
        void   setBins(const vector[double]& bins)
        void   setCnts(const vector[double]& cnts)

        int  findLastLesserOrEqualIndex(double point) const
        void shiftRightAndInsert(int idx, double point, double count)
        void shiftLeftAndOverride(int idx)

cdef extern from "digest.hpp" namespace "ranvar":
    double _rand()
    int    _randint(double l, double h)
    double _randnorm(double mu, double stdev)

# ---------------------------------------------------------------------------
# Extern declarations for the C++ ranvar::DigestArray class
# ---------------------------------------------------------------------------

cdef extern from "digest.hpp" namespace "ranvar":

    cdef cppclass CppDigestArray "ranvar::DigestArray":
        CppDigestArray(int length, int maxBins) except +

        int size()       const
        int getMaxBins() const

        int    getMaxBinsAt(int idx)        const
        int    getActiveBinCountAt(int idx) const
        const vector[double]& getBinsAt(int idx) const
        const vector[double]& getCntsAt(int idx) const

        void set(int idx, const CppDigest& d)
        void remove(int idx)
        void append(const CppDigest& d)
        void appendEmpty()

        vector[double] sample()

# ---------------------------------------------------------------------------
# Python extension type declarations (for cimport by other Cython modules)
# ---------------------------------------------------------------------------

cdef class Digest:
    cdef CppDigest* _digest
    cdef void _set_bins_list(self, list bins_list)
    cdef void _set_cnts_list(self, list cnts_list)

cdef class DigestArray:
    cdef CppDigestArray* _array
