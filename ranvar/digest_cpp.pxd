# Cython declaration file for the C++ Digest wrapper.
#
# Other Cython modules can cimport this to access the C++ class declarations
# and the Python extension type at C speed:
#
#   from ranvar.digest_cpp cimport CppDigest, Digest

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

cdef extern from "digest.hpp" namespace "ranvar":
    double _rand()
    int    _randint(double l, double h)
    double _randnorm(double mu, double stdev)

# ---------------------------------------------------------------------------
# Python extension type declaration (for cimport by other Cython modules)
# ---------------------------------------------------------------------------

cdef class Digest:
    cdef CppDigest* _digest
    cdef void _set_bins_list(self, list bins_list)
    cdef void _set_cnts_list(self, list cnts_list)
