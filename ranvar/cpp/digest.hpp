#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace ranvar {

class Digest {
public:
    explicit Digest(int maxBins = 32);

    void fit(const std::vector<double>& x);
    void add(double point, double count = 1.0);

    std::vector<double> centroids() const;
    std::vector<double> weights() const;

    int    lower() const;
    int    upper() const;
    double mean() const;
    double quantile(double p) const;
    double cdf(double k) const;
    double ccdf(double k) const;
    double dcdf(double k) const;
    double dccdf(double k) const;
    double pmf(int kk) const;
    int    sample();

    int    getMaxBins() const;
    int    getActiveBinCount() const;
    void   setActiveBinCount(int nActive);
    const std::vector<double>& getBins() const;
    const std::vector<double>& getCnts() const;
    void   setBins(const std::vector<double>& bins);
    void   setCnts(const std::vector<double>& cnts);

    // Exposed for testing — mirrors the cpdef helpers in the Cython Python version
    int    findLastLesserOrEqualIndex(double point) const;
    void   shiftRightAndInsert(int idx, double point, double count);
    void   shiftLeftAndOverride(int idx);

private:
    int maxBins_;
    int nActive_;
    std::vector<double> bins_;
    std::vector<double> cnts_;

    int    findMinimumDifference() const;
    double sumWeights() const;
    void   _add(double point, double count);
};

double _rand();
int    _randint(double l, double h);
double _randnorm(double mu, double stdev);

} // namespace ranvar
