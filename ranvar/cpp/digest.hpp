#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace ranvar {

// ---------------------------------------------------------------------------
// Module-level random helpers
// ---------------------------------------------------------------------------

static std::once_flag rand_seed_flag_;

static void ensureSeeded() {
    std::call_once(rand_seed_flag_, []() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    });
}

inline double _rand() {
    ensureSeeded();
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

inline int _randint(double l, double h) {
    double l2  = l - 1.0;
    double out = std::ceil((h - l2) * _rand() + l2);
    if (out < l) out = l;
    return static_cast<int>(out);
}

inline double _randnorm(double mu, double stdev) {
    double x, y, s;
    do {
        x = 2.0 * _rand() - 1.0;
        y = 2.0 * _rand() - 1.0;
        s = x * x + y * y;
    } while (s >= 1.0);
    double z = x * std::sqrt(-2.0 * std::log(s) / s);
    return z * stdev + mu;
}

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

class Digest {
public:
    explicit Digest(int maxBins = 32)
        : maxBins_(maxBins)
        , nActive_(0)
        , bins_(maxBins + 1, 0.0)
        , cnts_(maxBins + 1, 0.0)
    {}

    void fit(const std::vector<double>& x) {
        for (double val : x)
            add(val);
    }

    void add(double point, double count = 1.0) {
        _add(point, count);
    }

    std::vector<double> centroids() const {
        return std::vector<double>(bins_.begin(), bins_.begin() + nActive_);
    }

    std::vector<double> weights() const {
        return std::vector<double>(cnts_.begin(), cnts_.begin() + nActive_);
    }

    int lower() const { return static_cast<int>(bins_[0]); }
    int upper() const { return static_cast<int>(bins_[nActive_ - 1]); }

    double mean() const {
        double SxW = 0.0, W = 0.0;
        for (int i = 0; i < nActive_; ++i) {
            SxW += cnts_[i] * bins_[i];
            W   += cnts_[i];
        }
        return SxW / W;
    }

    double quantile(double p) const {
        if (p <= 0.0) return static_cast<double>(lower());
        if (p >= 1.0) return static_cast<double>(upper());

        double W  = sumWeights();
        double wi = 0.0;
        double w_ = p * W;

        for (int i = 0; i < nActive_ - 1; ++i) {
            double wGap;
            if (i == 0) {
                wGap = cnts_[i] + cnts_[i + 1] / 2.0;
            } else if (i == nActive_ - 1) {
                // unreachable: loop stops at nActive_-2, mirroring Python faithfully
                wGap = cnts_[i] / 2.0 + cnts_[i + 1];
            } else {
                wGap = cnts_[i] / 2.0 + cnts_[i + 1] / 2.0;
            }

            double wi_n = wi + wGap;

            if (wi <= w_ && w_ < wi_n) {
                double fraction = (w_ - wi) / wGap;
                return fraction * (bins_[i + 1] - bins_[i]) + bins_[i];
            }

            wi = wi_n;
        }

        return static_cast<double>(upper());
    }

    double cdf(double k) const {
        if (k <= static_cast<double>(lower())) return 0.0;
        if (k >= static_cast<double>(upper())) return 1.0;

        double som = 0.0;
        double W   = sumWeights();

        for (int i = 0; i < nActive_; ++i) {
            if (bins_[i] <= k && k < bins_[i + 1]) {
                double yi, yi_n;

                if (cnts_[i] > 1.0 && cnts_[i + 1] > 1.0) {
                    yi   = som + cnts_[i] / 2.0;
                    yi_n = yi  + (cnts_[i + 1] + cnts_[i]) / 2.0;
                } else if (cnts_[i] == 1.0 && cnts_[i + 1] > 1.0) {
                    yi   = som;
                    yi_n = yi + cnts_[i + 1] / 2.0;
                } else if (cnts_[i] > 1.0 && cnts_[i + 1] == 1.0) {
                    yi   = som + cnts_[i] / 2.0;
                    yi_n = yi  + cnts_[i] / 2.0;
                } else {
                    yi   = som;
                    yi_n = yi;
                }

                double g  = (yi_n - yi) / (bins_[i + 1] - bins_[i]);
                double yk = g * (k - bins_[i]) + yi;
                return yk / W;
            } else {
                som += cnts_[i];
            }
        }

        return 1.0;
    }

    double ccdf(double k) const { return 1.0 - cdf(k); }

    double dcdf(double k) const {
        if (k <= static_cast<double>(lower())) return 0.0;
        if (k >= static_cast<double>(upper())) return 0.0;

        double som = 0.0;
        double W   = sumWeights();

        for (int i = 0; i < nActive_; ++i) {
            if (bins_[i] <= k && k < bins_[i + 1]) {
                double yi, yi_n;

                if (cnts_[i] > 1.0 && cnts_[i + 1] > 1.0) {
                    yi   = som + cnts_[i] / 2.0;
                    yi_n = yi  + (cnts_[i + 1] + cnts_[i]) / 2.0;
                } else if (cnts_[i] == 1.0 && cnts_[i + 1] > 1.0) {
                    yi   = som;
                    yi_n = yi + cnts_[i + 1] / 2.0;
                } else if (cnts_[i] > 1.0 && cnts_[i + 1] == 1.0) {
                    yi   = som + cnts_[i] / 2.0;
                    yi_n = yi  + cnts_[i] / 2.0;
                } else {
                    yi   = som;
                    yi_n = yi;
                }

                double g = (yi_n - yi) / (bins_[i + 1] - bins_[i]);
                return g / W;
            } else {
                som += cnts_[i];
            }
        }

        return 0.0;
    }

    double dccdf(double k) const { return -dcdf(k); }

    double pmf(int kk) const { return cdf(kk + 0.5) - cdf(kk - 0.5); }

    int sample() {
        double p = _rand();
        return static_cast<int>(std::round(quantile(p)));
    }

    // --- Serialisation helpers -----------------------------------------------

    int    getMaxBins()        const { return maxBins_; }
    int    getActiveBinCount() const { return nActive_; }
    void   setActiveBinCount(int n)  { nActive_ = n; }
    const std::vector<double>& getBins() const { return bins_; }
    const std::vector<double>& getCnts() const { return cnts_; }
    void   setBins(const std::vector<double>& bins) { bins_ = bins; }
    void   setCnts(const std::vector<double>& cnts) { cnts_ = cnts; }

    // --- Exposed for testing / Cython ----------------------------------------

    int findLastLesserOrEqualIndex(double point) const {
        int idx = -1;
        while (true) {
            if ((bins_[idx + 1] > point) || (idx + 1 == nActive_))
                break;
            ++idx;
        }
        return idx;
    }

    void shiftRightAndInsert(int idx, double point, double count) {
        for (int j = nActive_ - 1; j > idx; --j) {
            bins_[j + 1] = bins_[j];
            cnts_[j + 1] = cnts_[j];
        }
        bins_[idx + 1] = point;
        cnts_[idx + 1] = count;
        ++nActive_;
    }

    void shiftLeftAndOverride(int idx) {
        for (int j = idx; j < nActive_ - 1; ++j) {
            bins_[j] = bins_[j + 1];
            cnts_[j] = cnts_[j + 1];
        }
        bins_[nActive_ - 1] = 0.0;
        cnts_[nActive_ - 1] = 0.0;
        --nActive_;
    }

private:
    int maxBins_;
    int nActive_;
    std::vector<double> bins_;
    std::vector<double> cnts_;

    int findMinimumDifference() const {
        int    minK    = -1;
        double minDiff = 9e9;
        for (int k = 1; k < nActive_ - 2; ++k) {
            double dB = bins_[k + 1] - bins_[k];
            if (dB < minDiff) {
                minDiff = dB;
                minK    = k;
            }
        }
        return minK;
    }

    double sumWeights() const {
        double sum = 0.0;
        for (int i = 0; i < nActive_; ++i)
            sum += cnts_[i];
        return sum;
    }

    void _add(double point, double count) {
        int idx = findLastLesserOrEqualIndex(point);

        if (idx >= 0 && bins_[idx] == point) {
            cnts_[idx] += count;
        } else {
            shiftRightAndInsert(idx, point, count);
        }

        if (nActive_ > maxBins_) {
            int    k    = findMinimumDifference();
            double sumC = cnts_[k] + cnts_[k + 1];
            bins_[k] = (bins_[k] * cnts_[k] + bins_[k + 1] * cnts_[k + 1]) / sumC;
            cnts_[k] = sumC;
            shiftLeftAndOverride(k + 1);
        }
    }
};

} // namespace ranvar
