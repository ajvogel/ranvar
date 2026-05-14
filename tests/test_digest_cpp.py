"""
Tests for ranvar.digest_cpp.Digest — the C++ backed Cython extension type.

The test suite mirrors test_digest.py so both implementations are held to the
same behavioural contract.
"""

import copy
import pickle

import numpy as np
import pytest

from ranvar.cdigest import Digest, DigestArray


def _make_digest(maxBins=32, N=10_000, seed=31337):
    np.random.seed(seed)
    data = np.random.randn(N) * 100 + 100
    d = Digest(maxBins=maxBins)
    d.fit(data)
    return d


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_nActiveCount():
    np.random.seed(0)
    data = np.random.randn(100) * 100 + 100
    d = Digest(maxBins=16)
    for v in data:
        d.add(v)
    assert d.getActiveBinCount() == 16


def test_freqAddsUp():
    N = 100
    np.random.seed(0)
    data = np.random.randn(N) * 100 + 100
    d = Digest(maxBins=16)
    for v in data:
        d.add(v)
    assert sum(d.getWeights()) == N


# ---------------------------------------------------------------------------
# Normal approximation — CDF (68-95-99.7 rule)
# ---------------------------------------------------------------------------

def test_normalApprox():
    d = _make_digest(maxBins=32)
    mu, sigma = 100.0, 100.0

    prob1 = d.cdf(mu + sigma)     - d.cdf(mu - sigma)
    prob2 = d.cdf(mu + 2 * sigma) - d.cdf(mu - 2 * sigma)
    prob3 = d.cdf(mu + 3 * sigma) - d.cdf(mu - 3 * sigma)

    assert abs(0.6827 - prob1) <= 1e-1
    assert abs(0.9545 - prob2) <= 1e-2
    assert abs(0.9973 - prob3) <= 1e-3


# ---------------------------------------------------------------------------
# Normal approximation — quantile
# ---------------------------------------------------------------------------

def test_normalApprox_quantile():
    d = _make_digest(maxBins=64)

    # Expected (value, probability) pairs for Normal(100, 100) with seed 31337
    DATA = [
        (-64.485, 0.05),
        (32.551,  0.25),
        (100.00,  0.50),
        (167.449, 0.75),
        (264.485, 0.95),
    ]

    for v, p in DATA:
        dv = abs((v - d.quantile(p)) / v)
        assert dv <= 2.5e-2, f"quantile({p}): expected≈{v}, got {d.quantile(p)}"


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_ccdf():
    d = _make_digest()
    mu, sigma = 100.0, 100.0
    for k in [mu - 2 * sigma, mu - sigma, mu, mu + sigma, mu + 2 * sigma]:
        assert abs(d.ccdf(k) - (1.0 - d.cdf(k))) < 1e-12

    assert d.ccdf(d.lower() - 1) == 1.0
    assert d.ccdf(d.upper() + 1) == 0.0


def test_dcdf():
    d = _make_digest()
    mu, sigma = 100.0, 100.0
    h = 0.001
    for k in [mu - sigma, mu, mu + sigma]:
        numerical = (d.cdf(k + h) - d.cdf(k - h)) / (2 * h)
        assert abs(d.dcdf(k) - numerical) < 1e-4

    for k in [mu - sigma, mu, mu + sigma]:
        assert d.dcdf(k) >= 0

    assert d.dcdf(d.lower() - 1) == 0.0
    assert d.dcdf(d.upper() + 1) == 0.0


def test_dccdf():
    d = _make_digest()
    mu, sigma = 100.0, 100.0
    for k in [mu - sigma, mu, mu + sigma]:
        assert abs(d.dccdf(k) - (-d.dcdf(k))) < 1e-12
        assert d.dccdf(k) <= 0

    assert d.dccdf(d.lower() - 1) == 0.0
    assert d.dccdf(d.upper() + 1) == 0.0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_pickle():
    d = _make_digest(maxBins=64)
    restored = pickle.loads(pickle.dumps(d))

    np.testing.assert_array_equal(restored.getWeights(), d.getWeights())
    np.testing.assert_array_equal(restored.getBins(),    d.getBins())


def test_copy():
    d = _make_digest(maxBins=64)
    cloned = copy.deepcopy(d)

    np.testing.assert_array_equal(cloned.getWeights(), d.getWeights())
    np.testing.assert_array_equal(cloned.getBins(),    d.getBins())


# ---------------------------------------------------------------------------
# Boundary behaviour
# ---------------------------------------------------------------------------

def test_cdf_boundaries():
    d = _make_digest()
    assert d.cdf(d.lower()) == 0.0
    assert d.cdf(d.upper()) == 1.0


def test_quantile_boundaries():
    d = _make_digest()
    assert d.quantile(0.0) == d.lower()
    assert d.quantile(1.0) == d.upper()


# ---------------------------------------------------------------------------
# sample() returns a numeric value within bounds
# ---------------------------------------------------------------------------

def test_sample():
    d = _make_digest()
    val = d.sample()
    assert isinstance(val, int)
    assert d.lower() - 1 <= val <= d.upper() + 1


# ---------------------------------------------------------------------------
# operator+ — Discrete Centroid Convolution
# ---------------------------------------------------------------------------

def test_add_returns_digest():
    d1 = _make_digest()
    d2 = _make_digest(seed=42)
    result = d1 + d2
    assert isinstance(result, Digest)


def test_add_mean_is_sum_of_means():
    """E[X + Y] == E[X] + E[Y] for independent X, Y."""
    d1 = _make_digest(seed=1)
    d2 = _make_digest(seed=2)
    result = d1 + d2
    assert abs(result.mean() - (d1.mean() + d2.mean())) < 1.0


def test_add_bounds():
    """lower(X+Y) == lower(X)+lower(Y) and upper(X+Y) == upper(X)+upper(Y)."""
    d1 = _make_digest(seed=10)
    d2 = _make_digest(seed=20)
    result = d1 + d2
    assert result.lower() == d1.lower() + d2.lower()
    assert result.upper() == d1.upper() + d2.upper()


def test_add_weights_sum_to_one():
    """Convolution weights are normalised probabilities summing to 1."""
    d1 = _make_digest(seed=3)
    d2 = _make_digest(seed=4)
    result = d1 + d2
    assert abs(sum(result.getWeights()) - 1.0) < 1e-10


def test_add_constant_shifts_distribution():
    """Adding a degenerate (single-point) digest shifts every quantile by that constant."""
    np.random.seed(0)
    data = np.random.randn(5_000) * 50 + 200
    d = Digest(maxBins=64)
    d.fit(data)

    constant = Digest(maxBins=64)
    constant.add(100.0)

    shifted = d + constant

    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        assert abs(shifted.quantile(p) - (d.quantile(p) + 100.0)) < 5.0


def test_add_symmetry():
    """X + Y and Y + X should have identical means and near-identical quantiles.

    Means are exactly equal because both orderings sum the same centroid pairs.
    Quantiles may differ by a small amount because t-digest centroid merging is
    insertion-order-sensitive, so the tolerance is set to a few distribution units.
    """
    d1 = _make_digest(seed=7)
    d2 = _make_digest(seed=8)
    ab = d1 + d2
    ba = d2 + d1
    assert abs(ab.mean() - ba.mean()) < 1e-10
    for p in [0.1, 0.5, 0.9]:
        assert abs(ab.quantile(p) - ba.quantile(p)) < 10.0


def test_add_two_normals_variance():
    """Sum of two Normal(mu, sigma) distributions has variance 2*sigma^2."""
    np.random.seed(99)
    mu, sigma = 0.0, 50.0
    data1 = np.random.randn(20_000) * sigma + mu
    data2 = np.random.randn(20_000) * sigma + mu

    d1 = Digest(maxBins=128)
    d1.fit(data1)
    d2 = Digest(maxBins=128)
    d2.fit(data2)

    result = d1 + d2

    # For N(0, sigma) + N(0, sigma) = N(0, sqrt(2)*sigma), the IQR ≈ 1.349 * sqrt(2)*sigma
    expected_iqr = 1.349 * (2 ** 0.5) * sigma
    actual_iqr = result.quantile(0.75) - result.quantile(0.25)
    assert abs(actual_iqr - expected_iqr) / expected_iqr < 0.05


# ---------------------------------------------------------------------------
# DigestArray operator+ — element-wise addition
# ---------------------------------------------------------------------------

def _make_array(length, seed=0, mu=100.0, sigma=50.0, N=5_000, maxBins=32):
    np.random.seed(seed)
    da = DigestArray(length, maxBins)
    for i in range(length):
        data = np.random.randn(N) * sigma + mu * (i + 1)
        da.fitAt(i, data)
    return da


def test_array_add_returns_digestarray():
    da1 = _make_array(4)
    da2 = _make_array(4, seed=1)
    result = da1 + da2
    assert isinstance(result, DigestArray)


def test_array_add_equal_length():
    da1 = _make_array(5)
    da2 = _make_array(5, seed=2)
    result = da1 + da2
    assert len(result) == 5


def test_array_add_truncates_to_shorter():
    da_long  = _make_array(6)
    da_short = _make_array(3, seed=3)
    assert len(da_long  + da_short) == 3
    assert len(da_short + da_long)  == 3


def test_array_add_element_means():
    """Each element's mean in the result equals the sum of the input element means."""
    da1 = _make_array(4, seed=10)
    da2 = _make_array(4, seed=20)
    result = da1 + da2
    for i in range(len(result)):
        expected = da1[i].mean() + da2[i].mean()
        assert abs(result[i].mean() - expected) < 1.0


def test_array_add_element_bounds():
    """Min/max centroids of each result element equal the sums of the input min/max centroids.

    The t-digest never merges its first or last centroid, so the extreme values
    are preserved exactly as floating-point sums.
    """
    da1 = _make_array(3, seed=5, N=10_000)
    da2 = _make_array(3, seed=6, N=10_000)
    result = da1 + da2
    for i in range(len(result)):
        assert result[i].centroids()[0]  == da1[i].centroids()[0]  + da2[i].centroids()[0]
        assert result[i].centroids()[-1] == da1[i].centroids()[-1] + da2[i].centroids()[-1]


def test_array_add_weights_sum_to_one():
    """Convolution preserves normalised weights at every position."""
    da1 = _make_array(4, seed=7)
    da2 = _make_array(4, seed=8)
    result = da1 + da2
    for i in range(len(result)):
        assert abs(sum(result[i].getWeights()) - 1.0) < 1e-10
