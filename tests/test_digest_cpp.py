"""
Tests for ranvar.digest_cpp.Digest — the C++ backed Cython extension type.

The test suite mirrors test_digest.py so both implementations are held to the
same behavioural contract.
"""

import copy
import pickle

import numpy as np
import pytest

from ranvar.digest_cpp import Digest


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
