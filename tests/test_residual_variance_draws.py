"""Exact boundary references for the independent inverse-gamma variance update."""

import numpy as np
import pytest
from scipy.stats import invgamma, kstest

from srvar.samplers_dl import _dl_sample_beta_sigma


class FixedGamma:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def standard_normal(self, size):
        return np.zeros(size)

    def gamma(self, *, shape, scale):
        self.calls.append((float(shape), float(scale)))
        return self.value


def variance_draw(rng, *, rate=1.0, nu0=2.0, y=None):
    if y is None:
        y = np.zeros((2, 1))
    return _dl_sample_beta_sigma(
        x=np.zeros((2, 1)),
        y=y,
        m0=np.zeros((1, 1)),
        inv_v0_vec=np.ones(1),
        s0=np.array([[rate]]),
        nu0=nu0,
        sigma=np.eye(1),
        rng=rng,
    )[1].item()


@pytest.mark.parametrize("expected", [5e-15, 2e13, 1e308])
def test_representable_variance_is_not_clipped_or_overflowed(expected):
    rng = FixedGamma(1.0 / expected)
    actual = variance_draw(rng)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0)


def test_variance_uses_posterior_rate_and_unit_gamma():
    rng = FixedGamma(4.0)
    # RSS=25; posterior rate=2.5+25/2=15 and shape=2+2/2=3.
    assert variance_draw(rng, rate=2.5, y=np.array([[3.0], [4.0]])) == 3.75
    assert rng.calls == [(3.0, 1.0)]


def test_subnormal_rate_does_not_require_its_reciprocal():
    rng = FixedGamma(1.0)
    assert variance_draw(rng, rate=1e-310) == 1e-310
    assert rng.calls == [(3.0, 1.0)]


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_invalid_gamma_draw_fails_explicitly(gamma):
    with pytest.raises(FloatingPointError, match="Gamma draw.*finite and positive"):
        variance_draw(FixedGamma(gamma))


@pytest.mark.parametrize("rate,gamma", [(1e308, 1e-308), (1e-308, 1e308)])
def test_unrepresentable_variance_fails_explicitly(rate, gamma):
    with pytest.raises(FloatingPointError, match="variance draw.*finite and positive"):
        variance_draw(FixedGamma(gamma), rate=rate)


@pytest.mark.parametrize("nu0", [-2.0, -1.0, np.nan, np.inf])
def test_invalid_conditional_shape_fails_before_gamma(nu0):
    rng = FixedGamma(1.0)
    with pytest.raises(ValueError, match="conditional shape.*finite and positive"):
        variance_draw(rng, nu0=nu0)
    assert rng.calls == []


@pytest.mark.parametrize("rate", [-1.0, 0.0, np.nan, np.inf])
def test_invalid_conditional_rate_fails_before_gamma(rate):
    rng = FixedGamma(1.0)
    with pytest.raises(ValueError, match="conditional rate.*finite and positive"):
        variance_draw(rng, rate=rate)
    assert rng.calls == []


@pytest.mark.parametrize("rate", [1e-14, 2.0, 1e14])
def test_variance_conditional_matches_independent_inverse_gamma_reference(rate):
    rng = np.random.default_rng(927)
    # X=0 decouples beta; each variance is an independent IG(5,rate) draw.
    draws = np.array([variance_draw(rng, rate=rate, nu0=4.0) for _ in range(4000)])
    normalised = draws / rate
    # A fixed seed and a conservative KS threshold avoid a fragile moment test
    # for the heavy upper tail. Mean tolerance is about six standard errors.
    assert kstest(normalised, invgamma(a=5).cdf).statistic < 0.035
    assert normalised.mean() == pytest.approx(0.25, rel=0.06)
