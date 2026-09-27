"""Forecast timing and paired-state contracts, independent of fitted smoke tests."""

from dataclasses import replace

import numpy as np
import pytest

from srvar import Dataset, ElbSpec, VolatilitySpec
from srvar.api import forecast
from srvar.results import FitResult
from srvar.scenario import conditional_forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig, ShockSpec


def _fit(mode: str = "homoskedastic", *, elb: bool = False) -> FitResult:
    ds = Dataset.from_arrays(values=np.zeros((3, 2)), variables=["r", "x"])
    beta = np.zeros((2, 2, 2))
    beta[:, 0, 0] = [0.5, 0.25] if elb else [0.0, 0.0]
    latent = np.zeros((2, 3, 2))
    latent[:, -1, 0] = [-2.0, -20.0]
    fit = FitResult(
        dataset=ds,
        model=ModelSpec(
            p=1,
            include_intercept=False,
            elb=ElbSpec(bound=0, applies_to=["r"]) if elb else None,
            volatility=None if mode == "homoskedastic" else VolatilitySpec(covariance=mode),
        ),
        prior=PriorSpec.niw_default(k=2, n=2),
        sampler=SamplerConfig(draws=2, burn_in=0),
        posterior=None,
        beta_draws=beta,
        latent_draws=latent if elb else None,
        latent_dataset=Dataset.from_arrays(values=latent[0], variables=ds.variables)
        if elb
        else None,
    )
    if mode == "homoskedastic":
        return replace(fit, sigma_draws=np.repeat(np.eye(2)[None] * 1e-24, 2, axis=0))
    fit = replace(
        fit,
        h_draws=np.full((2, 2, 2), -60.0 if elb else 0.0),
        sigma_eta2_draws=np.full((2, 2), 0.0 if elb else 0.25),
    )
    if mode == "triangular":
        return replace(fit, q_draws=np.repeat(np.eye(2)[None], 2, axis=0))
    if mode == "factor":
        return replace(
            fit,
            lambda_draws=np.ones((2, 2, 1)),
            h_factor_draws=np.full((2, 2, 1), -60.0 if elb else 0.0),
            sigma_eta2_factor_draws=np.full((2, 1), 0.0 if elb else 0.25),
        )
    return fit


class _UnitNormals:
    """Deterministic innovations expose state ordering without Monte Carlo error."""

    def integers(self, low, high=None, size=None):
        return np.zeros(size, dtype=int)

    def normal(self, size=None):
        return np.ones(size)


@pytest.mark.parametrize("mode", ["diagonal", "triangular", "factor"])
def test_future_volatility_precedes_each_observation(mode) -> None:
    result = forecast(_fit(mode), [1, 2], draws=1, rng=_UnitNormals())
    multiplier = 2 if mode == "factor" else 1
    expected = multiplier * np.exp(0.5 * np.array([0.5, 1.0]))
    np.testing.assert_allclose(result.draws[0, :, 0], expected)


@pytest.mark.parametrize("mode", ["diagonal", "triangular"])
def test_ar1_future_drift_and_persistence_precede_observation(mode) -> None:
    fit = _fit(mode)
    fit = replace(
        fit,
        model=replace(fit.model, volatility=VolatilitySpec(covariance=mode, dynamics="ar1")),
        h_draws=np.full((2, 2, 2), 2.0),
        sv_gamma0_draws=np.full((2, 2), 0.1),
        sv_phi_draws=np.full((2, 2), 0.5),
    )
    result = forecast(fit, [1, 2], draws=1, rng=_UnitNormals())
    np.testing.assert_allclose(result.draws[0, :, 0], np.exp(0.5 * np.array([1.6, 1.4])))


def test_one_step_sv_variance_integrates_future_state() -> None:
    fit = replace(_fit("diagonal"), sigma_eta2_draws=np.ones((2, 2)))
    result = forecast(fit, [1], draws=16000, rng=np.random.default_rng(19))
    # Heavy-tailed lognormal variance mixture: allow 10% at a fixed seed.
    assert np.var(result.draws[:, 0, 0]) == pytest.approx(np.exp(0.5), rel=0.10)


@pytest.mark.parametrize("mode", ["homoskedastic", "diagonal", "triangular", "factor"])
def test_latent_lags_are_paired_with_selected_coefficients(mode) -> None:
    result = forecast(_fit(mode, elb=True), [1], draws=100, rng=np.random.default_rng(2))
    latent = result.latent_draws[:, 0, 0]
    assert np.all(np.isclose(latent, -1.0) | np.isclose(latent, -5.0))
    assert np.any(np.isclose(latent, -1.0)) and np.any(np.isclose(latent, -5.0))


@pytest.mark.parametrize("mode", ["homoskedastic", "diagonal", "triangular", "factor"])
def test_latent_pairing_survives_stationarity_filter(mode) -> None:
    fit = _fit(mode, elb=True)
    beta = fit.beta_draws.copy()
    beta[0, 0, 0] = 1.2
    result = forecast(
        replace(fit, beta_draws=beta),
        [1],
        draws=20,
        stationarity="reject",
        rng=np.random.default_rng(5),
    )
    np.testing.assert_allclose(result.latent_draws[:, 0, 0], -5.0, atol=1e-9)


@pytest.mark.parametrize("latent", [None, np.zeros((1, 3, 2)), np.full((2, 3, 2), np.nan)])
def test_censored_terminal_lags_require_valid_paired_draws(latent) -> None:
    with pytest.raises(ValueError, match="latent_draws"):
        forecast(replace(_fit(elb=True), latent_draws=latent), [1], draws=2)


def test_uncensored_terminal_lags_need_no_latent_draws() -> None:
    fit = _fit(elb=True)
    ds = Dataset.from_arrays(values=np.ones((3, 2)), variables=["r", "x"])
    result = forecast(
        replace(fit, dataset=ds, latent_draws=None), [1], draws=20, rng=np.random.default_rng(2)
    )
    assert np.all(np.isclose(result.draws[:, 0, 0], 0.5) | np.isclose(result.draws[:, 0, 0], 0.25))


def test_gaussian_scenario_uses_paired_latent_lags() -> None:
    result = conditional_forecast(
        _fit(elb=True), [1], constraints={"x": {1: 1.0}}, draws=100, rng=np.random.default_rng(2)
    )
    latent = result.latent_draws[:, 0, 0]
    assert np.all(np.isclose(latent, -1.0) | np.isclose(latent, -5.0))
    assert np.any(np.isclose(latent, -5.0))
    np.testing.assert_allclose(result.draws[:, 0, 1], 1.0)


@pytest.mark.parametrize("family", ["student_t", "mixture_outlier"])
def test_robust_scenario_rejects_constraints_but_preserves_unconstrained(family) -> None:
    fit = _fit()
    fit = replace(
        fit,
        model=replace(fit.model, shocks=ShockSpec(family=family)),
        sigma_draws=np.repeat(np.eye(2)[None], 2, axis=0),
    )
    with pytest.raises(ValueError, match="Gaussian"):
        conditional_forecast(fit, [1], constraints={"x": {1: 0.0}}, draws=4)
    expected = forecast(fit, [1], draws=10, rng=np.random.default_rng(9))
    actual = conditional_forecast(fit, [1], constraints={}, draws=10, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(expected.draws, actual.draws)
