from dataclasses import replace

import numpy as np
import pytest

from srvar.api import fit
from srvar.config import ConfigError, build_prior
from srvar.data.dataset import Dataset
from srvar.elb import ElbSpec
from srvar.samplers_svcov import _fit_svcov, _sample_beta_triangular_svrw
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


def _ar_variances(y, p, intercept):
    # Independent lag-matrix construction; no production design/scale helper.
    result = []
    for series in y.T:
        regressors = [series[p - lag : -lag] for lag in range(1, p + 1)]
        if intercept:
            regressors.insert(0, np.ones(len(series) - p))
        x = np.column_stack(regressors)
        residual = series[p:] - x @ np.linalg.lstsq(x, series[p:], rcond=None)[0]
        result.append(residual @ residual / max(len(residual) - x.shape[1], 1))
    return np.array(result)


@pytest.mark.parametrize("intercept", [False, True])
def test_dl_empirical_bayes_matches_ar_reference_and_rescales(intercept):
    y = np.random.default_rng(72).normal(size=(40, 3)) * [0.2, 2, 10]
    args = dict(
        k=6 + int(intercept),
        n=3,
        p=2,
        include_intercept=intercept,
        residual_prior="empirical_bayes",
    )
    prior = PriorSpec.from_dl(y=y, **args)
    assert prior.niw.nu0 == 2
    assert prior.residual_prior == "empirical_bayes"
    np.testing.assert_allclose(np.diag(prior.niw.s0), _ar_variances(y, 2, intercept))
    scaled = PriorSpec.from_dl(y=y * [2, 3, 4], **args)
    np.testing.assert_allclose(np.diag(scaled.niw.s0), np.diag(prior.niw.s0) * [4, 9, 16])


def test_dl_short_window_and_active_floor():
    prior = PriorSpec.from_dl(
        k=3, n=1, p=2, y=np.zeros((3, 1)), residual_prior="empirical_bayes", min_sigma2=0.05
    )
    np.testing.assert_array_equal(prior.niw.s0, [[0.05]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"residual_prior": "explicit"},
        {"residual_prior": "explicit", "nu0": 2},
        {"residual_prior": "explicit", "nu0": 0, "s0": np.eye(2)},
        {"residual_prior": "explicit", "nu0": float("nan"), "s0": np.eye(2)},
        {"residual_prior": "explicit", "nu0": 2, "s0": [[1, 0.1], [0.1, 1]]},
        {"residual_prior": "explicit", "nu0": 2, "s0": np.eye(3)},
        {"residual_prior": "explicit", "nu0": 2, "s0": np.diag([1, -1])},
        {"residual_prior": "explicit", "nu0": 2, "s0": np.eye(2), "p": 1},
        {"residual_prior": "empirical_bayes", "p": 1, "y": np.ones((5, 2)), "nu0": 2},
        {"residual_prior": "empirical_bayes", "p": 1, "y": np.ones((5, 2)), "min_sigma2": 0},
        {"residual_prior": "empirical_bayes", "p": 1, "y": np.ones((5, 2)), "min_sigma2": np.inf},
        {"residual_prior": "empirical_bayes", "p": 1, "y": np.ones((1, 2))},
        {"residual_prior": "empirical_bayes", "p": 1, "y": np.full((5, 2), np.nan)},
        {"residual_prior": "empirical_bayes", "p": 2, "y": np.ones((5, 2))},
    ],
)
def test_dl_rejects_ambiguous_or_invalid_residual_prior(kwargs):
    with pytest.raises(ValueError):
        PriorSpec.from_dl(k=3, n=2, **kwargs)


def test_dl_config_default_explicit_and_unknown_inputs():
    ds = Dataset.from_arrays(
        values=np.random.default_rng(31).normal(size=(20, 2)), variables=["a", "b"]
    )
    model = ModelSpec(p=1)
    eb = build_prior({"prior": {"family": "dl"}}, dataset=ds, model=model)
    np.testing.assert_allclose(np.diag(eb.niw.s0), _ar_variances(ds.values, 1, True))
    assert eb.niw.nu0 == 2
    explicit = build_prior(
        {
            "prior": {
                "family": "dl",
                "dl": {"residual_prior": "explicit", "nu0": 3, "s0": [[2, 0], [0, 7]]},
            }
        },
        dataset=ds,
        model=model,
    )
    np.testing.assert_array_equal(explicit.niw.s0, [[2, 0], [0, 7]])
    for hyp in [{"nu0": 3}, {"residual_prior": "explicit", "nu0": 3}, {"typo": 1}, {"s0": None}]:
        with pytest.raises(ConfigError):
            build_prior({"prior": {"family": "dl", "dl": hyp}}, dataset=ds, model=model)


def _triangular_reference(intercept=True):
    rng = np.random.default_rng(612)
    x = rng.normal(size=(14, 2))
    if intercept:
        x[:, 0] = 1
    y = rng.normal(size=(14, 3))
    q = np.array([[1, 0.7, -0.4], [0, 1, 0.3], [0, 0, 1.0]])
    h = rng.normal(size=y.shape) * 0.5
    m = rng.normal(size=(2, 3))
    # Explicit equation blocks independent of flattening in production.
    blocks = [np.diag([0.5, 2.0]), np.diag([4.0, 0.8]), np.diag([1.3, 7.0])]
    precision = np.zeros((6, 6))
    for j, block in enumerate(blocks):
        precision[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = block
    rhs = precision @ m.T.reshape(-1)
    for row, response, ht in zip(x, y, h, strict=True):
        design = np.kron(np.eye(3), row[None, :])
        weight = q.T @ np.diag(np.exp(-ht)) @ q
        precision += design.T @ weight @ design
        rhs += design.T @ weight @ response
    args = dict(
        x=x, y=y, q=q, h=h, m0=m, v0=np.eye(2), inv_v0_vec=np.array([0.5, 2, 4, 0.8, 1.3, 7])
    )
    return args, precision, rhs


@pytest.mark.parametrize("intercept", [False, True])
def test_triangular_eqwise_conditionals_and_slice_negative_control(intercept):
    class Zero:
        def standard_normal(self, size):
            return np.zeros(size)

    args, precision, rhs = _triangular_reference(intercept)
    previous = np.array([[0.2, -0.9, 1.1], [0.7, 0.1, -0.8]])
    expected = previous.T.reshape(-1).copy()
    for j in [2, 1, 0]:
        indices = np.arange(2 * j, 2 * j + 2)
        rest = np.setdiff1d(np.arange(6), indices)
        expected[indices] = np.linalg.solve(
            precision[np.ix_(indices, indices)],
            rhs[indices] - precision[np.ix_(indices, rest)] @ expected[rest],
        )
    actual = _sample_beta_triangular_svrw(**args, beta=previous, rng=Zero())
    np.testing.assert_allclose(actual.T.reshape(-1), expected, atol=1e-12)
    bad_args = dict(args, inv_v0_vec=np.roll(args["inv_v0_vec"], 2))
    wrong = _sample_beta_triangular_svrw(**bad_args, beta=previous, rng=Zero())
    assert np.max(np.abs(wrong - actual)) > 0.01


def test_triangular_eqwise_joint_moments():
    args, precision, rhs = _triangular_reference()
    expected_mean = np.linalg.solve(precision, rhs)
    expected_cov = np.linalg.inv(precision)
    rng = np.random.default_rng(17)
    beta = args["m0"].copy()
    samples = []
    for it in range(8500):
        beta = _sample_beta_triangular_svrw(**args, beta=beta, rng=rng)
        if it >= 500:
            samples.append(beta.T.reshape(-1))
    samples = np.array(samples)
    # Fixed seed; tolerances include autocorrelation and covariance Monte Carlo error.
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.015)
    np.testing.assert_allclose(np.cov(samples.T), expected_cov, atol=0.006)


@pytest.mark.parametrize(
    "precision", [np.ones(5), np.zeros(6), np.full(6, np.nan), np.ones((2, 3))]
)
def test_triangular_rejects_invalid_precision(precision):
    args, _, _ = _triangular_reference()
    args["inv_v0_vec"] = precision
    with pytest.raises(ValueError, match="inv_v0_vec"):
        _sample_beta_triangular_svrw(**args, beta=args["m0"], rng=np.random.default_rng(1))


@pytest.mark.parametrize("dynamics", ["rw", "ar1"])
@pytest.mark.parametrize("elb", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
def test_canonical_triangular_fit_routes_precision(monkeypatch, dynamics, elb, intercept):
    import srvar.samplers_svcov as module

    y = np.random.default_rng(1).normal(size=(18, 3)) * [1, 3, 7]
    if elb:
        y[:, 0] = np.maximum(y[:, 0], 0)
    ds = Dataset.from_arrays(values=y, variables=["a", "b", "c"])
    model = ModelSpec(
        p=1,
        include_intercept=intercept,
        volatility=VolatilitySpec(covariance="triangular", dynamics=dynamics),
        elb=ElbSpec(bound=0, applies_to=["a"]) if elb else None,
    )
    prior = build_prior(
        {"prior": {"family": "niw", "method": "minnesota_canonical"}}, dataset=ds, model=model
    )
    calls = []
    original = module._sample_beta_triangular_svrw

    def checked(**kwargs):
        np.testing.assert_array_equal(kwargs["inv_v0_vec"], prior.minnesota_canonical.inv_v0_vec)
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(module, "_sample_beta_triangular_svrw", checked)
    result = fit(ds, model, prior, SamplerConfig(draws=4, burn_in=1), rng=np.random.default_rng(8))
    assert len(calls) == 4
    assert np.isfinite(result.beta_draws).all()


def test_triangular_rejects_legacy_and_tempered_even_internal():
    ds = Dataset.from_arrays(
        values=np.random.default_rng(2).normal(size=(15, 2)), variables=["a", "b"]
    )
    model = ModelSpec(p=1, volatility=VolatilitySpec(covariance="triangular"))
    for method in ["minnesota", "minnesota_legacy", "minnesota_tempered"]:
        with pytest.raises(ConfigError):
            build_prior({"prior": {"family": "niw", "method": method}}, dataset=ds, model=model)
    priors = [
        PriorSpec.niw_minnesota_legacy(y=ds.values, p=1),
        replace(PriorSpec.niw_minnesota_tempered(y=ds.values, p=1), method="custom"),
    ]
    for prior in priors:
        with pytest.raises(ValueError):
            _fit_svcov(
                dataset=ds,
                model=model,
                prior=prior,
                sampler=SamplerConfig(draws=3),
                rng=np.random.default_rng(1),
            )


def test_dl_backtest_estimates_once_per_training_window_without_future_data(monkeypatch, tmp_path):
    import pandas as pd
    import yaml

    import srvar.spec as spec_module
    from srvar.backtest import backtest_from_config

    y = np.random.default_rng(199).normal(size=(20, 2))
    dates = pd.date_range("2000", periods=20, freq="MS")
    seen = []
    original = spec_module._estimate_minnesota_sigma2

    def capture(**kwargs):
        seen.append(kwargs["y"].copy())
        return original(**kwargs)

    monkeypatch.setattr(spec_module, "_estimate_minnesota_sigma2", capture)
    cfg = {
        "data": {
            "csv_path": str(tmp_path / "data.csv"),
            "date_column": "date",
            "variables": ["a", "b"],
        },
        "model": {"p": 1},
        "prior": {"family": "dl"},
        "sampler": {"draws": 4, "burn_in": 1, "seed": 78},
        "backtest": {
            "mode": "rolling",
            "window": 10,
            "min_obs": 12,
            "step": 4,
            "horizons": [1],
            "draws": 4,
            "origin_start": str(dates[11].date()),
            "origin_end": str(dates[15].date()),
        },
        "output": {"save_plots": False, "save_forecasts": False},
        "evaluation": {
            "metrics_table": False,
            "coverage": {"enabled": False},
            "crps": {"enabled": False},
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg))
    for run in range(2):
        values = y.copy()
        if run:
            values[16:] += 1000
        pd.DataFrame({"date": dates, "a": values[:, 0], "b": values[:, 1]}).to_csv(
            tmp_path / "data.csv", index=False
        )
        backtest_from_config(path, out_dir=tmp_path / f"out{run}")
    assert len(seen) == 4  # two origins per run, independent of chain sweeps
    for first, second, expected in [(seen[0], seen[2], y[2:12]), (seen[1], seen[3], y[6:16])]:
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, expected, atol=1e-15)
    assert not np.array_equal(seen[0], seen[1])


def test_triangular_eqwise_diagonal_q_forgets_previous_state():
    args, _, _ = _triangular_reference()
    args["q"] = np.eye(3)
    a = _sample_beta_triangular_svrw(**args, beta=np.zeros((2, 3)), rng=np.random.default_rng(33))
    b = _sample_beta_triangular_svrw(
        **args, beta=np.ones((2, 3)) * 12, rng=np.random.default_rng(33)
    )
    np.testing.assert_allclose(a, b, atol=1e-14)
