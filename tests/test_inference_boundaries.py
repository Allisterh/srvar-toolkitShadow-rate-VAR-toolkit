from dataclasses import replace

import numpy as np
import pytest

from srvar.analysis import (
    fevd_cholesky,
    historical_decomposition_cholesky,
    irf_cholesky,
    irf_reduced_form,
    irf_sign_restricted,
)
from srvar.api import fit, forecast
from srvar.data.dataset import Dataset
from srvar.elb import ElbSpec
from srvar.scenario import conditional_forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


def dataset():
    return Dataset.from_arrays(
        values=np.random.default_rng(19).normal(size=(25, 2)), variables=["a", "b"]
    )


def prior(family):
    if family == "dl":
        return PriorSpec.from_dl(k=3, n=2, residual_prior="explicit", nu0=2, s0=np.eye(2))
    return PriorSpec.from_ssvs(k=3, n=2)


@pytest.mark.parametrize("family", ["ssvs", "dl"])
def test_nonzero_means_and_empty_retention_rejected(family):
    p = prior(family)
    nonzero = replace(p, niw=replace(p.niw, m0=np.ones((3, 2))))
    with pytest.raises(ValueError, match="zero prior means"):
        fit(dataset(), ModelSpec(p=1), nonzero, SamplerConfig(draws=5, burn_in=0))
    for burn_in in (5, 6):
        with pytest.raises(ValueError, match="retained draws"):
            fit(dataset(), ModelSpec(p=1), p, SamplerConfig(draws=5, burn_in=burn_in))
    valid = fit(
        dataset(),
        ModelSpec(p=1),
        p,
        SamplerConfig(draws=5, burn_in=2),
        rng=np.random.default_rng(91),
    )
    assert valid.posterior is None
    assert valid.beta_draws.shape == (3, 3, 2)
    assert forecast(valid, [1], draws=2, rng=np.random.default_rng(92)).draws.shape == (2, 1, 2)


OPERATIONS = [
    lambda f: forecast(f, [1], draws=2),
    lambda f: conditional_forecast(f, [1], constraints={"a": {1: 0}}, draws=2),
    lambda f: irf_reduced_form(f, horizons=1, draws=2),
    lambda f: irf_cholesky(f, horizons=1, draws=2),
    lambda f: irf_sign_restricted(f, horizons=1, restrictions={}, draws=2),
    lambda f: fevd_cholesky(f, horizons=1, draws=2),
    lambda f: historical_decomposition_cholesky(f, draws=2),
]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("invalid", ["blasso", "nonzero", "no_draws", "empty_draws"])
def test_archived_or_constructed_fits_cannot_bypass_boundaries(operation, invalid):
    base = fit(
        dataset(),
        ModelSpec(p=1),
        PriorSpec.niw_default(k=3, n=2),
        SamplerConfig(draws=4, burn_in=0),
        rng=np.random.default_rng(93),
    )
    if invalid == "blasso":
        bad = replace(base, prior=PriorSpec.from_blasso(k=3, n=2))
        message = "Bayesian LASSO"
    elif invalid == "nonzero":
        p = prior("dl")
        bad = replace(base, prior=replace(p, niw=replace(p.niw, m0=np.ones((3, 2)))))
        message = "zero prior means"
    else:
        bad = replace(
            base,
            prior=prior("dl"),
            beta_draws=None if invalid == "no_draws" else np.empty((0, 3, 2)),
        )
        assert bad.posterior is None
        message = "retained coefficient draws"
    with pytest.raises(ValueError, match=message):
        operation(bad)


def test_missing_covariance_never_substitutes_niw_but_reduced_form_needs_only_beta():
    base = fit(
        dataset(),
        ModelSpec(p=1),
        prior("dl"),
        SamplerConfig(draws=4, burn_in=0),
        rng=np.random.default_rng(94),
    )
    missing = replace(base, sigma_draws=None)
    with pytest.raises(ValueError, match="covariance states"):
        forecast(missing, [1], draws=2)
    assert irf_reduced_form(missing, horizons=1, rng=np.random.default_rng(95)).draws.shape[0] == 4


@pytest.mark.parametrize("use_latent", [None, True, False])
def test_elb_historical_decomposition_override_cannot_bypass_rejection(use_latent):
    base = fit(
        dataset(),
        ModelSpec(p=1),
        PriorSpec.niw_default(k=3, n=2),
        SamplerConfig(draws=4, burn_in=0),
        rng=np.random.default_rng(93),
    )
    elb = replace(
        base,
        model=ModelSpec(p=1, elb=ElbSpec(bound=0, applies_to=["a"])),
        latent_dataset=base.dataset,
    )
    with pytest.raises(ValueError, match="ELB historical decomposition is disabled"):
        historical_decomposition_cholesky(elb, use_latent=use_latent)


def test_conjugate_analytic_forecast_still_available():
    result = fit(
        dataset(),
        ModelSpec(p=1),
        PriorSpec.niw_default(k=3, n=2),
        SamplerConfig(draws=2, burn_in=2),
        rng=np.random.default_rng(96),
    )
    assert result.beta_draws is None or len(result.beta_draws) == 0
    assert result.posterior is not None
    assert forecast(result, [1], draws=2, rng=np.random.default_rng(97)).draws.shape == (2, 1, 2)


def test_cli_runner_rejects_lasso_before_writing_results(tmp_path):
    import pandas as pd
    import yaml

    from srvar.runner import run_from_config

    pd.DataFrame(
        {
            "date": pd.date_range("2000", periods=25, freq="MS"),
            "a": np.random.default_rng(1).normal(size=25),
            "b": np.random.default_rng(2).normal(size=25),
        }
    ).to_csv(tmp_path / "data.csv", index=False)
    config = {
        "data": {
            "csv_path": str(tmp_path / "data.csv"),
            "date_column": "date",
            "variables": ["a", "b"],
        },
        "model": {"p": 1},
        "prior": {"family": "blasso"},
        "sampler": {"draws": 5, "burn_in": 0},
        "output": {"save_plots": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Bayesian LASSO inference is disabled"):
        run_from_config(path, out_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "fit.npz").exists()


@pytest.mark.parametrize("family", ["ssvs", "dl", "blasso"])
def test_custom_shrinkage_cannot_dispatch_through_minnesota_metadata(family):
    import json

    from srvar._prior_io import prior_from_json, prior_to_json

    canonical = PriorSpec.niw_minnesota_canonical(y=dataset().values, p=1)
    shrinkage = PriorSpec.from_blasso(k=3, n=2) if family == "blasso" else prior(family)
    with pytest.raises(ValueError, match="Minnesota metadata requires family='niw'"):
        replace(shrinkage, method="custom", minnesota_canonical=canonical.minnesota_canonical)
    saved = json.loads(prior_to_json(shrinkage))
    saved["method"] = "custom"
    saved["minnesota_canonical"] = json.loads(prior_to_json(canonical))["minnesota_canonical"]
    with pytest.raises(ValueError, match="invalid saved prior metadata"):
        prior_from_json(json.dumps(saved))


def test_custom_niw_metadata_preserves_equationwise_dispatch():
    canonical = PriorSpec.niw_minnesota_canonical(y=dataset().values, p=1)
    outputs = [
        fit(
            dataset(),
            ModelSpec(p=1),
            p,
            SamplerConfig(draws=4, burn_in=0),
            rng=np.random.default_rng(101),
        )
        for p in (canonical, replace(canonical, method="custom"))
    ]
    np.testing.assert_array_equal(outputs[0].beta_draws, outputs[1].beta_draws)
    np.testing.assert_array_equal(outputs[0].sigma_draws, outputs[1].sigma_draws)


def test_sv_analysis_needs_contemporaneous_covariance_but_forecasts_need_innovations():
    from srvar.sv import VolatilitySpec

    full = fit(
        dataset(),
        ModelSpec(p=1, volatility=VolatilitySpec()),
        prior("dl"),
        SamplerConfig(draws=4, burn_in=0),
        rng=np.random.default_rng(102),
    )
    reduced = replace(full, sigma_eta2_draws=None)
    operations = [
        lambda f, rng: irf_cholesky(f, horizons=1, rng=rng),
        lambda f, rng: irf_sign_restricted(f, horizons=1, restrictions={}, rng=rng),
        lambda f, rng: fevd_cholesky(f, horizons=1, rng=rng),
        lambda f, rng: historical_decomposition_cholesky(f, draws=4, rng=rng),
    ]
    for operation in operations:
        expected = operation(full, np.random.default_rng(103))
        actual = operation(reduced, np.random.default_rng(103))
        np.testing.assert_array_equal(actual.mean, expected.mean)
        missing_h = replace(reduced, h_draws=None)
        with pytest.raises(ValueError, match="covariance states"):
            operation(missing_h, np.random.default_rng(103))
    for operation in (
        lambda f: forecast(f, [1], draws=2, rng=np.random.default_rng(104)),
        lambda f: conditional_forecast(
            f, [1], constraints={"a": {1: 0}}, draws=2, rng=np.random.default_rng(104)
        ),
    ):
        with pytest.raises(ValueError, match="covariance states"):
            operation(reduced)
