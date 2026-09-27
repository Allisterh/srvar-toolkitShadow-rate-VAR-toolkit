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
    assert forecast(valid, [1], draws=2).draws.shape == (2, 1, 2)


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
    base = fit(dataset(), ModelSpec(p=1), prior("dl"), SamplerConfig(draws=4, burn_in=0))
    missing = replace(base, sigma_draws=None)
    with pytest.raises(ValueError, match="covariance states"):
        forecast(missing, [1], draws=2)
    assert irf_reduced_form(missing, horizons=1).draws.shape[0] == 4


@pytest.mark.parametrize("use_latent", [None, True, False])
def test_elb_historical_decomposition_override_cannot_bypass_rejection(use_latent):
    base = fit(
        dataset(),
        ModelSpec(p=1),
        PriorSpec.niw_default(k=3, n=2),
        SamplerConfig(draws=4, burn_in=0),
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
    )
    assert result.beta_draws is None or len(result.beta_draws) == 0
    assert result.posterior is not None
    assert forecast(result, [1], draws=2).draws.shape == (2, 1, 2)


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
