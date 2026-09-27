from dataclasses import replace

import numpy as np
import pytest

from scripts import check_gaussian_units as units
from scripts import compare_dl_controls as controls
from scripts import qualify_dl_empirical_bayes as baseline
from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, SamplerConfig


def test_unfloored_rates_match_direct_regressions_and_preserve_tiny_scales():
    y, truth = baseline.generate_data(3, np.random.default_rng(8))
    expected = []
    for j in range(2):
        design = np.column_stack([np.ones(39), y[:-1, j]])
        beta = np.linalg.lstsq(design, y[1:, j], rcond=None)[0]
        residual = y[1:, j] - design @ beta
        expected.append(residual @ residual / 37)
    rates = controls.estimate_unfloored_rates(y)
    np.testing.assert_allclose(rates, expected, rtol=1e-13, atol=0)
    for name in controls.UNFLOORED_ARMS:
        prior = controls.make_prior(y, truth, name)
        np.testing.assert_array_equal(np.diag(prior.niw.s0), rates)
    assert np.all(rates < 1e-12)


def test_zero_estimates_are_failed_arms_and_preserved(tmp_path, monkeypatch):
    pytest.importorskip("arviz")
    monkeypatch.setattr(
        baseline, "generate_data", lambda *args: (np.zeros((40, 2)), np.r_[np.zeros(6), np.ones(2)])
    )
    (tmp_path / "chains").mkdir()
    records = controls.run_dataset(
        dict(
            out=str(tmp_path),
            cell_id=0,
            replicate=0,
            seed=8,
            chains=2,
            draws=20,
            warmup=10,
            include_unfloored=True,
        )
    )
    assert [r["status"] for r in records] == ["ok"] * 4 + ["failed"] * 2
    for record in records[-2:]:
        assert record["estimated_unfloored_rates"] == [0.0, 0.0]
        assert record["completed_chains"] == 0
        assert "finite and positive" in record["error"]
    _, counts = controls.arm_summaries(records, arms=controls.ARMS + controls.UNFLOORED_ARMS)
    assert len(counts) == 6 and counts.failed.sum() == 2
    contrasts = controls.paired_summaries(records, contrasts=controls.UNFLOORED_CONTRASTS)
    missing = contrasts[
        (contrasts.contrast == "oracle_minus_unfloored_dl") & (contrasts.metric == "coverage90")
    ]
    assert (missing.complete_pairs == 0).all()
    assert np.allclose(missing.failure_bound_high - missing.failure_bound_low, 1)


@pytest.mark.parametrize("scales", [[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1], [1]])
def test_invalid_unit_scales_rejected(scales):
    with pytest.raises(ValueError, match="two finite positive"):
        units.draw_factors(np.asarray(scales))


def test_prior_transformation_matches_dense_gaussian_reference_and_wrong_prior_fails():
    class ZeroNormals:
        def standard_normal(self, size):
            return np.zeros(size)

        def gamma(self, *, shape, scale):
            return 1.0

    y, truth = baseline.generate_data(0, np.random.default_rng(5))
    prior = controls.make_prior(y, truth, "gaussian_unfloored")
    m0 = np.array([[0.4, -0.2], [0.1, 0.3], [-0.3, 0.2]])
    prior = replace(prior, niw=replace(prior.niw, m0=m0))
    scales = np.array([0.01, 100.0])
    transformed = units.transform_prior(prior, scales)
    factors = scales[None, :] / np.r_[1.0, scales][:, None]
    np.testing.assert_allclose(transformed.niw.m0, m0 * factors)
    np.testing.assert_allclose(
        transformed.minnesota_canonical.inv_v0_vec, (1 / factors**2).reshape(-1, order="F")
    )
    x = np.column_stack([np.ones(39), y[:-1]])
    expected = np.column_stack(
        [
            np.linalg.solve(
                np.eye(3) + x.T @ x / prior.niw.s0[j, j],
                m0[:, j] + x.T @ y[1:, j] / prior.niw.s0[j, j],
            )
            for j in range(2)
        ]
    )
    dataset = Dataset.from_arrays(values=y * scales, variables=["a", "b"])
    result = fit(
        dataset, ModelSpec(p=1), transformed, SamplerConfig(draws=1, burn_in=0), rng=ZeroNormals()
    )
    np.testing.assert_allclose(result.beta_draws[0] / factors, expected, atol=1e-12)
    # Preserve transformed rates but deliberately retain the original coefficient prior.
    wrong = replace(
        transformed,
        niw=replace(transformed.niw, m0=m0),
        minnesota_canonical=prior.minnesota_canonical,
    )
    wrong_fit = fit(
        dataset, ModelSpec(p=1), wrong, SamplerConfig(draws=1, burn_in=0), rng=ZeroNormals()
    )
    assert np.max(np.abs(wrong_fit.beta_draws[0] / factors - expected)) > 0.01


def test_matched_gaussian_chains_and_prior_roundtrip(tmp_path):
    pytest.importorskip("arviz")
    y, truth = baseline.generate_data(0, np.random.default_rng(4))
    prior = controls.make_prior(y, truth, "gaussian_unfloored")
    scales = np.array([1e-7, 1e7])
    roundtrip = units.transform_prior(units.transform_prior(prior, scales), 1 / scales)
    np.testing.assert_allclose(roundtrip.niw.s0, prior.niw.s0)
    np.testing.assert_allclose(roundtrip.minnesota_canonical.inv_v0_vec, np.ones(6))
    records = units.run_dataset(
        dict(out=str(tmp_path), seed=20260928, replicate=0, chains=2, draws=20, warmup=10)
    )
    assert all(r["status"] == "ok" for r in records)
    base = np.load(tmp_path / records[0]["chain_file"])["samples"]
    for record in records:
        raw = np.load(tmp_path / record["chain_file"])["samples"]
        back = raw / units.draw_factors(np.asarray(record["scales"]))
        np.testing.assert_allclose(back, base, rtol=1e-10, atol=1e-12)
