import numpy as np
import pytest

from scripts import compare_dl_controls as controls
from srvar.config import ConfigError, build_prior
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec


def _prior(y, p=1, intercept=True):
    return PriorSpec.from_dl(
        k=y.shape[1] * p + int(intercept),
        n=y.shape[1],
        y=y,
        p=p,
        include_intercept=intercept,
        residual_prior="empirical_bayes",
    )


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("p", [1, 2])
def test_rates_follow_squared_units_and_independent_ols(p, intercept):
    y = np.random.default_rng(104).normal(size=(60, 2))
    base = np.diag(_prior(y, p, intercept).niw.s0)
    for scales in ([1e-100, 1e100], [1e-7, 1e7], [1e100, 1e-100]):
        scaled = np.diag(_prior(y * scales, p, intercept).niw.s0)
        np.testing.assert_allclose(scaled / np.square(scales), base, rtol=1e-12, atol=0)
    expected = []
    for col in y.T:
        x = np.column_stack([col[p - lag : len(col) - lag] for lag in range(1, p + 1)])
        if intercept:
            x = np.column_stack([np.ones(len(x)), x])
        residual = col[p:] - x @ np.linalg.lstsq(x, col[p:], rcond=None)[0]
        expected.append(residual @ residual / (len(x) - x.shape[1]))
    np.testing.assert_allclose(base, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "series,p,intercept,message",
    [
        (np.zeros(30), 1, True, "zero series"),
        (np.ones(30), 1, True, "rank-deficient"),
        (np.tile([1.0, 2.0], 15), 2, True, "rank-deficient"),
        (np.arange(30.0), 1, True, "numerically zero"),
        (0.9 ** np.arange(30), 1, False, "numerically zero"),
        (np.array([1.0, -2.0, 3.0, 4.0]), 2, True, "degrees of freedom"),
        (
            np.arange(30.0) + np.random.default_rng(1).normal(size=30) * 1e-14,
            1,
            True,
            "numerically zero",
        ),
    ],
)
def test_invalid_auxiliary_regressions_fail_in_all_units(series, p, intercept, message):
    for scale in (1.0, 1e-7, 1e7):
        with pytest.raises(ValueError, match=message) as caught:
            _prior((series * scale)[:, None], p, intercept)
        assert "equation 0" in str(caught.value)
        assert "residual_prior='explicit'" in str(caught.value)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_nonrepresentable_rates_fail_explicitly(scale):
    with pytest.raises(ValueError, match="not representable"):
        _prior(np.random.default_rng(8).normal(size=(40, 1)) * scale)


def test_small_valid_variances_are_not_floored_and_minnesota_is_unchanged():
    y = np.random.default_rng(8).normal(size=(40, 2)) * 1e-7
    dl = _prior(y)
    assert np.all(np.diag(dl.niw.s0) < 1e-12)
    canonical = PriorSpec.niw_minnesota_canonical(y=y, p=1)
    np.testing.assert_array_equal(canonical.minnesota_canonical.sigma2, [1e-12, 1e-12])
    tiny = _prior(y * 1e-143)
    np.testing.assert_allclose(
        np.diag(tiny.niw.s0) / 1e-286, np.diag(dl.niw.s0), rtol=1e-12, atol=0
    )


def test_removed_floor_option_and_config_error_migration():
    y = np.random.default_rng(8).normal(size=(40, 2))
    with pytest.raises(TypeError, match="min_sigma2"):
        PriorSpec.from_dl(k=3, n=2, residual_prior="empirical_bayes", y=y, p=1, min_sigma2=1e-12)
    dataset = Dataset.from_arrays(values=y, variables=["a", "b"])
    for mode in ("empirical_bayes", "explicit"):
        with pytest.raises(ConfigError, match="min_sigma2 was removed"):
            build_prior(
                {"prior": {"family": "dl", "dl": {"residual_prior": mode, "min_sigma2": 1e-12}}},
                dataset=dataset,
                model=ModelSpec(p=1),
            )
    with pytest.raises(ConfigError, match="rank-deficient"):
        build_prior(
            {"prior": {"family": "dl"}},
            dataset=Dataset.from_arrays(values=np.ones((40, 2)), variables=["a", "b"]),
            model=ModelSpec(p=1),
        )


def test_historical_study_targets_remain_floored_and_policy_uses_new_rates(tmp_path):
    pytest.importorskip("arviz")
    y = np.random.default_rng(8).normal(size=(40, 2)) * 1e-7
    truth = np.r_[np.zeros(6), [1e-14, 1e-14]]
    for arm in ("dl_eb", "gaussian_eb"):
        np.testing.assert_array_equal(
            np.diag(controls.make_prior(y, truth, arm).niw.s0), [1e-12, 1e-12]
        )
    for arm in controls.POLICY_ARMS:
        np.testing.assert_array_equal(controls.make_prior(y, truth, arm).niw.s0, _prior(y).niw.s0)
    (tmp_path / "chains").mkdir()
    records = controls.run_dataset(
        dict(
            out=str(tmp_path),
            cell_id=3,
            replicate=0,
            seed=20260928,
            chains=2,
            draws=20,
            warmup=10,
            policy_only=True,
        )
    )
    assert [r["arm"] for r in records] == list(controls.POLICY_ARMS)
    assert all(r["status"] == "ok" and r["completed_chains"] == 2 for r in records)
