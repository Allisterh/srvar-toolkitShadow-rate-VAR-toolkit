import numpy as np
import pytest

from scripts import diagnose_sv_state_block as state_study
from scripts import qualify_dl_empirical_bayes as dl_study
from srvar import sv


def test_fixed_dgp_truth_and_scaling():
    y, truth = dl_study.generate_data(0, np.random.default_rng(7))
    assert y.shape == (40, 2)
    np.testing.assert_array_equal(truth, [0, 0.3, 0, 0, 0, 0.3, 1, 1])
    small, small_truth = dl_study.generate_data(3, np.random.default_rng(7))
    np.testing.assert_allclose(small, y * 1e-7, atol=1e-20)
    np.testing.assert_allclose(small_truth[-2:], [1e-14, 1e-14])


def test_dl_coverage_keeps_failed_dataset_in_bounds(monkeypatch):
    row = dict(
        cell="short",
        parameter="beta",
        error=1,
        covered90=True,
        width90=2,
        flag=True,
        rhat=1.2,
        ess_bulk=20,
        ess_tail=30,
    )
    result = dl_study.summarise(
        [dict(cell="short", status="ok", rows=[row]), dict(cell="short", status="failed")]
    ).iloc[0]
    assert result.attempted == 2 and result.successful == 1 and result.failures == 1
    assert result.coverage90 == 1 and result.failure_bound_low == 0.5
    assert result.failure_bound_high == 1 and result.diagnostic_flags == 1

    def fail(job):
        raise ValueError("fixture")

    monkeypatch.setattr(dl_study, "run_replication", fail)
    assert dl_study.safe_replication(dict(cell_id=0, replicate=3))["status"] == "failed"


@pytest.mark.parametrize("length", [1, 2, 7])
def test_rw_state_gaussian_conditional_against_dense_difference_matrix(monkeypatch, length):
    labels = np.arange(length) % 7
    monkeypatch.setattr(sv, "sample_mixture_indicators", lambda **kwargs: labels)
    y = np.linspace(-1, 2, length)
    difference = np.eye(length)
    for t in range(1, length):
        difference[t, t - 1] = -1
    variance = 0.17
    h0 = -0.4
    obs_precision = np.diag(1 / sv._KSC_SIGI[labels])
    precision = difference.T @ difference / variance + obs_precision
    offset = np.zeros(length)
    offset[0] = h0
    rhs = difference.T @ offset / variance + obs_precision @ (y - sv._KSC_MI[labels])
    expected_mean = np.linalg.solve(precision, rhs)

    class FixedNormals:
        def __init__(self, value):
            self.value = value

        def standard_normal(self, size):
            return self.value

    args = dict(y_star=y, h=np.zeros(length), sigma_eta2=variance, h0=h0)
    actual = sv.sample_h_svrw(**args, rng=FixedNormals(np.zeros(length)))
    np.testing.assert_allclose(actual, expected_mean, atol=1e-13)
    root = np.column_stack(
        [sv.sample_h_svrw(**args, rng=FixedNormals(row)) - actual for row in np.eye(length)]
    )
    np.testing.assert_allclose(root @ root.T, np.linalg.inv(precision), atol=1e-13)


def test_rw_h0_and_innovation_variance_conditionals():
    class Capture:
        def normal(self):
            return 0

        def gamma(self, *, shape, scale):
            self.shape = shape
            self.scale = scale
            return 2.0

    rng = Capture()
    value = sv.sample_h0(h1=1.0, sigma_eta2=0.5, prior_mean=-1.0, prior_var=2.0, rng=rng)
    assert value == pytest.approx(0.6)
    h = np.array([1.0, 2.0, -0.5])
    assert sv.sample_sigma_eta2(h=h, h0=0.2, nu0=2.0, s0=0.1, rng=rng) == 0.5
    assert rng.shape == 3.5
    assert rng.scale == pytest.approx(1 / (0.1 + 0.5 * (0.8**2 + 1 + 2.5**2)))


def test_tiny_mixture_reference_grid_refines():
    coarse = state_study.tiny_reference(151)
    fine = state_study.tiny_reference(301)
    np.testing.assert_allclose(coarse["mean"], fine["mean"], atol=1e-5)
    assert fine["boundary_mass"] < 1e-6
    assert fine["grid_mass"].sum() == pytest.approx(1.0)
    assert np.all(fine["sd"] > 0)


def test_interwoven_scale_density_includes_transformation_jacobians():
    from scipy.stats import invgamma, norm

    from scripts.compare_sv_parameterisations import scale_log_density

    z = np.array([0.3, -0.7, 0.4])
    obs = np.array([1.0, -0.5, 0.8])
    omega = np.array([0.7, 1.3, 0.4])
    h0 = 0.2
    direct = []
    proposed = []
    for ell in [-2.0, 0.3]:
        scale = np.exp(ell)
        h = h0 + scale * z
        increments = np.diff(np.r_[h0, h])
        # Joint in h,var transformed to z,ell: dh/dz=s^T; dvar/dell=2*s^2.
        direct.append(
            invgamma.logpdf(scale**2, a=2, scale=0.1)
            + norm.logpdf(increments, scale=scale).sum()
            + norm.logpdf(obs, loc=h, scale=np.sqrt(omega)).sum()
            + len(z) * ell
            + np.log(2)
            + 2 * ell
        )
        proposed.append(scale_log_density(ell, z, obs, omega, h0, 2, 0.1))
    assert direct[1] - direct[0] == pytest.approx(proposed[1] - proposed[0])


def test_dl_study_reproduces_prior_and_intervals():
    pytest.importorskip("arviz")
    job = dict(cell_id=0, replicate=0, seed=17, chains=2, draws=20, warmup=10)
    first = dl_study.run_replication(job)
    second = dl_study.run_replication(job)
    assert first["rates"] == second["rates"]
    assert first["floor_active"] == second["floor_active"]
    assert len(first["rows"]) == 8
    for a, b in zip(first["rows"], second["rows"], strict=True):
        assert a["parameter"] == b["parameter"]
        for key in [
            "truth",
            "posterior_mean",
            "lower90",
            "upper90",
            "rhat",
            "ess_bulk",
            "ess_tail",
        ]:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0, equal_nan=True)
