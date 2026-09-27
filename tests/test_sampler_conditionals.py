"""Independent small-model references for sampler transitions."""

import numpy as np
import pytest
from scipy.integrate import quad

from srvar import Dataset, ElbSpec
from srvar.api import fit
from srvar.samplers_dl import _dl_sample_beta_sigma
from srvar.samplers_svcov import _sample_beta_triangular_svrw
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig, SteadyStateSpec


def test_independent_normal_ig_chain_matches_integrated_posterior() -> None:
    # Integrate IG(2,1) out exactly up to a one-dimensional quadrature.
    y = np.tile([-10.0, 10.0], 10).reshape(-1, 1)
    rng = np.random.default_rng(23)
    sigma = np.ones((1, 1))
    kept = []
    for it in range(12500):
        beta, sigma = _dl_sample_beta_sigma(
            x=np.ones((20, 1)),
            y=y,
            m0=np.zeros((1, 1)),
            inv_v0_vec=np.ones(1),
            s0=np.ones((1, 1)),
            nu0=2.0,
            sigma=sigma,
            rng=rng,
        )
        if it >= 500:
            kept.append(beta.item())

    def density(b: float) -> float:
        return float(np.exp(-0.5 * b * b) * (1 + 10 * b * b / 1001) ** -12)

    z = quad(density, -np.inf, np.inf)[0]
    expected_var = quad(lambda b: b * b * density(b), -np.inf, np.inf)[0] / z
    # Broad Monte Carlo tolerance; the faulty fixed variance is only 0.0476.
    assert abs(np.mean(kept)) < 0.04
    assert np.var(kept) == pytest.approx(expected_var, rel=0.06)


@pytest.mark.parametrize("variance", [0.25, 4.0])
def test_beta_conditional_uses_current_variance(variance: float) -> None:
    rng = np.random.default_rng(50)
    draws = [
        _dl_sample_beta_sigma(
            x=np.ones((3, 1)),
            y=np.full((3, 1), 2.0),
            m0=np.ones((1, 1)),
            inv_v0_vec=np.array([2.0]),
            s0=np.ones((1, 1)),
            nu0=2.0,
            sigma=np.array([[variance]]),
            rng=rng,
        )[0].item()
        for _ in range(6000)
    ]
    precision = 2 + 3 / variance
    assert np.mean(draws) == pytest.approx((2 + 6 / variance) / precision, abs=0.035)
    assert np.var(draws) == pytest.approx(1 / precision, rel=0.07)


@pytest.mark.parametrize("sigma", [np.zeros((1, 1)), np.array([[np.nan]]), np.eye(2)])
def test_independent_variance_rejects_invalid_state(sigma: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma"):
        _dl_sample_beta_sigma(
            x=np.ones((3, 1)),
            y=np.ones((3, 1)),
            m0=np.zeros((1, 1)),
            inv_v0_vec=np.ones(1),
            s0=np.ones((1, 1)),
            nu0=2.0,
            sigma=sigma,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("family", ["dl", "canonical"])
@pytest.mark.parametrize("elb", [False, True])
@pytest.mark.parametrize("steady", [False, True])
def test_every_independent_variance_caller_retains_state(monkeypatch, family, elb, steady):
    import srvar.samplers_homoskedastic as module

    original = module._dl_sample_beta_sigma
    previous = None
    calls = 0

    def tracked(**kwargs):
        nonlocal previous, calls
        assert "sigma" in kwargs, "coefficient update must receive retained residual variance"
        if previous is not None:
            np.testing.assert_array_equal(kwargs["sigma"], previous)
        result = original(**kwargs)
        previous = result[1].copy()
        calls += 1
        return result

    monkeypatch.setattr(module, "_dl_sample_beta_sigma", tracked)
    values = np.random.default_rng(4).normal(3, 0.2, (25, 1))
    values[-1] = 2.0
    ds = Dataset.from_arrays(values=values, variables=["r"])
    model = ModelSpec(
        p=1,
        elb=ElbSpec(bound=2.0, applies_to=["r"]) if elb else None,
        steady_state=SteadyStateSpec(mu0=np.zeros(1), v0_mu=1.0) if steady else None,
    )
    prior = (
        PriorSpec.from_dl(k=2, n=1)
        if family == "dl"
        else PriorSpec.niw_minnesota_canonical(p=1, y=values, n=1)
    )
    fit(ds, model, prior, SamplerConfig(draws=6, burn_in=1), rng=np.random.default_rng(7))
    assert calls == 6


def test_triangular_gibbs_matches_dense_joint_gaussian() -> None:
    x = np.array([[1.0, -1.0], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([[1.0, 3.0], [-1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    q = np.array([[1.0, 0.8], [0.0, 1.0]])
    h = np.array([[0.0, 0.3], [-0.2, 0.5], [0.1, -0.4], [0.5, 0.2]])
    m0 = np.array([[0.3, -0.4], [0.1, 0.2]])
    v0 = np.array([[0.7, 0.1], [0.1, 1.2]])
    # vec(B) is column-major; this oracle constructs the full likelihood.
    precision = np.kron(np.eye(2), np.linalg.inv(v0))
    rhs = precision @ m0.reshape(-1, order="F")
    for xt, yt, ht in zip(x, y, h, strict=True):
        design = np.kron(np.eye(2), xt.reshape(1, -1))
        innovation_precision = q.T @ np.diag(np.exp(-ht)) @ q
        precision += design.T @ innovation_precision @ design
        rhs += design.T @ innovation_precision @ yt
    covariance = np.linalg.inv(precision)
    mean = covariance @ rhs
    beta = m0.copy()
    rng = np.random.default_rng(17)
    kept = []
    for it in range(10500):
        beta = _sample_beta_triangular_svrw(
            x=x,
            y=y,
            q=q,
            h=h,
            m0=m0,
            v0=v0,
            beta=beta,
            rng=rng,
        )
        if it >= 500:
            kept.append(beta.reshape(-1, order="F"))
    samples = np.array(kept)
    # Includes Gibbs autocorrelation; incorrect sequential sampling misses means/covariances.
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.035)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=0.018)


def test_triangular_diagonal_q_does_not_depend_on_previous_beta() -> None:
    args = dict(
        x=np.ones((3, 1)),
        y=np.ones((3, 2)),
        q=np.eye(2),
        h=np.zeros((3, 2)),
        m0=np.zeros((1, 2)),
        v0=np.ones((1, 1)),
    )
    a = _sample_beta_triangular_svrw(**args, beta=np.zeros((1, 2)), rng=np.random.default_rng(1))
    b = _sample_beta_triangular_svrw(
        **args, beta=np.full((1, 2), 20.0), rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_triangular_sweep_conditional_means_match_joint_precision() -> None:
    class ZeroNormals:
        def standard_normal(self, size):
            return np.zeros(size)

    # With zero innovations each block must equal its conditional mean. The
    # second block is sampled first and must condition on the supplied first.
    q = np.array([[1.0, 0.8], [0.0, 1.0]])
    y = np.array([[1.0, 3.0], [-1.0, 1.0]])
    innovation_precision = q.T @ q
    precision = np.eye(2) + 2 * innovation_precision
    rhs = innovation_precision @ y.sum(axis=0)
    previous = np.array([[2.0, -3.0]])
    expected_second = (rhs[1] - precision[1, 0] * previous[0, 0]) / precision[1, 1]
    expected_first = (rhs[0] - precision[0, 1] * expected_second) / precision[0, 0]
    actual = _sample_beta_triangular_svrw(
        x=np.ones((2, 1)),
        y=y,
        q=q,
        h=np.zeros_like(y),
        m0=np.zeros((1, 2)),
        v0=np.ones((1, 1)),
        beta=previous,
        rng=ZeroNormals(),
    )
    np.testing.assert_allclose(actual, [[expected_first, expected_second]], atol=1e-14)
    np.testing.assert_array_equal(previous, [[2.0, -3.0]])
