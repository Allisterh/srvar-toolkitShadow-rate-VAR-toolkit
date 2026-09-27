import numpy as np

from srvar.spec import PriorSpec


def test_from_dl_shapes_with_explicit_legacy_parameters() -> None:
    k, n = 5, 2
    prior = PriorSpec.from_dl(k=k, n=n, residual_prior="explicit", nu0=n + 2, s0=np.eye(n))
    assert prior.family.lower() == "dl"
    assert prior.dl is not None
    assert prior.niw.m0.shape == (k, n)
    assert prior.niw.s0.shape == (n, n)


def test_dl_spec_rejects_nonpositive_hyperparams() -> None:
    k, n = 5, 2
    try:
        _ = PriorSpec.from_dl(
            k=k, n=n, residual_prior="explicit", nu0=n + 2, s0=np.eye(n), abeta=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        _ = PriorSpec.from_dl(
            k=k, n=n, residual_prior="explicit", nu0=n + 2, s0=np.eye(n), dl_scaler=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
