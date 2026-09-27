import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from srvar.artifacts import load_fit_npz, load_run_dir, save_fit_npz
from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import ModelSpec, NIWPrior, PriorSpec, SamplerConfig


def _priors():
    y = np.random.default_rng(51).normal(size=(24, 2)) * [1.0, 7.0]
    custom = PriorSpec(
        family="niw",
        niw=NIWPrior(
            m0=np.arange(6, dtype=float).reshape(3, 2),
            v0=np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]),
            s0=np.array([[4.0, 0.3], [0.3, 2.0]]),
            nu0=7.0,
        ),
    )
    return {
        "custom": custom,
        "niw_default": PriorSpec.niw_default(k=3, n=2),
        "minnesota_legacy": PriorSpec.niw_minnesota_legacy(p=1, y=y),
        "minnesota_canonical": PriorSpec.niw_minnesota_canonical(p=1, y=y),
        "minnesota_tempered": PriorSpec.niw_minnesota_tempered(p=1, y=y, alpha=0.3),
        "ssvs": PriorSpec.from_ssvs(k=3, n=2, inclusion_prob=0.35),
        "blasso": PriorSpec.from_blasso(k=3, n=2, mode="adaptive"),
        "dl": PriorSpec.from_dl(k=3, n=2, residual_prior="explicit", nu0=4, s0=np.eye(2)),
    }


def _fit(prior):
    ds = Dataset.from_arrays(
        values=np.arange(20, dtype=float).reshape(10, 2),
        variables=["a", "b"],
        time_index=pd.date_range("2000", periods=10, freq="QS"),
    )
    return FitResult(
        dataset=ds,
        model=ModelSpec(p=1),
        prior=prior,
        sampler=SamplerConfig(draws=2, burn_in=0),
        posterior=None,
        beta_draws=np.zeros((2, 3, 2)),
        sigma_draws=np.broadcast_to(np.eye(2), (2, 2, 2)).copy(),
    )


def _as_json(prior):
    return json.dumps(asdict(prior), default=lambda value: value.tolist(), sort_keys=True)


def _payload(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_constructor_methods_and_explicit_dl_inputs_are_recorded():
    for method, prior in _priors().items():
        assert prior.method == method
    default = _priors()["dl"]
    assert default.residual_prior == "explicit"
    assert default.niw.nu0 == 4.0
    np.testing.assert_array_equal(default.niw.s0, np.eye(2))
    explicit = PriorSpec.from_dl(
        k=3, n=2, residual_prior="explicit", nu0=3.0, s0=np.diag([2.0, 8.0])
    )
    assert explicit.residual_prior == "explicit"
    alias = PriorSpec.niw_minnesota(p=1, y=np.random.default_rng(4).normal(size=(20, 2)))
    assert alias.method == "minnesota_legacy"


@pytest.mark.parametrize("method", list(_priors()))
def test_fit_roundtrip_preserves_resolved_prior_exactly(tmp_path, method):
    original = _fit(_priors()[method])
    path = tmp_path / "fit.npz"
    save_fit_npz(path, original)
    data = _payload(path)
    assert data["format_version"].item() == 2
    assert data["prior_json"].dtype.kind == "U"
    assert all(not value.dtype.hasobject for value in data.values())
    loaded = load_fit_npz(path)
    assert loaded.prior is not None
    assert _as_json(loaded.prior) == _as_json(original.prior)
    np.testing.assert_array_equal(loaded.beta_draws, original.beta_draws)


@pytest.mark.parametrize("method", list(_priors()))
def test_run_roundtrip_preserves_each_prior_family(tmp_path, monkeypatch, method):
    import srvar.config as config

    prior = _priors()[method]
    save_fit_npz(tmp_path / "fit_result.npz", _fit(prior))
    # Deliberately different prior configuration: the saved resolved prior wins.
    (tmp_path / "config.yml").write_text(
        "model:\n  p: 1\nprior:\n  family: niw\nsampler:\n  draws: 2\n  burn_in: 0\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("saved priors must not be re-estimated")

    monkeypatch.setattr(config, "build_prior", forbidden)
    assert _as_json(load_run_dir(tmp_path).prior) == _as_json(prior)


def test_run_loader_rejects_model_dimension_change(tmp_path):
    save_fit_npz(tmp_path / "fit_result.npz", _fit(_priors()["custom"]))
    (tmp_path / "config.yml").write_text("model:\n  p: 2\nprior:\n  family: niw\n")
    with pytest.raises(ValueError, match="prior dimensions.*configured model"):
        load_run_dir(tmp_path)


@pytest.mark.parametrize("empirical", [False, True])
def test_run_loader_uses_saved_dl_prior_not_config_defaults(tmp_path, monkeypatch, empirical):
    import srvar.config as config

    prior = (
        PriorSpec.from_dl(
            k=3,
            n=2,
            residual_prior="empirical_bayes",
            y=np.random.default_rng(7).normal(size=(20, 2)),
            p=1,
        )
        if empirical
        else PriorSpec.from_dl(k=3, n=2, residual_prior="explicit", nu0=9.0, s0=np.diag([3, 11]))
    )
    original = _fit(prior)
    save_fit_npz(tmp_path / "fit_result.npz", original)
    (tmp_path / "config.yml").write_text(
        "model:\n  p: 1\nprior:\n  family: dl\nsampler:\n  draws: 2\n  burn_in: 0\n",
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("must not reconstruct saved prior using current constructors")

    monkeypatch.setattr(config, "build_prior", forbidden)
    loaded = load_run_dir(tmp_path)
    assert _as_json(loaded.prior) == _as_json(prior)


def test_v1_raw_draws_remain_readable_but_run_prior_is_not_invented(tmp_path):
    path = tmp_path / "fit_result.npz"
    save_fit_npz(path, _fit(_priors()["dl"]))
    data = _payload(path)
    data.pop("prior_json")
    data["format_version"] = np.asarray(1)
    np.savez_compressed(path, **data)
    assert load_fit_npz(path).prior is None
    (tmp_path / "config.yml").write_text("model:\n  p: 1\nprior:\n  family: dl\n")
    with pytest.raises(ValueError, match="saved prior.*load_fit_npz"):
        load_run_dir(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.pop("method"),
        lambda p: p.update(unknown="untrusted-payload"),
        lambda p: p.update(method="minnesota_legacy"),
        lambda p: p["niw"].update(nu0=float("nan")),
        lambda p: p["niw"].update(v0=[[1.0]]),
        lambda p: p["niw"].update(s0=[[1.0, 2.0], [2.0, 1.0]]),
        lambda p: p["niw"].update(m0=[[True, False]] * 3),
        lambda p: p["dl"].update(abeta="untrusted-payload"),
    ],
)
def test_invalid_saved_prior_is_rejected_without_echoing_values(tmp_path, change):
    path = tmp_path / "fit.npz"
    save_fit_npz(path, _fit(_priors()["dl"]))
    data = _payload(path)
    prior = json.loads(data["prior_json"].item())
    change(prior)
    data["prior_json"] = np.asarray(json.dumps(prior))
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="prior") as error:
        load_fit_npz(path)
    assert "untrusted-payload" not in str(error.value)


@pytest.mark.parametrize("bad_json", ['{"family":"niw","family":"dl"}', '{"niw": Infinity}', "[]"])
def test_prior_json_rejects_duplicate_keys_nonfinite_and_nonobject(tmp_path, bad_json):
    path = tmp_path / "fit.npz"
    save_fit_npz(path, _fit(_priors()["dl"]))
    data = _payload(path)
    data["prior_json"] = np.asarray(bad_json)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="prior"):
        load_fit_npz(path)


def test_v2_requires_prior_and_checks_it_against_draw_dimensions(tmp_path):
    path = tmp_path / "fit.npz"
    save_fit_npz(path, _fit(_priors()["custom"]))
    data = _payload(path)
    missing = {key: value for key, value in data.items() if key != "prior_json"}
    np.savez_compressed(path, **missing)
    with pytest.raises(ValueError, match="missing required"):
        load_fit_npz(path)
    other = _fit(PriorSpec.niw_default(k=4, n=2))
    save_fit_npz(path, other)
    with pytest.raises(ValueError, match="prior.*dimension"):
        load_fit_npz(path)


def test_provenance_tags_cannot_conflict_with_minnesota_metadata():
    prior = _priors()["minnesota_canonical"]
    with pytest.raises(ValueError, match="method"):
        replace(prior, method="minnesota_tempered")
    with pytest.raises(ValueError, match="method"):
        replace(prior, method="minnesota_legacy")


def test_archived_legacy_dl_prior_remains_readable(tmp_path):
    prior = replace(_priors()["dl"], residual_prior="legacy_default")
    path = tmp_path / "fit.npz"
    save_fit_npz(path, _fit(prior))
    restored = load_fit_npz(path).prior
    assert restored.residual_prior == "legacy_default"
    np.testing.assert_array_equal(restored.niw.s0, np.eye(2))
    assert restored.niw.nu0 == 4


def test_saved_empirical_bayes_marker_requires_matching_shape(tmp_path):
    path = tmp_path / "fit.npz"
    save_fit_npz(path, _fit(_priors()["dl"]))
    data = _payload(path)
    prior = json.loads(data["prior_json"].item())
    prior["residual_prior"] = "empirical_bayes"
    data["prior_json"] = np.asarray(json.dumps(prior))
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="prior"):
        load_fit_npz(path)
