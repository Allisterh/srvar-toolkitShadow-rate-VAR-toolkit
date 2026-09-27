import numpy as np
import pandas as pd
import pytest

from scripts import qualify_samplers as study
from scripts.diagnose_sampler_chains import monitored_draws
from scripts.qualify_local_benchmark import benchmark_configs, validate_source_root


def test_gaussian_reference_matches_scalar_conjugate_solution():
    mean, covariance = study.gaussian_reference(
        np.ones((3, 1)),
        np.array([[1.0], [2.0], [3.0]]),
        np.eye(1),
        np.zeros((3, 1)),
        np.array([[2.0]]),
        np.array([[0.5]]),
    )
    np.testing.assert_allclose(mean, [2.0])
    np.testing.assert_allclose(covariance, [[0.2]])


def test_summary_uses_replications_for_monte_carlo_uncertainty():
    frame = pd.DataFrame(
        {
            "cell": ["a", "a"],
            "parameter": ["b", "b"],
            "error": [-1.0, 1.0],
            "covered90": [True, False],
            "width90": [2.0, 3.0],
            "rhat": [1.0, 1.02],
            "ess_bulk": [500, 500],
            "ess_tail": [500, 500],
            "rank_u": [0.2, 0.8],
            "max_abs_thinned_acf1": [0.1, 0.2],
        }
    )
    result = study.summarise_rows(frame).iloc[0]
    assert result.bias == 0
    assert result.bias_mcse == pytest.approx(1)
    assert result.rmse == 1
    assert result.coverage90 == 0.5
    assert result.coverage_mcse == pytest.approx(np.sqrt(0.125))
    assert result.coverage_wilson_low < 0.5 < result.coverage_wilson_high
    assert result.diagnostic_flags == 1


def test_failures_remain_identifiable(monkeypatch):
    def fail(job):
        raise np.linalg.LinAlgError("test failure")

    monkeypatch.setattr(study, "run_replication", fail)
    result = study.safe_replication({"cell_id": 1, "replicate": 7})
    assert result["status"] == "failed"
    assert result["replicate"] == 7
    assert result["cell"] == study.CELLS[1][0]
    assert "LinAlgError" in result["error"]


@pytest.mark.parametrize("cell_id", [0, 2, 4])
def test_study_is_reproducible_and_keeps_truth_per_replication(cell_id):
    pytest.importorskip("arviz")
    job = dict(
        cell_id=cell_id,
        replicate=1,
        seed=29,
        chains=2,
        draws=40,
        warmup=20,
        rank_thin=5,
        reset_variance_control=False,
    )
    first = study.run_replication(job)
    assert first == study.run_replication(job)
    changed = study.run_replication(dict(job, seed=30))
    assert first["rows"][0]["truth"] != changed["rows"][0]["truth"]
    for row in first["rows"]:
        assert row["rank_draws"] == 16
        assert 0 < row["rank_u"] < 1


def test_monitor_excludes_structural_covariance_zeros():
    from types import SimpleNamespace

    result = SimpleNamespace(
        prior=SimpleNamespace(minnesota_canonical=object(), family="niw"),
        beta_draws=np.zeros((5, 2, 2)),
        sigma_draws=np.broadcast_to(np.eye(2), (5, 2, 2)),
        q_draws=None,
        sigma_eta2_draws=None,
        h_draws=None,
    )
    values, labels = monitored_draws(result)
    assert values.shape == (5, 6)
    assert "sigma_draws:1,0" not in labels
    assert "sigma_draws:1,1" in labels


def test_benchmark_changes_only_paths_and_declared_variants(tmp_path):
    pytest.importorskip("yaml")
    configs = benchmark_configs(tmp_path / "data.csv", tmp_path / "out")
    baseline = configs["legacy_homoskedastic"]
    candidate = configs["canonical_homoskedastic"]
    assert candidate["prior"]["method"] == "minnesota_canonical"
    assert baseline["prior"]["method"] == "minnesota_legacy"
    assert baseline["sampler"] == candidate["sampler"]
    assert baseline["backtest"] == candidate["backtest"]
    assert baseline["data"] == candidate["data"]
    assert baseline["backtest"]["origin_start"] == "1995-01-01"
    assert baseline["backtest"]["origin_end"] == "2019-01-01"


def test_benchmark_rejects_wrong_imported_source_tree(tmp_path):
    with pytest.raises(ValueError, match="differs from expected"):
        validate_source_root(tmp_path / "wrong-source")
