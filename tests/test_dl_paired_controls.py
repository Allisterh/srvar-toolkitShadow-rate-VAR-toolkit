import numpy as np
import pytest

from scripts import compare_dl_controls as study
from scripts import qualify_dl_empirical_bayes as baseline


def test_controls_change_only_specified_prior_blocks():
    y, truth = baseline.generate_data(3, np.random.default_rng(2))
    priors = {arm: study.make_prior(y, truth, arm) for arm in study.ARMS}
    for prior in priors.values():
        assert prior.niw.nu0 == 2
        np.testing.assert_array_equal(prior.niw.m0, np.zeros((3, 2)))
    for suffix in ("eb", "oracle"):
        np.testing.assert_array_equal(
            priors[f"dl_{suffix}"].niw.s0, priors[f"gaussian_{suffix}"].niw.s0
        )
        gaussian = priors[f"gaussian_{suffix}"]
        assert gaussian.method == "custom" and gaussian.dl is None
        np.testing.assert_array_equal(gaussian.minnesota_canonical.inv_v0_vec, np.ones(6))
    np.testing.assert_array_equal(np.diag(priors["dl_eb"].niw.s0), [1e-12, 1e-12])
    np.testing.assert_array_equal(np.diag(priors["dl_oracle"].niw.s0), truth[-2:])
    assert priors["dl_eb"].dl == priors["dl_oracle"].dl
    with pytest.raises(ValueError, match="unknown study arm"):
        study.make_prior(y, truth, "unknown")


def make_record(arm, replicate, covered=None):
    record = dict(arm=arm, replicate=replicate, cell="short", status="failed")
    if covered is not None:
        record.update(
            status="ok",
            rows=[
                dict(
                    parameter="beta00",
                    covered90=covered,
                    error=0.2,
                    width90=1.0,
                    flag=False,
                    rhat=1.0,
                    ess_bulk=500.0,
                    ess_tail=500.0,
                )
            ],
        )
    return record


def test_gaussian_control_routes_to_independent_normal_conditional():
    class FixedRandom:
        def standard_normal(self, size):
            return np.zeros(size)

        def gamma(self, *, shape, scale):
            assert shape == 2 + 39 / 2
            assert scale == 1
            return 1.0

    y, truth = baseline.generate_data(0, np.random.default_rng(5))
    prior = study.make_prior(y, truth, "gaussian_oracle")
    result = study.fit(
        study.Dataset.from_arrays(values=y, variables=["a", "b"]),
        study.ModelSpec(p=1),
        prior,
        study.SamplerConfig(draws=1, burn_in=0),
        rng=FixedRandom(),
    )
    x = np.column_stack([np.ones(39), y[:-1]])
    expected = np.linalg.solve(np.eye(3) + x.T @ x, x.T @ y[1:])
    np.testing.assert_allclose(result.beta_draws[0], expected, atol=1e-13)
    rates = 1 + 0.5 * np.sum((y[1:] - x @ expected) ** 2, axis=0)
    np.testing.assert_allclose(np.diag(result.sigma_draws[0]), rates)


def test_paired_differences_use_dataset_mcse_and_keep_missing_pairs():
    records = [
        make_record("dl_eb", 0, False),
        make_record("dl_oracle", 0, True),
        make_record("dl_eb", 1, True),
        make_record("dl_oracle", 1, False),
        make_record("dl_eb", 2),
        make_record("dl_oracle", 2),
    ]
    frame = study.paired_summaries(records)
    row = frame[
        (frame.contrast == "oracle_minus_eb_dl")
        & (frame.parameter == "beta00")
        & (frame.metric == "coverage90")
    ].iloc[0]
    assert row.attempted == 3 and row.complete_pairs == 2 and row.incomplete_pairs == 1
    assert row.difference == 0 and row.paired_mcse == 1
    assert row.failure_bound_low == pytest.approx(-1 / 3)
    assert row.failure_bound_high == pytest.approx(1 / 3)


def test_all_failed_arms_remain_in_census_and_bounds():
    records = [make_record(arm, i) for arm in study.ARMS for i in range(2)]
    summary, counts = study.arm_summaries(records)
    assert summary.empty
    assert len(counts) == 4 and (counts.failed == 2).all()
    paired = study.paired_summaries(records)
    assert (paired.complete_pairs == 0).all() and paired.difference.isna().all()
    interaction = paired[(paired.contrast == "interaction") & (paired.metric == "coverage90")]
    assert (interaction.failure_bound_low == -2).all()
    assert (interaction.failure_bound_high == 2).all()


def test_raw_chains_pairing_and_exact_policy_reproduction(tmp_path, monkeypatch):
    pytest.importorskip("arviz")
    (tmp_path / "chains").mkdir()
    job = dict(out=str(tmp_path), cell_id=0, replicate=0, seed=17, chains=2, draws=20, warmup=10)
    original = study.fit
    gaussian_calls = 0

    def fail_one_arm(dataset, model, prior, sampler, **kwargs):
        nonlocal gaussian_calls
        if prior.minnesota_canonical is not None:
            gaussian_calls += 1
            if gaussian_calls == 2:
                raise ValueError("controlled second-chain failure")
        return original(dataset, model, prior, sampler, **kwargs)

    monkeypatch.setattr(study, "fit", fail_one_arm)
    records = study.run_dataset(job)
    assert len({r["data_sha256"] for r in records}) == 1
    assert [r["status"] for r in records] == ["ok", "ok", "failed", "ok"]
    partial = np.load(tmp_path / records[2]["partial_chain_file"], allow_pickle=False)
    assert partial["samples"].shape == (1, 20, 8)
    assert records[2]["completed_chains"] == 1
    # The baseline uses current rates; historical EB controls preserve the old estimator.
    policy_records = study.run_dataset(dict(job, policy_only=True))
    assert [r["arm"] for r in policy_records] == ["dl_policy", "gaussian_policy"]
    assert [r["status"] for r in policy_records] == ["ok", "ok"]
    assert {r["data_sha256"] for r in records + policy_records} == {records[0]["data_sha256"]}
    reference = baseline.run_replication(job)
    np.testing.assert_array_equal(
        np.diag(policy_records[0]["prior"]["niw"]["s0"]), reference["rates"]
    )
    for row, expected in zip(policy_records[0]["rows"], reference["rows"], strict=True):
        for key in ("posterior_mean", "lower90", "upper90", "rhat", "ess_bulk", "ess_tail"):
            np.testing.assert_allclose(row[key], expected[key], rtol=0, atol=0, equal_nan=True)
    data = np.load(tmp_path / "chains/short-0000-data.npz", allow_pickle=False)
    for record in records + policy_records:
        if record["status"] == "ok":
            samples = np.load(tmp_path / record["chain_file"], allow_pickle=False)["samples"]
            assert samples.shape == (2, 20, 8)
            rows = study.summarise_samples(samples, data["truth"])
            for actual, saved in zip(rows, record["rows"], strict=True):
                assert actual == saved
