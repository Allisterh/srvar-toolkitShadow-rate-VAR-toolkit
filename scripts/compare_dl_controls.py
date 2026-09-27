"""Paired fixed-DGP controls for residual-prior and coefficient-prior choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import qualify_dl_empirical_bayes as baseline
from scripts.qualify_samplers import chain_diagnostics
from scripts.study_evidence import archive_sources
from srvar._prior_io import prior_to_json
from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import (
    MinnesotaCanonicalSpec,
    ModelSpec,
    PriorSpec,
    SamplerConfig,
    _estimate_minnesota_sigma2,
)

ARMS = ("dl_eb", "dl_oracle", "gaussian_eb", "gaussian_oracle")
POLICY_ARMS = ("dl_policy", "gaussian_policy")
UNFLOORED_ARMS = ("dl_unfloored", "gaussian_unfloored")
LABELS = ("beta00", "beta10", "beta20", "beta01", "beta11", "beta21", "variance0", "variance1")
CONTRASTS = {
    "oracle_minus_eb_dl": {"dl_oracle": 1, "dl_eb": -1},
    "oracle_minus_eb_gaussian": {"gaussian_oracle": 1, "gaussian_eb": -1},
    "gaussian_minus_dl_eb": {"gaussian_eb": 1, "dl_eb": -1},
    "gaussian_minus_dl_oracle": {"gaussian_oracle": 1, "dl_oracle": -1},
    "interaction": {"gaussian_oracle": 1, "gaussian_eb": -1, "dl_oracle": -1, "dl_eb": 1},
}
UNFLOORED_CONTRASTS = {
    "unfloored_minus_eb_dl": {"dl_unfloored": 1, "dl_eb": -1},
    "unfloored_minus_eb_gaussian": {"gaussian_unfloored": 1, "gaussian_eb": -1},
    "oracle_minus_unfloored_dl": {"dl_oracle": 1, "dl_unfloored": -1},
    "oracle_minus_unfloored_gaussian": {"gaussian_oracle": 1, "gaussian_unfloored": -1},
    "gaussian_minus_dl_unfloored": {"gaussian_unfloored": 1, "dl_unfloored": -1},
    "unfloored_interaction": {
        "gaussian_unfloored": 1,
        "gaussian_eb": -1,
        "dl_unfloored": -1,
        "dl_eb": 1,
    },
}


def estimate_unfloored_rates(y: np.ndarray) -> np.ndarray:
    """Return raw AR(1) residual estimates, including zero estimates for failure records."""
    return _estimate_minnesota_sigma2(y=y, p=1, include_intercept=True, min_sigma2=0.0)


def make_prior(y: np.ndarray, truth: np.ndarray, arm: str) -> PriorSpec:
    """Change only the declared prior factors; the Gaussian comparator is N(0,I)."""
    if arm not in ARMS + UNFLOORED_ARMS + POLICY_ARMS:
        raise ValueError(f"unknown study arm: {arm}")
    if arm.endswith("_policy"):
        prior = PriorSpec.from_dl(k=3, n=2, residual_prior="empirical_bayes", y=y, p=1)
    else:
        if arm.endswith("_eb"):
            # Preserve the historical floored study target independently of current defaults.
            rates = _estimate_minnesota_sigma2(y=y, p=1, include_intercept=True, min_sigma2=1e-12)
        else:
            rates = estimate_unfloored_rates(y) if arm.endswith("_unfloored") else truth[-2:]
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError(
                "unfloored/oracle IG rates must be finite and positive; no replacement is applied"
            )
        prior = PriorSpec.from_dl(k=3, n=2, residual_prior="explicit", nu0=2, s0=np.diag(rates))
    if arm.startswith("gaussian"):
        # Custom provenance: metadata routes to the existing independent-normal
        # kernel. No Minnesota constructor or scale-dependent variance map is used.
        return PriorSpec(
            family="niw",
            niw=prior.niw,
            minnesota_canonical=MinnesotaCanonicalSpec(
                sigma2=np.diag(prior.niw.s0), inv_v0_vec=np.ones(6)
            ),
            method="custom",
        )
    return prior


def summarise_samples(samples: np.ndarray, truth: np.ndarray) -> list[dict]:
    """Summarise paired coefficient/variance draws without diagnostic filtering."""
    diagnostics = chain_diagnostics(samples)
    flat = samples.reshape(-1, 8)
    lower, upper = np.quantile(flat, [0.05, 0.95], axis=0)
    rows = []
    for i, label in enumerate(LABELS):
        row = dict(
            parameter=label,
            truth=float(truth[i]),
            posterior_mean=float(flat[:, i].mean()),
            error=float(flat[:, i].mean() - truth[i]),
            covered90=bool(lower[i] <= truth[i] <= upper[i]),
            width90=float(upper[i] - lower[i]),
            lower90=float(lower[i]),
            upper90=float(upper[i]),
            **{key: float(value[i]) for key, value in diagnostics.items()},
        )
        row["flag"] = bool(
            row["rhat"] > 1.01
            or row["ess_bulk"] < 400
            or row["ess_tail"] < 400
            or not np.isfinite([row["rhat"], row["ess_bulk"], row["ess_tail"]]).all()
        )
        rows.append(row)
    return rows


def run_dataset(job: dict) -> list[dict]:
    """Run all arms on one dataset; retain successful arms when another fails."""
    streams = np.random.SeedSequence([job["seed"], job["cell_id"], job["replicate"]]).spawn(
        job["chains"] + 1
    )
    y, truth = baseline.generate_data(job["cell_id"], np.random.default_rng(streams[0]))
    cell = baseline.CELLS[job["cell_id"]][0]
    identifier = f"{cell}-{job['replicate']:04d}"
    out = Path(job["out"])
    np.savez_compressed(out / "chains" / f"{identifier}-data.npz", y=y, truth=truth)
    data_hash = hashlib.sha256(np.asarray(y, dtype="<f8").tobytes()).hexdigest()
    dataset = Dataset.from_arrays(values=y, variables=["a", "b"])
    records = []
    arms = ARMS + UNFLOORED_ARMS if job.get("include_unfloored", False) else ARMS
    if job.get("policy_only", False):
        arms = POLICY_ARMS
    for arm in arms:
        started = time.perf_counter()
        record = dict(cell=cell, replicate=job["replicate"], arm=arm, data_sha256=data_hash)
        samples = []
        try:
            if arm.endswith("_unfloored"):
                record["estimated_unfloored_rates"] = estimate_unfloored_rates(y).tolist()
            prior = make_prior(y, truth, arm)
            record["prior"] = json.loads(prior_to_json(prior))
            for stream in streams[1:]:
                result = fit(
                    dataset,
                    ModelSpec(p=1),
                    prior,
                    SamplerConfig(draws=job["draws"] + job["warmup"], burn_in=job["warmup"]),
                    rng=np.random.default_rng(stream),
                )
                samples.append(
                    np.column_stack(
                        [
                            result.beta_draws.transpose(0, 2, 1).reshape(job["draws"], 6),
                            np.diagonal(result.sigma_draws, axis1=1, axis2=2),
                        ]
                    )
                )
            arrays = np.stack(samples)
            relative = f"chains/{identifier}-{arm}.npz"
            np.savez_compressed(out / relative, samples=arrays)
            record.update(status="ok", chain_file=relative, rows=summarise_samples(arrays, truth))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if samples:
                relative = f"chains/{identifier}-{arm}-partial.npz"
                np.savez_compressed(out / relative, samples=np.stack(samples))
                record["partial_chain_file"] = relative
        record.update(completed_chains=len(samples), seconds=time.perf_counter() - started)
        records.append(record)
    return records


def arm_summaries(
    records: list[dict], *, arms: tuple[str, ...] = ARMS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse baseline marginal summaries; keep an explicit census of all-failed arms."""
    summaries, counts = [], []
    for cell in sorted({r["cell"] for r in records}):
        for arm in arms:
            selected = [r for r in records if r["cell"] == cell and r["arm"] == arm]
            successful = sum(r["status"] == "ok" for r in selected)
            counts.append(
                dict(
                    cell=cell,
                    arm=arm,
                    attempted=len(selected),
                    successful=successful,
                    failed=len(selected) - successful,
                )
            )
            adapted = [
                dict(r, rows=[dict(row, cell=cell) for row in r.get("rows", [])]) for r in selected
            ]
            summary = baseline.summarise(adapted)
            if not summary.empty:
                summary["arm"] = arm
                summaries.append(summary)
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.DataFrame(counts),
    )


def paired_summaries(records: list[dict], *, contrasts: dict | None = None) -> pd.DataFrame:
    """Dataset-paired contrasts, complete-case MCSE and failure-inclusive coverage bounds."""
    results = []
    if contrasts is None:
        contrasts = CONTRASTS
    for cell in sorted({r["cell"] for r in records}):
        selected = [r for r in records if r["cell"] == cell]
        replicates = sorted({r["replicate"] for r in selected})
        lookup = {
            (r["replicate"], r["arm"]): {row["parameter"]: row for row in r.get("rows", [])}
            for r in selected
            if r["status"] == "ok"
        }
        for contrast, weights in contrasts.items():
            for label in LABELS:
                values, lower, upper = [], [], []
                for replicate in replicates:
                    available = {
                        arm: lookup.get((replicate, arm), {}).get(label) for arm in weights
                    }
                    lo = hi = 0.0
                    for arm, weight in weights.items():
                        row = available[arm]
                        if row is None:
                            lo += min(weight, 0)
                            hi += max(weight, 0)
                        else:
                            lo += weight * row["covered90"]
                            hi += weight * row["covered90"]
                    lower.append(lo)
                    upper.append(hi)
                    if all(row is not None for row in available.values()):
                        values.append(
                            [
                                sum(
                                    weight * available[arm][metric]
                                    for arm, weight in weights.items()
                                )
                                for metric in ("covered90", "error", "width90")
                            ]
                        )
                array = np.asarray(values).reshape(-1, 3)
                for j, metric in enumerate(("coverage90", "bias", "width90")):
                    results.append(
                        dict(
                            cell=cell,
                            contrast=contrast,
                            parameter=label,
                            metric=metric,
                            attempted=len(replicates),
                            complete_pairs=len(array),
                            incomplete_pairs=len(replicates) - len(array),
                            difference=float(array[:, j].mean()) if len(array) else np.nan,
                            paired_mcse=float(array[:, j].std(ddof=1) / np.sqrt(len(array)))
                            if len(array) > 1
                            else np.nan,
                            failure_bound_low=float(np.mean(lower))
                            if metric == "coverage90"
                            else np.nan,
                            failure_bound_high=float(np.mean(upper))
                            if metric == "coverage90"
                            else np.nan,
                        )
                    )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cells", nargs="+", choices=[c[0] for c in baseline.CELLS], default=["short"]
    )
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--include-unfloored",
        action="store_true",
        help="add estimated rates without a floor as two extra arms",
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="fit only the current DL residual-rate policy arms",
    )
    args = parser.parse_args()
    if args.policy_only and args.include_unfloored:
        parser.error("--policy-only and --include-unfloored are mutually exclusive")
    if (
        args.replications < 2
        or args.chains < 2
        or args.draws < 20
        or min(args.warmup, args.seed) < 0
        or args.workers < 1
    ):
        parser.error(
            "require >=2 replications/chains, >=20 draws, nonnegative warmup/seed and positive workers"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "chains").mkdir()
    controls = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    arms = ARMS + UNFLOORED_ARMS if args.include_unfloored else ARMS
    contrasts = CONTRASTS | UNFLOORED_CONTRASTS if args.include_unfloored else CONTRASTS
    if args.policy_only:
        arms = POLICY_ARMS
        contrasts = {"gaussian_minus_dl_policy": {"gaussian_policy": 1, "dl_policy": -1}}
    manifest = dict(
        rate_policy="normalised_ar" if args.policy_only else "historical_absolute_floor_controls",
        controls=controls,
        arms=arms,
        contrasts=contrasts,
        coefficient_control="independent N(0,1) in the supplied measurement units",
        residual_shape=2,
        dgp_cells=baseline.CELLS,
        estimand="paired fixed-DGP frequentist 90% interval coverage; not SBC",
        initialisation="production deterministic starts; identical per-chain seeds across arms",
        **archive_sources(args.out, [Path(__file__), Path(baseline.__file__)]),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [
        dict(controls, cell_id=i, replicate=rep)
        for i, c in enumerate(baseline.CELLS)
        if c[0] in args.cells
        for rep in range(args.replications)
    ]
    records = []
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        (args.out / "replications.jsonl").open("w") as stream,
    ):
        for i, batch in enumerate(pool.map(run_dataset, jobs), 1):
            records.extend(batch)
            for record in batch:
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                f"datasets={i}/{len(jobs)} failed_arms={sum(r['status'] != 'ok' for r in records)}",
                flush=True,
            )
    summary, counts = arm_summaries(records, arms=arms)
    summary.to_csv(args.out / "summary.csv", index=False)
    counts.to_csv(args.out / "arm_counts.csv", index=False)
    paired_summaries(records, contrasts=contrasts).to_csv(
        args.out / "paired_summary.csv", index=False
    )
    failures = [r for r in records if r["status"] != "ok"]
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed arms retained")


if __name__ == "__main__":
    main()
