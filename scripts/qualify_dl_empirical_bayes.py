"""Fixed-DGP frequentist coverage of the plug-in DL procedure; this is not SBC."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts.qualify_samplers import chain_diagnostics
from scripts.study_evidence import archive_sources
from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig

CELLS = (
    ("short", 40, 0.3, (1.0, 1.0)),
    ("unequal_scale", 120, 0.3, (1.0, 5.0)),
    ("persistent", 120, 0.9, (1.0, 1.0)),
    ("floor_stress", 40, 0.3, (1e-7, 1e-7)),
)


def generate_data(cell_id: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate stationary diagonal VAR(1); truth order is column-major B then variances."""
    _, nobs, rho, scales = CELLS[cell_id]
    sd = np.asarray(scales)
    y = np.empty((nobs, 2))
    y[0] = rng.normal(size=2) * sd / np.sqrt(1 - rho * rho)
    for t in range(1, nobs):
        y[t] = rho * y[t - 1] + rng.normal(size=2) * sd
    beta = np.array([[0.0, 0.0], [rho, 0.0], [0.0, rho]])
    return y, np.concatenate([beta.T.reshape(-1), sd * sd])


def run_replication(job: dict) -> dict:
    """Estimate scales once per dataset; retain all chain diagnostics and coverage outcomes."""
    start = time.perf_counter()
    streams = np.random.SeedSequence([job["seed"], job["cell_id"], job["replicate"]]).spawn(
        job["chains"] + 1
    )
    y, truth = generate_data(job["cell_id"], np.random.default_rng(streams[0]))
    dataset = Dataset.from_arrays(values=y, variables=["a", "b"])
    prior = PriorSpec.from_dl(k=3, n=2, p=1, y=y, residual_prior="empirical_bayes")
    samples = []
    for stream in streams[1:]:
        result = fit(
            dataset,
            ModelSpec(p=1),
            prior,
            SamplerConfig(draws=job["warmup"] + job["draws"], burn_in=job["warmup"]),
            rng=np.random.default_rng(stream),
        )
        beta = result.beta_draws.transpose(0, 2, 1).reshape(job["draws"], 6)
        variance = np.diagonal(result.sigma_draws, axis1=1, axis2=2)
        samples.append(np.column_stack([beta, variance]))
    samples = np.stack(samples)
    diagnostics = chain_diagnostics(samples)
    flat = samples.reshape(-1, 8)
    lo, hi = np.quantile(flat, [0.05, 0.95], axis=0)
    rows = []
    for i, label in enumerate(
        ["beta00", "beta10", "beta20", "beta01", "beta11", "beta21", "variance0", "variance1"]
    ):
        row = dict(
            cell=CELLS[job["cell_id"]][0],
            replicate=job["replicate"],
            parameter=label,
            truth=float(truth[i]),
            posterior_mean=float(flat[:, i].mean()),
            error=float(flat[:, i].mean() - truth[i]),
            covered90=bool(lo[i] <= truth[i] <= hi[i]),
            width90=float(hi[i] - lo[i]),
            lower90=float(lo[i]),
            upper90=float(hi[i]),
            **{key: float(value[i]) for key, value in diagnostics.items()},
        )
        row["flag"] = bool(
            row["rhat"] > 1.01
            or row["ess_bulk"] < 400
            or row["ess_tail"] < 400
            or not np.isfinite([row["rhat"], row["ess_bulk"], row["ess_tail"]]).all()
        )
        rows.append(row)
    return dict(
        status="ok",
        cell=CELLS[job["cell_id"]][0],
        replicate=job["replicate"],
        rows=rows,
        rates=np.diag(prior.niw.s0).tolist(),
        shape=prior.niw.nu0,
        floor_active=(np.diag(prior.niw.s0) == 1e-12).tolist(),
        seconds=time.perf_counter() - start,
    )


def safe_replication(job: dict) -> dict:
    """Retain a failure record rather than silently dropping a dataset."""
    try:
        return run_replication(job)
    except Exception as exc:
        return dict(
            status="failed",
            cell=CELLS[job["cell_id"]][0],
            replicate=job["replicate"],
            error=f"{type(exc).__name__}: {exc}",
        )


def summarise(records: list[dict]) -> pd.DataFrame:
    """Report success-conditional coverage and bounds counting all failed datasets."""
    frame = pd.DataFrame([row for record in records for row in record.get("rows", [])])
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (cell, param), group in frame.groupby(["cell", "parameter"]):
        attempted = sum(record["cell"] == cell for record in records)
        successful = len(group)
        covered = int(group.covered90.sum())
        interval = binomtest(covered, successful).proportion_ci(method="wilson")
        rows.append(
            dict(
                cell=cell,
                parameter=param,
                attempted=attempted,
                successful=successful,
                failures=attempted - successful,
                coverage90=covered / successful,
                coverage_mcse=np.sqrt(
                    (covered / successful) * (1 - covered / successful) / successful
                ),
                wilson_low=interval.low,
                wilson_high=interval.high,
                failure_bound_low=covered / attempted,
                failure_bound_high=(covered + attempted - successful) / attempted,
                bias=group.error.mean(),
                bias_mcse=group.error.std(ddof=1) / np.sqrt(successful),
                rmse=np.sqrt(np.mean(group.error**2)),
                mean_width90=group.width90.mean(),
                diagnostic_flags=int(group.flag.sum()),
                max_rhat=group.rhat.max(),
                min_ess_bulk=group.ess_bulk.min(),
                min_ess_tail=group.ess_tail.min(),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.replications, args.workers) < 1
        or args.chains < 2
        or args.draws < 20
        or min(args.warmup, args.seed) < 0
    ):
        parser.error(
            "require positive replications/workers, >=2 chains, >=20 draws, nonnegative warmup/seed"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    controls = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(
        controls=controls,
        cells=CELLS,
        estimand="fixed-DGP frequentist 90% interval coverage",
        initialisation="full fit uses common deterministic initial state, independent RNG streams",
        **archive_sources(args.out, [Path(__file__)]),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [
        dict(controls, cell_id=cell, replicate=rep)
        for cell in range(len(CELLS))
        for rep in range(args.replications)
    ]
    records = []
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        (args.out / "replications.jsonl").open("w") as stream,
    ):
        for i, record in enumerate(pool.map(safe_replication, jobs), 1):
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if i % 5 == 0 or i == len(jobs):
                print(
                    f"completed={i}/{len(jobs)} failures={sum(r['status'] != 'ok' for r in records)}",
                    flush=True,
                )
    pd.DataFrame(
        [
            {
                "cell": cell[0],
                "attempted": sum(r["cell"] == cell[0] for r in records),
                "successful": sum(r["cell"] == cell[0] and r["status"] == "ok" for r in records),
                "failed": sum(r["cell"] == cell[0] and r["status"] != "ok" for r in records),
            }
            for cell in CELLS
        ]
    ).to_csv(args.out / "dataset_counts.csv", index=False)
    summarise(records).to_csv(args.out / "summary.csv", index=False)
    failures = [r for r in records if r["status"] != "ok"]
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed datasets retained")


if __name__ == "__main__":
    main()
