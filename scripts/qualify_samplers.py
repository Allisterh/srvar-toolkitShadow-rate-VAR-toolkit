"""Component SBC for repaired transitions; not a certification of full model families."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import srvar
from srvar.samplers_dl import _dl_sample_beta_sigma
from srvar.samplers_svcov import _sample_beta_triangular_svrw

CELLS = (
    ("independent_20_scale1", "independent", 20, 1.0),
    ("independent_80_scale25", "independent", 80, 25.0),
    ("triangular_20_q08", "triangular", 20, 0.8),
    ("triangular_80_q16", "triangular", 80, 1.6),
)


def gaussian_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    h: np.ndarray,
    m0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a small dense joint posterior, independently of the block sampler."""
    n = y.shape[1]
    precision = np.kron(np.eye(n), np.linalg.inv(v0))
    rhs = precision @ m0.ravel(order="F")
    for xt, yt, ht in zip(x, y, h, strict=True):
        design = np.kron(np.eye(n), xt[None, :])
        innovation_precision = q.T @ np.diag(np.exp(-ht)) @ q
        precision += design.T @ innovation_precision @ design
        rhs += design.T @ innovation_precision @ yt
    covariance = np.linalg.inv(precision)
    return np.linalg.solve(precision, rhs), covariance


def chain_diagnostics(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Rank-normalised diagnostics for (chain, draw, parameter) arrays."""
    import arviz as az

    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 4:
        raise ValueError("diagnostics require at least two chains and four draws")
    if not np.isfinite(samples).all():
        raise ValueError("non-finite posterior draws")
    data = az.from_dict(posterior={"theta": samples})
    return {
        "rhat": az.rhat(data, method="rank").theta.values,
        "ess_bulk": az.ess(data, method="bulk").theta.values,
        "ess_tail": az.ess(data, method="tail").theta.values,
        "mcse_mean": az.mcse(data, method="mean").theta.values,
    }


def source_manifest() -> dict:
    """Record numerical source and tool versions without serialising environment secrets."""
    root = Path(__file__).resolve().parents[1]
    package_root = Path(srvar.__file__).resolve().parent
    hashes = {
        str(p.relative_to(package_root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }
    hashes[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return {
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "arviz")
        },
        "package_path": str(package_root),
        "sha256": hashes,
    }


def run_replication(job: dict) -> dict:
    """Generate one prior-predictive dataset and sample each chain independently."""
    cell, family, nobs, scale = CELLS[job["cell_id"]]
    seeds = np.random.SeedSequence([job["seed"], job["cell_id"], job["replicate"]]).spawn(
        job["chains"] + 2
    )
    dgp_rng = np.random.default_rng(seeds[0])
    rank_rng = np.random.default_rng(seeds[-1])
    x = np.column_stack([np.ones(nobs), np.linspace(-1, 1, nobs)])
    m0 = np.array([[0.3, -0.2], [0.1, 0.4]])
    precision = np.array([[1.0, 2.0], [3.0, 4.0]])
    v0 = np.array([[0.7, 0.1], [0.1, 1.2]])
    rates = np.array([2.0, 3.0]) * scale
    q = np.array([[1.0, scale], [0.0, 1.0]])
    h = np.column_stack([0.5 * np.sin(np.arange(nobs)), np.linspace(-0.5, 0.5, nobs)])
    if family == "independent":
        beta_true = m0 + dgp_rng.normal(size=(2, 2)) / np.sqrt(precision)
        variance_true = rates / dgp_rng.gamma(shape=3.0, size=2)
        y = x @ beta_true + dgp_rng.normal(size=(nobs, 2)) * np.sqrt(variance_true)
        truth = np.concatenate([beta_true.ravel(order="F"), variance_true])
        labels = ["beta00", "beta10", "beta01", "beta11", "variance0", "variance1"]
        reference_mean = None
    else:
        beta_true = m0 + np.linalg.cholesky(v0) @ dgp_rng.normal(size=(2, 2))
        shocks = dgp_rng.normal(size=(nobs, 2)) * np.exp(h / 2)
        y = x @ beta_true + np.linalg.solve(q, shocks.T).T
        truth = beta_true.ravel(order="F")
        labels = ["beta00", "beta10", "beta01", "beta11"]
        reference_mean, _ = gaussian_reference(x, y, q, h, m0, v0)

    samples = np.empty((job["chains"], job["draws"], len(truth)))
    for chain in range(job["chains"]):
        rng = np.random.default_rng(seeds[chain + 1])
        beta = m0 + rng.normal(size=(2, 2)) * 3
        sigma = np.diag(rates / 2 * 4.0 ** (chain - 1))
        for iteration in range(job["warmup"] + job["draws"]):
            if family == "independent":
                beta, sigma = _dl_sample_beta_sigma(
                    x=x,
                    y=y,
                    m0=m0,
                    inv_v0_vec=precision.ravel(order="F"),
                    s0=np.diag(rates),
                    nu0=3.0,
                    sigma=np.diag(rates) if job["reset_variance_control"] else sigma,
                    rng=rng,
                )
                draw = np.concatenate([beta.ravel(order="F"), np.diag(sigma)])
            else:
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
                draw = beta.ravel(order="F")
            if iteration >= job["warmup"]:
                samples[chain, iteration - job["warmup"]] = draw

    diagnostics = chain_diagnostics(samples)
    flat = samples.reshape(-1, len(truth))
    thinned = samples[:, :: job["rank_thin"], :]
    rank_draws = thinned.reshape(-1, len(truth))
    ranks = (rank_draws < truth).sum(axis=0)
    lower, upper = np.quantile(flat, [0.05, 0.95], axis=0)
    mean = flat.mean(axis=0)
    rows = []
    for index, label in enumerate(labels):
        acf = [np.corrcoef(chain[:-1, index], chain[1:, index])[0, 1] for chain in thinned]
        rows.append(
            {
                "cell": cell,
                "replicate": job["replicate"],
                "parameter": label,
                "truth": float(truth[index]),
                "posterior_mean": float(mean[index]),
                "error": float(mean[index] - truth[index]),
                "covered90": bool(lower[index] <= truth[index] <= upper[index]),
                "width90": float(upper[index] - lower[index]),
                "rank": int(ranks[index]),
                "rank_draws": len(rank_draws),
                "rank_u": float((ranks[index] + rank_rng.random()) / (len(rank_draws) + 1)),
                "max_abs_thinned_acf1": float(np.max(np.abs(acf))),
                **{key: float(value[index]) for key, value in diagnostics.items()},
                "reference_mean": None if reference_mean is None else float(reference_mean[index]),
            }
        )
    return {"status": "ok", "cell": cell, "replicate": job["replicate"], "rows": rows}


def safe_replication(job: dict) -> dict:
    """Keep numerical failures in the study denominator and evidence stream."""
    try:
        return run_replication(job)
    except Exception as exc:
        return {
            "status": "failed",
            "cell": CELLS[job["cell_id"]][0],
            "replicate": job["replicate"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise independent replications, including Monte Carlo uncertainty."""
    summaries = []
    for (cell, parameter), group in frame.groupby(["cell", "parameter"], sort=True):
        count = len(group)
        covered = int(group.covered90.sum())
        interval = binomtest(covered, count).proportion_ci(method="wilson")
        coverage = covered / count
        summaries.append(
            {
                "cell": cell,
                "parameter": parameter,
                "successful_replications": count,
                "bias": group.error.mean(),
                "bias_mcse": group.error.std(ddof=1) / np.sqrt(count),
                "rmse": np.sqrt(np.mean(group.error**2)),
                "coverage90": coverage,
                "coverage_mcse": np.sqrt(coverage * (1 - coverage) / count),
                "coverage_wilson_low": interval.low,
                "coverage_wilson_high": interval.high,
                "mean_width90": group.width90.mean(),
                "max_rhat": group.rhat.max(),
                "min_ess_bulk": group.ess_bulk.min(),
                "min_ess_tail": group.ess_tail.min(),
                "diagnostic_flags": int(
                    (
                        (group.rhat > 1.01)
                        | (group.ess_bulk < 400)
                        | (group.ess_tail < 400)
                        | ~np.isfinite(group.rhat)
                        | ~np.isfinite(group.ess_bulk)
                        | ~np.isfinite(group.ess_tail)
                    ).sum()
                ),
                "mean_rank_u": group.rank_u.mean(),
                "max_abs_thinned_acf1": group.max_abs_thinned_acf1.max(),
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--rank-thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", nargs="+", choices=[cell[0] for cell in CELLS])
    parser.add_argument("--reset-variance-control", action="store_true")
    args = parser.parse_args()
    if (
        args.replications < 2
        or args.chains < 2
        or args.draws < 20
        or args.warmup < 0
        or args.rank_thin < 1
        or args.draws // args.rank_thin < 4
        or args.workers < 1
        or args.seed < 0
    ):
        parser.error(
            "require >=2 replications/chains, >=20 draws, >=4 thinned draws and valid positive controls"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    controls = vars(args).copy()
    controls["out"] = str(args.out)
    (args.out / "manifest.json").write_text(
        json.dumps({"controls": controls, **source_manifest()}, indent=2) + "\n", encoding="utf-8"
    )
    cell_ids = (
        [index for index, cell in enumerate(CELLS) if cell[0] in args.cells]
        if args.cells
        else list(range(2) if args.reset_variance_control else range(len(CELLS)))
    )
    if args.reset_variance_control and any(index >= 2 for index in cell_ids):
        parser.error("variance-reset control applies only to independent-variance cells")
    jobs = [
        dict(controls, cell_id=cell_id, replicate=replicate)
        for cell_id in cell_ids
        for replicate in range(args.replications)
    ]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with (args.out / "replications.jsonl").open("w", encoding="utf-8") as stream:
            for number, result in enumerate(executor.map(safe_replication, jobs), start=1):
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                rows.extend(result.get("rows", []))
                if result["status"] != "ok":
                    failures.append(result)
                if number % 10 == 0 or number == len(jobs):
                    print(f"completed={number}/{len(jobs)} failures={len(failures)}", flush=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(args.out / "parameters.csv", index=False)
        summarise_rows(frame).to_csv(args.out / "summary.csv", index=False)
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} numerical failures; inspect failures.json")


if __name__ == "__main__":
    main()
