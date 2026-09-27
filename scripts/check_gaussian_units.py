"""Check Gaussian VAR posterior draws under consistent changes of measurement units."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import compare_dl_controls as controls
from scripts import qualify_dl_empirical_bayes as baseline
from scripts.study_evidence import archive_sources
from srvar._prior_io import prior_to_json
from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig

UNITS = {
    "base": (1.0, 1.0),
    "small": (1e-7, 1e-7),
    "large": (1e7, 1e7),
    "mixed": (1e-7, 1e7),
}
TOLERANCE = 1e-8


def draw_factors(scales: np.ndarray) -> np.ndarray:
    """Return forward factors for column-major beta and diagonal variances."""
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("unit scales must be two finite positive values")
    factors = scales[None, :] / np.r_[1.0, scales][:, None]
    return np.r_[factors.reshape(-1, order="F"), scales**2]


def transform_prior(prior: PriorSpec, scales: np.ndarray) -> PriorSpec:
    """Transform the actual independent Gaussian columns and IG rates together."""
    factors = draw_factors(scales)
    if prior.family != "niw" or prior.minnesota_canonical is None:
        raise ValueError("unit study requires an equation-wise independent Gaussian prior")
    return replace(
        prior,
        niw=replace(
            prior.niw,
            m0=prior.niw.m0 * factors[:6].reshape(3, 2, order="F"),
            s0=np.diag(np.diag(prior.niw.s0) * factors[-2:]),
        ),
        minnesota_canonical=replace(
            prior.minnesota_canonical,
            sigma2=prior.minnesota_canonical.sigma2 * factors[-2:],
            inv_v0_vec=prior.minnesota_canonical.inv_v0_vec / factors[:6] ** 2,
        ),
    )


def run_dataset(job: dict) -> list[dict]:
    """Retain raw scaled draws and compare matched seeds after conversion to base units."""
    out = Path(job["out"])
    streams = np.random.SeedSequence([job["seed"], 0, job["replicate"]]).spawn(job["chains"] + 1)
    y, truth = baseline.generate_data(0, np.random.default_rng(streams[0]))
    base_prior = controls.make_prior(y, truth, "gaussian_unfloored")
    np.savez_compressed(out / f"data-{job['replicate']:04d}.npz", y=y, truth=truth)
    reference = None
    records = []
    for name, units in UNITS.items():
        record = dict(replicate=job["replicate"], units=name, scales=units)
        samples = []
        try:
            scales = np.asarray(units)
            prior = transform_prior(base_prior, scales)
            record["prior"] = json.loads(prior_to_json(prior))
            raw_rates = controls.estimate_unfloored_rates(y * scales)
            record["estimated_rates_back"] = (raw_rates / scales**2).tolist()
            record["floored_rates_back"] = (np.maximum(raw_rates, 1e-12) / scales**2).tolist()
            dataset = Dataset.from_arrays(values=y * scales, variables=["a", "b"])
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
            samples_array = np.stack(samples)
            relative = f"chains-{job['replicate']:04d}-{name}.npz"
            np.savez_compressed(out / relative, samples=samples_array)
            record["chain_file"] = relative
            back = samples_array / draw_factors(scales)
            record["rows"] = controls.summarise_samples(back, truth)
            if name == "base":
                reference = back
            if reference is None:
                raise ValueError("base-unit fit failed; paired comparison is unavailable")
            errors = np.max(
                np.abs(back - reference) / np.maximum(1.0, np.abs(reference)), axis=(0, 1)
            )
            record["normalised_max_error"] = errors.tolist()
            record["status"] = (
                "ok" if np.isfinite(errors).all() and np.all(errors <= TOLERANCE) else "mismatch"
            )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if samples:
                relative = f"partial-{job['replicate']:04d}-{name}.npz"
                np.savez_compressed(out / relative, samples=np.stack(samples))
                record["partial_chain_file"] = relative
        record["completed_chains"] = len(samples)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if (
        args.replications < 1
        or args.chains < 2
        or args.draws < 20
        or min(args.seed, args.warmup) < 0
    ):
        parser.error(
            "require positive replications, >=2 chains, >=20 draws and nonnegative seed/warmup"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    options = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest = dict(
        controls=options,
        units=UNITS,
        tolerance=TOLERANCE,
        metric="max over draws of abs(back-base)/max(1,abs(base)), per parameter",
        estimand="paired Gaussian posterior equivariance in base units; not coverage",
        **archive_sources(
            args.out, [Path(__file__), Path(controls.__file__), Path(baseline.__file__)]
        ),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.out / "replications.jsonl").open("w") as stream:
        for replicate in range(args.replications):
            batch = run_dataset(dict(options, replicate=replicate))
            records.extend(batch)
            for record in batch:
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                f"datasets={replicate + 1}/{args.replications} unsuccessful={sum(r['status'] != 'ok' for r in records)}",
                flush=True,
            )
    pd.DataFrame(
        [
            dict(
                replicate=r["replicate"],
                units=r["units"],
                status=r["status"],
                max_error=max(r.get("normalised_max_error", [np.nan])),
                diagnostic_flags=sum(row["flag"] for row in r.get("rows", [])),
            )
            for r in records
        ]
    ).to_csv(args.out / "summary.csv", index=False)
    failures = [r for r in records if r["status"] != "ok"]
    (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed/mismatched fits retained")


if __name__ == "__main__":
    main()
