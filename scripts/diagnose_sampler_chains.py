"""Run reproducible multi-chain diagnostics at one local empirical forecast origin."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.qualify_samplers import chain_diagnostics
from scripts.study_evidence import archive_sources
from srvar._prior_io import prior_to_json
from srvar.api import fit, forecast
from srvar.config import build_model, build_prior, load_config, load_dataset_from_csv
from srvar.data.dataset import Dataset
from srvar.results import FitResult
from srvar.spec import SamplerConfig
from srvar.sv import VolatilitySpec


def monitored_draws(result: FitResult) -> tuple[np.ndarray, list[str]]:
    """Monitor coefficients and covariance parameters, plus terminal SV states."""
    parts, labels = [], []
    for name in ("beta_draws", "sigma_draws", "q_draws", "sigma_eta2_draws", "h_draws", "h0_draws"):
        value = getattr(result, name, None)
        if value is None:
            continue
        if name == "h_draws":
            value = value[:, -1, :]
        if name == "sigma_draws":
            if result.prior.minnesota_canonical is not None or result.prior.family == "dl":
                indices = [(i, i) for i in range(value.shape[-1])]
            else:
                indices = list(zip(*np.tril_indices(value.shape[-1]), strict=True))
        elif name == "q_draws":
            indices = list(zip(*np.triu_indices(value.shape[-1], 1), strict=True))
        else:
            indices = list(np.ndindex(value.shape[1:]))
        for index in indices:
            parts.append(value[(slice(None), *index)])
            labels.append(f"{name}:{','.join(str(i) for i in index)}")
    return np.stack(parts, axis=-1), labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", default="2010-01-01")
    parser.add_argument("--triangular", action="store_true")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()
    if args.chains < 2 or args.draws < 20 or args.warmup < 0 or args.seed < 0:
        parser.error("require >=2 chains, >=20 draws, non-negative warmup and seed")
    cfg = load_config(args.config)
    full = load_dataset_from_csv(cfg)
    mask = pd.DatetimeIndex(full.time_index) <= pd.Timestamp(args.end)
    dataset = Dataset.from_arrays(
        values=full.values[mask], time_index=full.time_index[mask], variables=full.variables
    )
    model = build_model(cfg, dataset=dataset)
    if args.triangular:
        model = replace(model, volatility=VolatilitySpec(covariance="triangular", dynamics="rw"))
    prior = build_prior(cfg, dataset=dataset, model=model)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "controls": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": cfg,
        "dataset_T": dataset.T,
        "dataset_N": dataset.N,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "training_values_sha256": hashlib.sha256(dataset.values.tobytes()).hexdigest(),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolved_prior": json.loads(prior_to_json(prior)),
        "initialisation": "common deterministic full-fit initial state; independent RNG streams",
        **archive_sources(args.out, [Path(__file__)]),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    chains = []
    forecasts = []
    seeds = np.random.SeedSequence(args.seed).spawn(args.chains * 2)
    elapsed_seconds = []
    for chain in range(args.chains):
        start = time.perf_counter()
        result = fit(
            dataset,
            model,
            prior,
            SamplerConfig(draws=args.warmup + args.draws, burn_in=args.warmup, thin=1),
            rng=np.random.default_rng(seeds[2 * chain]),
        )
        elapsed_seconds.append(time.perf_counter() - start)
        values, labels = monitored_draws(result)
        chains.append(values)
        predictive = forecast(
            result, [1, 2, 4], draws=1000, rng=np.random.default_rng(seeds[2 * chain + 1])
        )
        forecasts.append(predictive.draws)
        np.savez_compressed(
            args.out / f"chain_{chain}.npz",
            monitored=values,
            labels=np.asarray(labels),
            forecast=predictive.draws,
        )
        print(
            f"completed_chain={chain + 1}/{args.chains} monitored_parameters={len(labels)}",
            flush=True,
        )
    samples = np.stack(chains)
    diagnostic = chain_diagnostics(samples)
    frame = pd.DataFrame(
        {
            "parameter": labels,
            "mean": samples.mean(axis=(0, 1)),
            "sd": samples.reshape(-1, samples.shape[-1]).std(axis=0),
            **diagnostic,
        }
    )
    frame["ess_bulk_per_fit_second"] = frame.ess_bulk / sum(elapsed_seconds)
    frame["flag"] = (
        (frame.rhat > 1.01)
        | (frame.ess_bulk < 400)
        | (frame.ess_tail < 400)
        | ~np.isfinite(frame.rhat)
        | ~np.isfinite(frame.ess_bulk)
        | ~np.isfinite(frame.ess_tail)
    )
    frame.to_csv(args.out / "diagnostics.csv", index=False)
    np.savez_compressed(
        args.out / "forecast_comparison.npz",
        mean=np.stack(forecasts).mean(axis=1),
        sd=np.stack(forecasts).std(axis=1),
        variables=np.asarray(dataset.variables),
    )
    (args.out / "timing.json").write_text(
        json.dumps({"fit_seconds_per_chain": elapsed_seconds}, indent=2) + "\n"
    )
    print(f"flagged_parameters={int(frame.flag.sum())}/{len(frame)}", flush=True)


if __name__ == "__main__":
    main()
