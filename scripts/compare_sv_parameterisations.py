"""Experimental RW-SV interweaving, isolated from production fits and empirical claims."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import diagnose_sv_state_block as oracle
from scripts.qualify_samplers import chain_diagnostics
from scripts.study_evidence import archive_sources
from srvar import sv


def scale_log_density(
    log_scale: float,
    z: np.ndarray,
    centred_y: np.ndarray,
    omega: np.ndarray,
    h0: float,
    a: float,
    b: float,
) -> float:
    """Conditional density in log sqrt(variance), including the IG Jacobian."""
    with np.errstate(over="ignore", invalid="ignore"):
        residual = centred_y - h0 - np.exp(log_scale) * z
        value = -2 * a * log_scale - b * np.exp(-2 * log_scale) - 0.5 * np.sum(residual**2 / omega)
    return float(value) if np.isfinite(value) else -np.inf


def interweave(
    y: np.ndarray, h: np.ndarray, h0: float, variance: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, float, bool]:
    """Apply a valid non-centred MH-within-Gibbs transition for the declared study prior."""
    labels = sv.sample_mixture_indicators(y_star=y, h=h, rng=rng)
    centred_y = y - sv._KSC_MI[labels]
    omega = sv._KSC_SIGI[labels]
    scale = np.sqrt(variance)
    z = (h - h0) / scale
    precision = 0.5 + np.sum(1 / omega)  # h0 prior N(0,2)
    h0 = float(
        np.sum((centred_y - scale * z) / omega) / precision + rng.normal() / np.sqrt(precision)
    )
    current = float(np.log(scale))
    proposal = current + 0.15 * rng.normal()
    ratio = scale_log_density(proposal, z, centred_y, omega, h0, 2.0, 0.1) - scale_log_density(
        current, z, centred_y, omega, h0, 2.0, 0.1
    )
    accepted = bool(np.log(rng.random()) < ratio)
    if accepted:
        scale = float(np.exp(proposal))
    return h0 + scale * z, h0, scale * scale, accepted


def run_chains(
    y: np.ndarray, *, draws: int, warmup: int, seed: int, use_interweaving: bool
) -> tuple[np.ndarray, float, float]:
    """Monitor h0, first/terminal h and innovation variance; retain dispersed starts."""
    start = time.perf_counter()
    samples = np.empty((4, draws, 4))
    accepted = 0
    for chain, stream in enumerate(np.random.SeedSequence(seed).spawn(4)):
        rng = np.random.default_rng(stream)
        h = np.full(len(y), [-3.0, -1.0, 1.0, 3.0][chain])
        h0 = float(h[0])
        variance = [0.001, 0.02, 0.5, 4.0][chain]
        for iteration in range(warmup + draws):
            h = sv.sample_h_svrw(y_star=y, h=h, sigma_eta2=variance, h0=h0, rng=rng)
            h0 = sv.sample_h0(h1=h[0], sigma_eta2=variance, prior_mean=0, prior_var=2, rng=rng)
            variance = sv.sample_sigma_eta2(h=h, h0=h0, nu0=2, s0=0.1, rng=rng)
            if use_interweaving:
                h, h0, variance, accept = interweave(y, h, h0, variance, rng)
                accepted += int(accept)
            if iteration >= warmup:
                samples[chain, iteration - warmup] = [h0, h[0], h[-1], variance]
    return samples, time.perf_counter() - start, accepted / (4 * (warmup + draws))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if args.draws < 20 or min(args.warmup, args.seed) < 0:
        parser.error("require >=20 draws and nonnegative warmup/seed")
    args.out.mkdir(parents=True, exist_ok=False)
    controls = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "manifest.json").write_text(
        json.dumps(
            dict(
                controls=controls,
                target="KSC approximate observation model; h0 N(0,2); variance IG(2,.1)",
                dgp_seed=161803,
                mh_log_scale_sd=0.15,
                **archive_sources(args.out, [Path(__file__), Path(oracle.__file__)]),
            ),
            indent=2,
        )
        + "\n"
    )
    rng = np.random.default_rng(161803)
    h = -0.5 + np.cumsum(rng.normal(size=200) * np.sqrt(0.005))
    labels = rng.choice(7, size=200, p=sv._KSC_PI / sv._KSC_PI.sum())
    long_y = h + sv._KSC_MI[labels] + rng.normal(size=200) * np.sqrt(sv._KSC_SIGI[labels])
    reference = oracle.tiny_reference()
    records = []
    for case, y in [("tiny", np.array([-0.7, 0.9])), ("long", long_y)]:
        for kernel in ["centred", "interwoven"]:
            samples, elapsed, acceptance = run_chains(
                y,
                draws=args.draws,
                warmup=args.warmup,
                seed=args.seed,
                use_interweaving=kernel == "interwoven",
            )
            diagnostic = chain_diagnostics(samples)
            np.savez_compressed(args.out / f"{case}-{kernel}.npz", samples=samples, y_star=y)
            for i, label in enumerate(["h0", "h_first", "h_terminal", "sigma_eta2"]):
                records.append(
                    dict(
                        case=case,
                        kernel=kernel,
                        parameter=label,
                        mean=float(samples[:, :, i].mean()),
                        seconds=elapsed,
                        acceptance=acceptance,
                        reference_mean=float(reference["mean"][i]) if case == "tiny" else None,
                        **{k: float(v[i]) for k, v in diagnostic.items()},
                    )
                )
            print(f"completed={case}-{kernel} seconds={elapsed:.2f}", flush=True)
    frame = pd.DataFrame(records)
    frame["ess_per_sweep"] = frame.ess_bulk / (4 * args.draws)
    frame["ess_per_second"] = frame.ess_bulk / frame.seconds
    frame["flag"] = (
        (frame.rhat > 1.01)
        | (frame.ess_bulk < 400)
        | (frame.ess_tail < 400)
        | ~np.isfinite(frame.rhat)
    )
    frame.to_csv(args.out / "diagnostics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
