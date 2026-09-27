"""Compare the RW-SV state block with an enumerated T=2 mixture-model posterior."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.qualify_samplers import chain_diagnostics
from scripts.study_evidence import archive_sources
from srvar import sv


def tiny_reference(points: int = 601) -> dict:
    """Integrate h0/h analytically, mixture labels exactly and log variance by quadrature.

    Target: h0~N(0,2), s~IG(2,.1), ht|hprev,s~N(hprev,s),
    y*|h,z~N(h+m_z,v_z), with y*=(-.7,.9) and the production mixture constants.
    This oracle tests the approximate observation model, not the original SV likelihood.
    """
    grid = np.linspace(-12, 6, points)
    y = np.array([-0.7, 0.9])
    times = np.arange(3)
    logs = []
    means = []
    seconds = []
    grid_indices = []
    for index, log_s in enumerate(grid):
        s = np.exp(log_s)
        prior_cov = 2 * np.ones((3, 3)) + s * np.minimum.outer(times, times)
        for labels in itertools.product(range(7), repeat=2):
            labels = np.asarray(labels)
            obs_cov = prior_cov[1:, 1:] + np.diag(sv._KSC_SIGI[labels])
            resid = y - sv._KSC_MI[labels]
            solved = np.linalg.solve(obs_cov, resid)
            log_lik = -0.5 * (
                np.linalg.slogdet(obs_cov)[1] + resid @ solved + 2 * np.log(2 * np.pi)
            )
            # IG(s)*ds/dlog(s), normalisation cancels across grid/allocation weights.
            log_prior = -2 * log_s - 0.1 / s
            log_weight = log_lik + log_prior + np.log(sv._KSC_PI[labels]).sum()
            if index in [0, points - 1]:
                log_weight += np.log(0.5)
            mean = prior_cov[:, 1:] @ solved
            covariance = prior_cov - prior_cov[:, 1:] @ np.linalg.solve(obs_cov, prior_cov[1:, :])
            logs.append(log_weight)
            means.append(np.r_[mean, s])
            seconds.append(np.r_[np.diag(covariance) + mean**2, s**2])
            grid_indices.append(index)
    weights = np.exp(np.array(logs) - logsumexp(logs))
    mean = weights @ np.array(means)
    sd = np.sqrt(weights @ np.array(seconds) - mean**2)
    grid_mass = np.bincount(grid_indices, weights=weights, minlength=points)
    return dict(
        mean=mean,
        sd=sd,
        log_variance=grid,
        grid_mass=grid_mass,
        boundary_mass=float(grid_mass[:2].sum() + grid_mass[-2:].sum()),
    )


def sample_block(*, draws: int, warmup: int, seed: int) -> np.ndarray:
    """Use dispersed starts and the unchanged production h, h0 and variance updates."""
    samples = np.empty((4, draws, 4))
    for chain, stream in enumerate(np.random.SeedSequence(seed).spawn(4)):
        rng = np.random.default_rng(stream)
        h = np.full(2, [-3.0, -1.0, 1.0, 3.0][chain])
        h0 = float(h[0])
        variance = [0.001, 0.02, 0.5, 4.0][chain]
        for iteration in range(warmup + draws):
            h = sv.sample_h_svrw(
                y_star=np.array([-0.7, 0.9]), h=h, sigma_eta2=variance, h0=h0, rng=rng
            )
            h0 = sv.sample_h0(h1=h[0], sigma_eta2=variance, prior_mean=0, prior_var=2, rng=rng)
            variance = sv.sample_sigma_eta2(h=h, h0=h0, nu0=2, s0=0.1, rng=rng)
            if iteration >= warmup:
                samples[chain, iteration - warmup] = np.r_[h0, h, variance]
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.draws < 20 or min(args.warmup, args.seed) < 0:
        parser.error("require >=20 draws and nonnegative warmup/seed")
    args.out.mkdir(parents=True, exist_ok=False)
    controls = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "manifest.json").write_text(
        json.dumps(
            dict(
                controls=controls,
                target="T=2 KSC mixture observation model, h0 N(0,2), innovation variance IG(2,.1)",
                **archive_sources(args.out, [Path(__file__)]),
            ),
            indent=2,
        )
        + "\n"
    )
    coarse = tiny_reference(301)
    reference = tiny_reference(601)
    start = time.perf_counter()
    samples = sample_block(draws=args.draws, warmup=args.warmup, seed=args.seed)
    elapsed = time.perf_counter() - start
    diagnostics = chain_diagnostics(samples)
    labels = ["h0", "h1", "h2", "sigma_eta2"]
    frame = pd.DataFrame(
        dict(
            parameter=labels,
            mean=samples.mean(axis=(0, 1)),
            reference_mean=reference["mean"],
            reference_sd=reference["sd"],
            **diagnostics,
        )
    )
    frame["ess_bulk_per_second"] = frame.ess_bulk / elapsed
    frame["mean_error_mcse"] = abs(frame["mean"] - frame.reference_mean) / frame.mcse_mean
    frame["flag"] = (
        (frame.rhat > 1.01)
        | (frame.ess_bulk < 400)
        | (frame.ess_tail < 400)
        | ~np.isfinite(frame.rhat)
    )
    frame.to_csv(args.out / "diagnostics.csv", index=False)
    np.savez_compressed(args.out / "chains.npz", samples=samples, labels=np.asarray(labels))
    np.savez_compressed(args.out / "reference.npz", **reference)
    report = dict(
        seconds=elapsed,
        max_grid_mean_change=float(abs(reference["mean"] - coarse["mean"]).max()),
        boundary_mass=reference["boundary_mass"],
        diagnostic_flags=int(frame.flag.sum()),
        max_mean_error_mcse=float(frame.mean_error_mcse.max()),
    )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
