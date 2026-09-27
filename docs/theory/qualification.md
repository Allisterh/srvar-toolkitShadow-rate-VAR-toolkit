# Statistical qualification studies

Use the standalone studies to assess the repaired sampler transitions and selected empirical fits. Unit tests, component calibration, full-fit convergence and paper replication answer different questions. Human scientific review remains required before substantive use of changed samplers.

## Component calibration

Install the existing diagnostic extras and run from the repository root:

```bash
python -m pip install -e '.[dev,arviz,cli,excel]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.qualify_samplers \
  --out .planning/private/sbc-study --replications 100 --workers 2
```

The output directory must be new. Each run records numerical source hashes, dependency versions, controls, every replication and any numerical failures. `parameters.csv` contains ranks, 90% coverage, posterior errors and chain diagnostics. `summary.csv` reports Monte Carlo uncertainty across independently generated datasets. Failed replications remain in `replications.jsonl` and `failures.json`; they must not be silently excluded from interpretation.

The four fixed-design cells exercise independent normal/inverse-gamma coefficient and variance transitions and triangular coefficient transitions conditional on known covariance states. They use non-zero prior means, sample sizes of 20 and 80, and two scale/coupling settings. Parameters are generated from the same fixed priors used for inference. This avoids treating data-dependent Minnesota prior estimation as ordinary prior-predictive SBC.

These component studies do not validate the complete DL shrinkage hierarchy, the SV state sampler, ELB augmentation or every public model combination. In particular, a well-calibrated coefficient block does not establish that the full SV chain mixes adequately.

The default is four chains, 250 warm-up sweeps and 500 retained sweeps per chain. Ranks use every tenth draw; unthinned draws supply R-hat, effective sample size and Monte Carlo standard errors. Inspect dependence before interpreting rank histograms. Longer runs for selected cells can be requested explicitly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.qualify_samplers \
  --out .planning/private/sbc-triangular-long --cells triangular_80_q16 \
  --replications 100 --draws 2000 --warmup 500 --rank-thin 20 --workers 2
```

`--reset-variance-control` deliberately restores the old variance reset in the independent-variance cells. This negative control helps assess sensitivity: good chain diagnostics can coexist with sampling from the wrong target. It has no effect on production model code.

## Local empirical reruns

The shipped quarterly benchmark can run from local workbook vintages without fetching data:

```bash
python -m scripts.prepare_vintage_macro15_benchmark \
  --out .planning/private/vintage_macro15.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.qualify_local_benchmark \
  --data .planning/private/vintage_macro15.csv \
  --out .planning/private/local-benchmark --source-label working-tree \
  --expected-package-root srvar
```

This reruns the shipped legacy homoskedastic and diagonal-SV settings, plus an explicitly selected canonical homoskedastic variant. It preserves the original origin window, lag order, horizons and short-chain settings. Results are useful for detecting changed outputs; those settings do not establish convergence or forecast superiority. The prepared 2022Q3 vintage supplies a revised historical panel, so this is not a real-time forecast experiment.

To compare revisions, load each source tree explicitly and use the same data hash. Run outside both source directories because Python prepends the current directory to its import search path. `--expected-package-root` checks the actual imported package and rejects a mismatch before computing results. The manifest records the imported source hashes, not merely a version label.

## Multi-chain empirical diagnostics

Use a generated benchmark configuration to diagnose a fixed training origin:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.diagnose_sampler_chains \
  --config .planning/private/local-benchmark/canonical_homoskedastic.yaml \
  --out .planning/private/canonical-chains --end 2010-01-01
```

For triangular RW SV, pass the generated `legacy_homoskedastic.yaml` configuration and `--triangular`. This is a named extension of the local benchmark, not one of its shipped settings. The script defaults to four independently seeded chains, 1,000 warm-up sweeps and 2,000 retained sweeps. Initial states follow the fit implementation; independent seeds do not guarantee overdispersed initial states.

The diagnostic table covers coefficients, free covariance/factor entries, terminal SV states and volatility-innovation variances. It excludes structural covariance zeros and the fixed diagonal of Q. It does not cover every historical latent state. Monitored draws and forecast paths are retained in non-object NPZ files. R-hat above 1.01, bulk/tail ESS below 400 or undefined diagnostics trigger review flags. Those thresholds identify runs needing investigation; they are not posterior-correctness proofs.

For method details see the [Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html) and [Vehtari et al. on rank-normalised diagnostics](https://arxiv.org/abs/1903.08008). The sampler targets and unresolved shrinkage contracts are described in {doc}`mcmc`.
