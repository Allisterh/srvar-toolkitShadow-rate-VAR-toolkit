# Changelog

All notable changes to the Python SRVAR toolkit will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- Fixed-DGP empirical-Bayes DL coverage harness with failure-inclusive bounds,
  per-dataset prior rates and diagnostics; this is not SBC.
- An isolated RW-SV mixture-model posterior oracle with quadrature checks and
  dispersed chains, plus a study-only interweaving prototype. Full-chain diagnostics now retain h0, resolved priors,
  fitting time and exact source snapshots. These studies do not certify empirical use.

- Constructor provenance on `PriorSpec` and complete resolved-prior metadata in
  fit artifacts, including explicit/default DL inputs and Minnesota precisions.
- Standalone component calibration, revision-specific local benchmark and empirical
  multi-chain diagnostic scripts, with source manifests and retained failure evidence.

### Fixed

- Retain residual variance state between coefficient updates in homoskedastic DL and
  canonical Minnesota samplers, including ELB and steady-state paths.
- Include every coupled likelihood term in triangular SV coefficient Gibbs updates.
- Advance volatility states before each forecast observation, including the first horizon.
- Pair ELB terminal histories with retained parameter draws, including after stationarity
  filtering, and preserve observed/latent draw pairing when pooling forecasts.
- Reject constrained Student-t and outlier-mixture scenarios instead of using Gaussian shocks.
- Centre scalar Newey-West inputs and normalise all lagged products by the retained sample size.
- Restrict the ArviZ extra to `>=0.17,<1` for the supported `InferenceData` interface.
- Correct triangular SV descriptions: a fixed triangular factor generally permits changing
  correlations.

### Changed

- DL Python construction requires `residual_prior="empirical_bayes"` with training
  data and lag order, or `"explicit"` with both IG shape and diagonal rates.
  YAML defaults to training-window IG(2, AR residual variance). Earlier
  default-configured DL runs used IG(N+2, 1); deliberately reproduce those values
  through explicit mode or regenerate under the new prior in fresh directories.
- Canonical Minnesota now supplies equation-specific coefficient precisions for
  triangular RW/AR(1) SV, including ELB. Legacy Minnesota is rejected on this path.
  Shared custom Gaussian priors remain supported; factor-SV and tempered-prior
  boundaries are retained. Changed prior targets require separate scientific review.

- Fit artifact writers now emit version 2; forecasts remain version 1.
  `load_run_dir` restores the saved prior instead of re-estimating it from config.
  Older files without a saved prior remain readable through `load_fit_npz`, but
  full run reconstruction now rejects them. Regenerate with a verified prior to
  migrate. This persistence change does not alter sampler transitions or defaults.
- Document the dimension-dependent DL residual-variance default and the missing
  dependent-variable scaling of legacy Minnesota coefficient covariances in triangular
  SV. Prior repairs remain separate from the approved component transitions; this
  documentation change leaves all numerical source and qualification harnesses unchanged.
- CI exercises xarray, ArviZ and Numba; release builds require tests at the exact selected tag.
- Numerical results and seeded random-number sequences change. Refit affected models and
  regenerate forecasts and comparisons. Censored terminal ELB lags now require aligned
  `latent_draws`; refit if these were not retained. Public result schemas are unchanged.
- Added analytical regression checks and documented sampler semantics. These checks do not
  replace human scientific review, convergence assessment, full calibration or replication.

## [0.3.1] - 2026-07-18

### Added

- Controlled GitHub release workflow for tag-validated build artifacts and manual PyPI
  publication.
- Local release preflight checks for tag format, metadata synchronisation, and changelog release
  headings.
- Trusted-publishing documentation covering PyPI project setup, GitHub environment protection, and
  release verification.

### Security

- Fit and forecast artifact writers now emit a schema-marked format without object arrays.
  Artifact readers reject pre-migration pickle-backed output by default; reopening a trusted old
  artifact requires the explicit `allow_legacy_pickle=True` API option or
  `--allow-legacy-pickle` comparison-script flag.

## [0.3.0] - 2026-04-06

### Added

- Explicit legacy Minnesota-style NIW prior path via `PriorSpec.niw_minnesota_legacy(...)` and
  `prior.method: "minnesota_legacy"` in YAML configs. `PriorSpec.niw_minnesota(...)` and
  `prior.method: "minnesota"` remain as backward-compatible aliases.
- Explicit canonical Minnesota prior path via `PriorSpec.niw_minnesota_canonical(...)` and
  `prior.method: "minnesota_canonical"` for homoskedastic models and diagonal stochastic
  volatility.
- Explicit experimental tempered Minnesota bridge via `PriorSpec.niw_minnesota_tempered(...)`
  and `prior.method: "minnesota_tempered"` for diagonal stochastic-volatility models.
- Reproducible Minnesota backtest comparison harness via
  `scripts/compare_minnesota_backtests.py`, which runs paired legacy/canonical backtests and
  writes a combined `metrics_comparison.csv`.
- Consolidated Minnesota benchmark summary script via
  `scripts/summarize_minnesota_comparisons.py`, which scans paired comparison bundles and writes
  repo-level `summary.csv` and `summary.md` tables.
- Variable-level Minnesota comparison summary script via
  `scripts/summarize_metrics_comparison_by_variable.py`, which aggregates one
  `metrics_comparison.csv` file across horizons and writes `variable_summary.csv` and
  `variable_summary.md`.
- Forecast-dispersion comparison script via `scripts/compare_forecast_dispersion.py`, plus a
  `--save-forecasts` option on `scripts/compare_minnesota_backtests.py` for diagnostic reruns
  that need per-origin predictive draw artifacts.
- Forecast-mean-vs-realized comparison script via `scripts/compare_forecast_means_to_realized.py`
  for origin-by-origin diagnostic summaries from saved forecast bundles.
- `scripts/compare_forecast_means_to_realized.py` now supports `--cases` and optional detail
  outputs for narrow origin-level deep dives on selected variable/horizon pairs.
- Single-origin Minnesota fit diagnostic via `scripts/diagnose_minnesota_origin.py`, which
  reproduces one scheduled backtest origin as paired baseline/candidate fits and writes fit
  artifacts plus state, forecast, and coefficient comparison tables.
- Posterior coefficient-draw comparison script via `scripts/compare_fit_coefficients.py` for
  selected `VARIABLE:REGRESSOR` cases from paired fit runs.
- Prior-scale diagnostic via `scripts/diagnose_minnesota_prior_scales.py` for comparing legacy
  and canonical Minnesota coefficient variances at one scheduled backtest origin.
- Tempered-origin experiment via `scripts/experiment_tempered_minnesota_origin.py`, which runs a
  three-way legacy/canonical/tempered Minnesota comparison for one scheduled origin.
- Local quarterly benchmark prep/config via `scripts/prepare_term_nfci_benchmark.py` and
  `config/term_nfci_backtest.yaml` for a second fully local Minnesota comparison run.
- Homoskedastic companion benchmark config via `config/term_nfci_backtest_homoskedastic.yaml`
  to compare canonical vs legacy Minnesota on the same local panel without stochastic volatility.
- Richer three-variable local benchmark prep/config via
  `scripts/prepare_term_nfci_wuxia_benchmark.py` and `config/term_nfci_wuxia_backtest.yaml`.
- Local transformed quarterly 15-variable macro benchmark prep/config via
  `scripts/prepare_vintage_macro15_benchmark.py` and
  `config/vintage_macro15_backtest_homoskedastic.yaml`.
- Diagonal-SV companion config for the local transformed 15-variable vintage benchmark via
  `config/vintage_macro15_backtest_diagonal_sv.yaml`.

### Changed

- Source docs and example configs now label the shipped Minnesota-style NIW construction as a
  legacy, non-canonical compatibility path, and document the support boundary for the explicit
  canonical path.

### Fixed

- Backtest metrics and plotting diagnostics now exclude missing realized values from evaluation
  denominators instead of treating them as misses.
- `config/backtest_demo_config.yaml` now parses as valid YAML again; the sample had a top-level
  indentation error before the `output` block.

## [0.2.0] - 2026-01-19

### Added

- `srvar fetch-fred` command to fetch FRED series to a cached CSV (config-driven).
- `--dry-run` flag for `fetch-fred` (prints planned fetch/output without network calls).
- `--validate-series` flag for `fetch-fred` (preflight series existence check via FRED metadata).
- Transformation support in the fetch pipeline, including `processing.transform_order`.
- Runtime warnings for non-positive values when applying log-based tcodes (4/5/6).
- Unit tests covering `tcode_1d` and the `fetch_fred` helpers (mocked, no network).
- Steady-state VAR parameterization (SSP) with Gibbs sampling of the steady-state mean `mu`.
- Optional spike-and-slab selection on `mu` (mu-SSVS).
- YAML-only configuration support for SSP via `model.steady_state`.
- SSP example script (`examples/ssp_fit_forecast.py`).
- SSP test coverage (`tests/test_ssp.py`).
- Robust shock models for homoskedastic VARs and factor SV via `model.shocks` (Student‑t and outlier-mixture innovations).
- Full-covariance stochastic volatility via factor SV (`model.volatility.covariance: "factor"`, `k_factors`) with RW dynamics (v1: `prior.family: "niw"`), including ELB shadow-rate data augmentation and steady-state support.
- Structural analysis stack (`srvar.analysis`, `srvar.identification`) including Cholesky IRFs, sign-restricted IRFs, FEVD, and historical decomposition (supports factor SV covariance draws).
- Conditional / scenario forecasting utilities (`srvar.scenario`).
- Optional `xarray` conversion utilities for labeled outputs (`srvar.xarray`).
- Optional ArviZ conversion utilities for `InferenceData` outputs (`srvar.arviz`).
- Labeled output hardening for factor SV loadings (`FitResult.loading_draws` and `ds_fit["loadings"]` alias of `ds_fit["lambda"]`).
- Expository notebooks in `examples/notebooks/` (quickstart, ELB, FSV, structural analysis, and backtesting/evaluation conventions).
- Factor SV demo config (`config/fsv_demo_config.yaml`) and example script (`examples/fsv_fit_forecast.py`).
- `srvar.artifacts.load_run_dir(out_dir)` to reconstruct a `FitResult` from `config.yml` + `fit_result.npz` (including factor SV draws and optional latent dataset / NIW posterior blocks).
- ELB-censored backtest evaluation via `evaluation.elb_censor` (censor realized values and optionally forecast draws).
- Streaming backtest evaluation for `metrics.csv` via `output.store_forecasts_in_memory` (reduces RAM for long runs).
- Additional scoring rules and comparison utilities:
  - weighted interval score (WIS), pinball (quantile) loss, and Gaussian log score / LPD approximation (`srvar.metrics`, `srvar.evaluation`)
  - Diebold–Mariano test and Giacomini–White CPA test (`srvar.stats`)
- Simple forecast pooling / ensembles (`srvar.ensemble`).
- Synthetic memory benchmark script (`scripts/benchmark_backtest_memory.py`).

### Changed

- Refactored `srvar.samplers` into smaller modules and re-exported the public API.
- `forecast()` now requires stored `beta_draws` when `steady_state` is enabled.
- Split config/backtest/evaluation/artifacts logic into `srvar.config`, `srvar.backtest`, `srvar.evaluation`, `srvar.artifacts` (keeping `srvar.runner` as a thin façade).
- Backtest evaluation flags now control both computation and outputs (`evaluation.coverage.enabled`, `evaluation.crps.enabled`, `evaluation.pit.enabled`).

### Fixed

- NIW posterior sampling now correctly handles the univariate case (`N=1`) when drawing from the inverse-Wishart distribution.
- `srvar.xarray.fit_to_xarray` now handles SV/FSV time-varying draws defined on the effective sample (`T - p`).

## [0.1.0] - 2025-12-22

### Added

- Conjugate NIW Bayesian VAR (BVAR) estimation.
- ELB / shadow-rate data augmentation.
- Diagonal stochastic volatility (SVRW) via KSC mixture + precision-based state sampling.
- Combined SV + ELB model.
- Forecasting API.
- Example scripts in `examples/`.
