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

## Empirical-Bayes DL interval coverage

The plug-in prior requires frequentist coverage checks that repeat scale estimation
inside every simulated training window. This is a fixed-DGP study, not SBC:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/qualify_dl_empirical_bayes.py \
  --out .planning/private/dl-eb-coverage \
  --replications 30 --chains 4 --draws 1000 --warmup 500 --workers 4
```

The four declared scenarios vary sample size, stationary AR persistence, equation
scale, including a tiny-scale cell that activated the historical variance floor. They use two independent AR(1) series,
zero intercept and Gaussian innovations; they do not cover cross-equation dynamics,
ELB, SV or non-zero prior means. Initial observations are drawn from the stationary
distribution. Each dataset supplies one empirical-Bayes prior shared by its chains.

`replications.jsonl` retains prior rates and interval endpoints,
coverage, rate-policy identity and parameter diagnostics. `dataset_counts.csv` records every attempted
cell, including cells with no successful fits. `summary.csv` reports coverage
conditional on numerical success, Wilson intervals and worst/best coverage bounds
when failed datasets are included. Diagnostics never remove a dataset from the
summary. Multiple chains improve posterior estimation; they do not increase the
number of independent coverage replications. Full-fit starts remain common and
deterministic, with independent random streams.

## Paired residual-prior and shrinkage controls

Compare specified procedure choices on the same datasets using a four-arm study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.compare_dl_controls \
  --out .planning/private/dl-paired-controls \
  --cells short floor_stress --replications 30 \
  --chains 4 --draws 1000 --warmup 500 --seed 20260928 --workers 4
```

The historical arms cross floored empirical-Bayes versus known-DGP residual-prior rates with the
production DL hierarchy versus independent Gaussian N(0,1) coefficient priors.
All use IG shape 2 and zero coefficient-prior means. The historical estimated-rate arms explicitly retain the old 1e-12 floor, independently of the current DL constructor. The Gaussian control has
custom provenance and uses the existing equation-wise precision route; it is
not a Minnesota construction. Its fixed variances are in the supplied units,
so the stress cell is not an equivalent rescaling of the ordinary prior.
Oracle rates use information unavailable in empirical applications.

Dataset seeds and chain seeds match the earlier DL coverage harness. Each arm
receives the same data and starts new generators from those chain seeds; distinct
RNG consumption can limit the benefit of shared seeds. Data SHA-256 values refer
to C-order little-endian float64 observation bytes. Input data, truth, full priors
and every retained coefficient/variance chain are saved, together with exact
source snapshots. If an arm fails, its previously completed chains and exception
are retained; draws inside the failing fit call are unavailable. Other arms
continue. A run with failed arms exits non-zero after writing
its summaries. Output directories must be new.

`summary.csv` reports marginal coverage, Wilson intervals, bias, RMSE, widths and
diagnostics. `arm_counts.csv` includes all-failed arms. `paired_summary.csv` reports
oracle-minus-EB differences within each coefficient prior, Gaussian-minus-DL
differences within each residual prior, and their interaction (the oracle effect
under Gaussian minus the oracle effect under DL). Coverage, bias and width
contrasts use complete dataset pairs with sample-based MCSE. Missing outcomes
also receive worst/best coverage-difference bounds over all attempted datasets;
these are missing-result bounds, not confidence intervals. Flagged chains remain
in every summary.

Thirty independent datasets give nominal marginal coverage MCSE about 0.055 at
coverage 0.90. This supports investigation of large differences, not precise
calibration certification. Repeated datasets are controlled comparisons, not
independent replication of earlier coverage estimates. Differences concern these
specified priors and DGPs; they neither prove the DL hierarchy's target nor
identify a universal effect of shrinkage. Separate prior-floor estimation effects
and changes of measurement units before changing production defaults.

### Qualify the current DL rate policy

Use `--policy-only` instead of `--include-unfloored` to fit the current constructor
rates under DL and fixed Gaussian coefficients. The `dl_policy` and
`gaussian_policy` arms use the same normalised AR rate estimator and IG shape 2;
their paired contrast compares coefficient-prior choices. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.compare_dl_controls \
  --out .planning/private/dl-rate-policy --policy-only \
  --cells short floor_stress --replications 30 --chains 4 \
  --draws 1000 --warmup 500 --seed 20260928 --workers 4
```

Rates have no absolute floor. The constructor rejects deficient designs,
non-positive residual degrees of freedom, unresolved residuals and rates that
cannot be represented as finite and positive. See {doc}`mcmc` for the exact rule.
The harness retains failed arms and excludes no draws based on diagnostics.
Compare with archived controls by dataset identity and resolved rate, not merely
by a prior-mode label. Rate and covariance transformations do not qualify the
full DL hyperparameter hierarchy.

### Separate estimation from the residual-prior floor

Add `--include-unfloored` to the paired command to fit six arms. The extra DL
and Gaussian arms use the same AR estimator with its floor set to zero inside
the study, then pass the rates through the explicit prior interface. This does
not change production defaults. Zero or non-finite rates are failed arms;
available raw estimates and exceptions are retained without replacements.

The additional paired contrasts compare unfloored minus floored estimates and
oracle minus unfloored estimates within each coefficient prior. They also report
Gaussian minus DL under unfloored rates and the difference between the two
unfloored-minus-floored effects. This separates floor and estimation changes on
the declared DGPs. Use `--cells floor_stress` to target the active-floor case.

### Equivalent Gaussian models in different units

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.check_gaussian_units \
  --out .planning/private/gaussian-units --replications 3 \
  --chains 4 --draws 1000 --warmup 500 --seed 20260928
```

This study rescales the same two-variable VAR(1) data by positive diagonal $S$.
With $D=\operatorname{diag}(1,s_1,s_2)$, the corresponding design is $X^*=XD$,
coefficients are $B^*=D^{-1}BS$, and residual variances are $v_j^*=s_j^2v_j$.
It transforms Gaussian prior means and covariances as
$m_j^*=s_jD^{-1}m_j$ and $V_j^*=s_j^2D^{-1}V_jD^{-1}$, and multiplies each
IG rate by $s_j^2$ while preserving shape. This also transforms the production
initial residual variances. The shared `v0` container is inert on this path;
the actual transformed coefficient covariances use equation-wise precisions.

The base prior uses independent N(0,1) coefficients and current DL training-window
rates at IG shape 2. Re-estimated unit checks also use the current DL rate policy. Base units are compared with common factors 1e-7 and 1e7
and mixed factors (1e-7,1e7). With the same per-chain seeds, every retained draw
is converted back to base units. The declared tolerance is 1e-8 for
`max(abs(back-base)/max(1,abs(base)))`, separately for each parameter. Arrays,
priors, diagnostics, errors and unsuccessful fits are retained. A mismatch or
exception makes the command exit non-zero after writing evidence.

Current AR rates and a historical absolute-floor control are recomputed from the scaled data and
converted back, exposing the absolute floor's unit dependence separately from
posterior sampling. Passing this finite grid checks numerical equivariance for
the specified Gaussian target; it does not establish coverage or a transformation
rule for the complete DL hierarchy.

## Isolated RW-SV investigation

Check state-block mixing against a tractable posterior before attributing full-fit
failures to a particular transition:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/diagnose_sv_state_block.py \
  --out .planning/private/sv-state-oracle --draws 10000 --warmup 3000
```

This harness fixes two transformed observations, enumerates the 49 KSC mixture
allocations, integrates initial/state values analytically and integrates innovation
variance on a refined log grid. Four chains start at dispersed states and variances.
The output retains chains, reference grid mass, refinement error, boundary mass,
R-hat/ESS, ESS per second and mean errors in estimated MCSE units. It tests the
KSC approximate-observation target; it does not establish approximation accuracy
against the original Gaussian-innovation SV likelihood or diagnose all full-fit
couplings. AR(1) state dynamics are outside this oracle.

The empirical chain harness also archives the resolved prior and exact source,
monitors initial volatility states and records fitting time. Choose canonical
Minnesota explicitly for triangular models. Common full-fit initial states and
finite retained draws remain limits, even if thresholds pass. Preserve earlier
failed studies and use a fresh output directory for every run.

## Experimental state interweaving

`scripts/compare_sv_parameterisations.py` compares the centred RW state sweep with
an experimental interweaving step on fixed synthetic mixture-model observations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/compare_sv_parameterisations.py \
  --out .planning/private/sv-interweaving --draws 10000 --warmup 3000
```

The extra step resamples mixture labels, sets $z=(h-h_0)/s$ with
$s=\sqrt{\sigma_\eta^2}$, draws $h_0$ given $z,s$, and takes one symmetric
Metropolis proposal for $\ell=\log s$. For an IG(a,b) innovation-variance prior,
the transformed conditional includes $-2a\ell-b\exp(-2\ell)$ as well as the
observation quadratic. The state transformation cancels the random-walk scale
normalisation. The proposal standard deviation is fixed at 0.15; there is no
output-dependent tuning.

The two-observation case is compared with the enumerated posterior; a 200-state
case compares mixing and posterior means under the same declared target. Both
use $h_0\sim N(0,2)$ and innovation variance $IG(2,0.1)$, which differ from the
library's default SV hyperparameters. Results therefore do not identify the
cause of a particular empirical failure. This kernel exists only in the study
script. It is not selected by `fit` and requires separate review before any
production integration.
