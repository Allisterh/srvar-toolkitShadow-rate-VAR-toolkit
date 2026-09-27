# Limitations and performance

This project targets transparency and reproducibility and is currently in an **alpha** stage.

## Enforced inference boundaries

Bayesian LASSO fitting and inference on its archived fits raise errors because
its scale conditional needs repair. SSVS/DL require zero coefficient-prior means
and retained posterior draws. Their `FitResult.posterior` is always `None`;
forecasting and analysis cannot substitute a conditional NIW block. Conjugate
NIW analytic inference remains available. Raw artefact inspection and prior
construction do not imply that inference with those objects is supported.

ELB Cholesky historical decomposition raises an error until latent histories are
paired with parameter draws. Neither `use_latent=True` nor `False` bypasses it.
Full DL hierarchy and SV fits remain experimental: diagnostics and component
checks do not establish calibration or empirical validity. The maintainer approved
the specified DL rate-construction policy and public inference contracts for
0.4.0 on 27 September 2026; that decision does not extend to these broader claims.

## Modeling limitations

- **Stochastic volatility coverage is still evolving**: diagonal SV, triangular SV with a fixed factor and generally time-varying correlations, and factor SV are supported. Factor SV is currently limited to `prior.family: "niw"` with RW dynamics; ELB, steady-state, and robust shocks are supported.
- **Structural analysis coverage is partial**: reduced-form, Cholesky, and sign-restricted IRFs are supported via `srvar.analysis` (plus FEVD and Cholesky historical decompositions), but other workflows (e.g. sign-restricted historical decompositions) are not yet first-class.
- **Conditional/scenario forecasts are limited**: constraints in `srvar.scenario.conditional_forecast` require homoskedastic Gaussian VARs. Constrained Student-t and outlier-mixture models raise an error because Gaussian conditioning would change the model. Empty constraints delegate to ordinary forecasting for homoskedastic models. For ELB models, conditioning is applied to the latent (unfloored) process.
- **ELB treatment**: ELB handling is implemented via latent shadow-rate augmentation for selected series.
- **Robust shocks limitations**: Student‑t and outlier-mixture innovations are supported for homoskedastic VARs and for factor SV. They are not yet supported for diagonal/triangular SV; ELB/steady-state combinations require factor SV.

## Statistical limitations / caveats

- **MCMC diagnostics are your responsibility**: the toolkit returns draws and includes qualification scripts for R-hat and ESS. Diagnose each intended fit; passing thresholds does not prove the target is correct.
- **Sensitivity to prior settings**: results can change meaningfully with Minnesota hyperparameters, SSVS spike/slab variances, and SV priors.
- **Scientific review remains necessary**: the corrected sampler conditionals have analytical regression checks, but full simulation-based calibration and empirical replication remain outstanding. Bayesian LASSO and non-zero-mean shrinkage are disabled pending repair; see {doc}`../theory/mcmc`.
- **ELB histories must remain paired**: censored terminal lags require retained latent histories aligned with parameter draws. Forecasts cannot substitute the last Gibbs history. ELB Cholesky historical decomposition is disabled because it does not yet integrate paired latent-history uncertainty.

## Performance considerations

Runtime depends primarily on:

- `T`: number of observations
- `N`: number of variables
- `p`: lag order
- `draws`, `burn_in`, `thin`: sampler configuration
- model features enabled (ELB and SV are more expensive than conjugate NIW)

Rules of thumb:

- Start with small samplers to validate data plumbing and model stability.
- Increase draws only once the model runs end-to-end and outputs look reasonable.

Backtesting can also be memory-heavy. For long backtests, prefer streaming evaluation:

- `output.save_plots: false`
- `output.store_forecasts_in_memory: false`

## Numerical considerations

- Some numerical constants/initializations are chosen for stability (e.g., latent ELB initialization uses a small offset below the bound).
- The SV implementation uses an auxiliary mixture approximation (KSC) and banded linear algebra; extreme data scaling can still cause numerical issues.
