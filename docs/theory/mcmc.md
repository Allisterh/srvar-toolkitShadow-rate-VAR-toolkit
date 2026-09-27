# MCMC overview

This page summarises the Markov chain Monte Carlo (MCMC) logic used in the toolkit. The implementation is a pragmatic Gibbs sampler for a reduced-form VAR with optional ELB augmentation and diagonal stochastic volatility.

## BVAR (NIW) without ELB/SV

1. Compute NIW posterior parameters.
2. Sample $(\beta, \Sigma)$ from the matrix-normal inverse-Wishart posterior.

## ELB only (shadow-rate augmentation)

At each iteration:

1. Sample VAR parameters $(\beta, \Sigma)$ conditional on the current latent series.
2. For each ELB-constrained observation, sample a latent shadow value from its conditional distribution subject to the bound.

## SV only (diagonal SVRW)

At each iteration:

1. Sample coefficients $\beta$ conditional on the current log-volatilities $h$.
2. Sample log-volatilities $h$ using the auxiliary-mixture approach.
3. Sample SV hyperparameters (initial log-volatility and innovation variance).

## SV + ELB (combined)

At each iteration:

1. Sample $\beta$ conditional on $h$ and the current latent series.
2. Sample ELB latent values conditional on $\beta$ and $h$.
3. Sample $h$ conditional on residuals.
4. Sample SV hyperparameters.

## Practical notes

### Independent coefficient priors and residual variances

Homoskedastic Dirichlet-Laplace (DL) and canonical Minnesota paths use independent normal coefficient priors conditional on their prior precisions. Their residual covariance is diagonal. For equation $i$, write

$$
y_i\mid\beta_i,s_i \sim N(X\beta_i,s_i I),\qquad
\beta_i\mid D_i \sim N(m_i,D_i^{-1}),\qquad
s_i \sim IG(a_i,b_i),
$$

where the inverse-gamma density is proportional to $s_i^{-a_i-1}\exp(-b_i/s_i)$. The implementation uses $a_i=\texttt{nu0}$ and $b_i=\texttt{s0}[i,i]$ from the prior container. These are equation-wise inverse-gamma hyperparameters; they do not inherit the NIW degrees-of-freedom interpretation.

The canonical Minnesota constructor supplies $IG(2,\widehat{\sigma}_i^2)$, where $\widehat{\sigma}_i^2$ is the equation's autoregressive residual-variance estimate from the supplied training data. Its prior mean is $\widehat{\sigma}_i^2$ and its prior variance is infinite. This is an empirical-Bayes, scale-adaptive choice; infinite variance alone does not establish weak informativeness or calibration.

`PriorSpec.from_dl` requires a residual-prior mode. `empirical_bayes` supplies $IG(2,\widehat{\sigma}_i^2)$ from training-only univariate AR(p) regressions. Each series is divided by its maximum absolute observation before fitting; the residual variance is then restored to the original units. The denominator is $T-p-(p+c)>0$, where $c$ indicates an intercept, and the normalised design must have full column rank. Each backtest origin estimates its rates once from its training window.

There is no absolute variance floor on this DL path. Zero series, unidentified designs and numerically unresolved residuals raise an error. For normalised response $z$, design $X$ and fitted coefficient $b$, the resolution rule rejects $\|z-Xb\|_2 \leq \epsilon\max(X.shape)(\|z\|_2+\|X\|_F\|b\|_2)$, with double-precision machine epsilon. This dimensionless rule is a numerical check, not a statistical test; it can reject nearly deterministic data. Under a positive change of units by $a_i$, valid rates transform by $a_i^2$ in exact arithmetic. Rates that underflow to zero or overflow are rejected. Supply an explicitly justified proper prior when the training window cannot identify a rate.

Earlier DL empirical-Bayes constructors applied an absolute floor (default $10^{-12}$) and a denominator fallback of one. The DL `min_sigma2` option is removed. Reproduce a previous target with its saved resolved rates and shape in explicit mode, or use its archived source; regenerate new-policy outputs separately. Minnesota constructors retain their distinct floor semantics. Rate equivariance does not establish equivariance or calibration of the full DL hierarchy.

`explicit` mode requires `nu0` and a positive diagonal `s0`, with no training inputs. The former implicit default $IG(N+2,1)$ can be reproduced deliberately using `nu0=N+2, s0=I`. Earlier default-configured results and new empirical-Bayes results have different targets and must not be pooled or relabelled. Fixed-hyperparameter component SBC does not qualify the plug-in scale-estimation procedure or the complete DL hierarchy; the empirical-Bayes procedure needs a separate frequentist coverage study.

The coefficient conditional has precision $P_i=D_i+X'X/s_i$ and mean $P_i^{-1}(D_i m_i+X'y_i/s_i)$. The next variance is sampled from $IG(a_i+T_\mathrm{eff}/2,b_i+\lVert y_i-X\beta_i\rVert^2/2)$. Each iteration retains this variance for the next coefficient update. Resetting $s_i$ to a prior scale at every iteration does not target this joint posterior.

The variance draw uses the posterior rate divided by a unit-scale Gamma draw.
It has no imposed lower or upper bound. Earlier implementations clipped draws
to $[10^{-12},10^{12}]$, introducing boundary atoms whenever clipping was active.
Invalid conditional parameters raise `ValueError`; non-finite or non-positive
numerical draws raise `FloatingPointError` instead of being replaced. Finite
floating-point range still limits computation. This repair changes results at
the former boundaries and can change seeded values through rounding, so refit
affected models in fresh output directories. The empirical-Bayes prior-rate
floor and the DL hierarchy's other safeguards remain separate; removing posterior
clipping does not establish full-procedure interval calibration.

DL updates its shrinkage precisions between coefficient updates; canonical Minnesota holds them fixed. The normal prior is independent of $s_i$, so the variance conditional contains no coefficient-prior quadratic term. This differs from the matrix-normal NIW model described above.

### Triangular SV coefficient sweep

Triangular SV uses independent coefficient-column priors $\beta_i\sim N(m_i,V_{0i})$ and transformed residuals $e_t=Q(y_t-B'x_t)$. Each column affects every transformed equation whose corresponding entry in column $i$ of $Q$ is non-zero.

For column $i$, let $r_{tj}^{(-i)}=e_{tj}+Q_{ji}x_t'\beta_i$ remove its current contribution and let $w_{tj}=\exp(-h_{tj})$. Its Gaussian full conditional is

$$
P_i=V_{0i}^{-1}+\sum_t x_tx_t'\sum_j w_{tj}Q_{ji}^2,
\qquad
m_i^*=P_i^{-1}\left(V_{0i}^{-1}m_i+\sum_t x_t\sum_j w_{tj}Q_{ji}r_{tj}^{(-i)}\right).
$$

The sampler updates columns in reverse order, updates the transformed residual after each draw, and retains the full coefficient matrix between sweeps. This is a block Gibbs transition; successive sweeps are not independent draws from the joint Gaussian conditional. Production solves use coefficient-sized matrices, while tests compare stationary moments against a dense joint Gaussian reference.

#### Legacy Minnesota covariance limitation

The custom shared-covariance path uses `prior.niw.v0` directly as $V_0$. Under the matrix-normal NIW model, the corresponding coefficient-column covariance is $\Sigma_{ii}V_0$; that dependent-variable scale is absent from the triangular path. The legacy constructor's lag variances are proportional to $1/\widehat{\sigma}_k^2$ with an averaged own/cross weight, and its intercept entry is $(\lambda_1\lambda_4)^2$. Used unscaled for triangular SV, these entries do not provide the intended equation-specific Minnesota ratios $\widehat{\sigma}_i^2/\widehat{\sigma}_k^2$. For example, $\lambda_1=0.2$ and $\lambda_4=25$ give an intercept variance of 25 in every equation, regardless of that equation's residual scale.

`minnesota_legacy` is therefore rejected for triangular SV, including its `minnesota` alias. Canonical Minnesota supplies $V_{0i}^{-1}=\operatorname{diag}(\texttt{inv_v0_vec}[iK:(i+1)K])$, with the same precision used in the prior linear term. This retains own-lag variances and the dependent/predictor residual-variance ratios already defined by the canonical constructor. The shared custom Gaussian path remains available. The container's `s0` and `nu0` do not parameterise the triangular coefficient or volatility target. The existing NIW mean calculation supplies an initial coefficient state only.

Canonical support covers triangular RW and AR(1), including ELB coefficient updates. Factor SV, tempered triangular priors and triangular steady-state models remain unsupported. The new prior mapping requires separate scientific review; existing triangular diagnostic runs retain their unresolved convergence failures. No SV state, initial-state or innovation-variance transition is repaired by this coefficient-prior change.

### Forecasts from retained ELB states

If any of the last `p` observations are censored, prediction selects `latent_draws[d, -p:, :]` with the same posterior index as coefficients and other state draws. This pairing is preserved after stationarity filtering. The final `latent_dataset` represents one Gibbs state and cannot replace retained histories. Forecasting rejects censored terminal histories when paired draws are absent or misaligned; refit with retained histories. Uncensored terminal histories use observed data.

### Verification boundary

Regression tests check selected Gaussian conditionals, an integrated inverse-gamma posterior reference, volatility timing and posterior-state pairing. They do not establish convergence, full simulation-based calibration or empirical replication. Human scientific review is required before substantive use of changed samplers. Bayesian LASSO covariance/dimension coherence and shrinkage updates with non-zero prior means remain separate methodological review items.

- Use `burn_in` and `thin` to control storage and reduce autocorrelation in retained draws.
- For long runs, profile your model and consider multiple shorter chains rather than a single very long chain.

Related:
- {doc}`../getting-started/quickstart`
- {doc}`../reference/api`
