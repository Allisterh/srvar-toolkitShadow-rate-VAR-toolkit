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

`PriorSpec.from_dl` currently defaults to `nu0=N+2` and `s0=I`, giving $IG(N+2,1)$ with mean $1/(N+1)$. That default depends on the number of variables and assumes a fixed residual scale. It is not recommended as an intended scientific prior merely because the coefficient/variance transition is correct. The Python constructor accepts explicit `nu0` and `s0` as IG shape and diagonal rates. Replacing the default is a separate model change; the existing component calibration uses explicit hyperparameters and does not qualify this default or the complete DL hierarchy.

The coefficient conditional has precision $P_i=D_i+X'X/s_i$ and mean $P_i^{-1}(D_i m_i+X'y_i/s_i)$. The next variance is sampled from $IG(a_i+T_\mathrm{eff}/2,b_i+\lVert y_i-X\beta_i\rVert^2/2)$. Each iteration retains this variance for the next coefficient update. Resetting $s_i$ to a prior scale at every iteration does not target this joint posterior.

DL updates its shrinkage precisions between coefficient updates; canonical Minnesota holds them fixed. The normal prior is independent of $s_i$, so the variance conditional contains no coefficient-prior quadratic term. This differs from the matrix-normal NIW model described above.

### Triangular SV coefficient sweep

Triangular SV uses independent coefficient-column priors $\beta_i\sim N(m_i,V_0)$ and transformed residuals $e_t=Q(y_t-B'x_t)$. Each column affects every transformed equation whose corresponding entry in column $i$ of $Q$ is non-zero.

For column $i$, let $r_{tj}^{(-i)}=e_{tj}+Q_{ji}x_t'\beta_i$ remove its current contribution and let $w_{tj}=\exp(-h_{tj})$. Its Gaussian full conditional is

$$
P_i=V_0^{-1}+\sum_t x_tx_t'\sum_j w_{tj}Q_{ji}^2,
\qquad
m_i^*=P_i^{-1}\left(V_0^{-1}m_i+\sum_t x_t\sum_j w_{tj}Q_{ji}r_{tj}^{(-i)}\right).
$$

The sampler updates columns in reverse order, updates the transformed residual after each draw, and retains the full coefficient matrix between sweeps. This is a block Gibbs transition; successive sweeps are not independent draws from the joint Gaussian conditional. Production solves use coefficient-sized matrices, while tests compare stationary moments against a dense joint Gaussian reference.

#### Legacy Minnesota covariance limitation

The triangular path uses the shared `prior.niw.v0` directly as $V_0$. Under the matrix-normal NIW model, the corresponding coefficient-column covariance is $\Sigma_{ii}V_0$; that dependent-variable scale is absent from the triangular path. The legacy constructor's lag variances are proportional to $1/\widehat{\sigma}_k^2$ with an averaged own/cross weight, and its intercept entry is $(\lambda_1\lambda_4)^2$. Used unscaled for triangular SV, these entries do not provide the intended equation-specific Minnesota ratios $\widehat{\sigma}_i^2/\widehat{\sigma}_k^2$. For example, $\lambda_1=0.2$ and $\lambda_4=25$ give an intercept variance of 25 in every equation, regardless of that equation's residual scale.

Consequently, `minnesota_legacy` with triangular SV must not be described as implementing the intended Minnesota prior. The sweep is correct for the supplied $N(m_i,V_0)$ column priors, but that fact does not endorse the constructor's covariance mapping. The container's `s0` and `nu0` do not parameterise the triangular coefficient or volatility target. Equation-specific canonical Minnesota precisions are not currently supported on this path; extending that support requires a separate implementation and scientific review. Existing triangular diagnostic runs also remain subject to their unresolved convergence failures.

### Forecasts from retained ELB states

If any of the last `p` observations are censored, prediction selects `latent_draws[d, -p:, :]` with the same posterior index as coefficients and other state draws. This pairing is preserved after stationarity filtering. The final `latent_dataset` represents one Gibbs state and cannot replace retained histories. Forecasting rejects censored terminal histories when paired draws are absent or misaligned; refit with retained histories. Uncensored terminal histories use observed data.

### Verification boundary

Regression tests check selected Gaussian conditionals, an integrated inverse-gamma posterior reference, volatility timing and posterior-state pairing. They do not establish convergence, full simulation-based calibration or empirical replication. Human scientific review is required before substantive use of changed samplers. Bayesian LASSO covariance/dimension coherence and shrinkage updates with non-zero prior means remain separate methodological review items.

- Use `burn_in` and `thin` to control storage and reduce autocorrelation in retained draws.
- For long runs, profile your model and consider multiple shorter chains rather than a single very long chain.

Related:
- {doc}`../getting-started/quickstart`
- {doc}`../reference/api`
