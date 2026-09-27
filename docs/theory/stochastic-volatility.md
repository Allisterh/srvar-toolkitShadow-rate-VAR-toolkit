# Stochastic volatility (SV)

## Overview

Stochastic volatility (SV) allows the variance of VAR residuals to change over time. This is important in macro/financial data where volatility can shift markedly across regimes.

This toolkit implements SV variants commonly used in the Bayesian VAR literature:

- **log-volatility dynamics**: random walk (`SVRW`) or AR(1)
- **residual covariance**: diagonal covariance (independent shocks), a fixed triangular factor with generally time-varying correlations, or factor SV

## Model sketch

Let $h_{t,j}$ be the log-variance state for series $j$ at time $t$.

### Random walk (SVRW)

For each series $j$ the log-variance evolves as:

$$
 h_{t,j} = h_{t-1,j} + \eta_{t,j}, \qquad \eta_{t,j} \sim \mathcal{N}(0, \sigma_{\eta,j}^2).
$$

### AR(1)

Optionally, the log-variance follows an AR(1):

$$
 h_{t,j} = \gamma_{0,j} + \phi_j h_{t-1,j} + \eta_{t,j}.
$$

### Residual covariance structure

By default, conditional residual covariance is diagonal:

$$
\Sigma_t = \mathrm{diag}(\exp(h_t)).
$$

With the triangular factorization, the model uses:

$$
\Sigma_t = Q^{-1}\,\mathrm{diag}(\exp(h_t))\,(Q^{-1})',
$$

where $Q$ is upper-triangular with ones on the diagonal. The triangular factor is fixed within a parameter draw, but changing relative innovation variances generally changes correlations.

For example, if $Q_{12}=q$ in a two-variable model with diagonal variances $d_1,d_2$, then

$$
\rho_{12,t} = \frac{-q\sqrt{d_{2,t}}}{\sqrt{d_{1,t}+q^2d_{2,t}}}.
$$

This correlation changes with the variance ratio unless special restrictions hold. The model therefore does not impose constant conditional correlations.

## Forecast state timing

The first forecast observation uses $h_{T+1}$, sampled from the state transition conditional on the retained $h_T$. Each subsequent observation advances the state once more. This applies to diagonal and triangular RW/AR(1) models and to both the factor and idiosyncratic RW states in factor SV.

For a Gaussian diagonal RW model, the one-step innovation variance conditional on $h_T$ and $\sigma_\eta^2$ is $\exp(h_T+\sigma_\eta^2/2)$. Using $\exp(h_T)$ would omit the first forecast state innovation.

## Inference approach (KSC mixture)

The toolkit uses a standard auxiliary-mixture method (Kim, Shephard and Chib) to sample log-volatilities efficiently by approximating the log-$\chi^2$ distribution with a discrete mixture.

Implementation notes:
- the log-volatility state is sampled with a banded precision representation;
- when `covariance="triangular"`, the triangular factor is updated via a Gaussian prior on off-diagonal elements.

Related:
- {doc}`mcmc`
