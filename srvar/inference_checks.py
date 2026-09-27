"""Public inference boundaries for configurations excluded by scientific review."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .spec import PriorSpec

if TYPE_CHECKING:
    from .results import FitResult


def validate_prior_for_inference(prior: PriorSpec) -> None:
    """Reject known inconsistent shrinkage targets before numerical work starts."""
    family = prior.family.lower()
    if family == "blasso":
        raise ValueError(
            "Bayesian LASSO inference is disabled: its scale conditional requires repair; "
            "choose a supported prior and refit"
        )
    if family in {"ssvs", "dl"} and np.any(prior.niw.m0 != 0):
        raise ValueError(
            "SSVS/DL inference requires zero prior means; non-zero-mean shrinkage "
            "conditionals are not supported"
        )


def validate_fit_for_inference(
    fit: FitResult,
    *,
    require_covariance: bool = True,
    require_volatility_innovations: bool = True,
) -> None:
    """Validate loaded or constructed fits without substituting a different posterior."""
    validate_prior_for_inference(fit.prior)
    if fit.prior.family.lower() not in {"ssvs", "dl"}:
        return
    beta = fit.beta_draws
    if beta is None or beta.ndim != 3 or beta.shape[0] == 0:
        raise ValueError(
            "shrinkage inference requires retained coefficient draws; refit with burn_in < draws"
        )
    if not require_covariance:
        return
    vol = fit.model.volatility
    states = [fit.sigma_draws]
    if vol is not None and vol.enabled:
        states = [fit.h_draws]
        if require_volatility_innovations:
            states.append(fit.sigma_eta2_draws)
    if any(state is None or state.ndim < 1 or state.shape[0] != beta.shape[0] for state in states):
        raise ValueError(
            "shrinkage inference requires aligned retained covariance states; refit the model"
        )
