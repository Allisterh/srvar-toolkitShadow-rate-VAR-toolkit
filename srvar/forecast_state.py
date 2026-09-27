"""Terminal histories aligned with retained posterior parameter draws."""

from __future__ import annotations

import numpy as np

from .results import FitResult


def _terminal_lags(fit: FitResult, *, indices: np.ndarray | None, draws: int) -> np.ndarray:
    """Return (draw, lag, variable) histories, preserving ELB state pairing.

    Observed terminal histories suffice unless one of the last p observations
    is censored. In that case each parameter draw needs its own retained latent
    history; a final Gibbs state is not a posterior substitute.
    """
    p = fit.model.p
    observed = fit.dataset.values[-p:, :]
    if observed.shape != (p, fit.dataset.N) or not np.isfinite(observed).all():
        raise ValueError("forecast requires p finite terminal observations")
    elb = fit.model.elb
    censored = False
    if elb is not None and elb.enabled:
        columns = [fit.dataset.variables.index(name) for name in elb.applies_to]
        censored = bool(np.any(observed[:, columns] <= elb.bound + elb.tol))
    if not censored:
        return np.broadcast_to(observed, (draws, p, fit.dataset.N))

    if fit.latent_draws is None or fit.beta_draws is None or indices is None:
        raise ValueError(
            "censored terminal observations require paired latent_draws and beta_draws; "
            "refit with retained posterior histories"
        )
    latent = np.asarray(fit.latent_draws, dtype=float)
    expected = (fit.beta_draws.shape[0], fit.dataset.T, fit.dataset.N)
    if latent.shape != expected:
        raise ValueError("latent_draws must have shape (D, T, N) aligned with beta_draws")
    terminal = latent[:, -p:, :]
    if not np.isfinite(terminal).all():
        raise ValueError("latent_draws must contain finite terminal histories")
    return terminal[indices]
