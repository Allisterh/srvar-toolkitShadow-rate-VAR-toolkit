"""Strict, non-executable encoding of the resolved prior in fit artefacts."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Any

import numpy as np

from .spec import BLassoSpec, DLSpec, MinnesotaCanonicalSpec, NIWPrior, PriorSpec, SSVSSpec


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate prior field")
        result[key] = value
    return result


def _finite_number(value: Any) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError("prior scalar must be finite and numeric")
    return float(value)


def _block(value: Any, cls: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError("prior block has missing or unknown fields")
    return dict(value)


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in {"i", "u", "f"} or not np.isfinite(result).all():
        raise ValueError("prior arrays must be finite and numeric")
    if shape is not None and result.shape != shape:
        raise ValueError("prior array has incompatible shape")
    return result.astype(float)


def _covariance(value: Any, size: int) -> np.ndarray:
    result = _array(value, (size, size))
    if not np.allclose(result, result.T, rtol=1e-12, atol=1e-12):
        raise ValueError("prior covariance must be symmetric")
    np.linalg.cholesky(result)
    return result


def _hyperparameters(value: Any, cls: Any) -> Any:
    block = _block(value, cls)
    for name, item in block.items():
        if name == "fix_intercept":
            if type(item) is not bool:
                raise ValueError("fix_intercept must be boolean")
        elif name == "mode":
            if not isinstance(item, str):
                raise ValueError("mode must be a string")
        elif name == "intercept_slab_var" and item is None:
            continue
        else:
            block[name] = _finite_number(item)
    return cls(**block)


def prior_from_json(text: str) -> PriorSpec:
    """Restore validated parameters directly, without re-estimating a prior."""
    try:
        raw = _block(json.loads(text, object_pairs_hook=_object), PriorSpec)
        if not isinstance(raw["family"], str):
            raise ValueError("unsupported prior family")
        family = raw["family"].lower()
        if family not in {"niw", "dl", "ssvs", "blasso"}:
            raise ValueError("unsupported prior family")
        if not isinstance(raw["method"], str):
            raise ValueError("method must be a string")
        if raw["residual_prior"] is not None and not isinstance(raw["residual_prior"], str):
            raise ValueError("residual_prior must be a string or null")

        niw = _block(raw["niw"], NIWPrior)
        m0 = _array(niw["m0"])
        if m0.ndim != 2 or min(m0.shape) < 1:
            raise ValueError("m0 must have positive coefficient and equation dimensions")
        k, n = m0.shape
        nu0 = _finite_number(niw["nu0"])
        if nu0 <= 0:
            raise ValueError("nu0 must be positive")
        raw["niw"] = NIWPrior(
            m0=m0,
            v0=_covariance(niw["v0"], k),
            s0=_covariance(niw["s0"], n),
            nu0=nu0,
        )
        if raw["residual_prior"] == "legacy_default" and (
            nu0 != n + 2 or not np.array_equal(raw["niw"].s0, np.eye(n))
        ):
            raise ValueError("legacy_default provenance conflicts with resolved IG parameters")

        for name, cls in (("dl", DLSpec), ("ssvs", SSVSSpec), ("blasso", BLassoSpec)):
            if family == name:
                raw[name] = _hyperparameters(raw[name], cls)
            elif raw[name] is not None:
                raise ValueError("shrinkage metadata conflicts with family")

        canonical = raw["minnesota_canonical"]
        if canonical is not None:
            if family != "niw":
                raise ValueError("Minnesota metadata conflicts with family")
            canonical = _block(canonical, MinnesotaCanonicalSpec)
            canonical["sigma2"] = _array(canonical["sigma2"], (n,))
            canonical["inv_v0_vec"] = _array(canonical["inv_v0_vec"], (k * n,))
            if not isinstance(canonical["mode"], str):
                raise ValueError("Minnesota mode must be a string")
            if canonical["tempered_alpha"] is not None:
                canonical["tempered_alpha"] = _finite_number(canonical["tempered_alpha"])
            raw["minnesota_canonical"] = MinnesotaCanonicalSpec(**canonical)
        return PriorSpec(**raw)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, np.linalg.LinAlgError):
        # Do not echo untrusted JSON, scalar values or decoder exception payloads.
        raise ValueError(
            "invalid saved prior metadata: check schema, values and provenance"
        ) from None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("unsupported prior value")


def prior_to_json(prior: PriorSpec) -> str:
    """Encode all resolved fields and reject metadata the reader cannot restore."""
    text = json.dumps(asdict(prior), default=_json_value, allow_nan=False, sort_keys=True)
    prior_from_json(text)
    return text
