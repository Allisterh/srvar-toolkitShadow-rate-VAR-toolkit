"""Reproduce the local final-vintage benchmark without replacing existing outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import srvar
from scripts.qualify_samplers import source_manifest
from srvar.backtest import backtest_from_config
from srvar.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def validate_source_root(expected: Path) -> None:
    """Fail before running if import precedence selected the wrong source tree."""
    actual = Path(srvar.__file__).resolve().parent
    if actual != expected.resolve():
        raise ValueError(
            f"loaded source {actual} differs from expected source {expected.resolve()}"
        )


def benchmark_configs(data: Path, out: Path, origin: str | None = None) -> dict[str, dict]:
    """Preserve shipped controls, changing paths and the explicit canonical variant."""
    homo = load_config(ROOT / "config/vintage_macro15_backtest_homoskedastic.yaml")
    canonical = deepcopy(homo)
    canonical["prior"]["method"] = "minnesota_canonical"
    diagonal = load_config(ROOT / "config/vintage_macro15_backtest_diagonal_sv.yaml")
    configs = {
        "legacy_homoskedastic": homo,
        "canonical_homoskedastic": canonical,
        "diagonal_sv": diagonal,
    }
    for name, cfg in configs.items():
        cfg["data"]["csv_path"] = str(data.resolve())
        cfg["output"]["out_dir"] = str((out / name).resolve())
        if origin is not None:
            cfg["backtest"]["origin_start"] = origin
            cfg["backtest"]["origin_end"] = origin
    return configs


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--origin", help="Optional single-origin runtime probe")
    args = parser.parse_args()
    validate_source_root(args.expected_package_root)
    if not args.data.is_file():
        parser.error("--data must name a prepared local CSV")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_label": args.source_label,
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "origin_override": args.origin,
        **source_manifest(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, cfg in benchmark_configs(args.data, args.out, args.origin).items():
        path = args.out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"starting={name}", flush=True)
        backtest_from_config(path, out_dir=args.out / name)
        metrics = pd.read_csv(args.out / name / "metrics.csv")
        numeric = metrics.select_dtypes(include="number")
        if metrics.empty or not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError(f"{name}: empty or non-finite metrics")
        print(f"completed={name} metric_rows={len(metrics)}", flush=True)


if __name__ == "__main__":
    main()
