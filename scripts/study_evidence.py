"""Archive exact numerical source and maintained harnesses before a study starts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.qualify_samplers import source_manifest


def archive_sources(out: Path, scripts: list[Path]) -> dict:
    """Return a source manifest after verifying and archiving every mapped file."""
    root = Path(__file__).resolve().parents[1]
    manifest = source_manifest()
    for script in [Path(__file__), *scripts]:
        relative = str(script.resolve().relative_to(root))
        manifest["sha256"][relative] = hashlib.sha256(script.read_bytes()).hexdigest()
    for relative, digest in manifest["sha256"].items():
        source = (root / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError("source changed while archiving study inputs")
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    return manifest
