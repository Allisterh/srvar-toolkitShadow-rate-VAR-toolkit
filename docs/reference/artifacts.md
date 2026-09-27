# `srvar.artifacts`

## Artifact compatibility and security

New fit artifacts use schema version 2 and forecast artifacts remain version 1.
Both use Unicode strings and numeric arrays, so loading them requires no pickle
deserialisation. Version-2 fits include a mandatory `prior_json` Unicode scalar.
It records the resolved NIW matrices and shape, shrinkage hyperparameters,
equation-specific Minnesota metadata, constructor `method` and DL residual-prior
input provenance. JSON is decoded as validated data, never executable code.

`load_run_dir` uses the saved prior as the authoritative fitted target. It reads
model and sampler settings from `config.yml` and checks prior dimensions against
the model, but does not re-evaluate the configuration's prior section. Preserve
the original configuration: this format does not yet persist the complete model
or detect every semantic edit to a same-dimension model configuration.

Version-1 fit artifacts lack a resolved prior. `load_fit_npz` continues to read
their raw draws with `prior=None`, while `load_run_dir` rejects reconstruction.
Regenerate a run with verified prior parameters to create a version-2 artifact;
do not relabel old arrays as a fit under new defaults. Constructor tags are
declared provenance, not proof of scientific qualification. Existing DL defaults
and numerical transitions are unchanged by the persistence format.

Markerless artifacts predating the safe version-1 format may contain pickle-backed arrays. The loaders reject them by
default. If, and only if, you have verified an old artifact's source and integrity, use the
explicit compatibility option:

```python
from srvar.artifacts import load_fit_npz

raw_draws = load_fit_npz("outputs/trusted_old_run/fit_result.npz", allow_legacy_pickle=True)
```

This option can execute pickle code. It is not appropriate for files received from an untrusted
or unknown source.

The pickle opt-in does not recover a missing prior or bypass `load_run_dir`'s
version-2 requirement.

## Availability limits

Every artifact loader applies metadata-based limits before NumPy opens the archive: a maximum
archive size (512 MiB), member count (128), per-member uncompressed size (512 MiB), aggregate
uncompressed size (1 GiB), and ZIP expansion ratio (100:1). These defaults allow ordinary
research runs while bounding accidental or maliciously oversized inputs. The same checks apply
when `allow_legacy_pickle=True`; trusted legacy compatibility does not bypass them.

Use an immutable `ArtifactLoadLimits` value to set stricter or larger local bounds:

```python
from srvar.artifacts import ArtifactLoadLimits, load_fit_npz

limits = ArtifactLoadLimits(max_archive_bytes=64 * 1024 * 1024)
fit = load_fit_npz("outputs/fit_result.npz", limits=limits)
```

These are early availability controls, not complete payload validation. ZIP metadata is untrusted
and is used only to bound work before NumPy access: parsing the ZIP directory and subsequent
`.npy` headers can still consume resources, and a malformed `.npy` shape or header can request an
allocation inconsistent with its ZIP member size. The limits reduce this exposure but cannot
eliminate it; do not treat them as a safe way to load untrusted pickle-backed files.

## Versioned payload validation

After supported format markers and archive limits pass, loaders allow only fields emitted by the matching
writer and validate stored dtype families, ranks, duplicate member names, and cross-field shapes
before creating result objects. Fit artifacts must have consistent variable, time, draw,
coefficient, and factor dimensions; forecast artifacts must have consistent variable, horizon,
and draw dimensions. Forecast quantile fields use the writer's canonical `q_<float>` spelling and
must be finite levels strictly between zero and one.

Each version has a strict field set. Version 2 adds mandatory prior metadata to
the version-1 fit fields. Prior decoding rejects duplicate, missing or unknown
JSON fields, invalid provenance, non-finite parameters, inconsistent dimensions
and non-positive-definite covariance matrices. The resolved prior is restored
without running data-dependent constructors.

A future field requires a format-version change; unknown or malformed fields
are rejected rather than silently ignored. Explicitly trusted,
markerless legacy artifacts retain their compatibility path and are not represented as validated
v1 data. The archive and `.npy` parsing residual risks described above still apply.

```{eval-rst}
.. automodule:: srvar.artifacts
   :members:
   :undoc-members:
   :show-inheritance:
```
