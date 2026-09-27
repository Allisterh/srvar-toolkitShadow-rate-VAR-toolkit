# Releasing srvar-toolkit

Releases are built from a version tag and published only through GitHub Actions trusted publishing.
Tag pushes build and retain distributions; they cannot publish a package to PyPI.

## Current candidate

Version 0.4.0 is an alpha candidate with explicitly authorised pre-1.0 breaking
changes. The maintainer approved its specified DL rate policy, public inference
contracts, integration and tag on 27 September 2026. PyPI publication remains a
separate decision after inspection of the tag build. Approval excludes full-model
calibration and empirical claims. Final-head CI must pass before integration.
Enforced and experimental boundaries are listed in
{doc}`../user-guide/limitations`.

## First-time administrator setup

This repository cannot configure PyPI or GitHub protection by itself. Before the first release, a
PyPI project owner must configure a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for:

- PyPI project: `srvar-toolkit`
- GitHub owner: `shawcharles`
- Repository: `srvar-toolkit`
- Workflow filename: `release.yml`
- Environment: `pypi`

A GitHub administrator must also create the `pypi` Environment and require the appropriate
reviewers and tag deployment policy. GitHub's
[OIDC in PyPI guide](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi)
describes the same environment-bound publishing model. The workflow merely names that Environment;
it does not create reviewers, deployment policy, or PyPI authority. Do not add a PyPI API token or
password as a repository secret.

## Prepare a release

1. Update the package version in `pyproject.toml` and `srvar/__init__.py`, and add exactly one
   dated `## [X.Y.Z] - YYYY-MM-DD` heading to `CHANGELOG.md`.
2. Run the normal project quality gates and then run:

   ```bash
   python scripts/release_preflight.py --tag vX.Y.Z
   ```

3. Obtain review for the release commit. Create and push `vX.Y.Z` at that approved commit. An
   annotated or lightweight tag is accepted, but it must point at the approved commit.
4. Inspect the tag-triggered workflow run. It must pass preflight and the full pytest suite at the
   exact selected tag with xarray, ArviZ and Numba installed. Explicit import checks prevent
   missing extras from silently skipping coverage. It must then build exactly one wheel and one
   source distribution, pass `twine check`, and import the installed wheel at the tagged version.
   Download the retained artifact if an independent inspection is needed.

## Publish an inspected build

Start the workflow at the tag itself—not from `main`—and give the same tag as input:

```bash
gh workflow run release.yml --ref vX.Y.Z -f tag=vX.Y.Z -f confirm_publish=PUBLISH
```

Normal publication rejects a differing tag input, malformed tags, or any checkout
that is not the selected tag commit. Branch dispatch is rejected except for the
explicit recovery path below. The `publish-pypi` job runs only for this manual confirmation, waits
for the protected `pypi` Environment, downloads the validated artifact from its own build job, and
uses a short-lived GitHub OIDC token as described in PyPI's
[trusted publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/). No tag-push
run can publish.

After approval, verify the version on PyPI and install it independently in a fresh environment.

## Recovery

PyPI distributions cannot be replaced or routinely deleted after publication. If a release is
wrong, assess the impact, yank it where appropriate, publish a corrected higher version, and
communicate the incident. Deleting or replacing the Git tag does not retract an installed PyPI
distribution.

## Recover publication without moving a tag

If a tagged workflow cannot publish because its publishing tool is obsolete,
review and merge the tooling repair on main. Keep the package tag immutable.
Dispatch the repaired workflow from main with both the package tag and its exact
approved 40-character commit SHA:

```bash
gh workflow run release.yml --ref main -f tag=vX.Y.Z \
  -f expected_commit=FULL_APPROVED_COMMIT_SHA -f confirm_publish=PUBLISH
```

This recovery interface accepts only main or the selected tag as the workflow
ref. Main requires expected_commit. The workflow checks out the package tag and
verifies both its commit and the supplied SHA before tests or builds. The
protected pypi environment and OIDC publishing job remain unchanged; inspect
this run's artefacts before approving the environment. Arbitrary branches and
short or mismatched commit identities are rejected.

The publisher is pinned to v1.14.2, whose Twine 7 supports Core Metadata 2.5. See
the [PyPA release notes](https://github.com/pypa/gh-action-pypi-publish/releases/tag/v1.14.2).
