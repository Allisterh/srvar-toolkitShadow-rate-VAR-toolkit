from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")


def _run_bodies() -> list[str]:
    bodies: list[str] = []
    lines = TEXT.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^(?P<indent>\s*)run: \|$", line)
        if not match:
            index += 1
            continue
        run_indent = len(match.group("indent"))
        index += 1
        body_lines: list[str] = []
        while index < len(lines):
            next_line = lines[index]
            if next_line.strip() and len(next_line) - len(next_line.lstrip(" ")) <= run_indent:
                break
            body_lines.append(next_line)
            index += 1
        bodies.append("\n".join(body_lines))
    return bodies


def test_tag_trigger_is_routing_only() -> None:
    assert 'tags: ["v*"]' in TEXT
    assert "TAG_RE = re.compile" in TEXT
    assert r"\Av\d+\.\d+\.\d+\Z" in TEXT


def test_no_github_expressions_inside_run_scripts() -> None:
    for body in _run_bodies():
        assert "${{" not in body


def test_validates_dispatch_ref_matches_input_before_checkout() -> None:
    validation_index = TEXT.index("id: validated-ref")
    checkout_index = TEXT.index("name: Check out selected tag")

    assert validation_index < checkout_index
    assert "EVENT_NAME: ${{ github.event_name }}" in TEXT
    assert "WORKFLOW_REF: ${{ github.ref }}" in TEXT
    assert "DISPATCH_TAG: ${{ inputs.tag }}" in TEXT
    assert 'expected_ref = f"refs/tags/{tag}"' in TEXT
    assert "workflow_dispatch must run at the selected tag ref" in TEXT
    assert "GITHUB_OUTPUT" in TEXT


def test_downstream_steps_use_validated_tag_output() -> None:
    assert "ref: ${{ steps.validated-ref.outputs.tag }}" in TEXT
    assert "TAG: ${{ steps.validated-ref.outputs.tag }}" in TEXT
    assert "name: ${{ steps.validated-ref.outputs.artifact_name }}" in TEXT
    assert "name: ${{ needs.build.outputs.artifact-name }}" in TEXT


def test_checkout_identity_and_exact_payload_are_enforced() -> None:
    assert 'git rev-parse --verify "refs/tags/${TAG}^{commit}"' in TEXT
    assert "git rev-parse --verify HEAD^{commit}" in TEXT
    assert "rm -rf dist" in TEXT
    assert "exactly one wheel and one sdist" in TEXT
    assert "if-no-files-found: error" in TEXT
    assert "python -m twine check dist/*" in TEXT
    assert "python -m venv" in TEXT
    assert "import srvar" in TEXT


def test_publish_job_is_manual_artifact_only_oidc() -> None:
    publish = TEXT[TEXT.index("publish-pypi:") :]

    assert "needs: build" in publish
    assert "runs-on: ubuntu-latest" in publish
    assert "container:" not in publish
    assert "environment: pypi" in publish
    assert "id-token: write" in publish
    assert "contents: write" not in publish
    assert "actions/checkout" not in publish
    assert "password:" not in publish
    assert "secret" not in publish.lower()
    assert "github.event_name == 'workflow_dispatch'" in publish
    assert "inputs.confirm_publish == 'PUBLISH'" in publish
    assert "pypa/gh-action-pypi-publish@" in publish


def test_third_party_actions_are_full_sha_pinned() -> None:
    action_uses = re.findall(r"uses: ([^\s#]+)", TEXT)

    assert action_uses
    for use in action_uses:
        assert re.fullmatch(r"[-\w]+/[-\w]+@[0-9a-f]{40}", use), use


def test_selected_release_commit_is_tested_before_build_and_upload() -> None:
    checkout = TEXT.index("name: Verify checkout is the selected tag commit")
    qualify = TEXT.index("name: Test selected tag with optional integrations")
    build = TEXT.index("name: Build and validate distributions")
    upload = TEXT.index("name: Upload validated distributions")
    assert checkout < qualify < build < upload
    qualification = TEXT[qualify:build]
    assert "python -m pytest" in qualification
    assert "import arviz, numba, xarray" in qualification
    assert "continue-on-error" not in TEXT


def test_ci_exercises_optional_integrations() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    optional = ci[ci.index("  optional-integrations:") :]
    assert "xarray,arviz,accel" in optional
    assert "import arviz, numba, xarray" in optional
    assert "python -m pytest" in optional


def _execute_body(index, *, cwd, env):
    import os
    import subprocess
    import textwrap

    return subprocess.run(
        ["bash", "-e", "-c", textwrap.dedent(_run_bodies()[index])],
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


def test_dispatch_validation_executes_with_explicit_recovery_binding(tmp_path):
    source = "c3cc59b06df3c86a9b1120db48dc68e6e7199144"
    cases = [
        ("push", "refs/tags/v0.4.0", "", "", True),
        ("workflow_dispatch", "refs/tags/v0.4.0", "v0.4.0", "", True),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", source, True),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", "", False),
        ("workflow_dispatch", "refs/heads/feature", "v0.4.0", source, False),
        ("workflow_dispatch", "refs/tags/v0.3.1", "v0.4.0", source, False),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", source[:7], False),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", source.upper(), False),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", "x" * 40, False),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0", source + "\n", False),
        ("workflow_dispatch", "refs/heads/main", "v0.4.0\n", source, False),
        ("push", "refs/heads/main", "v0.4.0", source, False),
        ("pull_request", "refs/tags/v0.4.0", "v0.4.0", source, False),
    ]
    for index, (event, ref, tag, expected, accepted) in enumerate(cases):
        output = tmp_path / f"output-{index}"
        result = _execute_body(
            0,
            cwd=tmp_path,
            env={
                "EVENT_NAME": event,
                "WORKFLOW_REF": ref,
                "DISPATCH_TAG": tag,
                "EXPECTED_COMMIT": expected,
                "GITHUB_OUTPUT": str(output),
            },
        )
        assert (result.returncode == 0) == accepted, (event, ref, result.stderr)
        if accepted:
            assert output.read_text().splitlines() == [
                "tag=v0.4.0",
                "version=0.4.0",
                "artifact_name=python-distributions-v0.4.0",
            ]
        else:
            assert not output.exists()


def test_checkout_check_executes_against_real_tag_and_expected_commit(tmp_path):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    commit = [
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
    ]
    git(*commit, "-m", "approved source")
    approved = git("rev-parse", "HEAD")
    git("tag", "v0.4.0")
    for expected, accepted in ((approved, True), ("", True), ("0" * 40, False)):
        result = _execute_body(1, cwd=tmp_path, env={"TAG": "v0.4.0", "EXPECTED_COMMIT": expected})
        assert (result.returncode == 0) == accepted
    git(*commit, "-m", "different source")
    other = git("rev-parse", "HEAD")
    for expected in (approved, other, ""):
        result = _execute_body(1, cwd=tmp_path, env={"TAG": "v0.4.0", "EXPECTED_COMMIT": expected})
        assert result.returncode != 0
