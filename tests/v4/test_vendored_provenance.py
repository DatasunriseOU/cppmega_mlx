"""Provenance / drift tests for vendored mlx-lm snapshots (Path E + mHC/MTP).

These keep the ``*_vendored.py`` copies pinned to the PRs recorded in
``VENDORED_MANIFEST.json`` so a verbatim snapshot cannot silently diverge.
"""

from __future__ import annotations

import json

from cppmega_v4.nn._external._vendored_provenance import (
    MANIFEST_PATH,
    check_vendored_drift,
    load_manifest,
)

_PATH_E_PRS = {1217, 1224}


def test_manifest_is_valid_json_with_entries():
    manifest = load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["entries"], "manifest must record at least one vendored file"
    for entry in manifest["entries"]:
        assert entry["file"]
        assert entry["pr"]
        assert entry["sha256"]
        assert entry["kind"] in {"verbatim", "derived"}


def test_path_e_prs_are_pinned():
    """The task-critical Path E provenance (PR #1217 + #1224) must be present."""
    manifest = load_manifest()
    prs = {e["pr"] for e in manifest["entries"]}
    assert _PATH_E_PRS <= prs, f"missing Path E PRs: {_PATH_E_PRS - prs}"


def test_no_vendored_drift():
    """Every vendored file matches its pinned sha256 (no silent edits)."""
    findings = check_vendored_drift()
    assert findings == [], "\n".join(
        f"{f.code} {f.file}: {f.detail}" for f in findings
    )


def test_drift_check_detects_edit(tmp_path):
    """A modified vendored byte must be flagged as drift."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    # Point the manifest at a temp dir where one file has been tampered with.
    src_dir = MANIFEST_PATH.parent
    first = manifest["entries"][0]["file"]
    for entry in manifest["entries"]:
        original = (src_dir / entry["file"]).read_bytes()
        (tmp_path / entry["file"]).write_bytes(original)
    # Tamper with the first vendored file.
    tampered = tmp_path / first
    tampered.write_bytes(tampered.read_bytes() + b"\n# tampered\n")
    tmp_manifest = tmp_path / "VENDORED_MANIFEST.json"
    tmp_manifest.write_text(MANIFEST_PATH.read_text())

    findings = check_vendored_drift(manifest_path=tmp_manifest, base_dir=tmp_path)
    codes = {(f.file, f.code) for f in findings}
    assert (first, "hash_mismatch") in codes
