"""Drift check for vendored mlx-lm snapshots.

The ``*_vendored.py`` files under this package are point-in-time copies (or
close derivatives) of upstream mlx-lm code, pinned to specific PRs in
``VENDORED_MANIFEST.json``. Because nothing in Python enforces that a
"verbatim" snapshot stays verbatim, an edit could silently diverge from the
recorded upstream snapshot — exactly the DRIFT risk this guards against.

``check_vendored_drift()`` recomputes the sha256 of every manifest entry and
reports any mismatch (edited file) or missing file. It is exercised by
``tests/v4/test_vendored_provenance.py`` and can also be run standalone:

    python -m cppmega_v4.nn._external._vendored_provenance

If you intentionally re-vendor a file, recompute its hash and update both the
``sha256`` and any provenance fields in ``VENDORED_MANIFEST.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
MANIFEST_PATH = _HERE / "VENDORED_MANIFEST.json"


@dataclass(frozen=True)
class DriftFinding:
    file: str
    code: str  # "hash_mismatch" | "missing_file" | "missing_hash"
    detail: str


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    return json.loads(manifest_path.read_text())


def check_vendored_drift(
    *,
    manifest_path: Path = MANIFEST_PATH,
    base_dir: Path | None = None,
) -> list[DriftFinding]:
    """Return a list of drift findings; empty list means no drift."""
    manifest = load_manifest(manifest_path)
    base = base_dir if base_dir is not None else manifest_path.parent
    findings: list[DriftFinding] = []
    for entry in manifest.get("entries", []):
        name = entry.get("file")
        recorded = entry.get("sha256")
        target = base / name
        if not recorded:
            findings.append(
                DriftFinding(name, "missing_hash", "manifest entry has no sha256")
            )
            continue
        if not target.exists():
            findings.append(
                DriftFinding(
                    name, "missing_file", f"vendored file {target} does not exist"
                )
            )
            continue
        actual = _sha256_of(target)
        if actual != recorded:
            findings.append(
                DriftFinding(
                    name,
                    "hash_mismatch",
                    (
                        f"vendored file diverged from pinned snapshot "
                        f"(PR #{entry.get('pr')}): recorded {recorded[:12]}…, "
                        f"got {actual[:12]}…. If this re-vendor is intentional, "
                        f"update sha256 in VENDORED_MANIFEST.json."
                    ),
                )
            )
    return findings


def _main() -> int:
    findings = check_vendored_drift()
    if not findings:
        print("vendored provenance: ok (no drift)")
        return 0
    for f in findings:
        print(f"DRIFT [{f.code}] {f.file}: {f.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
