"""Consume-side verification for the Nebius sidecar download.

These tests use REAL parquet shards and the REAL ``audit_sidecar_parquet.py``
audit (run as a subprocess) to produce a genuine audit receipt -- no mocks of
the audit, no mocked object storage. They prove the fail-closed gate added to
``download_verified_sidecar_from_nebius_s3._verify_downloaded_set``:

  * a complete, green, fully-downloaded set passes;
  * a *missing-shards SET* (which ``aws s3 sync`` would treat as success)
    RAISES -- this is the gap the finding called out;
  * a non-green receipt, a manifest/receipt token divergence, and a manifest
    that is required to carry ``verified_valid_tokens`` but lacks it all RAISE;
  * the ``--use-default-manifest`` path (no uploaded token total) is still
    proven via the green receipt + per-bucket shard-count reconciliation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import download_verified_sidecar_from_nebius_s3 as download


REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "4"  # rows are this long; tiny so the real audit stays fast
SHARDS = 2


def _write_green_shard(path: Path) -> None:
    """Write a single-row parquet shard that passes every audit value check.

    The audit derives the canonical loss-target rules from the columns below
    (see ``audit_sidecar_parquet`` value-level validators):
      * ``target_ids == input_ids[1:] + PAD``
      * ``loss_mask[pos] == 1`` iff next token is the same doc and within valid
      * ``trained_token_count == sum(loss_mask > 0)``
    """
    length = int(BUCKET)
    input_ids = list(range(10, 10 + length))  # in-vocab, distinct
    target_ids = input_ids[1:] + [0]  # next-token shift + pad
    loss_mask = [1] * (length - 1) + [0]  # single doc; last token excluded
    doc_ids = [7] * length  # one document spanning the whole row
    table = pa.table(
        {
            "input_ids": pa.array([input_ids], pa.list_(pa.int64())),
            "target_ids": pa.array([target_ids], pa.list_(pa.int64())),
            "loss_mask": pa.array([loss_mask], pa.list_(pa.int64())),
            "doc_ids": pa.array([doc_ids], pa.list_(pa.int64())),
            "valid_token_count": pa.array([length], pa.int64()),
            "trained_token_count": pa.array([length - 1], pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_verified_tree(dest: Path) -> int:
    """Lay out a downloaded set + a REAL green audit receipt; return valid_tokens."""
    code_bucket = dest / "reindexed" / "code" / BUCKET
    for idx in range(SHARDS):
        _write_green_shard(code_bucket / f"shard{idx}.parquet")
    (dest / "reindexed_commits").mkdir(parents=True, exist_ok=True)
    (dest / "reindexed_pr").mkdir(parents=True, exist_ok=True)

    out_dir = dest / "audit"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "audit_sidecar_parquet.py"),
        "--code-root",
        str(dest / "reindexed" / "code"),
        "--commit-root",
        str(dest / "reindexed_commits"),
        "--pr-root",
        str(dest / "reindexed_pr"),
        "--buckets",
        BUCKET,
        "--out-dir",
        str(out_dir),
        "--vocab-size",
        "65536",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"real audit failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    report = json.loads((out_dir / "sidecar_parquet_audit.json").read_text())
    assert report["total"]["bad_files"] == 0
    assert report["total"]["bad_rows"] == 0
    assert report["by_kind_bucket"][f"code/{BUCKET}"]["files"] == SHARDS
    return int(report["total"]["valid_tokens"])


@pytest.fixture(scope="module")
def pristine(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    base = tmp_path_factory.mktemp("sidecar_pristine")
    tokens = _build_verified_tree(base)
    return base, tokens


def _fresh(pristine: tuple[Path, int], where: Path) -> tuple[Path, int]:
    base, tokens = pristine
    dest = where / "dl"
    shutil.copytree(base, dest)
    return dest, tokens


def _manifest(tokens: int, *, include_token_total: bool = True) -> dict:
    selections = [
        {"remote": "parquet/code/4", "local": "reindexed/code/4"},
        {"remote": download.RECEIPT_REMOTE, "local": "audit"},
    ]
    manifest: dict = {"selections": selections}
    if include_token_total:
        manifest["verified_valid_tokens"] = tokens
    return manifest


def test_verify_passes_on_complete_green_set(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    result = download._verify_downloaded_set(
        dest=dest, manifest=_manifest(tokens), require_token_total=True
    )
    assert result["bad_files"] == 0
    assert result["bad_rows"] == 0
    assert result["verified_valid_tokens"] == tokens
    assert result["buckets_verified"] == 1


def test_verify_raises_on_missing_shard_set(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    # Drop a shard to simulate an incomplete remote prefix that `aws s3 sync`
    # would otherwise treat as a successful (return-0) download.
    shards = sorted((dest / "reindexed" / "code" / BUCKET).glob("*.parquet"))
    shards[0].unlink()
    with pytest.raises(SystemExit) as exc:
        download._verify_downloaded_set(
            dest=dest, manifest=_manifest(tokens), require_token_total=True
        )
    assert "does not match the audit receipt" in str(exc.value)


def test_verify_raises_on_token_total_divergence(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    manifest = _manifest(tokens)
    manifest["verified_valid_tokens"] = tokens + 1
    with pytest.raises(SystemExit) as exc:
        download._verify_downloaded_set(
            dest=dest, manifest=manifest, require_token_total=True
        )
    assert "verified_valid_tokens" in str(exc.value)


def test_verify_raises_on_nongreen_receipt(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    receipt_path = dest / "audit" / "sidecar_parquet_audit.json"
    report = json.loads(receipt_path.read_text())
    report["total"]["bad_rows"] = 1
    receipt_path.write_text(json.dumps(report))
    with pytest.raises(SystemExit) as exc:
        download._verify_downloaded_set(
            dest=dest, manifest=_manifest(tokens), require_token_total=True
        )
    assert "not green" in str(exc.value)


def test_verify_raises_when_token_total_required_but_absent(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    with pytest.raises(SystemExit) as exc:
        download._verify_downloaded_set(
            dest=dest,
            manifest=_manifest(tokens, include_token_total=False),
            require_token_total=True,
        )
    assert "missing verified_valid_tokens" in str(exc.value)


def test_verify_default_manifest_still_proves_set(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    # --use-default-manifest (require_token_total=False): there is no uploaded
    # token total, but the set is NOT silently trusted -- it is still proven via
    # the green receipt and per-bucket shard-count reconciliation.
    dest, tokens = _fresh(pristine, tmp_path)
    result = download._verify_downloaded_set(
        dest=dest,
        manifest=_manifest(tokens, include_token_total=False),
        require_token_total=False,
    )
    assert result["buckets_verified"] == 1

    # ...and it still catches a missing-shards SET without any token total.
    dest2, _ = _fresh(pristine, tmp_path / "second")
    next(iter((dest2 / "reindexed" / "code" / BUCKET).glob("*.parquet"))).unlink()
    with pytest.raises(SystemExit):
        download._verify_downloaded_set(
            dest=dest2,
            manifest=_manifest(tokens, include_token_total=False),
            require_token_total=False,
        )
