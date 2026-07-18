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

from cppmega_mlx.data.source_identity import source_identity
from scripts import download_verified_sidecar_from_nebius_s3 as download
from scripts.nanochat_data.pack_enriched_rows import PACKED_ROW_OUTPUT_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "4"  # rows are this long; tiny so the real audit stays fast
SHARDS = 2


def _write_green_shard(path: Path) -> None:
    """Write a complete CASE5 row that passes the real audit gate."""
    length = int(BUCKET)
    input_ids = list(range(10, 10 + length))
    identity = source_identity({"source_path": "fixture.cpp"})
    row = {
        "input_ids": input_ids,
        "target_ids": input_ids[1:] + [0],
        "loss_mask": [1] * (length - 1) + [0],
        "doc_ids": [1] * length,
        "valid_token_count": length,
        "trained_token_count": length - 1,
        "num_docs": 1,
        "slack_tokens": 0,
        "source_doc_ids": [1],
        "source_doc_token_lengths": [length],
        "source_platform_ids": [[7]],
        "source_repo_stable_ids": ["9"],
        "source_filepath_stable_ids": ["11"],
        "source_file_local_commit_indices": [0],
        "platform_ids": [7],
        "token_platform_ids": [0] * length,
        "token_structure_ids": [1] * length,
        "token_dep_levels": [0] * length,
        "token_ast_depth": [1] * length,
        "token_sibling_index": list(range(length)),
        "token_ast_node_type": [3] * length,
        "token_symbol_ids": [0] * length,
        "token_call_targets": [0] * length,
        "token_type_refs": [0] * length,
        "token_domain_ids": [0] * length,
        "token_role_ids": [0] * length,
        "token_entity_ids": [0] * length,
        "token_scope_ids": [0] * length,
        "token_source_doc_ids": [1] * length,
        "token_source_identity_ids": [identity.source_identity_id] * length,
        "token_confidence_ids": [0] * length,
        "token_def_use": [0] * length,
        "token_change_mask_pre": [0] * length,
        "token_change_mask_post": [0] * length,
        "hunk_id_per_token": [-1] * length,
        "edit_op_per_token": [0] * length,
        "token_chunk_starts": [0],
        "token_chunk_ends": [length],
        "token_chunk_kinds": [1],
        "token_chunk_dep_levels": [0],
        "token_call_edges": [],
        "token_type_edges": [],
        "token_domain_edges": [],
        "token_build_edges": [],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
        "source_identity_registry": [identity.as_dict()],
        "changed_chunk_ids": [],
        "changed_chunk_spans": [],
    }
    table = pa.Table.from_pylist([row], schema=PACKED_ROW_OUTPUT_SCHEMA)
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
    report["by_kind_bucket"][f"code/{BUCKET}"]["bad_rows"] = 1
    receipt_path.write_text(json.dumps(report))
    with pytest.raises(SystemExit) as exc:
        download._verify_downloaded_set(
            dest=dest, manifest=_manifest(tokens), require_token_total=True
        )
    assert f"code/{BUCKET}" in str(exc.value)


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
