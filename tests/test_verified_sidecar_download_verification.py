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
  * the planning-only default manifest is rejected by the real verification
    path; only a schema-bound v2 inventory can certify a download.
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
from scripts.sidecar_manifest_contract import (
    AUDIT_FILENAME,
    AUDIT_REMOTE,
    AUDIT_SCHEMA,
    GRAPH_CONTRACT,
    MANIFEST_SCHEMA,
    OBJECTIVE_CONTRACT,
    finalize_manifest,
    inventory_directory,
    inventory_sha256,
    selection_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_BUCKETS = ("1024", "2048", "4096", "8192")
COMMIT_BUCKETS = ("1024", "2048", "4096", "8192", "16384")
SHARDS = 2
PROFILE_SELECTIONS = tuple(
    [(f"parquet/code/{bucket}", f"reindexed/code/{bucket}") for bucket in CODE_BUCKETS]
    + [
        (f"parquet/commits/{bucket}", f"reindexed_commits/{bucket}")
        for bucket in COMMIT_BUCKETS
    ]
    + [(AUDIT_REMOTE, "audit")]
)


def _write_green_shard(path: Path, *, length: int, source_id: int) -> None:
    """Write a complete CASE5 row that passes the real audit gate."""
    input_ids = [1 + (index % 4) for index in range(length)]
    identity = source_identity({"source_path": f"fixture-{source_id}.cpp"})
    row = {
        "input_ids": input_ids,
        "target_ids": input_ids[1:] + [0],
        "loss_mask": [1] * (length - 1) + [0],
        "doc_ids": [1] * length,
        "valid_token_count": length,
        "trained_token_count": length - 1,
        "num_docs": 1,
        "slack_tokens": 0,
        "source_doc_ids": [source_id],
        "source_doc_token_lengths": [length],
        "source_platform_ids": [[7]],
        "source_repo_stable_ids": ["9"],
        "source_filepath_stable_ids": [str(source_id)],
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
        "token_source_doc_ids": [source_id] * length,
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
    assert "source_doc_token_lengths" in table.column_names
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_verified_tree(dest: Path) -> int:
    """Lay out a downloaded set + a REAL green audit receipt; return valid_tokens."""
    source_id = 7
    for remote, local_relative in PROFILE_SELECTIONS:
        if not remote.startswith("parquet/"):
            continue
        kind, bucket = remote.split("/")[1:]
        shard_count = SHARDS if kind == "code" and bucket == "1024" else 1
        for index in range(shard_count):
            _write_green_shard(
                dest / local_relative / f"shard{index}.parquet",
                length=int(bucket),
                source_id=source_id,
            )
            source_id += 1
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
        ",".join((*CODE_BUCKETS, *COMMIT_BUCKETS)),
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
    assert report["by_kind_bucket"]["code/1024"]["files"] == SHARDS
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


def _manifest(
    dest: Path, tokens: int, *, include_token_total: bool = True
) -> dict:
    selections: list[dict[str, object]] = []
    for remote, local_relative in PROFILE_SELECTIONS:
        local = dest / local_relative
        inventory = inventory_directory(local, remote=remote)
        selections.append(
            {"remote": remote, "local": local_relative, **inventory}
        )
    audit_selection = next(
        selection for selection in selections if selection["remote"] == AUDIT_REMOTE
    )
    audit_record = next(
        record
        for record in audit_selection["files"]
        if record["path"] == AUDIT_FILENAME
    )
    payload: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "bucket": "bucket",
        "prefix": "prefix/run-1",
        "endpoint_url": "https://storage.example",
        "profile": "code_commits_integrated_pr",
        "standalone_pr_included": False,
        "selections": selections,
        "inventory_sha256": inventory_sha256(selections),
        "selection_policy": selection_policy(
            [str(selection["remote"]) for selection in selections],
            include_standalone_pr=False,
        ),
        "audit_receipt": {
            "schema": AUDIT_SCHEMA,
            "status": "verified",
            "remote": AUDIT_REMOTE,
            "path": AUDIT_FILENAME,
            "sha256": audit_record["sha256"],
        },
        "graph_contract": GRAPH_CONTRACT,
        "objective_contract": OBJECTIVE_CONTRACT,
    }
    if include_token_total:
        payload["verified_valid_tokens"] = tokens
    return finalize_manifest(payload)


def test_verify_passes_on_complete_green_set(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    result = download._verify_downloaded_set(
        dest=dest, manifest=_manifest(dest, tokens), require_token_total=True
    )
    assert result["bad_files"] == 0
    assert result["bad_rows"] == 0
    assert result["verified_valid_tokens"] == tokens
    assert result["buckets_verified"] == 9


def test_verify_raises_on_missing_shard_set(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    manifest = _manifest(dest, tokens)
    # Drop a shard to simulate an incomplete remote prefix that `aws s3 sync`
    # would otherwise treat as a successful (return-0) download.
    shards = sorted((dest / "reindexed" / "code" / "1024").glob("*.parquet"))
    shards[0].unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        download._verify_downloaded_set(
            dest=dest, manifest=manifest, require_token_total=True
        )


def test_verify_raises_on_token_total_divergence(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    manifest = _manifest(dest, tokens)
    manifest["verified_valid_tokens"] = tokens + 1
    manifest = finalize_manifest(manifest)
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
    report["by_kind_bucket"]["code/1024"]["bad_rows"] = 1
    receipt_path.write_text(json.dumps(report))
    manifest = _manifest(dest, tokens)
    with pytest.raises(ValueError) as exc:
        download._verify_downloaded_set(
            dest=dest, manifest=manifest, require_token_total=True
        )
    assert "code/1024" in str(exc.value)


def test_verify_raises_when_token_total_required_but_absent(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    with pytest.raises(ValueError, match="verified_valid_tokens must be positive"):
        download._verify_downloaded_set(
            dest=dest,
            manifest=_manifest(dest, tokens, include_token_total=False),
            require_token_total=True,
        )


def test_verify_rejects_planning_manifest_and_still_checks_v2_inventory(
    pristine: tuple[Path, int], tmp_path: Path
) -> None:
    dest, tokens = _fresh(pristine, tmp_path)
    with pytest.raises(ValueError, match="unsupported sidecar manifest schema"):
        download._verify_downloaded_set(
            dest=dest,
            manifest=download._default_manifest(
                "bucket", "prefix", "https://storage.example"
            ),
            require_token_total=False,
        )

    manifest = _manifest(dest, tokens)
    result = download._verify_downloaded_set(
        dest=dest,
        manifest=manifest,
        require_token_total=False,
    )
    assert result["buckets_verified"] == 9

    # The compatibility argument cannot bypass exact inventory verification.
    dest2, _ = _fresh(pristine, tmp_path / "second")
    manifest2 = _manifest(dest2, tokens)
    next(
        iter((dest2 / "reindexed" / "code" / "1024").glob("*.parquet"))
    ).unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        download._verify_downloaded_set(
            dest=dest2,
            manifest=manifest2,
            require_token_total=False,
        )
