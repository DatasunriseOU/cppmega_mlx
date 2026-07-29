from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.pr_primary_membership import (
    PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME,
    PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA,
    PRIMARY_PR_MEMBERSHIP_POLICY,
    PRIMARY_PR_MEMBERSHIP_RECEIPT_NAME,
    PRIMARY_PR_MEMBERSHIP_SCHEMA,
)

from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)


MLX_ROOT = Path(__file__).resolve().parents[1]
PR_INGEST = MLX_ROOT / "scripts" / "pr_ingest"
if str(PR_INGEST) not in sys.path:
    sys.path.insert(0, str(PR_INGEST))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_primary_membership(
    root: Path,
    *,
    scan_id: str,
    keys: list[tuple[str, int]],
) -> Path:
    root.mkdir(parents=True)
    keys = sorted(keys)
    digest = hashlib.sha256()
    for repo, pr_number in keys:
        encoded = f"{repo}\0{pr_number}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    membership_sha256 = digest.hexdigest()
    schema = pa.schema(
        [
            pa.field("repo", pa.string(), nullable=False),
            pa.field("pr_number", pa.int64(), nullable=False),
        ],
        metadata={
            b"cppmega.primary_pr_membership_schema": (
                PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA.encode("ascii")
            ),
            b"cppmega.primary_pr_membership_policy": (
                PRIMARY_PR_MEMBERSHIP_POLICY.encode("ascii")
            ),
            b"cppmega.primary_pr_membership_scan_id": scan_id.encode("ascii"),
            b"cppmega.primary_pr_membership_sha256": (
                membership_sha256.encode("ascii")
            ),
        },
    )
    artifact = root / PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"repo": repo, "pr_number": pr_number}
                for repo, pr_number in keys
            ],
            schema=schema,
        ),
        artifact,
        compression="zstd",
    )
    count = len(keys)
    membership = {
        "schema": PRIMARY_PR_MEMBERSHIP_SCHEMA,
        "policy": PRIMARY_PR_MEMBERSHIP_POLICY,
        "scan_id": scan_id,
        "commit_artifacts": {
            "schema": "cppmega_primary_commit_artifact_binding_v1",
            "source_composition_sha256": "2" * 64,
            "source_composition_plan_sha256": "3" * 64,
            "buckets": [1024],
            "files": 1,
            "rows": count,
            "byte_size": 1,
            "artifact_set_sha256": "4" * 64,
            "by_bucket": {
                "1024": {
                    "files": 1,
                    "rows": count,
                    "byte_size": 1,
                }
            },
        },
        "rows": count,
        "source_docs": count,
        "source_docs_with_pr_number": count,
        "source_docs_with_pr_discussion": count,
        "ignored_unverified_pr_number_source_docs": 0,
        "source_docs_with_commit_sha": count,
        "selected_pr_count": count,
        "sha_only_matched_source_docs": 0,
        "unmatched_commit_sha_source_docs": 0,
        "selected_membership_sha256": membership_sha256,
        "validation": {
            "source_composition_complete": True,
            "exact_allowlisted_commit_artifacts": True,
            "exact_source_doc_shapes": True,
            "exact_scan_membership": True,
            "direct_pr_sha_conflicts": 0,
        },
        "artifact": {
            "schema": PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA,
            "path": PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME,
            "rows": count,
            "byte_size": artifact.stat().st_size,
            "sha256": _sha256(artifact),
            "membership_sha256": membership_sha256,
        },
    }
    receipt = root / PRIMARY_PR_MEMBERSHIP_RECEIPT_NAME
    receipt.write_text(
        json.dumps(membership, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _record(pr_number: int, *, repo: str = "owner/repo", body: str = "body") -> dict:
    return {
        "repo": repo,
        "pr_number": pr_number,
        "merge_commit_sha": f"sha{pr_number}",
        "pr_title": f"title {pr_number}",
        "pr_body": body,
        "comments": [
            {
                "user": "alice",
                "body": f"comment {pr_number}",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "reviews": [],
        "linked_issues": [],
    }


def _verified_pr_inputs(
    tmp_path: Path,
    records: list[dict],
    *,
    stale_records: list[dict] | None = None,
    primary_pr_numbers: set[int] | None = None,
) -> tuple[Path, Path, Path, str, Path, Path]:
    import pr_store
    from scripts.pr_ingest.graphql_pr_stream import (
        GRAPHQL_MANIFEST_SCHEMA,
        GRAPHQL_QUERY_CONTRACT_SHA256,
    )
    from scripts.pr_ingest.verify_pr_completion import verify_pr_completion

    scan_id = "1" * 64
    store = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(store), create=True)
    try:
        for record in records:
            pr_store.upsert_record(
                conn,
                record,
                scan_id=scan_id,
            )
        for record in stale_records or []:
            pr_store.upsert_record(
                conn,
                record,
                scan_id="2" * 64,
            )
    finally:
        conn.close()

    repos = sorted({str(record["repo"]) for record in records})
    repo_rows = [
        {
            "bare_name": f"repo-{index}",
            "project_identity": repo,
            "owner_repo": repo,
        }
        for index, repo in enumerate(repos)
    ]
    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repos": repo_rows,
                "by_bare_name": {
                    row["bare_name"]: row["project_identity"]
                    for row in repo_rows
                },
                "project_identities": sorted(repos),
                "repo_names": sorted(repos),
                "unresolved": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    repo_counts = {
        repo: sum(1 for record in records if record["repo"] == repo)
        for repo in repos
    }
    manifest = tmp_path / "graphql_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": GRAPHQL_MANIFEST_SCHEMA,
                "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
                "scan_id": scan_id,
                "repos": {
                    repo: {
                        "status": "done",
                        "cursor": None,
                        "prs": count,
                        "initial_total_count": count,
                        "total_count": count,
                        "source_growth_count": 0,
                        "truncated": 0,
                    }
                    for repo, count in sorted(repo_counts.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "pr_completion.json"
    verify_pr_completion(
        repo_list_path=repo_list,
        graphql_manifest_path=manifest,
        store_path=store,
        output_path=receipt,
    )
    membership_root = tmp_path / "primary_membership"
    membership_receipt = _write_primary_membership(
        membership_root,
        scan_id=scan_id,
        keys=[
            (str(record["repo"]), int(record["pr_number"]))
            for record in records
            if (
                primary_pr_numbers is None
                or int(record["pr_number"]) in primary_pr_numbers
            )
        ],
    )
    return (
        store,
        repo_list,
        receipt,
        scan_id,
        membership_root,
        membership_receipt,
    )


def test_pr_export_all_batches_writes_manifest_and_shard(tmp_path):
    import export_pr_parquet

    (
        store,
        repo_list,
        receipt,
        scan_id,
        membership_root,
        membership_receipt,
    ) = _verified_pr_inputs(
        tmp_path,
        [_record(1), _record(2)],
        stale_records=[_record(3, repo="stale/repo")],
        primary_pr_numbers={1},
    )

    out = tmp_path / "out"
    manifest = out / "_done.json"
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        primary_membership_receipt=str(membership_receipt),
        primary_membership_root=str(membership_root),
        output_root=str(out),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=10_000,
        all=True,
        batch_size=1,
        max_shards=None,
        manifest=str(manifest),
        no_resume=False,
        memory_limit_gb=4.0,
    )

    result = export_pr_parquet.export_pr_parquet_batches(args)

    assert result["n_shards"] == 1
    assert result["next_offset"] == 1
    assert result["selected_pr_count"] == 1
    assert result["scan_id"] == scan_id
    shard = (
        out
        / "1024"
        / f"pr_discussions_all_{scan_id[:12]}_00000000.parquet"
    )
    assert shard.exists()
    schema = pq.read_schema(shard)
    assert schema.metadata is not None
    assert schema.metadata[
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    ] == str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii")
    identities = pq.read_table(shard, columns=[SYMBOL_IDENTITIES_COLUMN]).column(0)
    assert identities.to_pylist() == [[]]
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    assert blob["schema"] == export_pr_parquet.EXPORT_MANIFEST_SCHEMA
    assert blob["status"] == "complete"
    assert "all:0" in blob["done"]
    receipt_blob = json.loads(
        (out / "export_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_blob["schema"] == export_pr_parquet.EXPORT_RECEIPT_SCHEMA
    assert receipt_blob["selected_pr_count"] == 1
    assert receipt_blob["primary_membership"]["selected_pr_count"] == 1
    assert receipt_blob["validation"][
        "portable_primary_membership_verified"
    ] is True


def test_pr_export_partial_runs_cannot_publish_global_receipt(tmp_path):
    import export_pr_parquet

    (
        store,
        repo_list,
        receipt,
        _scan_id,
        membership_root,
        membership_receipt,
    ) = _verified_pr_inputs(
        tmp_path / "inputs",
        [_record(1), _record(2)],
    )

    subset_out = tmp_path / "subset"
    subset_args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        primary_membership_receipt=str(membership_receipt),
        primary_membership_root=str(membership_root),
        output_root=str(subset_out),
        target_lengths="1024",
        repo=None,
        offset=1,
        limit=10_000,
        all=True,
        batch_size=1,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )
    subset_result = export_pr_parquet.export_pr_parquet_batches(subset_args)

    assert "completion_receipt" not in subset_result
    assert not (subset_out / "export_receipt.json").exists()
    subset_manifest = json.loads(
        (subset_out / "_done.json").read_text(encoding="utf-8")
    )
    assert subset_manifest["status"] == "selection_complete"
    assert subset_manifest["completed_pr_count"] == 1

    bounded_out = tmp_path / "bounded"
    bounded_args = argparse.Namespace(
        **{
            **vars(subset_args),
            "output_root": str(bounded_out),
            "offset": 0,
            "max_shards": 1,
        }
    )
    bounded_result = export_pr_parquet.export_pr_parquet_batches(bounded_args)

    assert "completion_receipt" not in bounded_result
    assert not (bounded_out / "export_receipt.json").exists()
    bounded_manifest = json.loads(
        (bounded_out / "_done.json").read_text(encoding="utf-8")
    )
    assert "status" not in bounded_manifest
    assert "completed_pr_count" not in bounded_manifest


def test_pr_export_losslessly_splits_large_discussion(tmp_path):
    import export_pr_parquet

    body = "\n".join(
        f"review item {index}: preserve diagnostic and parser state"
        for index in range(5_000)
    )
    (
        store,
        repo_list,
        receipt,
        _scan_id,
        membership_root,
        membership_receipt,
    ) = _verified_pr_inputs(
        tmp_path,
        [_record(99, body=body)],
    )
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        primary_membership_receipt=str(membership_receipt),
        primary_membership_root=str(membership_root),
        output_root=str(tmp_path / "out"),
        target_lengths="1024,2048,4096,8192,16384",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=10_000,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )

    result = export_pr_parquet.export_pr_parquet(args)

    assert result["materialize_stats"]["split_input_docs"] == 1
    assert result["materialize_stats"]["dropped_input_docs"] == 0
    assert sum(item["rows"] for item in result["lengths"].values()) > 1


def test_pr_export_rejects_non_zstd_primary_membership(tmp_path):
    import export_pr_parquet

    (
        store,
        repo_list,
        receipt,
        _scan_id,
        membership_root,
        membership_receipt,
    ) = _verified_pr_inputs(tmp_path, [_record(1)])
    artifact = membership_root / PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME
    table = pq.read_table(artifact)
    pq.write_table(table, artifact, compression="snappy")
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        primary_membership_receipt=str(membership_receipt),
        primary_membership_root=str(membership_root),
        output_root=str(tmp_path / "out"),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=1,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )

    with pytest.raises(RuntimeError, match="must use ZSTD"):
        export_pr_parquet.export_pr_parquet(args)


def test_pr_export_rejects_membership_key_outside_verified_scan(tmp_path):
    import export_pr_parquet

    (
        store,
        repo_list,
        receipt,
        scan_id,
        _membership_root,
        _membership_receipt,
    ) = _verified_pr_inputs(tmp_path, [_record(1)])
    bad_root = tmp_path / "bad_membership"
    bad_receipt = _write_primary_membership(
        bad_root,
        scan_id=scan_id,
        keys=[("owner/repo", 999)],
    )
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
        primary_membership_receipt=str(bad_receipt),
        primary_membership_root=str(bad_root),
        output_root=str(tmp_path / "out"),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=1,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )

    with pytest.raises(RuntimeError, match="absent from the exact verified scan"):
        export_pr_parquet.export_pr_parquet(args)
