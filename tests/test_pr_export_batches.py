from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)


MLX_ROOT = Path(__file__).resolve().parents[1]
PR_INGEST = MLX_ROOT / "scripts" / "pr_ingest"
if str(PR_INGEST) not in sys.path:
    sys.path.insert(0, str(PR_INGEST))


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
) -> tuple[Path, Path, Path, str]:
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
    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps({"repos": [{"owner_repo": repo} for repo in repos]}) + "\n",
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
    return store, repo_list, receipt, scan_id


def test_pr_export_all_batches_writes_manifest_and_shard(tmp_path):
    import export_pr_parquet

    store, repo_list, receipt, scan_id = _verified_pr_inputs(
        tmp_path,
        [_record(1), _record(2)],
        stale_records=[_record(3, repo="stale/repo")],
    )

    out = tmp_path / "out"
    manifest = out / "_done.json"
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
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

    assert result["n_shards"] == 2
    assert result["next_offset"] == 2
    assert result["selected_pr_count"] == 2
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
    assert receipt_blob["selected_pr_count"] == 2


def test_pr_export_partial_runs_cannot_publish_global_receipt(tmp_path):
    import export_pr_parquet

    store, repo_list, receipt, _scan_id = _verified_pr_inputs(
        tmp_path / "inputs",
        [_record(1), _record(2)],
    )

    subset_out = tmp_path / "subset"
    subset_args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
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
    store, repo_list, receipt, _scan_id = _verified_pr_inputs(
        tmp_path,
        [_record(99, body=body)],
    )
    args = argparse.Namespace(
        store=str(store),
        pr_completion_receipt=str(receipt),
        repo_list=str(repo_list),
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
