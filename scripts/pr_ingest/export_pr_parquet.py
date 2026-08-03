#!/usr/bin/env python3
"""Export PR discussions from prs.sqlite into route-by-fit training parquet.

This is the standalone PR stream companion to the commit pipeline. Commit docs
already inject ``record['pr_discussion']`` at the head of PRE/POST/diff samples;
this script emits the same PR discussion content as its own document stream for
curriculum mixes that want explicit ``pr`` batches. A PR is eligible only when
the canonical cppmega primary-membership receipt binds it to an allowlisted
primary commit document. Merely existing in the GraphQL scan is not sufficient.

The output is intentionally compatible with the existing conveyor layout:

    outputs/reindexed_pr/{1024,2048,4096,...}/pr_discussions_<repo>_<offset>.parquet

RULE #1: every stage uses the existing materializer/packer path and raises on
missing input, empty output, or malformed store rows. There is no fallback path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_local_symbol_identity() -> ModuleType:
    module_path = REPO_ROOT / "cppmega_mlx" / "data" / "symbol_identity.py"
    module = importlib.import_module("cppmega_mlx.data.symbol_identity")
    loaded_path = Path(getattr(module, "__file__", "")).resolve()
    if loaded_path != module_path.resolve():
        raise ImportError(
            "cppmega_mlx.data.symbol_identity resolved outside this checkout: "
            f"loaded={loaded_path} expected={module_path}"
        )
    return module


_symbol_identity = _load_local_symbol_identity()
SYMBOL_IDENTITIES_COLUMN = _symbol_identity.SYMBOL_IDENTITIES_COLUMN
SYMBOL_IDENTITY_SCHEMA_VERSION = _symbol_identity.SYMBOL_IDENTITY_SCHEMA_VERSION

import streaming_reindex as sr  # noqa: E402
from streaming_reindex_commits import route_by_fit, recompress_zstd_max  # noqa: E402
from scripts.pr_ingest.pr_store import connect, get_by_pr  # noqa: E402
from scripts.pr_ingest.render_discussion import render_discussion  # noqa: E402
from cppmega_mlx.data.pr_primary_membership import (  # noqa: E402
    PRIMARY_PR_MEMBERSHIP_TABLE,
    load_primary_pr_membership,
    revalidate_primary_pr_membership,
)


DEFAULT_STORE = REPO_ROOT / "outputs" / "pr_ingest" / "prs.sqlite"
DEFAULT_REPO_LIST = REPO_ROOT / "outputs" / "pr_ingest" / "repo_list.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reindexed_pr"
ZSTD_LEVELS = (1024, 2048, 4096, 8192, 16384)
EXPORT_MANIFEST_SCHEMA = "cppmega_pr_parquet_export_manifest_v3"
EXPORT_RECEIPT_SCHEMA = "cppmega_pr_case5_export_v2"
MATERIALIZED_ROW_RESERVED_TOKENS = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pipeline():
    return sr, route_by_fit, recompress_zstd_max


def _lossless_stage_materialize(
    shard_name: str,
    jsonl: Path,
    work: Path,
    *,
    memory_limit_gb: float,
    max_fixed_tokens: int,
) -> tuple[Path, dict[str, int | str]]:
    max_raw_tokens = int(max_fixed_tokens) - MATERIALIZED_ROW_RESERVED_TOKENS
    if max_raw_tokens <= 0:
        raise ValueError(
            f"largest target length is too small: {max_fixed_tokens}"
        )
    tok = work / f"{shard_name}.tok.parquet"
    sr.run_checked(
        shard_name,
        "materialize",
        [
            str(sr.VENV_PYTHON),
            str(sr.MATERIALIZER),
            "--input-file",
            str(jsonl),
            "--output-file",
            str(tok),
            "--tokenizer-path",
            str(sr.TOKENIZER_PATH),
            "--materialize-tokenized-enriched",
            "--overflow-policy",
            "split",
            "--size",
            str(max_raw_tokens),
            "--memory-limit-gb",
            str(memory_limit_gb),
        ],
        log_path=work / f"{shard_name}.materialize.log",
    )
    if not tok.is_file() or tok.stat().st_size == 0:
        raise RuntimeError(f"empty tokenized PR parquet: {tok}")
    source_counts: Counter[str] = Counter()
    materialized_rows = 0
    materialized_output_tokens = 0
    max_materialized_tokens = 0
    parquet = pq.ParquetFile(tok)
    for batch in parquet.iter_batches(
        batch_size=128,
        columns=["source_doc_id", "token_ids"],
    ):
        source_ids = [str(value) for value in batch.column(0).to_pylist()]
        if any(not value for value in source_ids):
            raise RuntimeError("materialized PR rows lost source_doc_id")
        source_counts.update(source_ids)
        token_lengths = [
            len(tokens or [])
            for tokens in batch.column(1).to_pylist()
        ]
        materialized_rows += batch.num_rows
        materialized_output_tokens += sum(token_lengths)
        max_materialized_tokens = max(
            max_materialized_tokens,
            max(token_lengths, default=0),
        )
    with jsonl.open("r", encoding="utf-8") as handle:
        docs_in = sum(1 for line in handle if line.strip())
    source_docs_emitted = len(source_counts)
    if source_docs_emitted != docs_in:
        raise RuntimeError(
            "lossless PR materialization dropped or merged source documents: "
            f"input={docs_in} emitted={source_docs_emitted}"
        )
    if max_materialized_tokens > max_fixed_tokens:
        raise RuntimeError(
            "materialized PR row exceeds the largest fixed bucket: "
            f"{max_materialized_tokens} > {max_fixed_tokens}"
        )
    stats: dict[str, int | str] = {
        "schema": "cppmega_pr_lossless_materialize_stats_v1",
        "docs_in": docs_in,
        "docs_out": materialized_rows,
        "source_docs_emitted": source_docs_emitted,
        "split_input_docs": sum(1 for count in source_counts.values() if count > 1),
        "split_output_docs": sum(count for count in source_counts.values() if count > 1),
        "dropped_input_docs": docs_in - source_docs_emitted,
        "materialized_rows": materialized_rows,
        "materialized_output_tokens": materialized_output_tokens,
        "max_materialized_tokens": max_materialized_tokens,
        "output_file_size": tok.stat().st_size,
        "output_file_sha256": _sha256_file(tok),
    }
    return tok, stats


def _summarize_materialize_stats(entries: list[dict]) -> dict[str, int]:
    fields = (
        "docs_in",
        "docs_out",
        "source_docs_emitted",
        "split_input_docs",
        "split_output_docs",
        "dropped_input_docs",
        "materialized_rows",
        "materialized_output_tokens",
        "output_file_size",
    )
    totals = {"receipts": 0, **{field: 0 for field in fields}}
    totals["max_materialized_tokens"] = 0
    for info in entries:
        stats = info.get("materialize_stats")
        if not isinstance(stats, dict):
            continue
        if stats.get("schema") != "cppmega_pr_lossless_materialize_stats_v1":
            raise RuntimeError("unexpected PR materialize_stats schema")
        totals["receipts"] += 1
        for field in fields:
            totals[field] += int(stats[field])
        totals["max_materialized_tokens"] = max(
            totals["max_materialized_tokens"],
            int(stats["max_materialized_tokens"]),
        )
    return totals


def _completion_receipt_schema(receipt_path: Path) -> str:
    max_bytes = 4 * 1024 * 1024
    try:
        with receipt_path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as exc:
        raise RuntimeError(
            f"cannot read PR completion receipt {receipt_path}: {exc}"
        ) from exc
    if len(payload) > max_bytes:
        raise RuntimeError(
            f"PR completion receipt exceeds the 4 MiB metadata bound: {receipt_path}"
        )
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid PR completion receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict) or not isinstance(receipt.get("schema"), str):
        raise RuntimeError("PR completion receipt lacks a schema discriminator")
    return str(receipt["schema"])


def _load_completion_api(schema: str):
    if schema == "cppmega_pr_completion_v2":
        try:
            from streaming_reindex_commits import (
                load_pr_completion_binding,
                revalidate_pr_completion_binding,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PR parquet export requires root scripts/streaming_reindex_commits.py "
                "to verify cppmega_pr_completion_v2; no unverified fallback is "
                "permitted"
            ) from exc
        return load_pr_completion_binding, revalidate_pr_completion_binding
    if schema == "cppmega_gitlab_mr_completion_v1":
        try:
            from scripts.pr_ingest.gitlab_mr_stream import (
                load_gitlab_completion_binding,
                revalidate_gitlab_completion_binding,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PR parquet export requires scripts/pr_ingest/gitlab_mr_stream.py "
                "to verify cppmega_gitlab_mr_completion_v1; no unverified fallback "
                "is permitted"
            ) from exc
        return load_gitlab_completion_binding, revalidate_gitlab_completion_binding
    raise RuntimeError(f"unsupported PR completion schema: {schema!r}")


def _load_pr_completion(args: argparse.Namespace) -> dict[str, object]:
    raw_receipt = getattr(args, "pr_completion_receipt", None)
    raw_repo_list = getattr(args, "repo_list", None)
    if not raw_receipt:
        raise ValueError("--pr-completion-receipt is required")
    if not raw_repo_list:
        raise ValueError("--repo-list is required")
    receipt_path = Path(raw_receipt)
    load_binding, _revalidate = _load_completion_api(
        _completion_receipt_schema(receipt_path)
    )
    return load_binding(
        receipt_path,
        pr_store=Path(args.store),
        repo_list=Path(raw_repo_list),
    )


def _revalidate_pr_completion(
    args: argparse.Namespace,
    binding: dict[str, object],
) -> None:
    schema = binding.get("schema")
    if not isinstance(schema, str):
        raise RuntimeError("verified PR completion binding lacks its schema")
    _load_binding, revalidate = _load_completion_api(schema)
    revalidate(
        binding,
        Path(args.pr_completion_receipt),
        pr_store=Path(args.store),
        repo_list=Path(args.repo_list),
    )


def _comment_safe(text: str) -> str:
    return text.replace("*/", "* /")


def _render_training_doc(rec: dict) -> str:
    # The standalone PR corpus must retain the complete assembled discussion.
    discussion = render_discussion(
        rec,
        max_comments=len(rec.get("comments") or []),
        max_reviews=len(rec.get("reviews") or []),
        max_body_chars=sys.maxsize,
        max_item_chars=sys.maxsize,
        max_total_chars=sys.maxsize,
    )
    lines = [
        "/*",
        "@doc_type pr_discussion",
        f"@repo {rec['repo']}",
        f"@pr {rec['pr_number']}",
    ]
    sha = rec.get("merge_commit_sha")
    if sha:
        lines.append(f"@merge_commit_sha {sha}")
    for field in ("state", "author", "created_at", "merged_at"):
        value = rec.get(field)
        if value not in (None, ""):
            lines.append(f"@{field} {_comment_safe(str(value))}")
    lines.extend(["@discussion", _comment_safe(discussion), "*/", ""])
    return "\n".join(lines)


def _iter_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    scan_id: str,
    offset: int,
    limit: int | None,
):
    sql = (
        "SELECT p.repo, p.pr_number FROM prs AS p "
        f"JOIN {PRIMARY_PR_MEMBERSHIP_TABLE} AS m "
        "ON m.repo=p.repo AND m.pr_number=p.pr_number "
        "WHERE p.scan_id=?"
    )
    params: list[object] = [scan_id]
    if repo:
        sql += " AND p.repo=?"
        params.append(repo)
    sql += " ORDER BY p.repo, p.pr_number LIMIT ? OFFSET ?"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    yield from conn.execute(sql, params)


def _count_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    scan_id: str,
    offset: int,
    limit: int | None,
) -> int:
    sql = (
        "SELECT COUNT(*) AS n FROM (SELECT 1 FROM prs AS p "
        f"JOIN {PRIMARY_PR_MEMBERSHIP_TABLE} AS m "
        "ON m.repo=p.repo AND m.pr_number=p.pr_number "
        "WHERE p.scan_id=?"
    )
    params: list[object] = [scan_id]
    if repo:
        sql += " AND p.repo=?"
        params.append(repo)
    sql += " ORDER BY p.repo, p.pr_number LIMIT ? OFFSET ?)"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    return int(conn.execute(sql, params).fetchone()["n"])


def _write_pr_jsonl(
    conn: sqlite3.Connection,
    out_jsonl: Path,
    *,
    repo: str | None,
    scan_id: str,
    offset: int,
    limit: int | None,
) -> int:
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in _iter_pr_keys(
            conn,
            repo=repo,
            scan_id=scan_id,
            offset=offset,
            limit=limit,
        ):
            rec = get_by_pr(
                conn,
                row["repo"],
                int(row["pr_number"]),
                scan_id=scan_id,
            )
            if rec is None:
                raise RuntimeError(
                    f"PR key disappeared during export: {row['repo']}#{row['pr_number']}"
                )
            text = _render_training_doc(rec)
            payload = {
                "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
                SYMBOL_IDENTITIES_COLUMN: [],
                "text": text,
                "source_text": text,
                "repo": rec["repo"],
                "filepath": f"PR#{rec['pr_number']}",
                "doc_type": "pr_discussion",
                "language": "pr",
                "pr_number": int(rec["pr_number"]),
                "commit_hash": rec.get("merge_commit_sha") or "",
                "source_doc_id": (
                    f"{scan_id}:{rec['repo']}#{int(rec['pr_number'])}"
                ),
                "constituent_provenance_json": json.dumps(
                    [{
                        "kind": "pr_discussion",
                        "repo": rec["repo"],
                        "pr_number": int(rec["pr_number"]),
                        "merge_commit_sha": rec.get("merge_commit_sha") or "",
                        "scan_id": scan_id,
                        "state": rec.get("state") or "",
                        "author": rec.get("author") or "",
                        "created_at": rec.get("created_at") or "",
                        "merged_at": rec.get("merged_at") or "",
                    }],
                    separators=(",", ":"),
                ),
            }
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    if n == 0:
        raise RuntimeError(
            f"PR export produced zero rendered docs (repo={repo!r}, offset={offset}, limit={limit})"
        )
    return n


def _append_output(
    shard_name: str,
    packed: Path,
    target_length: int,
    output_root: Path,
    *,
    pipeline,
) -> dict:
    sr, _route_by_fit, recompress_zstd_max = pipeline
    out_dir = output_root / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{shard_name}.parquet"
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite an unverified PR parquet artifact: {dest}"
        )
    staged = out_dir / f".{dest.name}.staging-{os.getpid()}"
    if staged.exists() or staged.is_symlink():
        raise FileExistsError(f"stale PR parquet staging artifact: {staged}")
    try:
        shutil.copy2(packed, staged)
        recompress_zstd_max(staged)
        os.replace(staged, dest)
    finally:
        staged.unlink(missing_ok=True)
    stats = sr._parquet_stats(dest, target_length)
    return {
        **stats,
        "path": str(dest.resolve()),
        "byte_size": dest.stat().st_size,
        "sha256": _sha256_file(dest),
    }


def _repo_slug(repo: str | None) -> str:
    if not repo:
        return "all"
    return repo.replace("/", "__").replace(":", "_")


def _shard_name(repo: str | None, scan_id: str, offset: int) -> str:
    return (
        f"pr_discussions_{_repo_slug(repo)}_{scan_id[:12]}_"
        f"{int(offset):08d}"
    )


def _manifest_input(
    args: argparse.Namespace,
    binding: dict[str, object],
    primary_membership: dict[str, object],
    primary_membership_input: dict[str, object],
    lengths: tuple[int, ...],
) -> dict[str, object]:
    return {
        "pr_completion": binding,
        "primary_membership": primary_membership,
        "primary_membership_input": primary_membership_input,
        "exporter_script_sha256": _sha256_file(Path(__file__).resolve()),
        "pr_completion_receipt": str(
            Path(args.pr_completion_receipt).resolve()
        ),
        "pr_store": str(Path(args.store).resolve()),
        "repo_list": str(Path(args.repo_list).resolve()),
        "primary_membership_receipt": str(
            Path(args.primary_membership_receipt).resolve()
        ),
        "primary_membership_root": str(
            Path(args.primary_membership_root).resolve()
        ),
        "repo": args.repo,
        "start_offset": int(args.offset),
        "target_lengths": list(lengths),
        "batch_size": int(args.batch_size),
    }


def _load_manifest(
    path: Path,
    *,
    expected_input: dict[str, object],
) -> dict:
    if not path.exists():
        return {
            "schema": EXPORT_MANIFEST_SCHEMA,
            "input": expected_input,
            "done": {},
            "failed": {},
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != EXPORT_MANIFEST_SCHEMA
        or manifest.get("input") != expected_input
        or not isinstance(manifest.get("done"), dict)
        or not isinstance(manifest.get("failed"), dict)
    ):
        raise RuntimeError(
            f"{path}: existing PR export manifest is legacy, malformed, or "
            "bound to different immutable inputs"
        )
    return manifest


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _verify_done_artifacts(info: dict) -> None:
    lengths = info.get("lengths")
    if not isinstance(lengths, dict) or not lengths:
        raise RuntimeError("completed PR shard lacks length artifacts")
    for length, artifact in lengths.items():
        if not isinstance(artifact, dict):
            raise RuntimeError(f"completed PR shard {length} artifact is malformed")
        path = Path(str(artifact.get("path", "")))
        expected_sha256 = artifact.get("sha256")
        expected_size = artifact.get("byte_size")
        if (
            not path.is_file()
            or not isinstance(expected_sha256, str)
            or path.stat().st_size != expected_size
            or _sha256_file(path) != expected_sha256
        ):
            raise RuntimeError(
                f"completed PR shard artifact drifted or disappeared: {path}"
            )


def _publish_complete_receipt(
    *,
    output_root: Path,
    manifest_path: Path,
    manifest: dict,
    binding: dict[str, object],
    primary_membership: dict[str, object],
    primary_membership_input: dict[str, object],
    scan_id: str,
    target_lengths: tuple[int, ...],
    selected_pr_count: int,
) -> dict[str, object]:
    done = manifest["done"]
    artifacts: list[dict[str, object]] = []
    rendered_docs = 0
    for done_key, info in sorted(done.items()):
        if not isinstance(info, dict):
            raise RuntimeError(f"malformed completed PR export shard: {done_key}")
        _verify_done_artifacts(info)
        docs = info.get("rendered_docs")
        if not isinstance(docs, int) or isinstance(docs, bool) or docs < 1:
            raise RuntimeError(
                f"completed PR export shard has invalid rendered_docs: {done_key}"
            )
        rendered_docs += docs
        for length, artifact in sorted(
            info["lengths"].items(),
            key=lambda item: int(item[0]),
        ):
            path = Path(str(artifact["path"])).resolve()
            try:
                relative = path.relative_to(output_root.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"PR export artifact is outside output root: {path}"
                ) from exc
            bucket = int(length)
            if relative.parts[:1] != (str(bucket),):
                raise RuntimeError(
                    f"PR export artifact bucket path drifted: {relative}"
                )
            artifacts.append(
                {
                    "path": relative.as_posix(),
                    "bucket": bucket,
                    "rows": int(artifact["rows"]),
                    "valid_tokens": int(artifact["valid_tokens"]),
                    "pad_tokens": int(artifact["pad_tokens"]),
                    "capacity_tokens": int(artifact["capacity_tokens"]),
                    "byte_size": int(artifact["byte_size"]),
                    "sha256": str(artifact["sha256"]),
                }
            )
    if rendered_docs != selected_pr_count:
        raise RuntimeError(
            "PR export document conservation failed: "
            f"rendered={rendered_docs} selected={selected_pr_count}"
        )
    observed_buckets: set[int] = set()
    for item in artifacts:
        raw_bucket = item.get("bucket")
        if isinstance(raw_bucket, bool) or not isinstance(raw_bucket, int):
            raise RuntimeError("PR export artifact bucket is malformed")
        observed_buckets.add(raw_bucket)
    if observed_buckets != set(target_lengths):
        raise RuntimeError(
            "PR export did not publish every requested CASE5 bucket: "
            f"observed={sorted(observed_buckets)} "
            f"requested={list(target_lengths)}"
        )
    manifest_sha256 = _sha256_file(manifest_path)
    receipt: dict[str, object] = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "status": "complete",
        "source": "pr",
        "scan_id": scan_id,
        "pr_completion": binding,
        "primary_membership": primary_membership,
        "primary_membership_input": primary_membership_input,
        "exporter_script_sha256": _sha256_file(Path(__file__).resolve()),
        "target_lengths": list(target_lengths),
        "selected_pr_count": selected_pr_count,
        "rendered_docs": rendered_docs,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
        "artifacts": artifacts,
        "validation": {
            "exact_scan_membership": True,
            "exact_primary_commit_membership": True,
            "portable_primary_membership_verified": True,
            "input_revalidated_after_export": True,
            "document_conservation": True,
            "all_requested_buckets_present": True,
            "artifact_hashes_verified": True,
        },
    }
    receipt_path = output_root / "export_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise RuntimeError(
                f"existing PR export receipt differs from completed export: "
                f"{receipt_path}"
            )
        return existing
    _save_manifest(receipt_path, receipt)
    return receipt


def export_pr_parquet(
    args: argparse.Namespace,
    *,
    completion_binding: dict[str, object] | None = None,
    revalidate_at_finish: bool = True,
    primary_membership: dict[str, object] | None = None,
    primary_membership_input: dict[str, object] | None = None,
    connection: sqlite3.Connection | None = None,
) -> dict:
    store = Path(args.store)
    if not store.exists():
        raise FileNotFoundError(f"--store does not exist: {store}")
    output_root = Path(args.output_root)
    lengths = tuple(sorted(int(x) for x in args.target_lengths.split(",") if x.strip()))
    if not lengths:
        raise ValueError("--target-lengths produced no lengths")
    binding = (
        completion_binding
        if completion_binding is not None
        else _load_pr_completion(args)
    )
    scan_id = str(binding["scan_id"])

    pipeline = _load_pipeline()
    sr, route_by_fit, _recompress_zstd_max = pipeline
    owns_connection = connection is None
    conn = (
        connect(str(store), create=False, readonly=True)
        if connection is None
        else connection
    )
    try:
        if primary_membership is None:
            (
                primary_membership,
                primary_membership_input,
            ) = load_primary_pr_membership(
                conn,
                receipt_path=Path(args.primary_membership_receipt),
                input_root=Path(args.primary_membership_root),
                scan_id=scan_id,
            )
        elif primary_membership_input is None:
            raise ValueError(
                "provided primary membership requires its input binding"
            )
        shard_name = _shard_name(args.repo, scan_id, int(args.offset))
        with tempfile.TemporaryDirectory(prefix="pr_parquet_export_") as tmp:
            work = Path(tmp)
            jsonl = work / f"{shard_name}.jsonl"
            n_docs = _write_pr_jsonl(
                conn,
                jsonl,
                repo=args.repo,
                scan_id=scan_id,
                offset=int(args.offset),
                limit=args.limit,
            )
            tok, materialize_stats = _lossless_stage_materialize(
                shard_name,
                jsonl,
                work,
                memory_limit_gb=float(args.memory_limit_gb),
                max_fixed_tokens=max(lengths),
            )
            routed = route_by_fit(
                tok,
                lengths,
                work / "routed",
                repo=args.repo,
            )
            if not routed:
                raise RuntimeError(f"no PR docs routed for {jsonl}")
            per_length: dict[str, dict] = {}
            for length, route_parquet in sorted(routed.items()):
                packed = sr.stage_pack(shard_name, route_parquet, length, work)
                per_length[str(length)] = _append_output(
                    shard_name,
                    packed,
                    length,
                    output_root,
                    pipeline=pipeline,
                )
    finally:
        if owns_connection:
            conn.close()
    assert primary_membership is not None
    assert primary_membership_input is not None
    result = {
        "source": "pr",
        "store": str(store),
        "pr_completion": binding,
        "primary_membership": primary_membership,
        "primary_membership_input": primary_membership_input,
        "scan_id": scan_id,
        "output_root": str(output_root),
        "shard": shard_name,
        "rendered_docs": n_docs,
        "lengths": per_length,
        "materialize_stats": materialize_stats,
    }
    if revalidate_at_finish:
        _revalidate_pr_completion(args, binding)
        revalidate_primary_pr_membership(
            expected_membership=primary_membership,
            expected_input_binding=primary_membership_input,
            receipt_path=Path(args.primary_membership_receipt),
            input_root=Path(args.primary_membership_root),
            scan_id=scan_id,
        )
    return result


def export_pr_parquet_batches(args: argparse.Namespace) -> dict:
    store = Path(args.store)
    if not store.exists():
        raise FileNotFoundError(f"--store does not exist: {store}")
    lengths = tuple(
        sorted(int(x) for x in args.target_lengths.split(",") if x.strip())
    )
    if not lengths:
        raise ValueError("--target-lengths produced no lengths")
    binding = _load_pr_completion(args)
    scan_id = str(binding["scan_id"])
    conn = connect(str(store), create=False, readonly=True)
    primary_membership: dict[str, object] | None = None
    primary_membership_input: dict[str, object] | None = None
    try:
        (
            primary_membership,
            primary_membership_input,
        ) = load_primary_pr_membership(
            conn,
            receipt_path=Path(args.primary_membership_receipt),
            input_root=Path(args.primary_membership_root),
            scan_id=scan_id,
        )
        expected_input = _manifest_input(
            args,
            binding,
            primary_membership,
            primary_membership_input,
            lengths,
        )
        manifest_path = (
            Path(args.manifest)
            if args.manifest
            else Path(args.output_root) / "_done.json"
        )
        manifest = _load_manifest(
            manifest_path,
            expected_input=expected_input,
        )
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
    except Exception:
        conn.close()
        raise
    resume = not args.no_resume

    sr, _route_by_fit, _recompress_zstd_max = _load_pipeline()
    complete = False
    result: dict[str, object]
    try:
        raw_primary_selected = primary_membership.get("selected_pr_count")
        raw_stored_count = binding.get("stored_pr_count")
        if (
            isinstance(raw_primary_selected, bool)
            or not isinstance(raw_primary_selected, int)
            or raw_primary_selected < 1
            or isinstance(raw_stored_count, bool)
            or not isinstance(raw_stored_count, int)
            or raw_stored_count < 1
        ):
            raise RuntimeError(
                "primary membership or PR completion count is malformed"
            )
        selected_pr_count = _count_pr_keys(
            conn,
            repo=args.repo,
            scan_id=scan_id,
            offset=int(args.offset),
            limit=None,
        )
        global_selection = args.repo is None and int(args.offset) == 0
        if (
            global_selection
            and selected_pr_count
            != raw_primary_selected
        ):
            raise RuntimeError(
                "verified primary PR selection count differs from its membership "
                f"receipt: selected={selected_pr_count} "
                f"receipt={primary_membership['selected_pr_count']}"
            )
        if raw_stored_count < raw_primary_selected:
            raise RuntimeError(
                "primary PR membership exceeds the verified PR scan"
            )
        offset = int(args.offset)
        max_shards = args.max_shards
        shards: list[dict] = []
        totals_by_length: dict[str, dict[str, int]] = {}
        while True:
            shard_name = _shard_name(args.repo, scan_id, offset)
            done_key = f"{_repo_slug(args.repo)}:{offset}"
            n_keys = _count_pr_keys(
                conn,
                repo=args.repo,
                scan_id=scan_id,
                offset=offset,
                limit=batch_size,
            )
            if n_keys == 0:
                complete = True
                break
            if resume and done_key in manifest.get("done", {}):
                info = manifest["done"][done_key]
                _verify_done_artifacts(info)
                shards.append({"shard": shard_name, "skipped": True, **info})
            else:
                if done_key in manifest.get("done", {}):
                    raise RuntimeError(
                        f"--no-resume cannot overwrite completed shard {done_key}"
                    )
                shard_args = argparse.Namespace(**vars(args))
                shard_args.limit = batch_size
                shard_args.offset = offset
                try:
                    info = export_pr_parquet(
                        shard_args,
                        completion_binding=binding,
                        revalidate_at_finish=False,
                        primary_membership=primary_membership,
                        primary_membership_input=primary_membership_input,
                        connection=conn,
                    )
                except Exception as exc:
                    manifest.setdefault("failed", {})[done_key] = {
                        "offset": offset,
                        "limit": batch_size,
                        "repo": args.repo,
                        "stage": "export_pr_parquet",
                        "detail": str(exc)[:2000],
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    _save_manifest(manifest_path, manifest)
                    raise
                manifest.setdefault("done", {})[done_key] = info
                manifest.get("failed", {}).pop(done_key, None)
                _save_manifest(manifest_path, manifest)
                shards.append(info)

            last = shards[-1]
            for length, st in last.get("lengths", {}).items():
                agg = totals_by_length.setdefault(
                    length,
                    {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0},
                )
                for key in ("rows", "valid_tokens", "pad_tokens", "capacity_tokens"):
                    agg[key] += int(st.get(key, 0))

            offset += batch_size
            if max_shards is not None and len(shards) >= int(max_shards):
                break
        result = {
            "source": "pr",
            "store": str(store),
            "pr_completion": binding,
            "primary_membership": primary_membership,
            "primary_membership_input": primary_membership_input,
            "scan_id": scan_id,
            "output_root": str(Path(args.output_root)),
            "manifest": str(manifest_path),
            "repo": args.repo,
            "start_offset": int(args.offset),
            "batch_size": batch_size,
            "next_offset": offset,
            "shards": shards,
            "n_shards": len(shards),
            "selected_pr_count": selected_pr_count,
            "lengths": totals_by_length,
            "materialize_split_totals": _summarize_materialize_stats(shards),
        }
    finally:
        conn.close()
        _revalidate_pr_completion(args, binding)
        if (
            primary_membership is not None
            and primary_membership_input is not None
        ):
            revalidate_primary_pr_membership(
                expected_membership=primary_membership,
                expected_input_binding=primary_membership_input,
                receipt_path=Path(args.primary_membership_receipt),
                input_root=Path(args.primary_membership_root),
                scan_id=scan_id,
            )
    assert primary_membership is not None
    assert primary_membership_input is not None
    if complete:
        manifest["completed_pr_count"] = selected_pr_count
        if global_selection:
            manifest["status"] = "complete"
            _save_manifest(manifest_path, manifest)
            result["completion_receipt"] = _publish_complete_receipt(
                output_root=Path(args.output_root),
                manifest_path=manifest_path,
                manifest=manifest,
                binding=binding,
                primary_membership=primary_membership,
                primary_membership_input=primary_membership_input,
                scan_id=scan_id,
                target_lengths=lengths,
                selected_pr_count=selected_pr_count,
            )
        else:
            manifest["status"] = "selection_complete"
            _save_manifest(manifest_path, manifest)
    else:
        manifest.pop("status", None)
        manifest.pop("completed_pr_count", None)
        _save_manifest(manifest_path, manifest)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument(
        "--pr-completion-receipt",
        required=True,
        help=(
            "Verified cppmega_pr_completion_v2 or "
            "cppmega_gitlab_mr_completion_v1 receipt binding the exact scan."
        ),
    )
    p.add_argument("--repo-list", default=str(DEFAULT_REPO_LIST))
    p.add_argument(
        "--primary-membership-receipt",
        required=True,
        help=(
            "Canonical cppmega primary_pr_membership_receipt.json derived from "
            "the exact primary commit composition."
        ),
    )
    p.add_argument(
        "--primary-membership-root",
        required=True,
        help=(
            "Directory containing the canonical membership receipt and its "
            "ZSTD primary_pr_membership.parquet artifact."
        ),
    )
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--target-lengths", default=",".join(str(x) for x in ZSTD_LEVELS))
    p.add_argument("--repo", default=None, help="Optional owner/repo filter.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--all", action="store_true",
                   help="Export every PR row from --offset in resumable batches.")
    p.add_argument("--batch-size", type=int, default=10_000,
                   help="Rows per shard when --all is set.")
    p.add_argument("--max-shards", type=int, default=None,
                   help="Optional cap on number of shards exported in this run.")
    p.add_argument("--manifest", default=None,
                   help="Resume manifest for --all batched export. Default: "
                        "<output-root>/_done.json.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore completed PR export shards in --manifest.")
    p.add_argument("--memory-limit-gb", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = export_pr_parquet_batches(args) if args.all else export_pr_parquet(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
