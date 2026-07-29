"""Consume the canonical cppmega primary-PR membership without fallbacks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq


PRIMARY_PR_MEMBERSHIP_SCHEMA = "cppmega_primary_pr_membership_v1"
PRIMARY_PR_MEMBERSHIP_POLICY = (
    "exact_allowlisted_primary_commit_source_documents_v1"
)
PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA = (
    "cppmega_primary_pr_membership_parquet_v1"
)
PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME = "primary_pr_membership.parquet"
PRIMARY_PR_MEMBERSHIP_RECEIPT_NAME = "primary_pr_membership_receipt.json"
PRIMARY_PR_MEMBERSHIP_INPUT_SCHEMA = (
    "cppmega_mlx_primary_pr_membership_input_v1"
)
PRIMARY_PR_MEMBERSHIP_TABLE = "_cppmega_primary_pr_membership"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MEMBERSHIP_FIELDS = {
    "schema",
    "policy",
    "scan_id",
    "commit_artifacts",
    "rows",
    "source_docs",
    "source_docs_with_pr_number",
    "source_docs_with_pr_discussion",
    "ignored_unverified_pr_number_source_docs",
    "source_docs_with_commit_sha",
    "selected_pr_count",
    "sha_only_matched_source_docs",
    "unmatched_commit_sha_source_docs",
    "selected_membership_sha256",
    "validation",
    "artifact",
}
_COMMIT_BINDING_FIELDS = {
    "schema",
    "source_composition_sha256",
    "source_composition_plan_sha256",
    "buckets",
    "files",
    "rows",
    "byte_size",
    "artifact_set_sha256",
    "by_bucket",
}
_ARTIFACT_FIELDS = {
    "schema",
    "path",
    "rows",
    "byte_size",
    "sha256",
    "membership_sha256",
}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _require_int(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise RuntimeError(f"{field} must be an integer >= {minimum}")
    return value


def _membership_arrow_schema(
    *,
    scan_id: str,
    membership_sha256: str,
) -> pa.Schema:
    return pa.schema(
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


def _validate_commit_binding(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _COMMIT_BINDING_FIELDS:
        raise RuntimeError("primary PR commit-artifact binding is malformed")
    if value.get("schema") != "cppmega_primary_commit_artifact_binding_v1":
        raise RuntimeError("primary PR commit-artifact schema is unsupported")
    _require_sha256(
        value.get("source_composition_sha256"),
        field="commit_artifacts.source_composition_sha256",
    )
    _require_sha256(
        value.get("source_composition_plan_sha256"),
        field="commit_artifacts.source_composition_plan_sha256",
    )
    _require_sha256(
        value.get("artifact_set_sha256"),
        field="commit_artifacts.artifact_set_sha256",
    )
    buckets = value.get("buckets")
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(
            isinstance(bucket, bool)
            or not isinstance(bucket, int)
            or bucket < 1
            for bucket in buckets
        )
        or buckets != sorted(set(buckets))
    ):
        raise RuntimeError("primary PR commit-artifact buckets are malformed")
    files = _require_int(value.get("files"), field="commit_artifacts.files", minimum=1)
    rows = _require_int(value.get("rows"), field="commit_artifacts.rows", minimum=1)
    byte_size = _require_int(
        value.get("byte_size"),
        field="commit_artifacts.byte_size",
        minimum=1,
    )
    by_bucket = value.get("by_bucket")
    if (
        not isinstance(by_bucket, dict)
        or set(by_bucket) != {str(bucket) for bucket in buckets}
    ):
        raise RuntimeError("primary PR commit-artifact bucket summary drifted")
    observed_files = 0
    observed_rows = 0
    observed_bytes = 0
    for bucket in buckets:
        summary = by_bucket[str(bucket)]
        if not isinstance(summary, dict) or set(summary) != {
            "files",
            "rows",
            "byte_size",
        }:
            raise RuntimeError(
                f"primary PR commit-artifact bucket {bucket} is malformed"
            )
        observed_files += _require_int(
            summary.get("files"),
            field=f"commit_artifacts.by_bucket.{bucket}.files",
            minimum=1,
        )
        observed_rows += _require_int(
            summary.get("rows"),
            field=f"commit_artifacts.by_bucket.{bucket}.rows",
            minimum=1,
        )
        observed_bytes += _require_int(
            summary.get("byte_size"),
            field=f"commit_artifacts.by_bucket.{bucket}.byte_size",
            minimum=1,
        )
    if (observed_files, observed_rows, observed_bytes) != (
        files,
        rows,
        byte_size,
    ):
        raise RuntimeError("primary PR commit-artifact totals drifted")


def _validate_membership(
    value: object,
    *,
    expected_scan_id: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _MEMBERSHIP_FIELDS:
        raise RuntimeError("primary PR membership receipt is malformed")
    if (
        value.get("schema") != PRIMARY_PR_MEMBERSHIP_SCHEMA
        or value.get("policy") != PRIMARY_PR_MEMBERSHIP_POLICY
        or value.get("scan_id") != expected_scan_id
    ):
        raise RuntimeError(
            "primary PR membership schema, policy, or scan binding drifted"
        )
    _require_sha256(expected_scan_id, field="expected scan_id")
    membership_sha256 = _require_sha256(
        value.get("selected_membership_sha256"),
        field="selected_membership_sha256",
    )
    counters = {
        field: _require_int(value.get(field), field=field, minimum=0)
        for field in (
            "rows",
            "source_docs",
            "source_docs_with_pr_number",
            "source_docs_with_pr_discussion",
            "ignored_unverified_pr_number_source_docs",
            "source_docs_with_commit_sha",
            "sha_only_matched_source_docs",
            "unmatched_commit_sha_source_docs",
        )
    }
    selected = _require_int(
        value.get("selected_pr_count"),
        field="selected_pr_count",
        minimum=1,
    )
    if counters["rows"] < 1 or counters["source_docs"] < 1:
        raise RuntimeError("primary PR source accounting is empty")
    if (
        counters["source_docs_with_commit_sha"] != counters["source_docs"]
        or counters["source_docs_with_pr_discussion"]
        > counters["source_docs_with_pr_number"]
        or counters["source_docs_with_pr_number"] > counters["source_docs"]
        or counters["ignored_unverified_pr_number_source_docs"]
        > counters["source_docs_with_pr_number"]
        or (
            counters["sha_only_matched_source_docs"]
            + counters["unmatched_commit_sha_source_docs"]
            > counters["source_docs"]
        )
    ):
        raise RuntimeError("primary PR source counters are inconsistent")
    validation = value.get("validation")
    if (
        not isinstance(validation, dict)
        or set(validation)
        != {
            "source_composition_complete",
            "exact_allowlisted_commit_artifacts",
            "exact_source_doc_shapes",
            "exact_scan_membership",
            "direct_pr_sha_conflicts",
        }
        or validation.get("source_composition_complete") is not True
        or validation.get("exact_allowlisted_commit_artifacts") is not True
        or validation.get("exact_source_doc_shapes") is not True
        or validation.get("exact_scan_membership") is not True
        or validation.get("direct_pr_sha_conflicts") != 0
    ):
        raise RuntimeError("primary PR membership validation is not green")
    _validate_commit_binding(value.get("commit_artifacts"))
    artifact = value.get("artifact")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != _ARTIFACT_FIELDS
        or artifact.get("schema") != PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA
        or artifact.get("path") != PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME
        or artifact.get("rows") != selected
        or artifact.get("membership_sha256") != membership_sha256
    ):
        raise RuntimeError("primary PR membership artifact descriptor drifted")
    _require_int(
        artifact.get("byte_size"),
        field="artifact.byte_size",
        minimum=1,
    )
    _require_sha256(artifact.get("sha256"), field="artifact.sha256")
    return value


def _resolve_inputs(
    *,
    receipt_path: Path,
    input_root: Path,
) -> tuple[Path, Path, Path]:
    raw_root = input_root.expanduser()
    if raw_root.is_symlink():
        raise RuntimeError(f"primary PR membership root is a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    raw_receipt = receipt_path.expanduser()
    if raw_receipt.is_symlink():
        raise RuntimeError(
            f"primary PR membership receipt is a symlink: {raw_receipt}"
        )
    receipt = raw_receipt.resolve()
    expected_receipt = root / PRIMARY_PR_MEMBERSHIP_RECEIPT_NAME
    if receipt != expected_receipt or not receipt.is_file():
        raise RuntimeError(
            f"primary PR membership receipt must be {expected_receipt}"
        )
    artifact = root / PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME
    if artifact.is_symlink() or not artifact.is_file():
        raise RuntimeError(
            f"primary PR membership artifact is missing or symlinked: {artifact}"
        )
    return root, receipt, artifact


def _read_receipt(
    receipt: Path,
    *,
    expected_scan_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    size = receipt.stat().st_size
    if size < 1 or size > _MAX_RECEIPT_BYTES:
        raise RuntimeError(
            f"primary PR membership receipt size is invalid: {size}"
        )
    payload = receipt.read_bytes()
    if len(payload) != size:
        raise RuntimeError("primary PR membership receipt changed while reading")
    try:
        membership = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid primary PR membership receipt: {receipt}"
        ) from exc
    validated = _validate_membership(
        membership,
        expected_scan_id=expected_scan_id,
    )
    canonical_payload = (
        json.dumps(validated, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if payload != canonical_payload:
        raise RuntimeError(
            "primary PR membership receipt is not canonical JSON"
        )
    binding = {
        "path": str(receipt),
        "byte_size": size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return validated, binding


def _verify_artifact(
    membership: dict[str, object],
    *,
    artifact: Path,
    conn: sqlite3.Connection | None,
) -> dict[str, object]:
    scan_id = str(membership["scan_id"])
    selected = _require_int(
        membership.get("selected_pr_count"),
        field="selected_pr_count",
        minimum=1,
    )
    membership_sha256 = str(membership["selected_membership_sha256"])
    expected_schema = _membership_arrow_schema(
        scan_id=scan_id,
        membership_sha256=membership_sha256,
    )
    parquet = pq.ParquetFile(artifact)
    if parquet.schema_arrow != expected_schema:
        raise RuntimeError("primary PR membership Parquet schema drifted")
    if (
        parquet.metadata.num_rows != selected
        or parquet.metadata.num_row_groups < 1
    ):
        raise RuntimeError("primary PR membership Parquet row count drifted")
    codecs = {
        str(
            parquet.metadata.row_group(row_group)
            .column(column)
            .compression
        )
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    if codecs != {"ZSTD"}:
        raise RuntimeError(
            "primary PR membership Parquet must use ZSTD for every column"
        )

    if conn is not None:
        conn.execute(f"DROP TABLE IF EXISTS temp.{PRIMARY_PR_MEMBERSHIP_TABLE}")
        conn.execute(
            f"""
            CREATE TEMP TABLE {PRIMARY_PR_MEMBERSHIP_TABLE} (
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                PRIMARY KEY (repo, pr_number)
            ) WITHOUT ROWID
            """
        )
    digest = hashlib.sha256()
    observed_rows = 0
    previous: tuple[str, int] | None = None
    try:
        for batch in parquet.iter_batches(
            columns=["repo", "pr_number"],
            batch_size=8192,
        ):
            keys: list[tuple[str, int]] = []
            for row in batch.to_pylist():
                repo = row.get("repo")
                pr_number = row.get("pr_number")
                if (
                    not isinstance(repo, str)
                    or not repo.strip()
                    or isinstance(pr_number, bool)
                    or not isinstance(pr_number, int)
                    or pr_number < 1
                ):
                    raise RuntimeError(
                        "primary PR membership contains an invalid key"
                    )
                key = (repo, pr_number)
                if previous is not None and key <= previous:
                    raise RuntimeError(
                        "primary PR membership keys are not strictly sorted"
                    )
                previous = key
                encoded = f"{repo}\0{pr_number}".encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                keys.append(key)
            if conn is not None:
                conn.executemany(
                    f"""
                    INSERT INTO {PRIMARY_PR_MEMBERSHIP_TABLE}(repo, pr_number)
                    VALUES (?, ?)
                    """,
                    keys,
                )
            observed_rows += len(keys)
        if (
            observed_rows != selected
            or digest.hexdigest() != membership_sha256
        ):
            raise RuntimeError(
                "primary PR membership logical digest or row count drifted"
            )
        raw_descriptor = membership["artifact"]
        assert isinstance(raw_descriptor, dict)
        descriptor = {
            "schema": PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA,
            "path": PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME,
            "rows": observed_rows,
            "byte_size": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
            "membership_sha256": digest.hexdigest(),
        }
        if descriptor != raw_descriptor:
            raise RuntimeError(
                "primary PR membership artifact byte binding drifted"
            )
        return descriptor
    except Exception:
        if conn is not None:
            conn.execute(
                f"DROP TABLE IF EXISTS temp.{PRIMARY_PR_MEMBERSHIP_TABLE}"
            )
        raise


def _input_binding(
    *,
    root: Path,
    artifact: Path,
    receipt_binding: dict[str, object],
    artifact_descriptor: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": PRIMARY_PR_MEMBERSHIP_INPUT_SCHEMA,
        "root": str(root),
        "receipt": receipt_binding,
        "artifact": {
            **artifact_descriptor,
            "path": str(artifact),
        },
    }


def load_primary_pr_membership(
    conn: sqlite3.Connection,
    *,
    receipt_path: Path,
    input_root: Path,
    scan_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify and materialize the canonical membership into a TEMP table."""

    scan_id = _require_sha256(scan_id, field="scan_id")
    root, receipt, artifact = _resolve_inputs(
        receipt_path=receipt_path,
        input_root=input_root,
    )
    membership, receipt_binding = _read_receipt(
        receipt,
        expected_scan_id=scan_id,
    )
    descriptor = _verify_artifact(
        membership,
        artifact=artifact,
        conn=conn,
    )
    missing = conn.execute(
        f"""
        SELECT m.repo, m.pr_number
        FROM {PRIMARY_PR_MEMBERSHIP_TABLE} AS m
        LEFT JOIN prs AS p
          ON p.repo = m.repo
         AND p.pr_number = m.pr_number
         AND p.scan_id = ?
        WHERE p.pr_number IS NULL
        LIMIT 1
        """,
        (scan_id,),
    ).fetchone()
    if missing is not None:
        conn.execute(
            f"DROP TABLE IF EXISTS temp.{PRIMARY_PR_MEMBERSHIP_TABLE}"
        )
        raise RuntimeError(
            "primary PR membership key is absent from the exact verified scan: "
            f"{missing['repo']}#{missing['pr_number']}"
        )
    observed = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {PRIMARY_PR_MEMBERSHIP_TABLE}"
        ).fetchone()[0]
    )
    selected = _require_int(
        membership.get("selected_pr_count"),
        field="selected_pr_count",
        minimum=1,
    )
    if observed != selected:
        conn.execute(
            f"DROP TABLE IF EXISTS temp.{PRIMARY_PR_MEMBERSHIP_TABLE}"
        )
        raise RuntimeError("primary PR membership TEMP table lost keys")
    return membership, _input_binding(
        root=root,
        artifact=artifact,
        receipt_binding=receipt_binding,
        artifact_descriptor=descriptor,
    )


def revalidate_primary_pr_membership(
    *,
    expected_membership: dict[str, object],
    expected_input_binding: dict[str, object],
    receipt_path: Path,
    input_root: Path,
    scan_id: str,
) -> None:
    """Re-read all immutable membership inputs and reject any drift."""

    scan_id = _require_sha256(scan_id, field="scan_id")
    root, receipt, artifact = _resolve_inputs(
        receipt_path=receipt_path,
        input_root=input_root,
    )
    membership, receipt_binding = _read_receipt(
        receipt,
        expected_scan_id=scan_id,
    )
    descriptor = _verify_artifact(
        membership,
        artifact=artifact,
        conn=None,
    )
    current_binding = _input_binding(
        root=root,
        artifact=artifact,
        receipt_binding=receipt_binding,
        artifact_descriptor=descriptor,
    )
    if (
        membership != expected_membership
        or current_binding != expected_input_binding
    ):
        raise RuntimeError("primary PR membership input binding drifted")


__all__ = [
    "PRIMARY_PR_MEMBERSHIP_ARTIFACT_NAME",
    "PRIMARY_PR_MEMBERSHIP_ARTIFACT_SCHEMA",
    "PRIMARY_PR_MEMBERSHIP_INPUT_SCHEMA",
    "PRIMARY_PR_MEMBERSHIP_POLICY",
    "PRIMARY_PR_MEMBERSHIP_RECEIPT_NAME",
    "PRIMARY_PR_MEMBERSHIP_SCHEMA",
    "PRIMARY_PR_MEMBERSHIP_TABLE",
    "load_primary_pr_membership",
    "revalidate_primary_pr_membership",
]
