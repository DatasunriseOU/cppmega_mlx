#!/usr/bin/env python3
"""Tokenize CI enriched JSONL into packed parquet with sidecars.

Stage 4 of the CI pipeline: reads ci_logs_enriched.jsonl and
ci_paired_enriched.jsonl, classifies domain_kind, tokenizes text using the
project tokenizer (65536 vocab), produces token-level sidecars, routes to
length buckets, and packs into zstd parquet matching the reindexed schema.

Usage:
    python scripts/tokenize_ci_enriched.py \
        --input outputs/ci_enriched/ \
        --output outputs/reindexed_ci/ \
        --seq-lengths 1024,2048,4096

    python scripts/tokenize_ci_enriched.py \
        --input outputs/ci_enriched/ci_logs_enriched.jsonl \
        --output outputs/reindexed_ci/ \
        --seq-lengths 4096 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega_mlx.data.domain_schema import (
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    NUM_DOCS_COLUMN,
    PACK_ID_COLUMN,
    ROW_PLATFORM_IDS_COLUMN,
    SOURCE_DOC_IDS_COLUMN,
    SOURCE_PLATFORM_IDS_COLUMN,
    TARGET_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
)
from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from scripts.nanochat_data.pack_enriched_rows import (
    PACKED_ROW_OUTPUT_SCHEMA,
    NormalizedDoc,
    normalize_document_record,
    pack_documents,
    rows_to_table,
)
from scripts.nanochat_data.token_budget import (
    load_tokenizer,
    tokenizer_fingerprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tokenize_ci_enriched")

# ---------------------------------------------------------------------------
# Domain kind classification
# ---------------------------------------------------------------------------

_SANITIZER_PATTERN = re.compile(
    r"(AddressSanitizer|LeakSanitizer|MemorySanitizer|"
    r"UndefinedBehaviorSanitizer|ThreadSanitizer|"
    r"SUMMARY: \S*Sanitizer|"
    r"==\d+==ERROR: (Address|Leak|Memory|Thread)Sanitizer)",
)

# Linker errors: require line-start or after whitespace to avoid matching
# inside words like "Build:" or "experimental"
_LINKER_ERROR_PATTERN = re.compile(
    r"(^|\n)\s*(undefined reference to|undefined symbol|"
    r"(?:/usr/bin/)?ld: \S|lld: error|linker command failed|"
    r"cannot find -l\w+|fatal error LNK\d+|"
    r"unresolved external symbol|"
    r"collect2: error: ld returned)",
    re.IGNORECASE,
)

_COMPILER_ERROR_PATTERN = re.compile(
    r"(^|\n)\s*\S+:\d+:\d+: (fatal )?error:|"
    r"(^|\n)\s*error C\d+|"
    r"(^|\n)\s*fatal error:",
    re.IGNORECASE,
)

# Test output: require structured test framework markers, not bare PASS/FAIL
_TEST_OUTPUT_PATTERN = re.compile(
    r"(\[  PASSED  \]|\[  FAILED  \]|\[ RUN      \]|\[       OK \]|"
    r"^\d+ tests? from \d+ test|"
    r"^ok\s+\S+\s+\d+\.\d+s|"
    r"Tests? (passed|failed|run): \d+|"
    r"pytest.*\d+ passed|"
    r"^\d+ passing \(\d+)",
    re.IGNORECASE | re.MULTILINE,
)

_BUILD_ERROR_PATTERN = re.compile(
    r"(make(\[\d+\])?: \*\*\*|ninja: build stopped|"
    r"Build FAILED|BUILD FAILURE|"
    r"error: recipe for target .* failed|"
    r"MSBUILD : error|"
    r"CMake Error)",
    re.IGNORECASE,
)


def classify_domain_kind(doc: dict[str, Any]) -> int:
    """Classify a CI document into a DomainKind value (40-48).

    Uses ci_metadata fields and text pattern matching to determine the
    most specific diagnostic domain kind. Prioritizes structured metadata
    (conclusion, severities) over text heuristics to avoid false positives
    from CI logs that mention tools/paths without actual errors.
    """
    text = doc.get("text", "")
    ci_meta = doc.get("ci_metadata") or {}
    conclusion = str(ci_meta.get("conclusion", "")).lower()
    severities = [str(s).lower() for s in (ci_meta.get("severities") or [])]
    domain_sidecars = doc.get("domain_sidecars") or {}

    # Check domain_sidecars for explicit classification hints
    if domain_sidecars.get("sanitizer"):
        return int(DomainKind.SANITIZER_OUTPUT)

    # Sanitizer output (highest specificity, unambiguous markers)
    if _SANITIZER_PATTERN.search(text):
        return int(DomainKind.SANITIZER_OUTPUT)

    # For SUCCESSFUL builds, classify by warning/diagnostic content only.
    # A successful build cannot have linker/build/compiler errors by definition.
    if conclusion == "success":
        # Check for test output in successful builds (test runs that passed)
        if _TEST_OUTPUT_PATTERN.search(text):
            return int(DomainKind.TEST_OUTPUT)
        # Successful builds with warnings -> compiler diagnostic
        if "warning" in severities:
            return int(DomainKind.COMPILER_DIAGNOSTIC)
        # Successful builds with diagnostics_count > 0
        if ci_meta.get("diagnostics_count", 0) > 0:
            return int(DomainKind.COMPILER_DIAGNOSTIC)
        # Successful test-oriented builds
        build_cmd = str(ci_meta.get("build_command", "")).lower()
        if any(kw in build_cmd for kw in ("test", "ctest", "pytest", "check")):
            return int(DomainKind.TEST_OUTPUT)
        # Generic successful build log
        return int(DomainKind.BUILD_DIAGNOSTIC)

    # For FAILED/CANCELLED builds, identify the failure mode
    if conclusion in ("failure", "timed_out", "cancelled"):
        # Linker errors (check before compiler errors - more specific)
        if _LINKER_ERROR_PATTERN.search(text):
            return int(DomainKind.LINKER_ERROR)
        # Compiler errors
        if "error" in severities or _COMPILER_ERROR_PATTERN.search(text):
            return int(DomainKind.COMPILER_ERROR)
        # Build system failure
        if _BUILD_ERROR_PATTERN.search(text):
            return int(DomainKind.BUILD_ERROR)
        # Test failures
        if _TEST_OUTPUT_PATTERN.search(text):
            return int(DomainKind.TEST_OUTPUT)
        # Generic build failure
        return int(DomainKind.BUILD_ERROR)

    # Unknown conclusion: fall back to text heuristics
    if _LINKER_ERROR_PATTERN.search(text):
        return int(DomainKind.LINKER_ERROR)
    if _COMPILER_ERROR_PATTERN.search(text):
        return int(DomainKind.COMPILER_ERROR)
    if _TEST_OUTPUT_PATTERN.search(text):
        return int(DomainKind.TEST_OUTPUT)
    if "warning" in severities or ci_meta.get("diagnostics_count", 0) > 0:
        return int(DomainKind.COMPILER_DIAGNOSTIC)

    # Default: build diagnostic (generic CI log)
    return int(DomainKind.BUILD_DIAGNOSTIC)


# ---------------------------------------------------------------------------
# CI section structure classification
# ---------------------------------------------------------------------------

# Structure IDs for CI documents (extends code structure vocab 0-9):
# 10 = CI_HEADER (metadata comment block)
# 11 = CI_COMMAND (shell commands, build steps)
# 12 = CI_OUTPUT (compiler/tool output)
# 13 = CI_DIAGNOSTIC (warning/error messages)
# 14 = CI_TEST_RESULT (test pass/fail lines)
# 15 = CI_LOG (generic log lines)
CI_STRUCT_HEADER = 10
CI_STRUCT_COMMAND = 11
CI_STRUCT_OUTPUT = 12
CI_STRUCT_DIAGNOSTIC = 13
CI_STRUCT_TEST_RESULT = 14
CI_STRUCT_LOG = 15

_CI_HEADER_LINE = re.compile(r"^//\s*(CI Build Log|Job|Platform|Compiler|Conclusion|Build|Commit|Diagnostics):")
_CI_COMMAND_LINE = re.compile(r"^(##\[group\]|##\[command\]|\$\s|>\s|\+\s|Run\s)")
_CI_DIAGNOSTIC_LINE = re.compile(
    r"(warning:|error:|note:|fatal error:|remark:|"
    r"\[-W[a-z-]+\]|error C\d+|warning C\d+)"
)
_CI_TEST_LINE = re.compile(
    r"(\[  PASSED  \]|\[  FAILED  \]|\[ RUN      \]|\[       OK \]|"
    r"PASS|FAIL|ok\s+\S+)"
)


def classify_line_structure(line: str) -> int:
    """Classify a single line of CI output into a structure category."""
    if _CI_HEADER_LINE.match(line):
        return CI_STRUCT_HEADER
    if _CI_COMMAND_LINE.match(line):
        return CI_STRUCT_COMMAND
    if _CI_DIAGNOSTIC_LINE.search(line):
        return CI_STRUCT_DIAGNOSTIC
    if _CI_TEST_LINE.search(line):
        return CI_STRUCT_TEST_RESULT
    if line.startswith(("  ", "\t")) or line.strip() == "":
        return CI_STRUCT_OUTPUT
    return CI_STRUCT_LOG


def build_char_structure_ids(text: str) -> list[int]:
    """Build per-character structure IDs for a CI document."""
    result: list[int] = []
    for line in text.splitlines(keepends=True):
        struct_id = classify_line_structure(line)
        result.extend([struct_id] * len(line))
    # Handle case where text doesn't end with newline
    while len(result) < len(text):
        result.append(CI_STRUCT_LOG)
    return result[:len(text)]


# ---------------------------------------------------------------------------
# Token-level role classification for CI
# ---------------------------------------------------------------------------

_CI_ROLE_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(error|warning|note|fatal|remark)\b", re.IGNORECASE), int(DomainRoleKind.SEVERITY)),
    (re.compile(r"(/[^\s:]+\.\w+|\\[^\s:]+\.\w+)"), int(DomainRoleKind.FILE)),
    (re.compile(r":\d+:\d+:"), int(DomainRoleKind.LINE)),
    (re.compile(r"\[-W[^\]]+\]"), int(DomainRoleKind.OPTION)),
    (re.compile(r"\b(PASS|FAIL|OK|RUN)\b"), int(DomainRoleKind.TEST_NAME)),
]


def build_char_role_ids(text: str) -> list[int]:
    """Build per-character role IDs for CI text (simplified heuristic)."""
    roles = [int(DomainRoleKind.NONE)] * len(text)
    for pattern, role_id in _CI_ROLE_PATTERNS:
        for match in pattern.finditer(text):
            for i in range(match.start(), match.end()):
                if roles[i] == int(DomainRoleKind.NONE):
                    roles[i] = role_id
    return roles


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def prepare_ci_document(
    raw_doc: dict[str, Any],
    *,
    doc_index: int,
) -> dict[str, Any]:
    """Transform a raw CI enriched JSONL doc into the format expected by
    materialize_tokenized_enriched_batch and pack_enriched_rows.
    """
    text = raw_doc.get("text", "")
    domain_kind = classify_domain_kind(raw_doc)
    ci_meta = raw_doc.get("ci_metadata") or {}

    # Build per-char metadata
    char_structure_ids = build_char_structure_ids(text)
    char_domain_ids = [domain_kind] * len(text)
    char_role_ids = build_char_role_ids(text)
    char_confidence_ids = [int(ParseConfidence.HEURISTIC)] * len(text)

    # Prepare the enriched document dict compatible with the pipeline
    enriched: dict[str, Any] = {
        "text": text,
        "doc_type": raw_doc.get("doc_type", "diagnostic"),
        "domain_kind": domain_kind,
        "source_doc_id": raw_doc.get("source_doc_id", f"ci:{doc_index}"),
        "repo": raw_doc.get("repo", ""),
        "filepath": raw_doc.get("filepath", ""),
        "commit_hash": raw_doc.get("commit_hash", ""),
        # Char-level metadata for materialize_tokenized_enriched_batch
        "structure_ids": char_structure_ids,
        "domain_ids": char_domain_ids,
        "domain_role_ids": char_role_ids,
        "domain_confidence_ids": char_confidence_ids,
        # Empty graph metadata (CI docs have no AST)
        "chunk_boundaries": [],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [],
        "sibling_index": [],
        "ast_node_type": [],
        "symbol_ids": [],
        "call_targets": [],
        "type_refs": [],
        "def_use": [],
        "domain_entity_ids": [],
        "domain_scope_ids": [],
        "domain_source_doc_ids": [],
        "domain_source_identity_ids": [],
        # Diagnostic edges from domain_sidecars if available
        "diagnostic_edges": _extract_diagnostic_edges(raw_doc),
        "domain_edges": [],
        "build_edges": [],
        "shell_edges": [],
        "cross_domain_edges": [],
        # Symbol identity (empty for CI docs)
        SYMBOL_IDENTITIES_COLUMN: raw_doc.get("symbol_identities") or [],
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        # Platform info from CI metadata
        "platform_info": {
            "platform": ci_meta.get("platform", "unknown"),
            "compiler": ci_meta.get("compiler_info", "unknown"),
        },
    }
    return enriched


def _extract_diagnostic_edges(raw_doc: dict[str, Any]) -> list[dict[str, int]]:
    """Extract diagnostic edges from domain_sidecars if present."""
    sidecars = raw_doc.get("domain_sidecars") or {}
    edges = sidecars.get("diagnostic_edges")
    if not edges:
        return []
    normalized = []
    for edge in edges:
        if isinstance(edge, dict):
            normalized.append({
                "from_char": int(edge.get("from_char", edge.get("from", 0))),
                "to_char": int(edge.get("to_char", edge.get("to", 0))),
                "kind": int(edge.get("kind", 0)),
            })
    return normalized


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def assign_bucket(token_count: int, seq_lengths: list[int]) -> int | None:
    """Assign a document to the smallest bucket that fits it.

    Returns the bucket size, or None if the document exceeds all buckets.
    """
    for length in sorted(seq_lengths):
        if token_count <= length:
            return length
    return None


# ---------------------------------------------------------------------------
# Sidecar emission
# ---------------------------------------------------------------------------

def write_sidecar_manifest(
    output_dir: Path,
    *,
    bucket: int,
    num_docs: int,
    num_packed_rows: int,
    total_tokens: int,
    domain_kind_counts: dict[int, int],
    tokenizer_fp: str,
    timestamp: str,
) -> Path:
    """Write a JSON sidecar manifest for a bucket directory."""
    manifest = {
        "bucket_seq_length": bucket,
        "num_input_docs": num_docs,
        "num_packed_rows": num_packed_rows,
        "total_valid_tokens": total_tokens,
        "domain_kind_counts": {
            DomainKind(k).name if k in {int(d) for d in DomainKind} else str(k): v
            for k, v in sorted(domain_kind_counts.items())
        },
        "tokenizer_fingerprint": tokenizer_fp,
        "timestamp": timestamp,
        "schema_version": "case5_domain_routes_v1",
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
    }
    manifest_path = output_dir / f"manifest_{bucket}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def read_ci_jsonl_files(input_path: Path) -> list[dict[str, Any]]:
    """Read all CI enriched JSONL files from the input path."""
    docs: list[dict[str, Any]] = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix == ".jsonl" and "enriched" in p.name
        )
        if not files:
            # Fallback: any .jsonl file
            files = sorted(p for p in input_path.iterdir() if p.suffix == ".jsonl")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for filepath in files:
        log.info("Reading %s ...", filepath)
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    docs.append(doc)
                    count += 1
                except json.JSONDecodeError as e:
                    log.warning("JSON decode error in %s: %s", filepath.name, e)
        log.info("  %s: %d documents", filepath.name, count)

    return docs


def tokenize_and_pack(
    docs: list[dict[str, Any]],
    *,
    tokenizer,
    seq_lengths: list[int],
    output_dir: Path,
    timestamp: str,
    dry_run: bool = False,
    batch_size: int = 256,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Tokenize CI documents, bucket by length, and pack into parquet."""

    tokenizer_fp = tokenizer_fingerprint(tokenizer)

    # Phase 1: Prepare enriched documents
    log.info("Preparing %d CI documents for tokenization...", len(docs))
    enriched_docs = []
    domain_kind_counts: dict[int, int] = {}
    for i, raw_doc in enumerate(docs):
        enriched = prepare_ci_document(raw_doc, doc_index=i)
        enriched_docs.append(enriched)
        dk = enriched["domain_kind"]
        domain_kind_counts[dk] = domain_kind_counts.get(dk, 0) + 1

    log.info("Domain kind distribution:")
    for dk, count in sorted(domain_kind_counts.items()):
        try:
            name = DomainKind(dk).name
        except ValueError:
            name = f"UNKNOWN({dk})"
        log.info("  %s (%d): %d docs", name, dk, count)

    # Phase 2: Tokenize in batches
    log.info("Tokenizing %d documents (batch_size=%d)...", len(enriched_docs), batch_size)
    tokenized_docs: list[dict[str, Any]] = []
    for batch_start in range(0, len(enriched_docs), batch_size):
        batch_end = min(batch_start + batch_size, len(enriched_docs))
        batch = enriched_docs[batch_start:batch_end]
        tokenized_batch = materialize_tokenized_enriched_batch(batch, tokenizer)
        # Merge tokenized metadata back into enriched docs
        for enriched, tokenized in zip(batch, tokenized_batch):
            merged = {**enriched, **tokenized}
            tokenized_docs.append(merged)
        if (batch_start // batch_size) % 10 == 0:
            log.info("  Tokenized %d/%d documents", batch_end, len(enriched_docs))

    log.info("Tokenization complete. %d documents tokenized.", len(tokenized_docs))

    # Phase 3: Bucket by token count
    buckets: dict[int, list[dict[str, Any]]] = {length: [] for length in seq_lengths}
    oversized: list[dict[str, Any]] = []
    token_counts_by_bucket: dict[int, int] = {length: 0 for length in seq_lengths}

    for doc in tokenized_docs:
        token_ids = doc.get(TOKEN_IDS_COLUMN, [])
        token_count = len(token_ids)
        if token_count == 0:
            continue
        bucket = assign_bucket(token_count, seq_lengths)
        if bucket is None:
            oversized.append(doc)
        else:
            buckets[bucket].append(doc)
            token_counts_by_bucket[bucket] += token_count

    log.info("Bucket assignment:")
    for length in sorted(seq_lengths):
        log.info("  %d tokens: %d docs (%d total tokens)",
                 length, len(buckets[length]), token_counts_by_bucket[length])
    if oversized:
        log.info("  OVERSIZED: %d docs (exceed max bucket %d)",
                 len(oversized), max(seq_lengths))

    if dry_run:
        log.info("[DRY RUN] Would write packed parquet to %s", output_dir)
        return {
            "input_docs": len(docs),
            "tokenized_docs": len(tokenized_docs),
            "buckets": {str(k): len(v) for k, v in buckets.items()},
            "oversized": len(oversized),
            "domain_kind_counts": domain_kind_counts,
        }

    # Phase 4: Pack each bucket and write parquet
    output_base = output_dir / f"reindexed_ci_{timestamp}_code"
    output_base.mkdir(parents=True, exist_ok=True)

    total_packed_rows = 0
    summary: dict[str, Any] = {
        "input_docs": len(docs),
        "tokenized_docs": len(tokenized_docs),
        "domain_kind_counts": domain_kind_counts,
        "buckets": {},
        "output_dir": str(output_base),
        "timestamp": timestamp,
        "tokenizer_fingerprint": tokenizer_fp,
    }

    signature_to_id: dict[str, int] = {}

    for bucket_length in sorted(seq_lengths):
        bucket_docs = buckets[bucket_length]
        if not bucket_docs:
            log.info("Bucket %d: empty, skipping.", bucket_length)
            continue

        bucket_dir = output_base / str(bucket_length)
        bucket_dir.mkdir(parents=True, exist_ok=True)

        log.info("Packing bucket %d (%d docs)...", bucket_length, len(bucket_docs))

        # Normalize documents for packing
        normalized: list[NormalizedDoc] = []
        for doc_idx, doc in enumerate(bucket_docs):
            try:
                norm_doc = normalize_document_record(
                    doc,
                    source_doc_index=doc_idx,
                )
                normalized.append(norm_doc)
            except (ValueError, KeyError) as e:
                log.warning("  Skipping doc %d in bucket %d: %s",
                           doc_idx, bucket_length, e)

        if not normalized:
            log.warning("Bucket %d: no valid documents after normalization.", bucket_length)
            continue

        # Pack documents
        packed_rows, overflow = pack_documents(
            normalized,
            target_length=bucket_length,
            pad_token_id=pad_token_id,
            strategy="best_fit",
        )

        # Write packed parquet with zstd compression
        parquet_path = bucket_dir / f"ci_packed_{bucket_length}.parquet"
        table = rows_to_table(packed_rows)
        pq.write_table(
            table,
            parquet_path,
            compression="zstd",
            compression_level=3,
            row_group_size=min(128, len(packed_rows)),
        )

        # Write overflow sidecar
        overflow_path = bucket_dir / f"overflow_{bucket_length}.jsonl"
        if overflow:
            with open(overflow_path, "w", encoding="utf-8") as f:
                for record in overflow:
                    f.write(json.dumps(record, default=str) + "\n")

        # Write bucket manifest sidecar
        bucket_domain_counts: dict[int, int] = {}
        for doc in bucket_docs:
            dk = doc.get("domain_kind", 0)
            bucket_domain_counts[dk] = bucket_domain_counts.get(dk, 0) + 1

        write_sidecar_manifest(
            bucket_dir,
            bucket=bucket_length,
            num_docs=len(bucket_docs),
            num_packed_rows=len(packed_rows),
            total_tokens=token_counts_by_bucket[bucket_length],
            domain_kind_counts=bucket_domain_counts,
            tokenizer_fp=tokenizer_fp,
            timestamp=timestamp,
        )

        total_packed_rows += len(packed_rows)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        log.info(
            "  Bucket %d: %d docs -> %d packed rows (%.1f MB, %d overflow)",
            bucket_length, len(normalized), len(packed_rows),
            file_size_mb, len(overflow),
        )

        summary["buckets"][str(bucket_length)] = {
            "input_docs": len(bucket_docs),
            "normalized_docs": len(normalized),
            "packed_rows": len(packed_rows),
            "overflow_docs": len(overflow),
            "total_tokens": token_counts_by_bucket[bucket_length],
            "parquet_path": str(parquet_path),
            "parquet_size_mb": round(file_size_mb, 2),
        }

    # Handle oversized documents: pack into the largest bucket with expanded rows
    if oversized:
        max_bucket = max(seq_lengths)
        oversized_dir = output_base / "oversized"
        oversized_dir.mkdir(parents=True, exist_ok=True)

        log.info("Packing %d oversized docs...", len(oversized))
        normalized_oversized: list[NormalizedDoc] = []
        for doc_idx, doc in enumerate(oversized):
            try:
                norm_doc = normalize_document_record(
                    doc,
                    source_doc_index=doc_idx,
                )
                normalized_oversized.append(norm_doc)
            except (ValueError, KeyError) as e:
                log.warning("  Skipping oversized doc %d: %s", doc_idx, e)

        if normalized_oversized:
            # Pack with the max bucket length; oversized docs get their own rows
            packed_rows, _ = pack_documents(
                normalized_oversized,
                target_length=max_bucket,
                pad_token_id=pad_token_id,
                strategy="best_fit",
            )
            parquet_path = oversized_dir / "ci_packed_oversized.parquet"
            table = rows_to_table(packed_rows)
            pq.write_table(
                table,
                parquet_path,
                compression="zstd",
                compression_level=3,
                row_group_size=min(64, len(packed_rows)),
            )
            total_packed_rows += len(packed_rows)
            log.info("  Oversized: %d docs -> %d packed rows",
                     len(normalized_oversized), len(packed_rows))
            summary["oversized"] = {
                "input_docs": len(oversized),
                "packed_rows": len(packed_rows),
                "parquet_path": str(parquet_path),
            }

    summary["total_packed_rows"] = total_packed_rows
    log.info("Done. Total packed rows: %d", total_packed_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize CI enriched JSONL into packed parquet with sidecars.\n\n"
            "Reads ci_logs_enriched.jsonl and ci_paired_enriched.jsonl,\n"
            "classifies domain_kind, tokenizes, and packs into length-bucketed\n"
            "zstd parquet matching the reindexed schema."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input directory containing CI enriched JSONL files, or a single "
            "JSONL file path."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for packed parquet and sidecars.",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated sequence length buckets (default: 1024,2048,4096).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-resolved if not provided).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenization batch size (default: 256).",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=0,
        help="Token ID used for padding (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and classify but do not write output files.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum number of documents to process (0 = all).",
    )
    args = parser.parse_args()

    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",") if x.strip()]
    if not seq_lengths:
        parser.error("--seq-lengths must contain at least one value")
    seq_lengths.sort()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    log.info("Tokenizer loaded (fingerprint: %s)", tokenizer_fingerprint(tokenizer))

    log.info("Reading CI enriched JSONL from %s ...", input_path)
    docs = read_ci_jsonl_files(input_path)
    if not docs:
        log.error("No documents found in %s", input_path)
        sys.exit(1)

    if args.max_docs > 0:
        docs = docs[:args.max_docs]
        log.info("Limited to %d documents (--max-docs)", len(docs))

    log.info("Loaded %d CI documents total.", len(docs))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary = tokenize_and_pack(
        docs,
        tokenizer=tokenizer,
        seq_lengths=seq_lengths,
        output_dir=output_dir,
        timestamp=timestamp,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        pad_token_id=args.pad_token_id,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
