#!/usr/bin/env python3
"""Tokenize CI enriched JSONL into packed parquet with sidecars.

Stage 4 of the CI pipeline: reads ci_logs_enriched.jsonl and
ci_paired_enriched.jsonl, classifies domain_kind, tokenizes text using the
project tokenizer (65536 vocab), produces token-level sidecars, routes to
length buckets, and packs into zstd parquet matching the reindexed schema.

Usage:
    python scripts/tokenize_ci_enriched.py \
        --input outputs/ci_enriched/ \
        --output outputs/ \
        --seq-lengths 1024,2048,4096,8192,16384

    python scripts/tokenize_ci_enriched.py \
        --input outputs/ci_enriched/ci_logs_enriched.jsonl \
        --output outputs/ \
        --seq-lengths 4096 --dry-run
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import shutil
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
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.diagnostic_parsers import (
    parse_build_error,
    parse_clang_diagnostic,
    parse_gcc_diagnostic,
    parse_linker_error,
    parse_msvc_diagnostic,
    parse_sanitizer_output,
    parse_test_output,
)
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    PACK_ID_COLUMN,
    PACKED_ROWS_TOKEN_METADATA_COLUMNS,
    TARGET_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKENIZED_ENRICHED_OBJECTIVE_COLUMNS,
)
from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
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

DEFAULT_SEQ_LENGTHS = (1024, 2048, 4096, 8192, 16384)
CI_MANIFEST_SCHEMA = "cppmega_ci_fixed_buckets_manifest_v3"
CI_BUCKET_MANIFEST_SCHEMA = "cppmega_ci_fixed_bucket_v2"
CI_LOG_COMPLETION_SCHEMA = "cppmega_ci_log_extraction_v1"
CANONICAL_CI_INPUT_NAMES = (
    "ci_logs_enriched.jsonl",
    "ci_paired_enriched.jsonl",
)

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
    conclusion = str(ci_meta.get("conclusion") or "").lower()
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
        build_cmd = str(ci_meta.get("build_command") or "").lower()
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

_CI_LOCAL_BUILD_SIGNAL = re.compile(
    r"(error|failed|failure|build stopped|make(?:\[\d+\])?: \*\*\*|"
    r"FAILED:|CMake Error|MSBUILD : error)",
    re.IGNORECASE,
)


def _parse_build_ci_sidecars(
    text: str,
    *,
    domain_kind: int,
    tool: str,
) -> dict[str, Any]:
    """Parse build failures line-locally so a single log-wide hub is not invented."""

    text_length = len(text)
    role_ids = build_char_role_ids(text)
    confidence_ids = [int(ParseConfidence.RAW)] * text_length
    diagnostic_edges: list[dict[str, int]] = []
    offset = 0
    parsed_lines = 0
    for line in text.splitlines(keepends=True):
        if _CI_LOCAL_BUILD_SIGNAL.search(line):
            parsed = parse_build_error(line, tool=tool).to_enriched_document()
            parsed_lines += 1
            for local_index, role_id in enumerate(parsed["domain_role_ids"]):
                if int(role_id) != int(DomainRoleKind.NONE):
                    role_ids[offset + local_index] = int(role_id)
            for local_index, confidence in enumerate(
                parsed["domain_confidence_ids"]
            ):
                if int(confidence) != int(ParseConfidence.RAW):
                    confidence_ids[offset + local_index] = int(confidence)
            for edge in parsed["diagnostic_edges"]:
                diagnostic_edges.append(
                    {
                        "from_char": offset + int(edge["from_char"]),
                        "to_char": offset + int(edge["to_char"]),
                        "kind": int(edge["kind"]),
                    }
                )
        offset += len(line)
    if offset != text_length:
        raise AssertionError("build CI line parser lost character coverage")
    return {
        "text": text,
        "domain_kind": domain_kind,
        "domain_ids": [domain_kind] * text_length,
        "domain_role_ids": role_ids,
        "domain_entity_ids": [0] * text_length,
        "domain_scope_ids": [0] * text_length,
        "domain_source_doc_ids": [0] * text_length,
        "domain_source_identity_ids": [0] * text_length,
        "source_identity_registry": [],
        "domain_confidence_ids": confidence_ids,
        "domain_edges": [dict(edge) for edge in diagnostic_edges],
        "build_edges": [],
        "shell_edges": [],
        "diagnostic_edges": diagnostic_edges,
        "cross_domain_edges": [],
        "domain_parse_info": {
            "parser_adapter": "ci-line-local-build",
            "parsed_lines": parsed_lines,
        },
    }


def _parse_ci_domain_sidecars(
    *,
    text: str,
    domain_kind: int,
    ci_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run the matching production diagnostic parser on a CI log."""

    domain = DomainKind(domain_kind)
    compiler = str(ci_metadata.get("compiler_info") or "").lower()
    if domain in {DomainKind.COMPILER_DIAGNOSTIC, DomainKind.COMPILER_ERROR}:
        if "msvc" in compiler or compiler in {"cl", "cl.exe"}:
            parsed = parse_msvc_diagnostic(text)
        elif "gcc" in compiler or "g++" in compiler:
            parsed = parse_gcc_diagnostic(text)
        else:
            parsed = parse_clang_diagnostic(text, tool=compiler or "clang")
    elif domain in {DomainKind.LINKER_DIAGNOSTIC, DomainKind.LINKER_ERROR}:
        parsed = parse_linker_error(text)
    elif domain == DomainKind.SANITIZER_OUTPUT:
        parsed = parse_sanitizer_output(text)
    elif domain == DomainKind.TEST_OUTPUT:
        parsed = parse_test_output(text)
    elif domain in {DomainKind.BUILD_DIAGNOSTIC, DomainKind.BUILD_ERROR}:
        build_command = str(ci_metadata.get("build_command") or "").strip()
        tool = build_command.split(maxsplit=1)[0] if build_command else "build"
        return _parse_build_ci_sidecars(
            text,
            domain_kind=domain_kind,
            tool=tool,
        )
    else:
        raise ValueError(f"unsupported CI diagnostic domain {domain.name}")
    sidecars = parsed.to_enriched_document()
    # The metadata-aware classifier owns the document route. Parsers own roles,
    # confidence and typed edges, but may see only a subset of a mixed CI log.
    sidecars["domain_kind"] = domain_kind
    sidecars["domain_ids"] = [domain_kind] * len(text)
    return sidecars


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
    parsed_sidecars = _parse_ci_domain_sidecars(
        text=text,
        domain_kind=domain_kind,
        ci_metadata=ci_meta,
    )

    # Build per-char metadata
    char_structure_ids = build_char_structure_ids(text)
    char_domain_ids = parsed_sidecars["domain_ids"]
    char_role_ids = parsed_sidecars["domain_role_ids"]
    char_confidence_ids = parsed_sidecars["domain_confidence_ids"]
    explicit_diagnostic_edges = _extract_diagnostic_edges(
        raw_doc,
        text_length=len(text),
    )
    diagnostic_edges = (
        explicit_diagnostic_edges
        if explicit_diagnostic_edges
        else parsed_sidecars["diagnostic_edges"]
    )

    # Prepare the enriched document dict compatible with the pipeline
    enriched: dict[str, Any] = {
        "text": text,
        "doc_type": raw_doc.get("doc_type", "diagnostic"),
        "domain_kind": domain_kind,
        "source_doc_id": raw_doc.get("source_doc_id", f"ci:{doc_index}"),
        "repo": raw_doc.get("repo", ""),
        "filepath": raw_doc.get("filepath") or "",
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
        "domain_entity_ids": parsed_sidecars["domain_entity_ids"],
        "domain_scope_ids": parsed_sidecars["domain_scope_ids"],
        "domain_source_doc_ids": parsed_sidecars["domain_source_doc_ids"],
        "domain_source_identity_ids": parsed_sidecars[
            "domain_source_identity_ids"
        ],
        # Diagnostic edges from domain_sidecars if available
        "diagnostic_edges": diagnostic_edges,
        "domain_edges": parsed_sidecars["domain_edges"],
        "build_edges": parsed_sidecars["build_edges"],
        "shell_edges": parsed_sidecars["shell_edges"],
        "cross_domain_edges": parsed_sidecars["cross_domain_edges"],
        # Symbol identity (empty for CI docs)
        SYMBOL_IDENTITIES_COLUMN: raw_doc.get("symbol_identities") or [],
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        SOURCE_IDENTITY_REGISTRY_COLUMN: parsed_sidecars[
            SOURCE_IDENTITY_REGISTRY_COLUMN
        ],
        # Platform info from CI metadata
        "platform_info": {
            "platform": ci_meta.get("platform") or "unknown",
            "compiler": ci_meta.get("compiler_info") or "unknown",
        },
    }
    return enriched


def _extract_diagnostic_edges(
    raw_doc: dict[str, Any],
    *,
    text_length: int,
) -> list[dict[str, int]]:
    """Extract diagnostic edges from domain_sidecars if present."""
    sidecars = raw_doc.get("domain_sidecars")
    if sidecars is None:
        return []
    if not isinstance(sidecars, dict):
        raise ValueError("domain_sidecars must be an object when present")
    edges = sidecars.get("diagnostic_edges")
    if edges is None:
        return []
    if not isinstance(edges, list):
        raise ValueError("domain_sidecars.diagnostic_edges must be a list")

    normalized: list[dict[str, int]] = []
    for edge_index, edge in enumerate(edges):
        where = f"domain_sidecars.diagnostic_edges[{edge_index}]"
        if not isinstance(edge, dict):
            raise ValueError(f"{where} must be an object")
        if "from_char" in edge or "to_char" in edge:
            if "from_char" not in edge or "to_char" not in edge:
                raise ValueError(
                    f"{where} must contain both from_char and to_char"
                )
            source = edge["from_char"]
            target = edge["to_char"]
        else:
            if "from" not in edge or "to" not in edge:
                raise ValueError(f"{where} is missing edge endpoints")
            source = edge["from"]
            target = edge["to"]
        kind = edge.get("kind")
        for field, value in (
            ("from", source),
            ("to", target),
            ("kind", kind),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{where}.{field} must be an integer")
        if not 0 <= source < text_length or not 0 <= target < text_length:
            raise ValueError(
                f"{where} endpoints {source}->{target} outside text length "
                f"{text_length}"
            )
        if kind < 0:
            raise ValueError(f"{where}.kind must be non-negative")
        normalized.append(
            {
                "from_char": source,
                "to_char": target,
                "kind": kind,
            }
        )
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

def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def capture_ci_code_revision(
    expected_commit: str | None,
    *,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, object]:
    """Capture and enforce the canonical cppmega.mlx source-tree revision."""

    if expected_commit is None:
        raise ValueError(
            "--expected-code-revision is required for production CI tokenization"
        )
    expected = expected_commit.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError(
            "--expected-code-revision must be an exact 40-character Git commit"
        )

    # Keep the read-only/dry-run tokenizer lightweight, but production must
    # reuse the conveyor's one authoritative revision/pathspec implementation.
    from scripts.streaming_conveyor import (
        capture_code_revision as capture_canonical_code_revision,
    )

    revision = capture_canonical_code_revision(repo_root)
    actual_commit = revision.get("git_commit")
    if actual_commit != expected:
        raise RuntimeError(
            "expected CI producer revision does not match HEAD: "
            f"expected={expected} actual={actual_commit}"
        )
    if revision.get("dirty") is not False:
        raise RuntimeError(
            "production CI tokenization requires a clean canonical "
            f"cppmega.mlx source/config scope: {revision.get('dirty_fingerprint')}"
        )
    source_tree_sha256 = revision.get("source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256) is None
    ):
        raise RuntimeError(
            "cppmega.mlx revision receipt lacks a canonical source_tree_sha256"
        )
    return {
        **revision,
        "schema": "cppmega_ci_code_revision_v2",
        "repository_identity": "cppmega.mlx",
    }


def discover_ci_input_files(
    input_path: Path,
    *,
    allowed_auxiliary_jsonl: Iterable[Path] = (),
) -> list[Path]:
    """Resolve the exact CI source inventory without filename fallbacks."""

    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"CI input file must be JSONL: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"CI input path does not exist: {input_path}")

    required = [input_path / name for name in CANONICAL_CI_INPUT_NAMES]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{input_path}: canonical CI input directory is incomplete; "
            f"missing={missing}"
        )
    allowed_auxiliary = {path.resolve() for path in allowed_auxiliary_jsonl}
    unexpected = sorted(
        path.name
        for path in input_path.glob("*.jsonl")
        if path.name not in CANONICAL_CI_INPUT_NAMES
        and path.resolve() not in allowed_auxiliary
    )
    if unexpected:
        raise RuntimeError(
            f"{input_path}: unbound extra JSONL inputs present: {unexpected}; "
            "pass an explicit file or reconcile the canonical inventory"
        )
    return [path.resolve() for path in required]


def inventory_ci_inputs(files: Iterable[Path]) -> list[dict[str, object]]:
    inventory = []
    for path in files:
        stat = path.stat()
        inventory.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256(path),
            }
        )
    return inventory


def load_ci_log_completion(
    path: Path,
    *,
    logs_path: Path,
) -> dict[str, object]:
    """Validate the exact GitHub Actions extraction receipt for the log input."""

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid CI log completion receipt {path}: {error}") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != CI_LOG_COMPLETION_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("unresolved_count") != 0
    ):
        raise RuntimeError(f"{path}: CI log extraction is not complete")
    for name in (
        "unique_job_count",
        "fetched_count",
        "expired_count",
        "too_short_count",
    ):
        value = receipt.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise RuntimeError(f"{path}: invalid CI completion counter {name}")
    if (
        receipt["unique_job_count"]
        != receipt["fetched_count"]
        + receipt["expired_count"]
        + receipt["too_short_count"]
    ):
        raise RuntimeError(f"{path}: CI completion job accounting drifted")
    output = receipt.get("output")
    state = receipt.get("state")
    if not isinstance(output, dict) or not isinstance(state, dict):
        raise RuntimeError(f"{path}: CI completion lacks output/state bindings")
    if (
        Path(str(output.get("path", ""))).resolve() != logs_path.resolve()
        or output.get("row_count") != receipt["fetched_count"]
        or output.get("size") != logs_path.stat().st_size
        or output.get("sha256") != _sha256(logs_path)
    ):
        raise RuntimeError(f"{path}: CI log output binding drifted")
    state_path = Path(str(state.get("path", "")))
    if (
        not state_path.is_file()
        or state.get("row_count") != receipt["unique_job_count"]
        or state.get("size") != state_path.stat().st_size
        or state.get("sha256") != _sha256(state_path)
    ):
        raise RuntimeError(f"{path}: CI fetch-state binding drifted")
    expired_jobs = receipt.get("expired_jobs")
    if (
        not isinstance(expired_jobs, list)
        or len(expired_jobs) != receipt["expired_count"]
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("job_id"), int)
            or "HTTP 410" not in str(item.get("detail", ""))
            for item in expired_jobs
        )
    ):
        raise RuntimeError(f"{path}: expired CI jobs lack exact HTTP 410 evidence")
    return {
        "schema": CI_LOG_COMPLETION_SCHEMA,
        "status": "complete",
        "receipt_path": str(path.resolve()),
        "receipt_size": path.stat().st_size,
        "receipt_sha256": _sha256(path),
        "unique_job_count": receipt["unique_job_count"],
        "fetched_count": receipt["fetched_count"],
        "expired_count": receipt["expired_count"],
        "too_short_count": receipt["too_short_count"],
        "unresolved_count": 0,
        "source_inventory_sha256": receipt.get("source_inventory_sha256"),
        "job_set_sha256": receipt.get("job_set_sha256"),
        "output": output,
        "state": state,
        "expired_jobs": expired_jobs,
    }


def _validate_raw_ci_document(
    value: object,
    *,
    path: Path,
    line_number: int,
    seen_source_doc_ids: set[str],
) -> dict[str, Any]:
    where = f"{path}:{line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{where}: CI JSONL row must be an object")
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: CI document text must be a non-empty string")
    source_doc_id = value.get("source_doc_id")
    if not isinstance(source_doc_id, str) or not source_doc_id:
        raise ValueError(f"{where}: missing non-empty source_doc_id")
    if source_doc_id in seen_source_doc_ids:
        raise ValueError(f"{where}: duplicate source_doc_id={source_doc_id!r}")
    seen_source_doc_ids.add(source_doc_id)
    repo = value.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError(f"{where}: missing non-empty repo")
    doc_type = value.get("doc_type")
    if not isinstance(doc_type, str) or not doc_type:
        raise ValueError(f"{where}: missing non-empty doc_type")
    filepath = value.get("filepath")
    if filepath is not None and not isinstance(filepath, str):
        raise ValueError(f"{where}: filepath must be a string or null")
    commit_hash = value.get("commit_hash")
    if (
        not isinstance(commit_hash, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit_hash) is None
    ):
        raise ValueError(f"{where}: commit_hash must be an exact lowercase Git SHA")
    ci_metadata = value.get("ci_metadata")
    if not isinstance(ci_metadata, dict):
        raise ValueError(f"{where}: ci_metadata must be an object")
    for field in ("run_id", "job_id"):
        raw_id = ci_metadata.get(field)
        if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id <= 0:
            raise ValueError(f"{where}: ci_metadata.{field} must be a positive integer")
    for field in (
        "workflow",
        "job_name",
        "conclusion",
        "platform",
        "compiler_info",
        "build_command",
        "created_at",
    ):
        field_value = ci_metadata.get(field)
        if field_value is not None and not isinstance(field_value, str):
            raise ValueError(
                f"{where}: ci_metadata.{field} must be a string or null"
            )
    diagnostics_count = ci_metadata.get("diagnostics_count")
    if diagnostics_count is not None and (
        not isinstance(diagnostics_count, int)
        or isinstance(diagnostics_count, bool)
        or diagnostics_count < 0
    ):
        raise ValueError(
            f"{where}: ci_metadata.diagnostics_count must be a non-negative integer"
        )
    severities = ci_metadata.get("severities")
    if severities is not None and (
        not isinstance(severities, list)
        or any(
            not isinstance(severity, str) or not severity
            for severity in severities
        )
    ):
        raise ValueError(
            f"{where}: ci_metadata.severities must be a list of non-empty strings"
        )
    domain_sidecars = value.get("domain_sidecars")
    if domain_sidecars is not None and not isinstance(domain_sidecars, dict):
        raise ValueError(f"{where}: domain_sidecars must be an object or null")
    symbol_identities = value.get("symbol_identities")
    if symbol_identities is not None and not isinstance(symbol_identities, list):
        raise ValueError(f"{where}: symbol_identities must be a list or null")
    return value


def iter_ci_jsonl_files(
    files: Iterable[Path],
    *,
    max_docs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield strict UTF-8/JSON CI rows; malformed input is terminal."""

    seen_source_doc_ids: set[str] = set()
    emitted = 0
    for path in files:
        log.info("Reading %s ...", path)
        file_docs = 0
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: malformed JSON: {error.msg}"
                    ) from error
                yield _validate_raw_ci_document(
                    value,
                    path=path,
                    line_number=line_number,
                    seen_source_doc_ids=seen_source_doc_ids,
                )
                emitted += 1
                file_docs += 1
                if max_docs > 0 and emitted >= max_docs:
                    log.info("Stopped at --max-docs=%d", max_docs)
                    return
        log.info("  %s: %d documents", path.name, file_docs)


def read_ci_jsonl_files(input_path: Path) -> list[dict[str, Any]]:
    """Compatibility helper returning the strict, inventory-bound input."""

    return list(iter_ci_jsonl_files(discover_ci_input_files(input_path)))


def _batched(
    values: Iterable[dict[str, Any]],
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fragment_lengths(
    token_count: int,
    *,
    min_bucket: int,
    max_bucket: int,
    preferred_boundaries: Iterable[int] = (),
) -> list[int]:
    if token_count <= 0:
        raise ValueError("cannot fragment an empty token sequence")
    if token_count <= max_bucket:
        return [token_count]
    preferred = sorted(
        {
            int(boundary)
            for boundary in preferred_boundaries
            if 0 < int(boundary) < token_count
        }
    )
    lengths: list[int] = []
    offset = 0
    while token_count - offset > max_bucket:
        maximum_cut = offset + max_bucket
        if token_count - maximum_cut < min_bucket:
            maximum_cut = token_count - min_bucket
        candidates = [
            boundary
            for boundary in preferred
            if (
                offset + min_bucket <= boundary <= maximum_cut
                and token_count - boundary >= min_bucket
            )
        ]
        cut = candidates[-1] if candidates else maximum_cut
        if cut <= offset:
            raise AssertionError("invalid fixed-bucket split boundary")
        lengths.append(cut - offset)
        offset = cut
    lengths.append(token_count - offset)
    if sum(lengths) != token_count or any(
        length <= 0 or length > max_bucket for length in lengths
    ):
        raise AssertionError("fixed-bucket fragmentation lost tokens")
    return lengths


_TOKEN_EDGE_COLUMNS = (
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
)
_CHUNK_EDGE_COLUMNS = (TOKEN_CALL_EDGES_COLUMN, TOKEN_TYPE_EDGES_COLUMN)


def _edge_endpoints(edge: object, *, column: str) -> tuple[int, int]:
    if not isinstance(edge, dict):
        raise ValueError(f"{column}: edge must be an object, got {type(edge).__name__}")
    try:
        source = int(edge["from"])
        target = int(edge["to"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{column}: malformed edge={edge!r}") from error
    return source, target


def split_tokenized_ci_document(
    record: dict[str, Any],
    *,
    seq_lengths: list[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Losslessly split one tokenized CI row into representable fragments.

    Token-aligned sidecars are sliced exactly. Intra-fragment graph edges are
    rebased. Edges crossing a fixed-sequence boundary cannot be represented by
    the packed-row schema, so their exact count is returned for the manifest;
    tokens are never rejected or truncated.
    """

    token_ids = record.get(TOKEN_IDS_COLUMN)
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("tokenized CI document has missing/empty token_ids")
    token_count = len(token_ids)
    for column in PACKED_ROWS_TOKEN_METADATA_COLUMNS:
        values = record.get(column)
        if values and (not isinstance(values, list) or len(values) != token_count):
            raise ValueError(
                f"{column}: expected {token_count} token-aligned values, "
                f"got {len(values) if isinstance(values, list) else type(values).__name__}"
            )
    for column in TOKENIZED_ENRICHED_OBJECTIVE_COLUMNS:
        if record.get(column):
            raise ValueError(
                f"CI document unexpectedly contains objective section {column}; "
                "fragmentation semantics are undefined"
            )

    starts = record.get(TOKEN_CHUNK_STARTS_COLUMN) or []
    ends = record.get(TOKEN_CHUNK_ENDS_COLUMN) or []
    kinds = record.get(TOKEN_CHUNK_KINDS_COLUMN) or []
    dep_levels = record.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN) or []
    if len({len(starts), len(ends), len(kinds), len(dep_levels)}) != 1:
        raise ValueError("CI token chunk sidecars have inconsistent lengths")

    lengths = _fragment_lengths(
        token_count,
        min_bucket=min(seq_lengths),
        max_bucket=max(seq_lengths),
        preferred_boundaries=(int(end) for end in ends),
    )
    fragment_bounds: list[tuple[int, int]] = []
    running_offset = 0
    for length in lengths:
        fragment_bounds.append((running_offset, running_offset + length))
        running_offset += length

    def token_fragment_index(token_index: int, *, column: str) -> int:
        if not 0 <= token_index < token_count:
            raise ValueError(
                f"{column}: token edge endpoint {token_index} outside {token_count}"
            )
        return next(
            index
            for index, (start, end) in enumerate(fragment_bounds)
            if start <= token_index < end
        )

    chunk_fragment_memberships: dict[int, set[int]] = {}
    for chunk_index, (start, end) in enumerate(zip(starts, ends)):
        start_i = int(start)
        end_i = int(end)
        if not (0 <= start_i < end_i <= token_count):
            raise ValueError(
                f"chunk {chunk_index} out of range: "
                f"{start_i}:{end_i}/{token_count}"
            )
        chunk_fragment_memberships[chunk_index] = {
            fragment_index
            for fragment_index, (fragment_start, fragment_end) in enumerate(
                fragment_bounds
            )
            if max(start_i, fragment_start) < min(end_i, fragment_end)
        }

    origin_source_doc_id = str(record.get("source_doc_id", ""))
    fragments: list[dict[str, Any]] = []
    counters = {
        "source_tokens": token_count,
        "fragment_tokens": 0,
        "fragments": len(lengths),
        "split_source_docs": int(len(lengths) > 1),
        "cross_boundary_chunk_edges": 0,
        "cross_boundary_token_edges": 0,
    }
    for column in (*_CHUNK_EDGE_COLUMNS, *_TOKEN_EDGE_COLUMNS):
        counters[f"source_{column}"] = len(record.get(column) or [])
        counters[f"fragment_{column}"] = 0
    for column in _CHUNK_EDGE_COLUMNS:
        for edge in record.get(column) or []:
            source, target = _edge_endpoints(edge, column=column)
            if (
                source not in chunk_fragment_memberships
                or target not in chunk_fragment_memberships
            ):
                raise ValueError(
                    f"{column}: chunk edge endpoint out of range: {source}->{target}"
                )
            if chunk_fragment_memberships[source].isdisjoint(
                chunk_fragment_memberships[target]
            ):
                counters["cross_boundary_chunk_edges"] += 1
    for column in _TOKEN_EDGE_COLUMNS:
        for edge in record.get(column) or []:
            source, target = _edge_endpoints(edge, column=column)
            if token_fragment_index(source, column=column) != token_fragment_index(
                target,
                column=column,
            ):
                counters["cross_boundary_token_edges"] += 1

    offset = 0
    for fragment_index, length in enumerate(lengths):
        fragment_end = offset + length
        fragment = dict(record)
        fragment["origin_source_doc_id"] = origin_source_doc_id
        fragment["source_doc_id"] = (
            f"{origin_source_doc_id}#ci-fragment={fragment_index + 1}/{len(lengths)}"
        )
        fragment["ci_fragment_index"] = fragment_index
        fragment["ci_fragment_count"] = len(lengths)
        fragment[TOKEN_IDS_COLUMN] = list(token_ids[offset:fragment_end])
        for column in PACKED_ROWS_TOKEN_METADATA_COLUMNS:
            values = record.get(column)
            fragment[column] = (
                list(values[offset:fragment_end]) if values else []
            )

        old_to_new_chunk: dict[int, int] = {}
        new_starts: list[int] = []
        new_ends: list[int] = []
        new_kinds: list[int] = []
        new_dep_levels: list[int] = []
        for old_index, (start, end, kind, dep_level) in enumerate(
            zip(starts, ends, kinds, dep_levels)
        ):
            start_i = int(start)
            end_i = int(end)
            if not (0 <= start_i < end_i <= token_count):
                raise ValueError(
                    f"chunk {old_index} out of range: {start_i}:{end_i}/{token_count}"
                )
            clipped_start = max(start_i, offset)
            clipped_end = min(end_i, fragment_end)
            if clipped_start >= clipped_end:
                continue
            old_to_new_chunk[old_index] = len(new_starts)
            new_starts.append(clipped_start - offset)
            new_ends.append(clipped_end - offset)
            new_kinds.append(int(kind))
            new_dep_levels.append(int(dep_level))
        fragment[TOKEN_CHUNK_STARTS_COLUMN] = new_starts
        fragment[TOKEN_CHUNK_ENDS_COLUMN] = new_ends
        fragment[TOKEN_CHUNK_KINDS_COLUMN] = new_kinds
        fragment[TOKEN_CHUNK_DEP_LEVELS_COLUMN] = new_dep_levels

        for column in _CHUNK_EDGE_COLUMNS:
            rebased = []
            for edge in record.get(column) or []:
                source, target = _edge_endpoints(edge, column=column)
                if source in old_to_new_chunk and target in old_to_new_chunk:
                    rebased.append(
                        {
                            "from": old_to_new_chunk[source],
                            "to": old_to_new_chunk[target],
                        }
                    )
            fragment[column] = rebased
            counters[f"fragment_{column}"] += len(rebased)

        for column in _TOKEN_EDGE_COLUMNS:
            rebased = []
            for edge in record.get(column) or []:
                source, target = _edge_endpoints(edge, column=column)
                if offset <= source < fragment_end and offset <= target < fragment_end:
                    rebased.append(
                        {
                            "from": source - offset,
                            "to": target - offset,
                            "kind": int(edge["kind"]),
                        }
                    )
            fragment[column] = rebased
            counters[f"fragment_{column}"] += len(rebased)

        changed_ids = record.get(CHANGED_CHUNK_IDS_COLUMN) or []
        changed_spans = record.get(CHANGED_CHUNK_SPANS_COLUMN) or []
        if len(changed_ids) != len(changed_spans):
            raise ValueError("changed chunk ids/spans have inconsistent lengths")
        fragment_changed_ids: list[int] = []
        fragment_changed_spans: list[dict[str, int]] = []
        for changed_id, span in zip(changed_ids, changed_spans):
            old_id = int(changed_id)
            if old_id not in old_to_new_chunk:
                continue
            if not isinstance(span, dict) or "start" not in span or "end" not in span:
                raise ValueError(f"malformed changed chunk span: {span!r}")
            span_start = max(int(span["start"]), offset)
            span_end = min(int(span["end"]), fragment_end)
            if span_start >= span_end:
                continue
            fragment_changed_ids.append(old_to_new_chunk[old_id])
            fragment_changed_spans.append(
                {"start": span_start - offset, "end": span_end - offset}
            )
        fragment[CHANGED_CHUNK_IDS_COLUMN] = fragment_changed_ids
        fragment[CHANGED_CHUNK_SPANS_COLUMN] = fragment_changed_spans
        fragments.append(fragment)
        counters["fragment_tokens"] += length
        offset = fragment_end

    if offset != token_count or counters["fragment_tokens"] != token_count:
        raise AssertionError("CI split token conservation failure")
    return fragments, counters


@dataclass
class _BucketWriter:
    bucket: int
    path: Path
    writer: pq.ParquetWriter | None = None
    fragments: int = 0
    packed_rows: int = 0
    valid_tokens: int = 0
    trained_tokens: int = 0
    capacity_tokens: int = 0
    domain_kind_counts: dict[int, int] | None = None

    def __post_init__(self) -> None:
        self.domain_kind_counts = {}

    def write(
        self,
        fragments: list[dict[str, Any]],
        *,
        next_source_doc_index: int,
        pad_token_id: int,
        signature_to_id: dict[str, int],
        id_to_signature: dict[int, str],
        emit_output: bool,
    ) -> int:
        normalized: list[NormalizedDoc] = []
        for fragment in fragments:
            signature = str(fragment["source_doc_id"])
            stable_doc_id = int.from_bytes(
                hashlib.sha256(signature.encode("utf-8")).digest()[:4], "big"
            ) or 1
            collision = id_to_signature.get(stable_doc_id)
            if collision is not None and collision != signature:
                raise ValueError(
                    "stable CI fragment id collision: "
                    f"id={stable_doc_id} {collision!r} != {signature!r}"
                )
            id_to_signature[stable_doc_id] = signature
            signature_to_id[signature] = stable_doc_id
            normalized.append(
                normalize_document_record(
                    fragment,
                    source_doc_index=next_source_doc_index,
                    stable_doc_id=stable_doc_id,
                )
            )
            next_source_doc_index += 1

        rows, overflow = pack_documents(
            normalized,
            target_length=self.bucket,
            pad_token_id=pad_token_id,
            strategy="best_fit",
        )
        if overflow:
            raise RuntimeError(
                f"bucket {self.bucket}: fixed-fragment packer produced "
                f"{len(overflow)} overflow documents"
            )
        for row in rows:
            row[PACK_ID_COLUMN] = self.packed_rows + int(row[PACK_ID_COLUMN])
            valid_tokens = int(row[VALID_TOKEN_COUNT_COLUMN])
            if not 0 < valid_tokens <= self.bucket:
                raise RuntimeError(
                    f"bucket {self.bucket}: invalid valid_token_count={valid_tokens}"
                )
            for column in (INPUT_IDS_COLUMN, TARGET_IDS_COLUMN, LOSS_MASK_COLUMN):
                if len(row[column]) != self.bucket:
                    raise RuntimeError(
                        f"bucket {self.bucket}: {column} width={len(row[column])}"
                    )
            self.valid_tokens += valid_tokens
            self.trained_tokens += int(row.get("trained_token_count", 0))
            self.capacity_tokens += self.bucket
        if emit_output:
            table = rows_to_table(rows)
            if self.writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.writer = pq.ParquetWriter(
                    self.path,
                    table.schema,
                    compression="zstd",
                    compression_level=3,
                )
            self.writer.write_table(table, row_group_size=min(128, len(rows)))
        self.fragments += len(fragments)
        self.packed_rows += len(rows)
        assert self.domain_kind_counts is not None
        for fragment in fragments:
            domain_kind = int(fragment["domain_kind"])
            self.domain_kind_counts[domain_kind] = (
                self.domain_kind_counts.get(domain_kind, 0) + 1
            )
        return next_source_doc_index

    def close(self, *, create_empty: bool = False) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        elif create_empty and not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist([], schema=PACKED_ROW_OUTPUT_SCHEMA),
                self.path,
                compression="zstd",
                compression_level=3,
            )


def _verify_fixed_bucket_parquet(path: Path, bucket: int) -> dict[str, int]:
    rows = 0
    valid_tokens = 0
    for batch in pq.ParquetFile(path).iter_batches(
        columns=(
            VALID_TOKEN_COUNT_COLUMN,
            INPUT_IDS_COLUMN,
            TARGET_IDS_COLUMN,
            LOSS_MASK_COLUMN,
        ),
        batch_size=64,
    ):
        for row in batch.to_pylist():
            valid = int(row[VALID_TOKEN_COUNT_COLUMN])
            if not 0 < valid <= bucket:
                raise RuntimeError(f"{path}: invalid valid_token_count={valid}")
            for column in (INPUT_IDS_COLUMN, TARGET_IDS_COLUMN, LOSS_MASK_COLUMN):
                if len(row[column]) != bucket:
                    raise RuntimeError(
                        f"{path}: {column} has width {len(row[column])}, "
                        f"expected {bucket}"
                    )
            rows += 1
            valid_tokens += valid
    return {"rows": rows, "valid_tokens": valid_tokens}


def _domain_counts_json(counts: dict[int, int]) -> dict[str, int]:
    return {
        DomainKind(kind).name: count
        for kind, count in sorted(counts.items())
    }


def _assert_input_inventory_unchanged(
    inventory: list[dict[str, object]],
) -> None:
    for entry in inventory:
        path = Path(str(entry["path"]))
        stat = path.stat()
        if (
            stat.st_size != int(entry["size"])
            or stat.st_mtime_ns != int(entry["mtime_ns"])
            or _sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(f"CI source changed during tokenization: {path}")


def tokenize_and_pack(
    docs: Iterable[dict[str, Any]],
    *,
    tokenizer,
    seq_lengths: list[int],
    output_dir: Path,
    timestamp: str,
    dry_run: bool = False,
    batch_size: int = 16,
    pad_token_id: int = 0,
    source_inventory: list[dict[str, object]] | None = None,
    require_nonempty_buckets: bool = True,
    producer_revision: dict[str, object] | None = None,
    source_completion: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Tokenize CI documents into five fixed, manifest-bound bucket streams."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_lengths != sorted(set(seq_lengths)) or any(
        length <= 0 for length in seq_lengths
    ):
        raise ValueError("seq_lengths must be unique, sorted, positive integers")
    tokenizer_fp = tokenizer_fingerprint(tokenizer)
    output_base = output_dir / f"reindexed_ci_{timestamp}_ci"
    partial_base = output_dir / f".reindexed_ci_{timestamp}_ci.partial"
    if not dry_run:
        if output_base.exists() or partial_base.exists():
            raise FileExistsError(
                f"CI output already exists or is incomplete: {output_base}"
            )
        partial_base.mkdir(parents=True)

    writers = {
        bucket: _BucketWriter(
            bucket=bucket,
            path=partial_base / str(bucket) / f"ci_packed_{bucket}.parquet",
        )
        for bucket in seq_lengths
    }
    counters: dict[str, int] = {
        "input_docs": 0,
        "tokenized_docs": 0,
        "source_tokens": 0,
        "fragment_tokens": 0,
        "fragments": 0,
        "split_source_docs": 0,
        "cross_boundary_chunk_edges": 0,
        "cross_boundary_token_edges": 0,
        "malformed_json_rows": 0,
        "empty_text_docs": 0,
        "zero_token_docs": 0,
        "normalization_rejects": 0,
        "packing_overflow_docs": 0,
        "unexpected_rejects": 0,
    }
    for column in (*_CHUNK_EDGE_COLUMNS, *_TOKEN_EDGE_COLUMNS):
        counters[f"source_{column}"] = 0
        counters[f"fragment_{column}"] = 0
    domain_kind_counts: dict[int, int] = {}
    signature_to_id: dict[str, int] = {}
    id_to_signature: dict[int, str] = {}
    next_source_doc_index = 0
    published = False
    try:
        for raw_batch in _batched(docs, batch_size):
            counters["input_docs"] += len(raw_batch)
            enriched_batch = [
                prepare_ci_document(
                    raw_doc,
                    doc_index=counters["tokenized_docs"] + batch_index,
                )
                for batch_index, raw_doc in enumerate(raw_batch)
            ]
            tokenized_batch = materialize_tokenized_enriched_batch(
                enriched_batch,
                tokenizer,
            )
            if len(tokenized_batch) != len(enriched_batch):
                raise RuntimeError(
                    "materialize_tokenized_enriched_batch changed document count"
                )
            bucket_fragments: dict[int, list[dict[str, Any]]] = {
                bucket: [] for bucket in seq_lengths
            }
            for enriched, tokenized in zip(enriched_batch, tokenized_batch):
                merged = {**enriched, **tokenized}
                source_tokens = merged.get(TOKEN_IDS_COLUMN)
                if not isinstance(source_tokens, list) or not source_tokens:
                    counters["zero_token_docs"] += 1
                    counters["unexpected_rejects"] += 1
                    raise ValueError(
                        f"{merged.get('source_doc_id')}: tokenizer emitted zero tokens"
                    )
                fragments, split_counts = split_tokenized_ci_document(
                    merged,
                    seq_lengths=seq_lengths,
                )
                for key, value in split_counts.items():
                    counters[key] += value
                counters["tokenized_docs"] += 1
                domain_kind = int(merged["domain_kind"])
                domain_kind_counts[domain_kind] = (
                    domain_kind_counts.get(domain_kind, 0) + 1
                )
                for fragment in fragments:
                    bucket = assign_bucket(
                        len(fragment[TOKEN_IDS_COLUMN]),
                        seq_lengths,
                    )
                    if bucket is None:
                        counters["packing_overflow_docs"] += 1
                        counters["unexpected_rejects"] += 1
                        raise AssertionError(
                            "lossless splitter emitted an oversized fragment"
                        )
                    bucket_fragments[bucket].append(fragment)

            for bucket, fragments in bucket_fragments.items():
                if not fragments:
                    continue
                try:
                    next_source_doc_index = writers[bucket].write(
                        fragments,
                        next_source_doc_index=next_source_doc_index,
                        pad_token_id=pad_token_id,
                        signature_to_id=signature_to_id,
                        id_to_signature=id_to_signature,
                        emit_output=not dry_run,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    counters["normalization_rejects"] += 1
                    counters["unexpected_rejects"] += 1
                    raise RuntimeError(
                        f"CI normalization failed in bucket {bucket}: {error}"
                    ) from error
            if counters["tokenized_docs"] % (batch_size * 10) == 0:
                log.info(
                    "Tokenized %d docs -> %d fixed fragments",
                    counters["tokenized_docs"],
                    counters["fragments"],
                )

        if counters["input_docs"] == 0:
            raise RuntimeError("CI input contains zero documents")
        if counters["source_tokens"] != counters["fragment_tokens"]:
            raise RuntimeError(
                "CI token conservation failed: "
                f"source={counters['source_tokens']} "
                f"fragments={counters['fragment_tokens']}"
            )
        if counters["unexpected_rejects"] != 0:
            raise RuntimeError(f"CI pipeline rejected data: {counters}")
        if require_nonempty_buckets:
            empty = [
                bucket for bucket, writer in writers.items()
                if writer.fragments == 0
            ]
            if empty:
                raise RuntimeError(f"CI production buckets are empty: {empty}")

        if dry_run:
            if source_inventory is not None:
                _assert_input_inventory_unchanged(source_inventory)
            return {
                "schema": CI_MANIFEST_SCHEMA,
                "dry_run": True,
                "seq_lengths": seq_lengths,
                "source_inventory": source_inventory or [],
                "source_inventory_sha256": hashlib.sha256(
                    json.dumps(
                        source_inventory or [],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "source_completion": source_completion,
                "tokenizer_fingerprint": tokenizer_fp,
                "counters": counters,
                "domain_kind_counts": _domain_counts_json(domain_kind_counts),
                "buckets": {
                    str(bucket): {
                        "fragments": writer.fragments,
                        "rows": writer.packed_rows,
                        "valid_tokens": writer.valid_tokens,
                    }
                    for bucket, writer in writers.items()
                },
            }

        for writer in writers.values():
            writer.close(create_empty=True)
        if source_inventory is not None:
            _assert_input_inventory_unchanged(source_inventory)

        bucket_receipts: dict[str, dict[str, object]] = {}
        for bucket, writer in writers.items():
            verification = _verify_fixed_bucket_parquet(writer.path, bucket)
            if (
                verification["rows"] != writer.packed_rows
                or verification["valid_tokens"] != writer.valid_tokens
            ):
                raise RuntimeError(
                    f"bucket {bucket}: post-write verification drift: "
                    f"{verification} != rows={writer.packed_rows}, "
                    f"valid_tokens={writer.valid_tokens}"
                )
            bucket_manifest = {
                "schema": CI_BUCKET_MANIFEST_SCHEMA,
                "kind": "ci",
                "bucket_seq_length": bucket,
                "fragments": writer.fragments,
                "packed_rows": writer.packed_rows,
                "valid_tokens": writer.valid_tokens,
                "trained_tokens": writer.trained_tokens,
                "capacity_tokens": writer.capacity_tokens,
                "packing_overflow_docs": 0,
                "parquet": {
                    "path": writer.path.name,
                    "size": writer.path.stat().st_size,
                    "sha256": _sha256(writer.path),
                },
                "domain_kind_counts": _domain_counts_json(
                    writer.domain_kind_counts or {}
                ),
                "fixed_width_verified": True,
            }
            bucket_manifest_path = writer.path.parent / "manifest.json"
            _write_json_atomic(bucket_manifest_path, bucket_manifest)
            bucket_receipts[str(bucket)] = {
                **bucket_manifest,
                "manifest": {
                    "path": str(
                        bucket_manifest_path.relative_to(partial_base)
                    ),
                    "sha256": _sha256(bucket_manifest_path),
                },
                "parquet": {
                    **bucket_manifest["parquet"],
                    "path": str(writer.path.relative_to(partial_base)),
                },
            }

        manifest = {
            "schema": CI_MANIFEST_SCHEMA,
            "kind": "ci",
            "created_at": _utc_now(),
            "output_name": output_base.name,
            "seq_lengths": seq_lengths,
            "source_inventory": source_inventory or [],
            "source_inventory_sha256": hashlib.sha256(
                json.dumps(
                    source_inventory or [],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_completion": source_completion,
            "tokenizer": {
                "fingerprint": tokenizer_fp,
                "contract_sha256": TOKENIZER_CONTRACT_SHA256,
                "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
            },
            "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
            "sidecar_schema": "case5_domain_routes_v1",
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "producer": {
                "script": Path(__file__).name,
                "script_sha256": _sha256(Path(__file__).resolve()),
                "code_revision": producer_revision,
            },
            "counters": counters,
            "split_policy": {
                "schema": "cppmega_ci_lossless_token_fragmentation_v1",
                "token_loss": 0,
                "cross_boundary_edges_are_counted": True,
            },
            "domain_kind_counts": _domain_counts_json(domain_kind_counts),
            "buckets": bucket_receipts,
            "verification": {
                "fixed_width_all_rows": True,
                "source_tokens_equal_fragment_tokens": True,
                "unexpected_rejects": 0,
                "packing_overflow_docs": 0,
            },
        }
        _write_json_atomic(partial_base / "manifest.json", manifest)
        os.replace(partial_base, output_base)
        published = True
        manifest["output_dir"] = str(output_base)
        log.info(
            "Published fixed CI corpus: %s (%d docs, %d tokens, %d fragments)",
            output_base,
            counters["input_docs"],
            counters["source_tokens"],
            counters["fragments"],
        )
        return manifest
    finally:
        for writer in writers.values():
            writer.close()
        if not dry_run and not published and partial_base.exists():
            shutil.rmtree(partial_base)


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
        default=",".join(str(length) for length in DEFAULT_SEQ_LENGTHS),
        help=(
            "Comma-separated sequence length buckets "
            f"(default: {','.join(str(length) for length in DEFAULT_SEQ_LENGTHS)})."
        ),
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
        default=16,
        help="Bounded tokenization batch size (default: 16).",
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
        help="Dry-run-only document cap (0 = all).",
    )
    parser.add_argument(
        "--allow-empty-buckets",
        action="store_true",
        help=(
            "Allow a smoke/dry-run input to leave configured buckets empty. "
            "Production CI runs must not use this option."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Explicit output generation id. The output directory is "
            "reindexed_ci_<run-id>_ci. Defaults to the current UTC timestamp."
        ),
    )
    parser.add_argument(
        "--expected-code-revision",
        default=None,
        help=(
            "Required exact clean cppmega.mlx Git commit for production CI "
            "tokenization."
        ),
    )
    parser.add_argument(
        "--ci-log-completion-receipt",
        default=None,
        help=(
            "Required cppmega_ci_log_extraction_v1 receipt for production. "
            "Dry-runs may omit it."
        ),
    )
    args = parser.parse_args()

    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",") if x.strip()]
    if not seq_lengths:
        parser.error("--seq-lengths must contain at least one value")
    seq_lengths.sort()
    if len(seq_lengths) != len(set(seq_lengths)):
        parser.error("--seq-lengths must not contain duplicates")
    if args.max_docs > 0 and not args.dry_run:
        parser.error("--max-docs is allowed only with --dry-run")
    if args.allow_empty_buckets and not args.dry_run:
        parser.error("--allow-empty-buckets is allowed only with --dry-run")
    if not args.dry_run and tuple(seq_lengths) != DEFAULT_SEQ_LENGTHS:
        parser.error(
            "production CI requires the exact fixed ladder "
            f"{','.join(str(length) for length in DEFAULT_SEQ_LENGTHS)}"
        )
    if args.run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*",
        args.run_id,
    ):
        parser.error("--run-id must contain only [A-Za-z0-9_.-]")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not args.dry_run and input_path.is_file():
        parser.error(
            "production CI tokenization requires the canonical input directory "
            f"containing exactly {list(CANONICAL_CI_INPUT_NAMES)}"
        )
    if not args.dry_run and args.ci_log_completion_receipt is None:
        parser.error("--ci-log-completion-receipt is required for production")
    source_completion = (
        None
        if args.ci_log_completion_receipt is None
        else load_ci_log_completion(
            Path(args.ci_log_completion_receipt),
            logs_path=(
                input_path / "ci_logs_enriched.jsonl"
                if input_path.is_dir()
                else input_path
            ),
        )
    )
    allowed_auxiliary_jsonl = (
        ()
        if source_completion is None
        else (Path(str(source_completion["state"]["path"])),)
    )
    files = discover_ci_input_files(
        input_path,
        allowed_auxiliary_jsonl=allowed_auxiliary_jsonl,
    )
    source_inventory = inventory_ci_inputs(files)
    producer_revision = (
        None
        if args.dry_run
        else capture_ci_code_revision(args.expected_code_revision)
    )

    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    log.info("Tokenizer loaded (fingerprint: %s)", tokenizer_fingerprint(tokenizer))

    timestamp = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    summary = tokenize_and_pack(
        iter_ci_jsonl_files(files, max_docs=args.max_docs),
        tokenizer=tokenizer,
        seq_lengths=seq_lengths,
        output_dir=output_dir,
        timestamp=timestamp,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        pad_token_id=args.pad_token_id,
        source_inventory=source_inventory,
        require_nonempty_buckets=not args.allow_empty_buckets,
        producer_revision=producer_revision,
        source_completion=source_completion,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
