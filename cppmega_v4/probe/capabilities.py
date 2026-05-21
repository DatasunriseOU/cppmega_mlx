"""Read-only introspection of tokenizers and parquet datasets.

Both ``introspect_tokenizer`` and ``introspect_parquet`` return frozen
dataclass snapshots. They never mutate the source artifact and never
write to disk.

Tokenizer special-id contract (vendored nanochat BPE convention):
  0=PAD 1=UNK 2=BOS 3=EOS 4=FIM_PREFIX 5=FIM_MIDDLE 6=FIM_SUFFIX
  7=CODE_START 45=FIM_INSTRUCTION 46=SPACE 47=NL
We probe by the angle-bracketed literal — ``<FIM_PREFIX>`` etc. — so the
contract survives id renumbering as long as token strings stay stable.

Side-channel detection in parquet is purely name-driven: a column is
considered a side-channel if its name is NOT in :data:`_PARQUET_NON_SIDE`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Tokenizer side
# ---------------------------------------------------------------------------


_TOKENIZER_PROBES: Mapping[str, str] = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "BOS": "<BOS>",
    "EOS": "<EOS>",
    "FIM_PREFIX": "<FIM_PREFIX>",
    "FIM_MIDDLE": "<FIM_MIDDLE>",
    "FIM_SUFFIX": "<FIM_SUFFIX>",
    "CODE_START": "<CODE_START>",
    "FIM_INSTRUCTION": "<FIM_INSTRUCTION>",
    "SPACE": "<SPACE>",
    "NL": "<NL>",
}


@dataclass(frozen=True)
class TokenizerCapabilities:
    """Snapshot of what a tokenizer offers — read-only."""

    vocab_size: int
    special_ids: Mapping[str, int]
    has_fim: bool
    has_space_nl: bool
    has_code_start: bool
    has_instruction: bool
    byte_roundtrip: Literal["exact", "approx", "none"]
    decoder_kind: Literal["custom", "hf", "none"]
    source: str

    def has(self, *names: str) -> bool:
        """True iff every name is present in :attr:`special_ids`."""
        return all(name in self.special_ids for name in names)


def introspect_tokenizer(source: str | Path) -> TokenizerCapabilities:
    """Load a ``tokenizer.json`` and return its capability snapshot.

    Args:
      source: path to a Hugging Face-style ``tokenizer.json`` file.

    The function is purely declarative — no encode/decode round-trip is
    performed. ``byte_roundtrip`` is inferred from decoder presence:
    a custom decoder (our nanochat ``cpp_tokenizer.py``) is ``"approx"``;
    a built-in HF decoder is ``"exact"``; absent decoder is ``"none"``.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer source not found: {path}")
    stat = path.stat()
    return _introspect_tokenizer_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _introspect_tokenizer_cached(
    path_str: str,
    mtime_ns: int,
    size_bytes: int,
) -> TokenizerCapabilities:
    del mtime_ns, size_bytes
    path = Path(path_str)
    raw = json.loads(path.read_text())

    vocab = raw.get("model", {}).get("vocab", {}) or {}
    vocab_size = len(vocab)
    added = raw.get("added_tokens") or []

    found: dict[str, int] = {}
    for tok in added:
        content = tok.get("content")
        tid = tok.get("id")
        if not isinstance(content, str) or not isinstance(tid, int):
            continue
        for canonical, literal in _TOKENIZER_PROBES.items():
            if content == literal and canonical not in found:
                found[canonical] = tid

    decoder = raw.get("decoder")
    if decoder is None:
        decoder_kind: Literal["custom", "hf", "none"] = "custom"
        byte_roundtrip: Literal["exact", "approx", "none"] = "approx"
    elif isinstance(decoder, dict) and decoder.get("type"):
        decoder_kind = "hf"
        byte_roundtrip = "exact"
    else:
        decoder_kind = "none"
        byte_roundtrip = "none"

    return TokenizerCapabilities(
        vocab_size=vocab_size,
        special_ids=dict(found),
        has_fim=all(k in found for k in ("FIM_PREFIX", "FIM_MIDDLE", "FIM_SUFFIX")),
        has_space_nl="SPACE" in found and "NL" in found,
        has_code_start="CODE_START" in found,
        has_instruction="FIM_INSTRUCTION" in found,
        byte_roundtrip=byte_roundtrip,
        decoder_kind=decoder_kind,
        source=path_str,
    )


# ---------------------------------------------------------------------------
# Parquet side
# ---------------------------------------------------------------------------


# Columns that are NOT side-channels: the canonical text/token streams
# and bookkeeping the trainer always expects. Everything else in the
# schema is a side-channel that some brick or loss may consume.
_PARQUET_NON_SIDE: frozenset[str] = frozenset({
    "input_ids", "token_ids", "tokens",
    "text", "raw_text",
    "labels",
    "attention_mask",
    "row_id", "shard_id", "source_path",
})

# Known side-channel column names — used by *_has_* booleans.
_PARQUET_KNOWN: Mapping[str, str] = {
    "doc_ids": "has_doc_ids",
    "chunk_boundaries": "has_chunk_spans",
    "call_edges": "has_call_edges",
    "type_edges": "has_type_edges",
}

SideChannelFamilyStatus = Literal[
    "present",
    "partial",
    "missing",
    "derived",
    "dropped",
]
TokenAlignmentStatus = Literal["yes", "no", "unknown", "not_applicable"]
GraphRemapStatus = Literal["yes", "no", "missing", "not_applicable"]
SideChannelProvenance = Literal[
    "original",
    "derived",
    "missing",
    "dropped",
    "mixed",
    "unknown",
]

_PROVENANCE_COLUMNS: frozenset[str] = frozenset({
    "source_file_id",
    "language_id",
    "extractor_name",
    "extractor_version",
    "tokenizer_id",
    "side_channel_provenance",
})

_PACKED_UNIVERSAL_COLUMNS: tuple[str, ...] = (
    "input_ids",
    "target_ids",
    "loss_mask",
    "doc_ids",
    "pack_id",
    "valid_token_count",
    "num_docs",
)

_SIDE_CHANNEL_FAMILY_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "platform": (
        "platform_ids",
        "source_platform_ids",
    ),
    "syntax": (
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
    ),
    "structure": (
        "token_structure_ids",
        "token_dep_levels",
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
        "chunk_boundaries",
        "structure_ids",
    ),
    "semantic_graph": (
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_def_use",
        "token_call_edges",
        "token_type_edges",
        "call_edges",
        "type_edges",
    ),
    "temporal_diff": (
        "token_change_mask_pre",
        "token_change_mask_post",
        "hunk_id_per_token",
        "edit_op_per_token",
    ),
}

_TOKEN_COORDINATE_COLUMNS: frozenset[str] = frozenset({
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_structure_ids",
    "token_dep_levels",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
})

_SOURCE_LEVEL_ALIASES: frozenset[str] = frozenset({
    "structure_ids",
    "chunk_boundaries",
    "call_edges",
    "type_edges",
})

_TOKEN_GRAPH_COLUMNS: frozenset[str] = frozenset({
    "token_call_edges",
    "token_type_edges",
})

_SOURCE_GRAPH_COLUMNS: frozenset[str] = frozenset({
    "call_edges",
    "type_edges",
})

_TOKEN_SPAN_COLUMNS: frozenset[str] = frozenset({
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
})


@dataclass(frozen=True)
class SideChannelFamilyCoverage:
    """Coverage/provenance summary for one side-channel family."""

    family: str
    status: SideChannelFamilyStatus
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    token_alignment: TokenAlignmentStatus
    graph_remapping: GraphRemapStatus
    provenance: SideChannelProvenance
    non_null_ratio: float


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a parquet schema."""

    name: str
    arrow_dtype: str
    nullable: bool
    non_null_ratio: float


@dataclass(frozen=True)
class ParquetCapabilities:
    """Snapshot of what a parquet shard offers — read-only."""

    schema_columns: tuple[ColumnSpec, ...]
    row_count: int
    total_bytes: int
    has_token_ids: bool
    has_doc_ids: bool
    has_chunk_spans: bool
    has_call_edges: bool
    has_type_edges: bool
    has_provenance: bool
    side_channels: frozenset[str]
    sample_seq_lens: tuple[int, ...]
    source: str
    side_channel_families: Mapping[str, SideChannelFamilyCoverage] = field(
        default_factory=dict
    )

    def column(self, name: str) -> ColumnSpec | None:
        for c in self.schema_columns:
            if c.name == name:
                return c
        return None


def introspect_parquet(
    path: str | Path,
    *,
    sample_rows: int = 256,
) -> ParquetCapabilities:
    """Open a parquet file and return its capability snapshot.

    Args:
      path: path to a single ``.parquet`` shard.
      sample_rows: how many rows to read for non-null ratios and sample
        sequence lengths. Default 256 keeps the probe sub-second on
        production-sized shards.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"parquet source not found: {p}")

    pf = pq.ParquetFile(p)
    schema = pf.schema_arrow
    row_count = pf.metadata.num_rows
    total_bytes = p.stat().st_size

    sample_size = min(sample_rows, row_count) if row_count > 0 else 0
    if sample_size > 0:
        sample_table = pf.read_row_group(0).slice(0, sample_size)
    else:
        sample_table = pf.read().slice(0, 0)

    columns: list[ColumnSpec] = []
    for field in schema:
        col_name = field.name
        if col_name in sample_table.column_names and sample_size > 0:
            arr = sample_table.column(col_name)
            non_null = arr.length() - arr.null_count
            ratio = non_null / arr.length() if arr.length() else 0.0
        else:
            ratio = 0.0
        columns.append(
            ColumnSpec(
                name=col_name,
                arrow_dtype=str(field.type),
                nullable=field.nullable,
                non_null_ratio=float(ratio),
            )
        )

    schema_names = {c.name for c in columns}
    side_channels = frozenset(n for n in schema_names if n not in _PARQUET_NON_SIDE)

    # Sample seq lens — first sample-rows entries of the token stream.
    seq_lens: list[int] = []
    token_col = "input_ids" if "input_ids" in schema_names else (
        "token_ids" if "token_ids" in schema_names else None
    )
    if token_col is not None and sample_size > 0:
        for row in sample_table.column(token_col).to_pylist():
            seq_lens.append(len(row) if row is not None else 0)

    has_provenance = any(
        n.startswith("constituent_provenance")
        or n.endswith("_provenance")
        or n in _PROVENANCE_COLUMNS
        for n in schema_names
    )
    side_channel_families = _side_channel_family_coverage(
        columns,
        sample_table,
        schema_names=schema_names,
        token_col=token_col,
        has_provenance=has_provenance,
    )

    return ParquetCapabilities(
        schema_columns=tuple(columns),
        row_count=row_count,
        total_bytes=total_bytes,
        has_token_ids=token_col is not None,
        has_doc_ids="doc_ids" in schema_names,
        has_chunk_spans="chunk_boundaries" in schema_names,
        has_call_edges="call_edges" in schema_names,
        has_type_edges="type_edges" in schema_names,
        has_provenance=has_provenance,
        side_channels=side_channels,
        side_channel_families=side_channel_families,
        sample_seq_lens=tuple(seq_lens),
        source=str(p),
    )


def _side_channel_family_coverage(
    columns: list[ColumnSpec],
    sample_table,
    *,
    schema_names: set[str],
    token_col: str | None,
    has_provenance: bool,
) -> Mapping[str, SideChannelFamilyCoverage]:
    column_by_name = {c.name: c for c in columns}
    token_lengths = _sample_token_lengths(sample_table, token_col)
    out: dict[str, SideChannelFamilyCoverage] = {}
    out["universal"] = _universal_family_coverage(
        schema_names,
        column_by_name,
        token_col=token_col,
        has_provenance=has_provenance,
    )
    for family, expected in _SIDE_CHANNEL_FAMILY_COLUMNS.items():
        out[family] = _family_coverage(
            family,
            expected,
            schema_names,
            column_by_name,
            sample_table,
            token_lengths=token_lengths,
            has_provenance=has_provenance,
        )
    return out


def _sample_token_lengths(sample_table, token_col: str | None) -> tuple[int, ...]:
    if token_col is None or token_col not in sample_table.column_names:
        return ()
    return tuple(
        len(row) if row is not None and hasattr(row, "__len__") else 0
        for row in sample_table.column(token_col).to_pylist()
    )


def _universal_family_coverage(
    schema_names: set[str],
    column_by_name: Mapping[str, ColumnSpec],
    *,
    token_col: str | None,
    has_provenance: bool,
) -> SideChannelFamilyCoverage:
    present = tuple(c for c in _PACKED_UNIVERSAL_COLUMNS if c in schema_names)
    missing = tuple(c for c in _PACKED_UNIVERSAL_COLUMNS if c not in schema_names)
    if len(present) == len(_PACKED_UNIVERSAL_COLUMNS):
        status: SideChannelFamilyStatus = "present"
        provenance: SideChannelProvenance = "original" if has_provenance else "unknown"
    elif token_col is not None:
        status = "derived"
        provenance = "derived"
    else:
        status = "missing"
        provenance = "missing"
    return SideChannelFamilyCoverage(
        family="universal",
        status=status,
        columns=present,
        missing_columns=missing,
        dropped_columns=(),
        token_alignment="yes" if token_col is not None else "unknown",
        graph_remapping="not_applicable",
        provenance=provenance,
        non_null_ratio=_mean_non_null_ratio(present, column_by_name),
    )


def _family_coverage(
    family: str,
    expected: tuple[str, ...],
    schema_names: set[str],
    column_by_name: Mapping[str, ColumnSpec],
    sample_table,
    *,
    token_lengths: tuple[int, ...],
    has_provenance: bool,
) -> SideChannelFamilyCoverage:
    present_raw = tuple(c for c in expected if c in schema_names)
    missing = tuple(c for c in expected if c not in schema_names)
    dropped = tuple(
        c for c in present_raw
        if _column_is_dropped_for_family(c, sample_table, token_lengths)
    )
    usable = tuple(c for c in present_raw if c not in dropped)

    if usable and missing:
        status: SideChannelFamilyStatus = "partial"
    elif usable:
        status = "present"
    elif dropped:
        status = "dropped"
    else:
        status = "missing"

    token_alignment = _family_token_alignment(
        family,
        present_raw,
        dropped,
        token_lengths=token_lengths,
    )
    graph_remapping = _family_graph_remapping(family, present_raw)
    provenance = _family_provenance(status, has_provenance)

    return SideChannelFamilyCoverage(
        family=family,
        status=status,
        columns=usable,
        missing_columns=missing,
        dropped_columns=dropped,
        token_alignment=token_alignment,
        graph_remapping=graph_remapping,
        provenance=provenance,
        non_null_ratio=_mean_non_null_ratio(present_raw, column_by_name),
    )


def _column_is_dropped_for_family(
    column: str,
    sample_table,
    token_lengths: tuple[int, ...],
) -> bool:
    if column == "platform_ids":
        return False
    if column == "source_platform_ids":
        return not _source_platform_ids_are_remappable(sample_table, token_lengths)
    if column in _SOURCE_LEVEL_ALIASES:
        return True
    if column not in _TOKEN_COORDINATE_COLUMNS:
        return False
    return _column_token_alignment(column, sample_table, token_lengths) == "no"


def _column_token_alignment(
    column: str,
    sample_table,
    token_lengths: tuple[int, ...],
) -> TokenAlignmentStatus:
    if not token_lengths:
        return "unknown"
    if column not in sample_table.column_names:
        return "unknown"
    rows = sample_table.column(column).to_pylist()
    for row, expected_len in zip(rows, token_lengths, strict=False):
        if row is None:
            continue
        if not hasattr(row, "__len__"):
            return "not_applicable"
        if len(row) != expected_len:
            return "no"
    return "yes"


def _family_token_alignment(
    family: str,
    present_raw: tuple[str, ...],
    dropped: tuple[str, ...],
    *,
    token_lengths: tuple[int, ...],
) -> TokenAlignmentStatus:
    token_columns = tuple(c for c in present_raw if c in _TOKEN_COORDINATE_COLUMNS)
    if dropped:
        return "no"
    if token_columns:
        return "yes" if token_lengths else "unknown"
    if family == "platform":
        if "source_platform_ids" in present_raw:
            return "yes"
        if "platform_ids" in present_raw:
            return "not_applicable"
    return "not_applicable"


def _source_platform_ids_are_remappable(
    sample_table,
    token_lengths: tuple[int, ...],
) -> bool:
    if "source_platform_ids" not in sample_table.column_names:
        return False
    for doc_col in ("doc_ids", "document_ids", "packing_document_ids"):
        if _column_token_alignment(doc_col, sample_table, token_lengths) == "yes":
            return True
    return False


def _family_graph_remapping(
    family: str,
    present_raw: tuple[str, ...],
) -> GraphRemapStatus:
    present = set(present_raw)
    if family == "semantic_graph":
        if present & _TOKEN_GRAPH_COLUMNS:
            return "yes"
        if present & _SOURCE_GRAPH_COLUMNS:
            return "no"
        return "missing"
    if family == "structure":
        if present & _TOKEN_SPAN_COLUMNS:
            return "yes"
        if "chunk_boundaries" in present:
            return "no"
    return "not_applicable"


def _family_provenance(
    status: SideChannelFamilyStatus,
    has_provenance: bool,
) -> SideChannelProvenance:
    if status == "derived":
        return "derived"
    if status == "missing":
        return "missing"
    if status == "dropped":
        return "dropped"
    return "original" if has_provenance else "unknown"


def _mean_non_null_ratio(
    names: tuple[str, ...],
    column_by_name: Mapping[str, ColumnSpec],
) -> float:
    ratios = [column_by_name[name].non_null_ratio for name in names if name in column_by_name]
    return float(sum(ratios) / len(ratios)) if ratios else 0.0
