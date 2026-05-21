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
from dataclasses import dataclass
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
        n.startswith("constituent_provenance") for n in schema_names
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
        sample_seq_lens=tuple(seq_lens),
        source=str(p),
    )
