"""Backend data preview — F-H ``data.preview_parquet`` handler.

Loads a parquet shard, samples N rows, surfaces per-row token stream
plus every other column as a side-channel "ribbon". The GUI uses the
result to paint coloured strips under the token row so the researcher
can see exactly what enters ``model.forward()``.

Pyodide-friendly: only uses pyarrow + stdlib. The frontend can swap
to in-browser hyparquet for the same shape later (F-E follow-up).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import time
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.probe import introspect_parquet


_PRIMARY_TOKEN_COLS: tuple[str, ...] = ("input_ids", "token_ids", "tokens")
_EDGE_CHANNELS: frozenset[str] = frozenset(
    {
        "clang_call_edges",
        "clang_type_edges",
        "call_edges",
        "type_edges",
        "token_call_edges",
        "token_type_edges",
    }
)
_PARQUET_SUFFIXES: frozenset[str] = frozenset({".parquet", ".pq"})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PreviewParquetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    offset: int = 0
    limit: int = 32
    channels: list[str] | None = None  # None → all detected side-channels


class PreviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_index: int
    tokens: list[int]
    channels: dict[str, Any] = Field(default_factory=dict)


class SideChannelFamilyPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    status: str
    columns: list[str] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    dropped_columns: list[str] = Field(default_factory=list)
    token_alignment: str
    graph_remapping: str
    provenance: str
    non_null_ratio: float


class EdgeDistributionPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    edge_count: int
    row_count: int
    non_empty_rows: int
    min_node_id: int | None = None
    max_node_id: int | None = None
    distinct_node_count: int
    per_row_min: int
    per_row_avg: float
    per_row_max: int
    synthetic_0_to_7_only: bool
    sample_edges: list[dict[str, int]] = Field(default_factory=list)


class ShardPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    path: str
    byte_size: int
    row_count: int


class PreviewParquetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PreviewRow]
    token_column: str
    available_channels: list[str]
    side_channel_families: dict[str, SideChannelFamilyPreview] = Field(
        default_factory=dict
    )
    edge_distributions: dict[str, EdgeDistributionPreview] = Field(
        default_factory=dict
    )
    shards: list[ShardPreview] = Field(default_factory=list)
    bytes_per_token_avg: float
    bytes_per_token_p95: float
    bytes_per_token_max: int
    total_rows: int
    elapsed_ms: float
    # V7-G04: corpus stats sidecar (token coverage / doc-length / vocab).
    # Populated when the shard was emitted by clang_enriched_to_parquet
    # with token_ids materialized; absent for legacy shards.
    corpus_stats: dict | None = None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def preview_parquet(
    params: PreviewParquetParams,
    *,
    cache: LRUCache | None = None,
) -> PreviewParquetResult:
    """Return ``params.limit`` rows starting at ``params.offset``."""
    if params.limit < 1:
        raise ValueError(f"limit must be ≥ 1, got {params.limit}")
    if params.offset < 0:
        raise ValueError(f"offset must be ≥ 0, got {params.offset}")

    # Distinguish "no filter" (None → all channels) from "explicit empty
    # filter" ([] → drop all channels) in both cache key and selection.
    filter_tag = "ALL" if params.channels is None else \
                 f"FILTER:{','.join(sorted(params.channels))}"
    preview_path, shard_paths = _preview_path_and_shards(params.path)
    cache_key = (
        f"preview::{preview_path}::{params.offset}::{params.limit}::{filter_tag}"
    )
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    t0 = time.perf_counter()
    pf = pq.ParquetFile(preview_path)
    schema_names = [f.name for f in pf.schema_arrow]
    token_col = _pick_token_column(schema_names)
    if token_col is None:
        raise ValueError(
            f"parquet shard {params.path!r} has no token column "
            f"(expected one of {_PRIMARY_TOKEN_COLS})"
        )

    # Side-channel pool: every column that isn't the token stream or
    # bookkeeping. Caller may further restrict via ``params.channels``.
    caps = introspect_parquet(preview_path, sample_rows=min(params.limit, 64))
    available = sorted(c for c in caps.side_channels if c != token_col)
    if params.channels is None:
        selected = available
    else:
        wanted = set(params.channels)
        selected = [c for c in available if c in wanted]
    cols_to_read = [token_col, *selected]

    table = pf.read(columns=cols_to_read)
    total_rows = table.num_rows
    if params.offset >= total_rows:
        sliced = table.slice(0, 0)
    else:
        n = min(params.limit, total_rows - params.offset)
        sliced = table.slice(params.offset, n)

    rows: list[PreviewRow] = []
    token_lists: list[list[int]] = []
    for i in range(sliced.num_rows):
        token_val = sliced.column(token_col)[i].as_py() or []
        tokens = [int(t) for t in token_val]
        token_lists.append(tokens)
        channel_payload: dict[str, Any] = {}
        for ch in selected:
            channel_payload[ch] = sliced.column(ch)[i].as_py()
        rows.append(PreviewRow(
            row_index=params.offset + i,
            tokens=tokens,
            channels=channel_payload,
        ))

    bpt_avg, bpt_p95, bpt_max = _bytes_per_token_stats(token_lists)
    elapsed = (time.perf_counter() - t0) * 1000.0
    out = PreviewParquetResult(
        rows=rows,
        token_column=token_col,
        available_channels=available,
        side_channel_families={
            name: SideChannelFamilyPreview(
                family=coverage.family,
                status=coverage.status,
                columns=list(coverage.columns),
                missing_columns=list(coverage.missing_columns),
                dropped_columns=list(coverage.dropped_columns),
                token_alignment=coverage.token_alignment,
                graph_remapping=coverage.graph_remapping,
                provenance=coverage.provenance,
                non_null_ratio=coverage.non_null_ratio,
            )
            for name, coverage in sorted(caps.side_channel_families.items())
        },
        edge_distributions=_edge_distributions(rows, selected),
        shards=_shard_previews(shard_paths),
        bytes_per_token_avg=round(bpt_avg, 3),
        bytes_per_token_p95=round(bpt_p95, 3),
        bytes_per_token_max=int(bpt_max),
        total_rows=total_rows,
        elapsed_ms=elapsed,
        corpus_stats=_read_corpus_stats_sidecar(preview_path),
    )
    if cache is not None:
        cache.set(cache_key, out)
    return out


def _read_corpus_stats_sidecar(parquet_path) -> dict | None:
    """V7-G04: load the {parquet}.corpus_stats.json sidecar emitted by
    clang_enriched_to_parquet. Returns None when missing or malformed."""
    import json as _json
    import os
    sidecar = str(parquet_path) + ".corpus_stats.json"
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar) as f:
            data = _json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_token_column(schema_names: list[str]) -> str | None:
    for name in _PRIMARY_TOKEN_COLS:
        if name in schema_names:
            return name
    return None


def _preview_path_and_shards(path_text: str) -> tuple[Path, tuple[Path, ...]]:
    shards = _discover_parquet_shards(path_text)
    if not shards:
        return Path(path_text), (Path(path_text),)
    requested = Path(path_text)
    preview_path = requested if requested in shards else shards[0]
    return preview_path, shards


def _discover_parquet_shards(path_text: str) -> tuple[Path, ...]:
    if any(ch in path_text for ch in "*?[]"):
        return tuple(sorted(Path(item) for item in glob.glob(path_text)))

    path = Path(path_text)
    if path.is_dir():
        return _sorted_parquet_files(path)

    if path.suffix.lower() == ".json" and path.exists():
        return _manifest_parquet_shards(path)

    if path.suffix.lower() in _PARQUET_SUFFIXES and path.exists():
        siblings = _sorted_parquet_files(path.parent)
        return siblings or (path,)

    return (path,)


def _sorted_parquet_files(directory: Path) -> tuple[Path, ...]:
    files = [
        item
        for suffix in sorted(_PARQUET_SUFFIXES)
        for item in directory.glob(f"*{suffix}")
        if item.is_file()
    ]
    return tuple(sorted(files, key=lambda item: item.name))


def _manifest_parquet_shards(manifest_path: Path) -> tuple[Path, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("shards") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return ()
    shards: list[Path] = []
    for entry in entries:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(raw_path, str):
            continue
        shard_path = Path(raw_path)
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        if shard_path.suffix.lower() in _PARQUET_SUFFIXES:
            shards.append(shard_path)
    return tuple(sorted(shards, key=lambda item: item.name))


def _shard_previews(shard_paths: tuple[Path, ...]) -> list[ShardPreview]:
    previews: list[ShardPreview] = []
    for index, path in enumerate(shard_paths):
        row_count = 0
        if path.exists():
            row_count = int(pq.ParquetFile(path).metadata.num_rows)
        previews.append(
            ShardPreview(
                index=index,
                path=str(path),
                byte_size=path.stat().st_size if path.exists() else 0,
                row_count=row_count,
            )
        )
    return previews


def _bytes_per_token_stats(
    token_lists: list[list[int]],
) -> tuple[float, float, int]:
    """Compute per-token byte stats from the encoded id stream."""
    if not token_lists:
        return 0.0, 0.0, 0
    # Each token id encodes to its big-endian byte length — proxy for
    # the on-disk bytes/token cost (real bytes depend on vocab + BPE
    # merge stats; the GUI uses this as a heuristic, full stats need
    # the corresponding tokenizer).
    lengths: list[int] = []
    for row in token_lists:
        for tok in row:
            lengths.append(max(1, (tok.bit_length() + 7) // 8))
    if not lengths:
        return 0.0, 0.0, 0
    avg = sum(lengths) / len(lengths)
    sorted_lengths = sorted(lengths)
    p95_idx = max(0, int(len(sorted_lengths) * 0.95) - 1)
    p95 = float(sorted_lengths[p95_idx])
    return avg, p95, max(lengths)


def _edge_distributions(
    rows: list[PreviewRow],
    selected_channels: list[str],
) -> dict[str, EdgeDistributionPreview]:
    out: dict[str, EdgeDistributionPreview] = {}
    for channel in selected_channels:
        if channel not in _EDGE_CHANNELS:
            continue
        row_counts: list[int] = []
        node_ids: set[int] = set()
        samples: list[dict[str, int]] = []
        for row in rows:
            edges = _edge_pairs(row.channels.get(channel))
            row_counts.append(len(edges))
            for src, dst in edges:
                node_ids.add(src)
                node_ids.add(dst)
                if len(samples) < 8:
                    samples.append({"from": src, "to": dst})
        edge_count = sum(row_counts)
        out[channel] = EdgeDistributionPreview(
            column=channel,
            edge_count=edge_count,
            row_count=len(rows),
            non_empty_rows=sum(1 for count in row_counts if count > 0),
            min_node_id=min(node_ids) if node_ids else None,
            max_node_id=max(node_ids) if node_ids else None,
            distinct_node_count=len(node_ids),
            per_row_min=min(row_counts) if row_counts else 0,
            per_row_avg=round(edge_count / len(row_counts), 3) if row_counts else 0.0,
            per_row_max=max(row_counts) if row_counts else 0,
            synthetic_0_to_7_only=(
                bool(node_ids) and min(node_ids) >= 0 and max(node_ids) <= 7
            ),
            sample_edges=samples,
        )
    return out


def _edge_pairs(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, (list, tuple)):
        return []
    pairs: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, dict) and "from" in item and "to" in item:
            pairs.append((int(item["from"]), int(item["to"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((int(item[0]), int(item[1])))
    return pairs
