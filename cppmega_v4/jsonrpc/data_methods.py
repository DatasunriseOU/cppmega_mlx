"""Backend data preview — F-H ``data.preview_parquet`` handler.

Loads a parquet shard, samples N rows, surfaces per-row token stream
plus every other column as a side-channel "ribbon". The GUI uses the
result to paint coloured strips under the token row so the researcher
can see exactly what enters ``model.forward()``.

Pyodide-friendly: only uses pyarrow + stdlib. The frontend can swap
to in-browser hyparquet for the same shape later (F-E follow-up).
"""

from __future__ import annotations

import time
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.probe import introspect_parquet


_PRIMARY_TOKEN_COLS: tuple[str, ...] = ("input_ids", "token_ids", "tokens")


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


class PreviewParquetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PreviewRow]
    token_column: str
    available_channels: list[str]
    bytes_per_token_avg: float
    bytes_per_token_p95: float
    bytes_per_token_max: int
    total_rows: int
    elapsed_ms: float


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

    cache_key = f"preview::{params.path}::{params.offset}::{params.limit}::" \
                f"{','.join(sorted(params.channels or []))}"
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    t0 = time.perf_counter()
    pf = pq.ParquetFile(params.path)
    schema_names = [f.name for f in pf.schema_arrow]
    token_col = _pick_token_column(schema_names)
    if token_col is None:
        raise ValueError(
            f"parquet shard {params.path!r} has no token column "
            f"(expected one of {_PRIMARY_TOKEN_COLS})"
        )

    # Side-channel pool: every column that isn't the token stream or
    # bookkeeping. Caller may further restrict via ``params.channels``.
    caps = introspect_parquet(params.path, sample_rows=min(params.limit, 64))
    available = sorted(c for c in caps.side_channels if c != token_col)
    selected = [c for c in available if not params.channels
                or c in params.channels]
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
        bytes_per_token_avg=round(bpt_avg, 3),
        bytes_per_token_p95=round(bpt_p95, 3),
        bytes_per_token_max=int(bpt_max),
        total_rows=total_rows,
        elapsed_ms=elapsed,
    )
    if cache is not None:
        cache.set(cache_key, out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_token_column(schema_names: list[str]) -> str | None:
    for name in _PRIMARY_TOKEN_COLS:
        if name in schema_names:
            return name
    return None


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
