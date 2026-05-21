"""F-H data inspector tests — preview_parquet end-to-end."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc import LRUCache, dispatch
from cppmega_v4.jsonrpc.data_methods import (
    PreviewParquetParams,
    preview_parquet,
)


def _write_full_parquet(p: Path, n_rows: int = 32):
    pq.write_table(pa.table({
        "input_ids":   [list(range(8 + i % 4)) for i in range(n_rows)],
        "doc_ids":     [i // 4 for i in range(n_rows)],
        "loss_mask":   [[1] * (8 + i % 4) for i in range(n_rows)],
        "call_edges":  [[(0, 1)] for _ in range(n_rows)],
        "type_edges":  [[(0, 2)] for _ in range(n_rows)],
    }), p)


# ---------------------------------------------------------------------------
# preview_parquet
# ---------------------------------------------------------------------------


def test_preview_returns_first_page(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=16)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=0, limit=4))
    assert r.total_rows == 16
    assert len(r.rows) == 4
    assert r.token_column == "input_ids"
    assert "doc_ids" in r.available_channels
    assert "call_edges" in r.available_channels
    assert r.side_channel_families["semantic_graph"].status == "dropped"
    assert set(r.side_channel_families["semantic_graph"].dropped_columns) == {
        "call_edges",
        "type_edges",
    }
    assert r.side_channel_families["semantic_graph"].graph_remapping == "no"


def test_preview_carries_channel_payload_per_row(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=8)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=0, limit=2))
    first = r.rows[0]
    assert first.row_index == 0
    assert len(first.tokens) > 0
    assert "doc_ids" in first.channels
    assert "loss_mask" in first.channels


def test_preview_pagination_offset_works(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=10)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=5, limit=4))
    assert [row.row_index for row in r.rows] == [5, 6, 7, 8]


def test_preview_channel_filter_restricts_payload(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    r = preview_parquet(PreviewParquetParams(
        path=str(p), offset=0, limit=2, channels=["doc_ids"],
    ))
    for row in r.rows:
        assert set(row.channels.keys()) == {"doc_ids"}


def test_preview_bytes_per_token_stats_present(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=0, limit=4))
    assert r.bytes_per_token_avg > 0
    assert r.bytes_per_token_max >= r.bytes_per_token_avg


def test_preview_caches_repeat_calls(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    cache = LRUCache()
    params = PreviewParquetParams(path=str(p), offset=0, limit=2)
    preview_parquet(params, cache=cache)
    preview_parquet(params, cache=cache)
    assert cache.stats()["hits"] >= 1


def test_preview_offset_past_end_returns_empty(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=999, limit=8))
    assert r.rows == []
    assert r.total_rows == 4


def test_preview_rejects_invalid_limit(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    with pytest.raises(ValueError, match="limit"):
        preview_parquet(PreviewParquetParams(path=str(p), limit=0))


def test_preview_raises_when_token_column_missing(tmp_path: Path):
    p = tmp_path / "no_tokens.parquet"
    pq.write_table(pa.table({"some_col": [1, 2, 3]}), p)
    with pytest.raises(ValueError, match="no token column"):
        preview_parquet(PreviewParquetParams(path=str(p), limit=2))


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_dispatch_data_preview_parquet_round_trip(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=4)
    resp = dispatch({
        "jsonrpc": "2.0", "id": "d1", "method": "data.preview_parquet",
        "params": {"path": str(p), "offset": 0, "limit": 2},
    })
    assert resp.error is None
    assert len(resp.result["rows"]) == 2
    assert resp.result["token_column"] == "input_ids"
    assert "side_channel_families" in resp.result
    assert "universal" in resp.result["side_channel_families"]


def test_method_registry_includes_data_preview():
    from cppmega_v4.jsonrpc import METHOD_REGISTRY
    assert "data.preview_parquet" in METHOD_REGISTRY
