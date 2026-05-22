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


def _write_real_edge_parquet(p: Path):
    pq.write_table(pa.table({
        "input_ids": [
            list(range(8)),
            list(range(8, 16)),
        ],
        "call_edges": [
            [{"from": 5, "to": 10}, {"from": 12, "to": 3}],
            [],
        ],
        "type_edges": [
            [{"from": 2, "to": 42}],
            [{"from": 8, "to": 9}],
        ],
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


def test_preview_reports_ordered_sibling_shards(tmp_path: Path):
    shard_b = tmp_path / "val_00001.parquet"
    shard_a = tmp_path / "val_00000.parquet"
    _write_full_parquet(shard_b, n_rows=3)
    _write_full_parquet(shard_a, n_rows=2)

    r = preview_parquet(PreviewParquetParams(path=str(shard_a), offset=0, limit=1))

    assert [Path(shard.path).name for shard in r.shards] == [
        "val_00000.parquet",
        "val_00001.parquet",
    ]
    assert [shard.index for shard in r.shards] == [0, 1]
    assert [shard.row_count for shard in r.shards] == [2, 3]
    assert all(shard.byte_size > 0 for shard in r.shards)


def test_preview_directory_uses_first_shard_for_capabilities(tmp_path: Path):
    shard_a = tmp_path / "val_00000.parquet"
    shard_b = tmp_path / "val_00001.parquet"
    _write_full_parquet(shard_a, n_rows=2)
    _write_full_parquet(shard_b, n_rows=3)

    r = preview_parquet(PreviewParquetParams(path=str(tmp_path), offset=0, limit=1))

    assert r.token_column == "input_ids"
    assert "doc_ids" in r.available_channels
    assert [Path(shard.path).name for shard in r.shards] == [
        "val_00000.parquet",
        "val_00001.parquet",
    ]


def test_preview_carries_channel_payload_per_row(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    _write_full_parquet(p, n_rows=8)
    r = preview_parquet(PreviewParquetParams(path=str(p), offset=0, limit=2))
    first = r.rows[0]
    assert first.row_index == 0
    assert len(first.tokens) > 0
    assert "doc_ids" in first.channels
    assert "loss_mask" in first.channels


def test_preview_reports_real_clang_edge_distributions(tmp_path: Path):
    p = tmp_path / "real_edges.parquet"
    _write_real_edge_parquet(p)

    r = preview_parquet(PreviewParquetParams(path=str(p), offset=0, limit=2))

    call_edges = r.edge_distributions["call_edges"]
    assert call_edges.edge_count == 2
    assert call_edges.non_empty_rows == 1
    assert call_edges.min_node_id == 3
    assert call_edges.max_node_id == 12
    assert call_edges.distinct_node_count == 4
    assert call_edges.per_row_max == 2
    assert call_edges.sample_edges == [{"from": 5, "to": 10}, {"from": 12, "to": 3}]
    assert call_edges.synthetic_0_to_7_only is False

    type_edges = r.edge_distributions["type_edges"]
    assert type_edges.edge_count == 2
    assert type_edges.non_empty_rows == 2
    assert type_edges.max_node_id == 42
    assert type_edges.synthetic_0_to_7_only is False


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
