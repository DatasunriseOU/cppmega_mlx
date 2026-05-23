"""V7-G04: data.preview_parquet surfaces the corpus_stats sidecar.

Honest-closure: compute_corpus_stats existed in
cppmega_v4/data/corpus_stats.py but clang_enriched_to_parquet never
wrote its output. After wiring, the sidecar JSON lands next to the
shard and preview_parquet returns it under corpus_stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def _write_mini_shard(parquet_path: Path) -> None:
    schema = pa.schema([
        pa.field("token_ids", pa.list_(pa.int64())),
    ])
    table = pa.table({
        "token_ids": [[1, 2, 3], [2, 3, 4, 5], [1, 1, 5]],
    }, schema=schema)
    pq.write_table(table, parquet_path, compression="snappy")


def test_corpus_stats_field_returned_when_sidecar_present(client, tmp_path):
    p = tmp_path / "shard.parquet"
    _write_mini_shard(p)
    # Sidecar a synthetic stats blob.
    sidecar = p.with_suffix(".parquet.corpus_stats.json")
    sidecar.write_text(json.dumps({
        "token_coverage_pct": 12.5,
        "doc_length_p50": 3,
        "vocab_usage_topk": [[1, 3], [2, 2]],
    }))

    payload = {
        "jsonrpc": "2.0", "id": "g1", "method": "data.preview_parquet",
        "params": {"path": str(p), "limit": 5, "offset": 0},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body
    cs = body["result"]["corpus_stats"]
    assert cs is not None
    assert cs["token_coverage_pct"] == 12.5
    assert cs["doc_length_p50"] == 3


def test_corpus_stats_field_null_when_sidecar_missing(client, tmp_path):
    p = tmp_path / "shard.parquet"
    _write_mini_shard(p)
    # No sidecar written.

    payload = {
        "jsonrpc": "2.0", "id": "g2", "method": "data.preview_parquet",
        "params": {"path": str(p), "limit": 5, "offset": 0},
    }
    r = client.post("/rpc", json=payload)
    body = r.json()
    assert body["result"]["corpus_stats"] is None


# Silence unused-import warning — numpy is imported in case future tests
# extend the shard schema.
_ = np
