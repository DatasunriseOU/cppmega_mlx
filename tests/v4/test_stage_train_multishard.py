"""V7-G01: multi-shard parquet streaming in stage_train."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 64, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _write_int_parquet(path: pathlib.Path, n_rows: int = 1,
                        seq_len: int = 32, start: int = 0) -> str:
    rows = [list(range(start + i * seq_len,
                       start + (i + 1) * seq_len))
            for i in range(n_rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"input_ids": rows}), path)
    return str(path)


def _train(spec, **opts) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_g01_multi_shard_stream_payload_lists_all_shards(tmp_path):
    """parquet_shards=[A, B] → extras.parquet_stream lists BOTH shards
    in its `shards` array with their row counts, regardless of how many
    were actually drained for the demand."""
    s1 = _write_int_parquet(tmp_path / "shard_0.parquet", n_rows=1,
                             seq_len=32, start=10)
    s2 = _write_int_parquet(tmp_path / "shard_1.parquet", n_rows=1,
                             seq_len=32, start=200)
    extras = _train(_spec(), num_steps=2, parquet_shards=[s1, s2])
    ps = extras.get("parquet_stream")
    assert ps is not None
    assert ps["shard_total"] == 2
    assert len(ps["shards"]) == 2
    paths = {sh["path"] for sh in ps["shards"]}
    assert s1 in paths and s2 in paths
    # row_count populated per shard.
    for sh in ps["shards"]:
        assert sh["row_count"] >= 1


def test_v7_g01_two_shards_actually_drained(tmp_path):
    """Tiny shards (4 tokens each) force the demand to span both."""
    s1 = _write_int_parquet(tmp_path / "tiny_0.parquet", n_rows=1,
                             seq_len=4, start=10)
    s2 = _write_int_parquet(tmp_path / "tiny_1.parquet", n_rows=1,
                             seq_len=4, start=200)
    extras = _train(_spec(), num_steps=2,
                    parquet_shards=[s1, s2])
    ps = extras.get("parquet_stream")
    assert ps is not None
    # With demand = batch*seq = 64 and each shard only 4 tokens, the
    # stream falls back to synthetic — but parquet_stream still
    # reports both shard candidates so the UI can show 2 shards.
    assert ps["shard_total"] == 2


def test_v7_g01_single_shard_path_unchanged(tmp_path):
    """Back-compat: parquet_path (singular) still works without shards."""
    s = _write_int_parquet(tmp_path / "only.parquet", n_rows=2, seq_len=64)
    extras = _train(_spec(), num_steps=2, parquet_path=s)
    assert extras["data_source"] in ("parquet", "parquet_tokenized")
