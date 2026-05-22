"""V3-2: stage_train consumes real tokens from a parquet shard.

Previously stage_train hardcoded a synthetic random-targets tensor and
silently ignored any parquet/tokenizer the UI selected. Now opts may
supply parquet_path; extras report data_source ('synthetic'|'parquet')
and token_count for assertion.
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _verify_params() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _write_parquet(tmp_path: pathlib.Path, column: str = "input_ids",
                   n_rows: int = 4, n_tokens_per_row: int = 64) -> str:
    rows = []
    for r in range(n_rows):
        rows.append(list(range(r * n_tokens_per_row,
                               (r + 1) * n_tokens_per_row)))
    table = pa.table({column: rows})
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)
    return str(path)


def _run(stage_options: dict) -> dict:
    spec = _verify_params()
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": stage_options,
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_stage_train_synthetic_fallback_when_no_parquet():
    """No parquet_path → falls back to synthetic random targets.
    Backwards compat with E-4 train matrix."""
    extras = _run(stage_options={"train": {"num_steps": 2}})
    assert extras["data_source"] == "synthetic"
    assert extras["token_count"] == 0


def test_stage_train_reads_real_tokens_from_parquet(tmp_path):
    """parquet_path → tokens loaded; extras.data_source='parquet'."""
    path = _write_parquet(tmp_path)
    extras = _run(stage_options={"train": {
        "num_steps": 2, "parquet_path": path,
    }})
    assert extras["data_source"] == "parquet"
    assert extras["token_count"] == 1 * 8  # batch=1, seq=8


def test_stage_train_streams_parquet_shards_sequentially(tmp_path):
    """parquet_shards → stage_train consumes shard 0 then shard 1 in one run."""
    shard0 = _write_parquet(tmp_path / "s0", n_rows=1, n_tokens_per_row=4)
    shard1 = _write_parquet(tmp_path / "s1", n_rows=1, n_tokens_per_row=4)

    extras = _run(stage_options={"train": {
        "num_steps": 2,
        "parquet_path": shard0,
        "parquet_shards": [shard0, shard1],
    }})

    assert extras["data_source"] == "parquet_stream"
    assert extras["token_count"] == 8
    assert extras["parquet_stream"]["shard_total"] == 2
    assert extras["parquet_stream"]["shards_consumed"] == 2
    assert extras["parquet_stream"]["shard_index"] == 1


def test_stage_train_falls_back_when_parquet_missing_token_column(tmp_path):
    """Parquet without input_ids/token_ids/tokens column → fallback."""
    table = pa.table({"some_other_col": [[1, 2], [3, 4]]})
    p = tmp_path / "no_tokens.parquet"
    pq.write_table(table, p)
    extras = _run(stage_options={"train": {
        "num_steps": 2, "parquet_path": str(p),
    }})
    assert extras["data_source"] == "synthetic"
    assert extras["token_count"] == 0


def test_stage_train_falls_back_when_parquet_path_invalid():
    """Bad parquet path → synthetic fallback (no exception leaks)."""
    extras = _run(stage_options={"train": {
        "num_steps": 2, "parquet_path": "/nonexistent/file.parquet",
    }})
    assert extras["data_source"] == "synthetic"
    assert extras["token_count"] == 0


def test_stage_train_uses_alt_column_names(tmp_path):
    """token_ids and tokens are recognised in addition to input_ids."""
    for col in ("token_ids", "tokens"):
        p = _write_parquet(tmp_path / col, column=col)
        extras = _run(stage_options={"train": {
            "num_steps": 2, "parquet_path": p,
        }})
        assert extras["data_source"] == "parquet", f"col={col}"
