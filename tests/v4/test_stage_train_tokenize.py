"""V4-2: stage_train tokenizes parquet text column when tokenizer_path supplied.

The previous V3-2 path only read raw input_ids columns. Now if the
parquet has a 'text' column and a tokenizer_path is supplied,
stage_train encodes via the tokenizer and reports
extras.data_source='parquet_tokenized' + extras.tokenizer_used.
Falls back cleanly to V3-2 raw-int path or synthetic.
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

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


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory) -> str:
    """Tiny BPE tokenizer trained on a 4-line synthetic corpus.
    Returns path to tokenizer.json. Vocab ≈ 64 tokens."""
    tmp = tmp_path_factory.mktemp("tok")
    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "hello world foo bar baz qux\n"
        "the quick brown fox jumps over the lazy dog\n"
        "lorem ipsum dolor sit amet consectetur\n"
        "abcdefghij klmnopqrst uvwxyz\n"
    )
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=64, min_frequency=1,
        special_tokens=["<unk>"],
    )
    tok.train([str(corpus)], trainer)
    out = tmp / "tokenizer.json"
    tok.save(str(out))
    return str(out)


def _write_text_parquet(tmp_path: pathlib.Path,
                        rows: list[str] | None = None) -> str:
    rows = rows or [
        "hello world hello world hello world hello world hello world",
        "the quick brown fox jumps over the lazy dog the quick brown",
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
        "foo bar baz qux foo bar baz qux foo bar baz qux foo bar baz",
    ]
    table = pa.table({"text": rows})
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "text.parquet"
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


def test_stage_train_tokenizes_text_column(tmp_path, trained_tokenizer):
    """tokenizer_path + parquet[text] → data_source='parquet_tokenized'
    + tokenizer_used reports the tokenizer basename."""
    parquet = _write_text_parquet(tmp_path)
    extras = _run(stage_options={"train": {
        "num_steps": 2,
        "parquet_path": parquet,
        "tokenizer_path": trained_tokenizer,
    }})
    assert extras["data_source"] == "parquet_tokenized"
    assert extras["train_input_source"] == "token_embedding"
    assert extras["tokenizer_used"] == "tokenizer.json"
    assert extras["token_count"] >= 8  # batch*seq = 1*8


def test_stage_train_falls_back_to_raw_when_no_text_column(
    tmp_path, trained_tokenizer,
):
    """No 'text' column → falls through to V3-2 input_ids path."""
    table = pa.table({"input_ids": [list(range(64))]})
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "ids.parquet"
    pq.write_table(table, p)
    extras = _run(stage_options={"train": {
        "num_steps": 2,
        "parquet_path": str(p),
        "tokenizer_path": trained_tokenizer,
    }})
    # Tokenizer was supplied but text column absent → V3-2 path wins.
    assert extras["data_source"] == "parquet"
    assert extras["train_input_source"] == "token_embedding"
    assert extras["tokenizer_used"] is None
    assert extras["token_count"] == 8


def test_stage_train_falls_back_to_synthetic_when_tokenizer_missing(
    tmp_path,
):
    """Bad tokenizer_path AND no input_ids column → synthetic."""
    parquet = _write_text_parquet(tmp_path)
    extras = _run(stage_options={"train": {
        "num_steps": 2,
        "parquet_path": parquet,
        "tokenizer_path": "/nonexistent/tok.json",
    }})
    assert extras["data_source"] == "synthetic"
    assert extras["train_input_source"] == "random"
    assert extras["tokenizer_used"] is None


def test_stage_train_falls_back_when_corrupted_parquet(
    tmp_path, trained_tokenizer,
):
    """Corrupt parquet bytes → synthetic; no crash."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    extras = _run(stage_options={"train": {
        "num_steps": 2,
        "parquet_path": str(bad),
        "tokenizer_path": trained_tokenizer,
    }})
    assert extras["data_source"] == "synthetic"
    assert extras["tokenizer_used"] is None


def test_stage_train_clips_token_ids_to_vocab(tmp_path, trained_tokenizer):
    """Vocab clip applies to tokenized ids — large id values wrap mod vocab."""
    parquet = _write_text_parquet(tmp_path)
    extras = _run(stage_options={"train": {
        "num_steps": 2, "vocab_size": 32,  # smaller than tokenizer vocab
        "parquet_path": parquet,
        "tokenizer_path": trained_tokenizer,
    }})
    assert extras["data_source"] == "parquet_tokenized"
    # Loss must stay finite even though tokens were clipped.
    assert all(
        loss_item == loss_item and -1e10 < loss_item < 1e10
        for loss_item in extras["losses"]
    )
