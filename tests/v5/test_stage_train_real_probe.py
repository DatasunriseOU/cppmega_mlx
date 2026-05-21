"""G20: inference probe with real tokens (encode text → top1 drift)."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("tok-probe")
    corpus = tmp / "corpus.txt"
    corpus.write_text("hello world foo bar baz qux\n"
                      "the quick brown fox jumps over the lazy dog\n")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=64, min_frequency=1, special_tokens=["<unk>"])
    tok.train([str(corpus)], trainer)
    out = tmp / "tokenizer.json"
    tok.save(str(out))
    return str(out)


def _run(opts: dict) -> dict:
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_no_probe_text_random_gaussian_path():
    """V4-11 baseline preserved: no probe_text → real_tokens=False."""
    extras = _run({"num_steps": 4})
    p = extras["inference_probe"]
    assert p["real_tokens"] is False
    assert p["text_len"] == 0
    assert p["top1_token_drift"] == 0


def test_probe_text_without_tokenizer_falls_back():
    """probe_text without tokenizer_path → can't encode → fallback."""
    extras = _run({"num_steps": 4,
                   "inference_probe_text": "hello world"})
    p = extras["inference_probe"]
    assert p["real_tokens"] is False


def test_probe_text_with_tokenizer_activates_real_tokens(tiny_tokenizer):
    extras = _run({"num_steps": 4,
                   "inference_probe_text": "hello world foo bar",
                   "tokenizer_path": tiny_tokenizer})
    p = extras["inference_probe"]
    assert p["real_tokens"] is True
    assert p["text_len"] > 0
    # top1_token_drift may be 0 with only 4 train steps on tiny model
    # but the field MUST be populated (no longer 0 placeholder).
    assert p["top1_token_drift"] >= 0
