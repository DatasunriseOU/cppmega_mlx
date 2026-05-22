"""V7-A05: real corpus convergence across 2 epochs."""

from __future__ import annotations

import json
import pathlib

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
MATRIX = REPO / "tests" / "fixtures" / "MATRIX.json"


def _matrix_pair() -> tuple[str, str]:
    if not MATRIX.exists():
        pytest.skip("MATRIX.json missing")
    m = json.loads(MATRIX.read_text())
    parquet = m["parquets"]["T2_gpt2_small__P1_minimal"]["path"]
    tokenizer = m["tokenizers"]["T2_gpt2_small"]["path"]
    if not (pathlib.Path(parquet).exists()
            and pathlib.Path(tokenizer).exists()):
        pytest.skip("MATRIX fixture files missing")
    return parquet, tokenizer


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
        "dim_env": {"B": 1, "S": 16, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int, parquet: str, tokenizer: str) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {
            "num_steps": num_steps,
            "parquet_path": parquet,
            "tokenizer_path": tokenizer,
        }},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a05_real_corpus_2epoch_convergence():
    """Run for 60 steps (~2 logical epochs of the tiny fixture):
    second-half mean loss strictly below first-half mean loss."""
    parquet, tokenizer = _matrix_pair()
    e = _train(num_steps=60, parquet=parquet, tokenizer=tokenizer)
    losses = e["losses"]
    assert len(losses) == 60
    half = len(losses) // 2
    first = sum(losses[:half]) / half
    second = sum(losses[half:]) / (len(losses) - half)
    assert second < first, (
        f"real-corpus loss did not drop epoch-over-epoch: "
        f"epoch1≈{first:.4f}, epoch2≈{second:.4f}"
    )
    # Data source proves the real-corpus path activated.
    assert e["data_source"] in ("parquet", "parquet_tokenized")
