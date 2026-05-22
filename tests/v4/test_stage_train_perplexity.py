"""V7-A06: perplexity + bits_per_byte in extras."""

from __future__ import annotations

import math

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
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int = 4) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    return next(s for s in rep.stages if s.name == "train").extras


def test_v7_a06_perplexity_and_bpb_finite():
    e = _train(num_steps=4)
    assert e["perplexity"] is not None
    assert e["bits_per_byte"] is not None
    assert e["perplexity"] > 0
    assert math.isfinite(e["bits_per_byte"])
    # ppl == exp(tail-mean nll). Sanity: ppl ≈ exp(nll).
    tail = sum(e["losses"][-min(50, len(e["losses"])):]) / max(
        1, min(50, len(e["losses"])))
    assert abs(math.log(e["perplexity"]) - tail) < 1e-3


def test_v7_a06_bpb_relation_to_nll():
    e = _train(num_steps=4)
    tail = sum(e["losses"][-min(50, len(e["losses"])):]) / max(
        1, min(50, len(e["losses"])))
    expected = tail / math.log(2)
    assert abs(e["bits_per_byte"] - expected) < 1e-3
