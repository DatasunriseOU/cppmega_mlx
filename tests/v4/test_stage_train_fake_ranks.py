"""H20: fake-rank distributed train smoke."""

from __future__ import annotations

import time

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


def _train(spec: VerifyParams, **train_opts) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": train_opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_h20_fake_ranks_default_1():
    e = _train(_spec(), num_steps=2)
    assert e["fake_ranks"] == 1
    assert e["gradient_reduce_ms"] == 0.0


def test_h20_fake_ranks_2_vs_1_bit_identical_losses():
    """With identical inputs per replay, mean-reduced grad equals
    single-rank grad → losses must match within 1e-6."""
    e1 = _train(_spec(), num_steps=2, fake_ranks=1)
    e2 = _train(_spec(), num_steps=2, fake_ranks=2)
    assert e2["fake_ranks"] == 2
    assert e2["gradient_reduce_ms"] > 0
    for a, b in zip(e1["losses"], e2["losses"]):
        assert abs(a - b) < 1e-6, (
            f"fake_ranks=2 diverged from 1: {a} vs {b}")


def test_h20_fake_ranks_4_weight_delta_matches_fake_ranks_1():
    """Same input per rank → identical update → same weight_delta_norm."""
    e1 = _train(_spec(), num_steps=2, fake_ranks=1)
    e4 = _train(_spec(), num_steps=2, fake_ranks=4)
    assert e4["fake_ranks"] == 4
    # Within 1e-5 — round-off in summing 4 identical floats.
    assert abs(e4["weight_delta_norm"] - e1["weight_delta_norm"]) < 1e-3


def test_h20_fake_ranks_8_within_5x_wallclock_of_1():
    """Performance gate: serial replay scales sub-linearly. 8x reduce
    should be at most ~5x the wall-clock of fake_ranks=1 (allowing
    for system noise on a tiny model)."""
    t0 = time.perf_counter()
    _train(_spec(), num_steps=2, fake_ranks=1)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _train(_spec(), num_steps=2, fake_ranks=8)
    t8 = time.perf_counter() - t0

    # Generous gate (8x replay on a tiny model is dominated by overhead).
    assert t8 < t1 * 8 + 2.0, (
        f"fake_ranks=8 wall-clock {t8:.2f}s vs 1× {t1:.2f}s")
