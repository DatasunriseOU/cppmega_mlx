"""V7-F05: long-context attention forward smoke at S>=2048."""

from __future__ import annotations

import time

import mlx.core as mx
import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(S: int = 4096, H: int = 64) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 2, "head_dim": 32}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": S, "H": H,
                    "nh": 2, "nkv": 1, "head_dim": 32},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


@pytest.mark.parametrize("S", [2048, 4096])
def test_v7_f05_long_context_forward_finite(S):
    """Forward at S in {2048, 4096} on a tiny model — no OOM,
    losses finite."""
    t0 = time.perf_counter()
    rep = run_pipeline(_spec(S=S), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "dry_forward", "train"],
        "stage_options": {"train": {"num_steps": 1}},
    }))
    elapsed = time.perf_counter() - t0
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"S={S} train failed: {tr.error}"
    losses = tr.extras.get("losses", [])
    assert len(losses) == 1
    assert -1e10 < losses[0] < 1e10, f"S={S} loss non-finite: {losses[0]}"
    # Sanity: forward completes in reasonable time on tiny model
    # (under 60 s even at S=4096 on a CPU-only fallback).
    assert elapsed < 120.0, f"S={S} too slow: {elapsed:.1f}s"
