"""V7-H09: loss-surface (lr × wd) sweep."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.loss_surface_method import (
    LossSurfaceParams, loss_surface_run,
)
from cppmega_v4.jsonrpc.schema import VerifyParams


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


def test_v7_h09_3x3_grid_returns_matrix_with_best():
    r = loss_surface_run(LossSurfaceParams(
        spec=_spec(),
        lr_deltas=[0.5, 1.0, 2.0],
        wd_deltas=[0.5, 1.0, 2.0],
        k_steps=2,
    ))
    assert len(r.rows) == 3
    for row in r.rows:
        assert len(row) == 3
        for cell in row:
            assert cell.lr_mult in (0.5, 1.0, 2.0)
            assert cell.wd_mult in (0.5, 1.0, 2.0)
            assert cell.status == "ok"
            assert cell.final_loss is not None
            assert cell.elapsed_ms > 0
    # Best cell surfaced.
    assert r.best_lr_mult in (0.5, 1.0, 2.0)
    assert r.best_wd_mult in (0.5, 1.0, 2.0)
    assert r.best_loss is not None


def test_v7_h09_throughput_positive_on_ok_cells():
    r = loss_surface_run(LossSurfaceParams(
        spec=_spec(),
        lr_deltas=[1.0],
        wd_deltas=[1.0],
        k_steps=2,
    ))
    cell = r.rows[0][0]
    assert cell.status == "ok"
    assert cell.throughput_tok_s is not None
    assert cell.throughput_tok_s > 0
