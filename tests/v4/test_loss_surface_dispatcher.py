"""V7-H33: loss_surface.run RPC dispatcher route registration."""

from __future__ import annotations

from cppmega_v4.jsonrpc.dispatcher import dispatch


_SPEC = {
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
}


def test_v7_h33_loss_surface_run_method_is_routed():
    """Before this fix loss_surface_method.py existed but the route
    was orphan — UI got -32601 method_not_found."""
    response = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "loss_surface.run",
        "params": {
            "spec": _SPEC,
            "lr_deltas": [1.0],
            "wd_deltas": [1.0],
            "k_steps": 2,
        },
    })
    assert response.error is None, response.error
    assert response.result is not None
    assert "rows" in response.result
    assert len(response.result["rows"]) == 1
    assert response.result["lr_deltas"] == [1.0]
    assert response.result["wd_deltas"] == [1.0]
    assert response.result["best_lr_mult"] == 1.0
    assert response.result["best_wd_mult"] == 1.0
    assert response.result["best_loss"] is not None


def test_v7_h33_loss_surface_run_grid_returns_full_matrix():
    response = dispatch({
        "jsonrpc": "2.0", "id": "T2", "method": "loss_surface.run",
        "params": {
            "spec": _SPEC,
            "lr_deltas": [0.5, 1.0],
            "wd_deltas": [0.5, 1.0, 2.0],
            "k_steps": 2,
        },
    })
    assert response.error is None
    rows = response.result["rows"]
    assert len(rows) == 2
    assert all(len(r) == 3 for r in rows)
    for row in rows:
        for cell in row:
            assert cell["status"] in {"ok"} or cell["status"].startswith("fail")
