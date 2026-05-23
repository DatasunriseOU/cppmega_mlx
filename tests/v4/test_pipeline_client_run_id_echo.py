"""V7-H40: pipeline.run echoes client_run_id back in PipelineRunResult.

UI uses this token to correlate the response to its originating
request — stale retries / WS reconnects can land out of order, and
the UI drops responses whose echoed token doesn't match.
"""

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


def test_v7_h40_client_run_id_echoed_back_verbatim():
    response = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "pipeline.run",
        "params": {
            "spec": _SPEC,
            "pipeline": {
                "stages": ["parse", "verify_build_spec", "build_model"],
                "stage_options": {},
            },
            "client_run_id": "client-1700000000-abcdef",
        },
    })
    assert response.error is None, response.error
    assert response.result is not None
    assert response.result["client_run_id"] == "client-1700000000-abcdef"


def test_v7_h40_client_run_id_optional_none_when_absent():
    response = dispatch({
        "jsonrpc": "2.0", "id": "T2", "method": "pipeline.run",
        "params": {
            "spec": _SPEC,
            "pipeline": {
                "stages": ["parse"],
                "stage_options": {},
            },
        },
    })
    assert response.error is None
    assert response.result["client_run_id"] is None
