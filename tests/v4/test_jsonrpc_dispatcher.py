"""VBGui F-A dispatcher tests — envelope routing + error envelopes."""

from __future__ import annotations

from cppmega_v4.jsonrpc import (
    ErrorCode,
    JsonRpcRequest,
    LRUCache,
    dispatch,
)


_GRAPH = {
    "nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
    "edges": [{"src": "a", "dst": "b"}],
}
_DIM_ENV = {"B": 1, "S": 4, "H": 64, "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 8, "top_k": 2}
_LOSS = {"kind": "cross_entropy", "head_outputs": ["b"]}
_OPTIM = {"kind": "adamw",
          "groups": [{"matcher": "all", "lr": 3e-4,
                      "weight_decay": 0.01, "betas": [0.9, 0.95]}]}


def _verify_envelope(id_="r1"):
    return {
        "jsonrpc": "2.0", "id": id_, "method": "verify",
        "params": {"graph": _GRAPH, "dim_env": _DIM_ENV,
                   "loss": _LOSS, "optim": _OPTIM},
    }


def test_dispatch_verify_round_trip():
    resp = dispatch(_verify_envelope())
    assert resp.jsonrpc == "2.0"
    assert resp.id == "r1"
    assert resp.error is None
    assert resp.result is not None
    assert "resolved" in resp.result


def test_dispatch_method_not_found():
    resp = dispatch({"jsonrpc": "2.0", "id": "x", "method": "bogus_method",
                     "params": {}})
    assert resp.result is None
    assert resp.error.code == ErrorCode.METHOD_NOT_FOUND
    assert "available" in resp.error.data


def test_dispatch_invalid_envelope_returns_invalid_request():
    resp = dispatch({"jsonrpc": "3.0", "id": "x", "method": "verify"})
    assert resp.error.code == ErrorCode.INVALID_REQUEST


def test_dispatch_invalid_params_returns_invalid_params():
    envelope = {"jsonrpc": "2.0", "id": "y", "method": "verify",
                "params": {"graph": {"nodes": []}}}  # missing required fields
    resp = dispatch(envelope)
    assert resp.error.code == ErrorCode.INVALID_PARAMS


def test_dispatch_unknown_preset_returns_invalid_params():
    envelope = {"jsonrpc": "2.0", "id": "z", "method": "build_preset_specs",
                "params": {"preset_name": "nonexistent", "hidden_size": 64}}
    resp = dispatch(envelope)
    assert resp.error.code == ErrorCode.INVALID_PARAMS
    assert "unknown preset" in resp.error.data["detail"]


def test_dispatch_backend_status():
    resp = dispatch({"jsonrpc": "2.0", "id": "h", "method": "backend.status"})
    assert resp.result == {"status": "ok"}


def test_dispatch_accepts_pre_parsed_request():
    req = JsonRpcRequest(id="p", method="backend.status")
    resp = dispatch(req)
    assert resp.result == {"status": "ok"}


def test_dispatch_uses_cache_across_calls():
    cache = LRUCache(capacity=4)
    dispatch(_verify_envelope(id_=1), cache=cache)
    dispatch(_verify_envelope(id_=2), cache=cache)  # same canonical key
    stats = cache.stats()
    assert stats["hits"] == 1


def test_dispatch_preserves_request_id_on_error():
    resp = dispatch({"jsonrpc": "2.0", "id": 42, "method": "no_such"})
    assert resp.id == 42
    assert resp.error.code == ErrorCode.METHOD_NOT_FOUND


def test_dispatch_pipeline_run_round_trip():
    envelope = {
        "jsonrpc": "2.0", "id": "pp", "method": "pipeline.run",
        "params": {
            "spec": {"graph": _GRAPH, "dim_env": _DIM_ENV,
                     "loss": _LOSS, "optim": _OPTIM},
            "pipeline": {"stages": ["parse", "verify_build_spec"]},
        },
    }
    resp = dispatch(envelope)
    assert resp.error is None
    assert resp.result["overall_status"] == "ok"
    assert len(resp.result["stages"]) == 2
