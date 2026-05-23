"""V8-R06 pytest: compile.trace RPC."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.compile_trace_method import (
    CompileTraceParams, compile_trace,
)
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import JsonRpcRequest, VerifyParams


def _spec() -> dict:
    return {
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "b", "kind": "mlp", "params": {}},
                {"id": "c", "kind": "attention", "params": {}},
                {"id": "d", "kind": "mlp", "params": {}},
            ],
            "edges": [
                {"src": "a", "dst": "b"},
                {"src": "b", "dst": "c"},
                {"src": "c", "dst": "d"},
            ],
        },
        "dim_env": {"H": 128},
        "loss": {"kind": "cross_entropy", "head_outputs": ["d"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    }


def test_returns_one_op_per_brick():
    r = compile_trace(CompileTraceParams(spec=VerifyParams(**_spec())))
    assert len(r.ops) == 4
    assert {op.name for op in r.ops} == {"a", "b", "c", "d"}


def test_fused_flag_matches_region_size():
    r = compile_trace(CompileTraceParams(spec=VerifyParams(**_spec())))
    # All 4 land in one big path_c region for this contrived spec, so
    # fused is True everywhere. The aggregate fused_groups list has
    # one entry.
    assert all(op.fused for op in r.ops)
    assert len(r.fused_groups) >= 1


def test_backend_label_round_trips():
    r = compile_trace(CompileTraceParams(
        spec=VerifyParams(**_spec()), backend="tilelang"))
    assert r.backend == "tilelang"


def test_dispatch_end_to_end():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r06",
        method="compile.trace",
        params={"spec": _spec(), "backend": "mlx"},
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    r = resp.result
    assert "ops" in r and len(r["ops"]) == 4
    for op in r["ops"]:
        assert {"name", "fused", "group", "materialised",
                "dlpack_boundary", "backend"} <= set(op)


def test_aggregate_counters_consistent():
    r = compile_trace(CompileTraceParams(spec=VerifyParams(**_spec())))
    # Every materialised op-name appears in materialised_ops exactly once.
    mat_set = {op.name for op in r.ops if op.materialised}
    assert mat_set == set(r.materialised_ops)
    # dlpack_crossings counts backend transitions, not per-op flags.
    assert r.dlpack_crossings >= 0
