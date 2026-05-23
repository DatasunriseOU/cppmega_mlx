"""V8-R07 pytest: sync.check RPC + z3 sync-checker."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import JsonRpcRequest
from cppmega_v4.jsonrpc.sync_check_method import (
    SyncCheckParams, sync_check_method,
)
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.spec.sync_checker import run_sync_check


def _spec(n: int = 3) -> dict:
    nodes = []
    edges = []
    for i in range(n):
        kind = "attention" if i % 2 == 0 else "mlp"
        nodes.append({"id": f"op_{i}", "kind": kind, "params": {}})
        if i > 0:
            edges.append({"src": f"op_{i-1}", "dst": f"op_{i}"})
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "dim_env": {"H": 128},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"op_{n-1}"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    }


def test_last_op_always_marked_necessary():
    res = sync_check_method(SyncCheckParams(spec=VerifyParams(**_spec(3))))
    assert res.z3_solver_status == "sat"
    necessary_names = {s.after_op for s in res.necessary_syncs}
    assert "op_2" in necessary_names


def test_redundant_syncs_have_advice():
    res = sync_check_method(SyncCheckParams(spec=VerifyParams(**_spec(4))))
    # Every redundant op has a matching advice entry.
    redundant_names = {s.after_op for s in res.redundant_syncs}
    advice_names = {a.op for a in res.advice}
    assert redundant_names == advice_names
    for a in res.advice:
        assert "remove" in a.fix.lower() or "eval" in a.fix.lower()
        assert a.confidence in {"high", "medium", "low"}


def test_z3_elapsed_under_one_second():
    """The encoding is small; solving must finish within 1s."""
    res = sync_check_method(SyncCheckParams(spec=VerifyParams(**_spec(8))))
    assert res.z3_elapsed_ms < 1000


def test_empty_graph_returns_sat_with_no_entries():
    res = run_sync_check([], hidden_size=64)
    assert res.z3_solver_status == "sat"
    assert res.necessary_syncs == []
    assert res.redundant_syncs == []


def test_dispatch_end_to_end():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r07",
        method="sync.check",
        params={"spec": _spec(3)},
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    r = resp.result
    assert r["z3_solver_status"] == "sat"
    assert isinstance(r["necessary_syncs"], list)
    assert isinstance(r["redundant_syncs"], list)
    assert isinstance(r["advice"], list)
    assert isinstance(r["z3_elapsed_ms"], (int, float))
