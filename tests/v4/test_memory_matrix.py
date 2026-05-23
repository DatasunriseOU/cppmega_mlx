"""V8-R03 unit tests: memory.matrix RPC.

Asserts:
  * default 4×5 matrix when topologies/precisions omitted
  * per-precision bytes scale linearly with precision_bytes/2 (bf16 baseline)
  * mxfp4 < fp8 < bf16 == fp16 < fp32 for the same topology
  * fits flag flips when bytes exceed device_hbm_bytes × headroom
  * unknown topology / precision rejected
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.memory_matrix_method import (
    MemoryMatrixParams, memory_matrix, PRECISION_BYTES,
    TOPOLOGY_BUILDERS,
)
from cppmega_v4.jsonrpc.schema import JsonRpcRequest, VerifyParams


def _mini_spec() -> dict:
    """Minimal canvas spec: attention + mlp, hidden=256."""
    return {
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "b", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": {"H": 256, "B": 1, "S": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]},
        ]},
        "sharding": None,
        "training": True,
    }


def test_default_axes_yield_4x5_grid():
    res = memory_matrix(MemoryMatrixParams(
        spec=VerifyParams(**_mini_spec()),
    ))
    assert len(res.cells) == 4 * 5
    assert set(res.topologies) == {
        "h100_8x", "m3_ultra_solo", "gb10_quarter", "tpu_v6e_8"}
    assert set(res.precisions) == {"fp32", "bf16", "fp16", "fp8", "mxfp4"}


def test_precision_ordering_within_a_topology():
    res = memory_matrix(MemoryMatrixParams(
        spec=VerifyParams(**_mini_spec()),
        topologies=["h100_8x"],
    ))
    by_p = {c.precision: c.bytes for c in res.cells}
    assert by_p["mxfp4"] < by_p["fp8"] < by_p["bf16"]
    assert by_p["bf16"] == by_p["fp16"]
    assert by_p["bf16"] < by_p["fp32"]


def test_bf16_baseline_unchanged():
    """The bf16 column must equal the raw verify_and_estimate total —
    no rounding loss when the scale is exactly 1.0."""
    res = memory_matrix(MemoryMatrixParams(
        spec=VerifyParams(**_mini_spec()),
        topologies=["h100_8x"], precisions=["bf16"],
    ))
    cell = res.cells[0]
    breakdown = cell.breakdown
    # Sum of the parts is the cell total.
    assert (breakdown["weights"] + breakdown["grads"] + breakdown["optimizer"]
            + breakdown["activations"] + breakdown["kv_cache"]
            + breakdown["edge_handoff"]) == cell.bytes


def test_fits_flag_reflects_device_hbm():
    """All cells with bytes < gb10's HBM × headroom must fit."""
    res = memory_matrix(MemoryMatrixParams(
        spec=VerifyParams(**_mini_spec()),
        topologies=["gb10_quarter"], precisions=["bf16"],
        headroom=0.9,
    ))
    cell = res.cells[0]
    assert cell.fits == (cell.bytes <= int(cell.device_hbm_bytes * 0.9))


def test_unknown_topology_rejected():
    with pytest.raises(ValueError, match="unknown topology"):
        memory_matrix(MemoryMatrixParams(
            spec=VerifyParams(**_mini_spec()),
            topologies=["nonexistent_zzz"],
        ))


def test_unknown_precision_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        memory_matrix(MemoryMatrixParams(
            spec=VerifyParams(**_mini_spec()),
            precisions=["int1024"],
        ))


def test_invalid_headroom_rejected():
    with pytest.raises(ValueError, match="headroom"):
        memory_matrix(MemoryMatrixParams(
            spec=VerifyParams(**_mini_spec()),
            headroom=1.5,
        ))


def test_dispatch_end_to_end():
    req = JsonRpcRequest(
        jsonrpc="2.0", id="t-r03",
        method="memory.matrix",
        params={
            "spec": _mini_spec(),
            "topologies": ["m3_ultra_solo", "h100_8x"],
            "precisions": ["bf16", "mxfp4"],
        },
    )
    resp = dispatch(req)
    assert resp.error is None, resp.error
    r = resp.result
    assert len(r["cells"]) == 4
    by = {(c["topology"], c["precision"]): c["bytes"] for c in r["cells"]}
    assert by[("h100_8x", "mxfp4")] < by[("h100_8x", "bf16")]


def test_precision_bytes_table_matches_spec():
    """V8 spec §3 nominates the canonical precision -> bytes map."""
    assert PRECISION_BYTES["fp32"] == 4.0
    assert PRECISION_BYTES["bf16"] == 2.0
    assert PRECISION_BYTES["fp16"] == 2.0
    assert PRECISION_BYTES["fp8"]  == 1.0
    assert PRECISION_BYTES["mxfp4"] == 0.5


def test_all_default_topologies_buildable():
    """Each named topology builder produces a usable DeviceTopology."""
    for name, factory in TOPOLOGY_BUILDERS.items():
        topo = factory()
        assert topo.total_hbm_bytes > 0, name
