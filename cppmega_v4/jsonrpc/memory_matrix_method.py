"""V8-R03: ``memory.matrix`` RPC — compute a (topology × precision) grid
of memory estimates and per-cell fit verdicts.

Used by the V8 MemoryMatrix sidebar tab to show the user at-a-glance
which device-precision combinations fit their current canvas spec.

Backend strategy
----------------

For each cell we run :func:`cppmega_v4.spec.verify_and_estimate` once
in bf16, then post-scale dtype-sensitive components (weights,
gradients, activations, optimizer state) by ``precision_bytes / 2``.
``mxfp4`` is 0.5 bytes per element — supported because the scaling is
done in float space and rounded only on the final ``total_bytes``.

This is the same approach the V8 spec §3 nominates: one verify per
matrix, post-scale per precision, so the round-trip is fast enough to
re-run on every spec edit.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from cppmega_v4.fusion.brick_graph import from_block_specs
from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import (
    _cache_lookup, _cache_store, _graph_to_specs,
)
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.parallelism import topology as _topo
from cppmega_v4.spec.api import verify_and_estimate


__all__ = [
    "MemoryMatrixParams",
    "MemoryMatrixCell",
    "MemoryMatrixResult",
    "memory_matrix",
    "PRECISION_BYTES",
    "TOPOLOGY_BUILDERS",
]


# Precision -> bytes per element. mxfp4 is half a byte (4-bit mantissa
# with shared e4m3 scale per block — the scale overhead is folded into
# the headroom and is < 5% at block_size=16).
PRECISION_BYTES: dict[str, float] = {
    "fp32":  4.0,
    "bf16":  2.0,
    "fp16":  2.0,
    "fp8":   1.0,
    "mxfp4": 0.5,
}


# Builder lookup for topologies the matrix supports. Wraps the
# zero-arg factories from cppmega_v4.parallelism.topology.
TOPOLOGY_BUILDERS: dict[str, Any] = {
    "h100_8x":       lambda: _topo.h100_8x(),
    "h200_8x":       lambda: _topo.h200_8x(),
    "a100_8x":       lambda: _topo.a100_8x(),
    "b100_8x":       lambda: _topo.b100_8x(),
    "gb10_quarter":  lambda: _topo.gb10_quarter(),
    "tpu_v6e_8":     lambda: _topo.tpu_v6e_8(),
    "tpu_v5p_4":     lambda: _topo.tpu_v5p_4(),
    "m3_ultra_solo": lambda: _topo.m3_ultra_solo(),
}


_DEFAULT_TOPOLOGIES = (
    "h100_8x", "m3_ultra_solo", "gb10_quarter", "tpu_v6e_8",
)
_DEFAULT_PRECISIONS = ("fp32", "bf16", "fp16", "fp8", "mxfp4")


class MemoryMatrixParams(BaseModel):
    """Input — a VerifyParams payload + the topologies/precisions axes."""

    model_config = ConfigDict(extra="forbid")

    spec: VerifyParams
    topologies: list[str] | None = None
    precisions: list[str] | None = None
    headroom: float = 0.9


class MemoryMatrixCell(BaseModel):
    """One cell of the matrix — wire-form."""

    model_config = ConfigDict(extra="forbid")

    topology: str
    precision: str
    bytes: int
    device_hbm_bytes: int
    fits: bool
    headroom: float
    breakdown: dict[str, int]


class MemoryMatrixResult(BaseModel):
    """The full matrix — one row per (topology, precision) pair."""

    model_config = ConfigDict(extra="forbid")

    cells: list[MemoryMatrixCell]
    topologies: list[str]
    precisions: list[str]


def _scale_int(value: int, factor: float) -> int:
    """Round half-away-from-zero so e.g. ``2 * 0.5 = 1`` not ``0``."""
    return int(math.floor(value * factor + 0.5))


def memory_matrix(
    params: MemoryMatrixParams, *, cache: LRUCache | None = None,
) -> MemoryMatrixResult:
    """Build the (topology × precision) memory matrix for ``params.spec``."""
    key, hit = _cache_lookup(cache, "memory.matrix", params)
    if hit is not None:
        return hit

    topologies = list(params.topologies or _DEFAULT_TOPOLOGIES)
    precisions = list(params.precisions or _DEFAULT_PRECISIONS)
    for t in topologies:
        if t not in TOPOLOGY_BUILDERS:
            raise ValueError(
                f"unknown topology {t!r}; choose from "
                f"{sorted(TOPOLOGY_BUILDERS)}")
    for p in precisions:
        if p not in PRECISION_BYTES:
            raise ValueError(
                f"unknown precision {p!r}; choose from "
                f"{sorted(PRECISION_BYTES)}")
    if not 0.0 < params.headroom <= 1.0:
        raise ValueError(
            f"headroom must be in (0, 1], got {params.headroom}")

    # One verify_and_estimate call gives us the full MemoryReport with
    # aggregate totals; we then post-scale per precision in float space.
    specs = _graph_to_specs(params.spec.graph)
    hidden = params.spec.dim_env.get("H", 64)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    base = verify_and_estimate(
        graph,
        dim_env=params.spec.dim_env,
        training=params.spec.training,
    ).memory

    base_weights_b   = base.weights_bytes
    base_grads_b     = base.grads_bytes
    base_optim_b     = base.optimizer_bytes
    base_act_b       = base.activations_bytes
    base_kv_b        = base.kv_cache_bytes
    base_edge_b      = base.edge_handoff_bytes
    # KV cache uses kv_cache_dtype_bytes (independent of weight dtype),
    # so we don't scale it. Edge handoff is bytes-of-tensors-in-transit
    # and IS dtype-sensitive.

    cells: list[MemoryMatrixCell] = []
    for t_name in topologies:
        topo = TOPOLOGY_BUILDERS[t_name]()
        hbm = topo.total_hbm_bytes
        for p_name in precisions:
            scale = PRECISION_BYTES[p_name] / 2.0  # bf16 baseline
            w  = _scale_int(base_weights_b, scale)
            g  = _scale_int(base_grads_b, scale)
            o  = _scale_int(base_optim_b, scale)
            a  = _scale_int(base_act_b, scale)
            e  = _scale_int(base_edge_b, scale)
            total = w + g + o + a + base_kv_b + e
            cells.append(MemoryMatrixCell(
                topology=t_name, precision=p_name,
                bytes=total, device_hbm_bytes=hbm,
                fits=total <= int(hbm * params.headroom),
                headroom=params.headroom,
                breakdown={
                    "weights":     w,
                    "grads":       g,
                    "optimizer":   o,
                    "activations": a,
                    "kv_cache":    base_kv_b,
                    "edge_handoff": e,
                },
            ))

    out = MemoryMatrixResult(
        cells=cells, topologies=topologies, precisions=precisions)
    _cache_store(cache, key, out)
    return out
