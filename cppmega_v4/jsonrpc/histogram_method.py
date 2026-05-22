"""V7-H08: inspect.histogram RPC.

Returns a {bins, counts, min, max, mean} histogram for a given
brick's weight (or grad) tensor at a specified step. Server-side
bucketing avoids shipping the full tensor over the wire.
"""

from __future__ import annotations

from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


class HistogramParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: VerifyParams
    brick_id: str
    kind: Literal["weight", "grad"] = "weight"
    buckets: int = Field(64, ge=2, le=512)
    num_steps: int = Field(2, ge=1, le=64)


class HistogramResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    brick_id: str
    kind: str
    buckets: int
    bins: list[float]
    counts: list[int]
    min: float
    max: float
    mean: float
    n_values: int


def _flatten_brick_weights(model: Any, brick_id: str) -> mx.array | None:
    """Walk the trained model, find the brick by attribute path and
    return its concatenated flat weights."""
    # The training stage exposes all_modules as nn.Sequential.
    # Walk top-level submodules; each module's .name often carries
    # the brick id via the BLOCK_BUILDERS wrappers — fall back to
    # index-based positions.
    parts: list[mx.array] = []
    for sub in nn.utils.tree_flatten(model.parameters()):
        key, val = sub
        if brick_id in key and hasattr(val, "shape"):
            parts.append(mx.flatten(val))
    if not parts:
        return None
    return mx.concatenate(parts)


def inspect_histogram(params: HistogramParams,
                       *, cache: Any | None = None) -> HistogramResult:
    """Build the brick by kind and bucket its weight tensor.

    For V7-H08 backend MVP we instantiate the brick standalone via
    BLOCK_BUILDERS (avoiding the train pipeline so the call is fast
    enough for live-stream inspection). Weight-vs-grad differentiation
    is deferred to the WS-streaming sub-task; this entry returns the
    weight histogram at fresh init.
    """
    from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS

    spec_nodes = list(params.spec.graph.nodes or [])
    brick_node = next(
        (n for n in spec_nodes if getattr(n, "id", "") == params.brick_id),
        None,
    )
    if brick_node is None:
        raise ValueError(
            f"brick_id={params.brick_id!r} not in spec.graph.nodes"
        )
    kind = getattr(brick_node, "kind", "")
    if kind not in BLOCK_BUILDERS:
        raise ValueError(
            f"brick_id={params.brick_id!r} has unknown kind={kind!r}"
        )
    hidden = int(getattr(params.spec.dim_env, "H", 128))
    brick_params = dict(getattr(brick_node, "params", {}) or {})
    module = BLOCK_BUILDERS[kind](hidden, brick_params)
    flat_parts: list[mx.array] = []
    for _k, v in nn.utils.tree_flatten(module.parameters()):
        if hasattr(v, "shape"):
            flat_parts.append(mx.flatten(v))
    flat = (mx.concatenate(flat_parts) if flat_parts else None)
    if flat is None or flat.size == 0:
        raise ValueError(
            f"brick_id={params.brick_id!r} has no flat weights in the spec"
        )
    arr = flat.astype(mx.float32)
    n_values = int(arr.size)
    lo = float(mx.min(arr).item())
    hi = float(mx.max(arr).item())
    mean = float(mx.mean(arr).item())
    # Avoid degenerate single-bin range.
    if hi - lo < 1e-12:
        hi = lo + 1e-9
    width = (hi - lo) / params.buckets
    # Bin via integer indexing for speed.
    idx = ((arr - mx.array(lo)) / mx.array(width)).astype(mx.int32)
    idx = mx.clip(idx, 0, params.buckets - 1)
    counts = [0] * params.buckets
    for v in idx.tolist():
        counts[int(v)] += 1
    bins = [lo + i * width for i in range(params.buckets + 1)]
    # Stash the report so caller can correlate with extras if needed.
    return HistogramResult(
        brick_id=params.brick_id,
        kind=params.kind,
        buckets=params.buckets,
        bins=bins,
        counts=counts,
        min=lo,
        max=hi,
        mean=mean,
        n_values=n_values,
    )


__all__ = ["HistogramParams", "HistogramResult", "inspect_histogram"]
