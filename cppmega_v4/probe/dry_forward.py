"""Dry forward pass — synthetic input_ids on the instantiated graph.

Last gate after static checks. Each brick already has a unit test that
its forward preserves (B, S, H) at small sizes; here we chain the
instantiated graph and run **one** forward at hidden=64, batch=1, seq=8
to catch residual brick-to-brick coupling that the resolver missed.

Returns a one-word verdict + an optional exception trace. Never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from cppmega_v4.fusion.brick_graph import BrickGraph
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


@dataclass(frozen=True)
class DryForwardResult:
    verdict: Literal["ok", "shape_mismatch", "exception"]
    detail: str = ""


def dry_forward(
    graph: BrickGraph,
    *,
    hidden_size: int = 64,
    seq_len: int = 8,
    batch: int = 1,
) -> DryForwardResult:
    """Walk ``graph`` in declared order, forwarding a synthetic activation.

    Parallel-block topologies are handled by averaging branch outputs at
    the fan-in point — matches the convention BrickGraph uses for
    Tiny-Aya style ``GQA‖MLP``.
    """
    try:
        # Build modules fresh — caller may have passed instantiate=False.
        modules: dict[str, object] = {}
        for node in graph.nodes:
            builder = BLOCK_BUILDERS.get(node.kind)
            if builder is None:
                return DryForwardResult(
                    verdict="exception",
                    detail=f"unknown brick kind {node.kind!r}",
                )
            modules[node.name] = builder(hidden_size, dict(node.params))

        x0 = mx.random.normal((batch, seq_len, hidden_size))
        # Topo: pre-compute predecessors map; if a node has multiple
        # predecessors, mean-reduce their outputs before forwarding.
        outputs: dict[str, mx.array] = {}
        roots = [n.name for n in graph.nodes if not graph.predecessors(n.name)]
        for name in roots:
            outputs[name] = modules[name](x0)
        # Iterate remaining nodes in declared order (graph is already a
        # topological declaration thanks to from_block_specs).
        for node in graph.nodes:
            if node.name in outputs:
                continue
            preds = graph.predecessors(node.name)
            if not preds:
                outputs[node.name] = modules[node.name](x0)
                continue
            if len(preds) == 1:
                inp = outputs[preds[0]]
            else:
                stacked = mx.stack([outputs[p] for p in preds], axis=0)
                inp = mx.mean(stacked, axis=0)
            outputs[node.name] = modules[node.name](inp)

        last = graph.nodes[-1].name
        y = outputs[last]
        if y.shape != (batch, seq_len, hidden_size):
            return DryForwardResult(
                verdict="shape_mismatch",
                detail=f"final shape {y.shape} != "
                       f"(batch={batch}, seq={seq_len}, H={hidden_size})",
            )
        return DryForwardResult(verdict="ok")
    except Exception as exc:
        return DryForwardResult(
            verdict="exception",
            detail=f"{type(exc).__name__}: {exc}",
        )
