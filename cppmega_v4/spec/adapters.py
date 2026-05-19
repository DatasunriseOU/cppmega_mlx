"""Auto-bridge adapter library — small inserts that close shape gaps.

When the Stage B resolver finds a mismatch on an edge, this module
proposes (Stage E) or inserts (lenient resolver path, this stage) a
small *adapter brick* that turns the producer's output into something
the consumer expects. Adapters are first-class BrickShapeContract
entries (so they participate in fusion / memory accounting) and live
under reserved kind prefix ``adapter_*`` to avoid colliding with
BLOCK_BUILDERS.

The six rules shipped in this stage cover the most common GUI cases:

  - ``adapter_merge_heads``     (B,nh,S,d)   -> (B,S,nh*d)
  - ``adapter_split_heads``     (B,S,nh*d)   -> (B,nh,S,d)
  - ``adapter_transpose_bnsd``  (B,nh,S,d)   -> (B,S,nh,d)
  - ``adapter_linear_bridge``   (B,S,H_a)    -> (B,S,H_b)
  - ``adapter_rmsnorm``         (B,S,H)      -> (B,S,H) with norm
  - ``adapter_residual``        (B,S,H) skip -> (B,S,H) with residual

Every adapter is categorised as ``norm_or_proj`` in the fusion
compatibility table, so the planner happily groups it into a region
with its surrounding bricks. That means the *cost* of an auto-inserted
adapter is almost always zero at run-time (fused into the neighbour's
kernel).

Public surface:
  - :data:`ADAPTER_RULES` — the rule table the planner walks
  - :class:`AdapterRule`
  - :class:`AdapterSuggestion`
  - :func:`suggest_adapter_chain` — produce the (possibly empty) chain
    of adapters that turns ``producer_shape`` into ``consumer_shape``.
  - :func:`insert_adapter_chain` — splice the chain into a BrickGraph
    on a given edge, returning the rewritten graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.fusion.compatibility import _CATEGORY_BY_KIND
from cppmega_v4.spec.shape_contract import (
    BrickShapeContract,
    ShapeExpr,
    register_contract,
)


# ---------------------------------------------------------------------------
# Adapter brick contracts
# ---------------------------------------------------------------------------


def _zero() -> ShapeExpr:
    return ShapeExpr(("0",))


_ADAPTER_MERGE_HEADS = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "nh", "S", "head_dim"))},
    outputs={"y": ShapeExpr(("B", "S", "nh * head_dim"))},
    params_elems=_zero(),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=_zero(),
    description="reshape (B,nh,S,d) -> (B,S,nh*d)",
)


_ADAPTER_SPLIT_HEADS = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "S", "nh * head_dim"))},
    outputs={"y": ShapeExpr(("B", "nh", "S", "head_dim"))},
    params_elems=_zero(),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=_zero(),
    description="reshape (B,S,nh*d) -> (B,nh,S,d)",
)


_ADAPTER_TRANSPOSE_BNSD = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "nh", "S", "head_dim"))},
    outputs={"y": ShapeExpr(("B", "S", "nh", "head_dim"))},
    params_elems=_zero(),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=_zero(),
    description="permute (B,nh,S,d) -> (B,S,nh,d)",
)


_ADAPTER_LINEAR_BRIDGE = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "S", "H"))},
    outputs={"y": ShapeExpr(("B", "S", "H"))},
    # H_in * H_out (no bias); resolver fills `H` for both sides
    # asymmetrically by binding extra params on the BrickNode
    params_elems=ShapeExpr(("H * H",)),
    activations_elems=ShapeExpr(("B * S * H",)),
    kv_cache_elems=_zero(),
    description="Linear(H_in, H_out) projection bridge",
)


_ADAPTER_RMSNORM = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "S", "H"))},
    outputs={"y": ShapeExpr(("B", "S", "H"))},
    params_elems=ShapeExpr(("H",)),
    activations_elems=ShapeExpr(("B * S * H",)),
    kv_cache_elems=_zero(),
    description="RMSNorm normalisation adapter",
)


_ADAPTER_RESIDUAL = BrickShapeContract(
    inputs={"x": ShapeExpr(("B", "S", "H"))},
    outputs={"y": ShapeExpr(("B", "S", "H"))},
    params_elems=_zero(),
    activations_elems=ShapeExpr(("B * S * H",)),
    kv_cache_elems=_zero(),
    description="residual passthrough (x + previous skip)",
)


_ADAPTER_KINDS_TO_REGISTER: dict[str, BrickShapeContract] = {
    "adapter_merge_heads":     _ADAPTER_MERGE_HEADS,
    "adapter_split_heads":     _ADAPTER_SPLIT_HEADS,
    "adapter_transpose_bnsd":  _ADAPTER_TRANSPOSE_BNSD,
    "adapter_linear_bridge":   _ADAPTER_LINEAR_BRIDGE,
    "adapter_rmsnorm":         _ADAPTER_RMSNORM,
    "adapter_residual":        _ADAPTER_RESIDUAL,
}


def _register_all_adapter_contracts() -> None:
    """Idempotent registration — called on module import."""
    for kind, contract in _ADAPTER_KINDS_TO_REGISTER.items():
        register_contract(kind, contract)
    # Adapters are pointwise / pure projection — they fuse easily with
    # their neighbours. Mirror that in the fusion compatibility table.
    for kind in _ADAPTER_KINDS_TO_REGISTER:
        _CATEGORY_BY_KIND.setdefault(kind, "norm_or_proj")


_register_all_adapter_contracts()


# ---------------------------------------------------------------------------
# Adapter graph-walker support
# ---------------------------------------------------------------------------
#
# BrickGraph nodes refuse unknown kinds via BLOCK_BUILDERS + the
# _ADDITIONAL_FUSION_KINDS escape hatch. Add the adapter kinds there so
# planners / tests can build BrickNodes directly without a builder.

from cppmega_v4.fusion import brick_graph as _bg_module

_bg_module._ADDITIONAL_FUSION_KINDS = (
    _bg_module._ADDITIONAL_FUSION_KINDS | frozenset(_ADAPTER_KINDS_TO_REGISTER)
)


# ---------------------------------------------------------------------------
# Rules & suggestion logic
# ---------------------------------------------------------------------------


_TriggerFn = Callable[[tuple[int, ...], tuple[int, ...]], bool]


@dataclass(frozen=True)
class AdapterRule:
    """One row in the adapter table.

    Fields:
      kind: the BLOCK_BUILDERS-shaped kind name (always ``adapter_*``).
      when: predicate over (producer_shape, consumer_shape) — True if
        this rule can handle the mismatch.
      transform: function (producer_shape, consumer_shape) -> new shape
        the adapter emits. The resolver chains rules: each new shape
        becomes the next iteration's producer_shape until consumer_shape
        is reached or no rule matches.
      description: human-readable label (shown in GUI tooltips).
    """

    kind: str
    when: _TriggerFn
    transform: Callable[[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]
    description: str


def _merge_heads_trigger(
    p: tuple[int, ...], c: tuple[int, ...]
) -> bool:
    return (
        len(p) == 4 and len(c) == 3
        and p[0] == c[0] and p[2] == c[1]
        and p[1] * p[3] == c[2]
    )


def _split_heads_trigger(
    p: tuple[int, ...], c: tuple[int, ...]
) -> bool:
    """Accept both consumer layouts:

      Option A: consumer is (B, nh, S, d) — split emits c directly.
      Option B: consumer is (B, S, nh, d) — split emits (B,nh,S,d), then
        a transpose rule completes the chain.
    """
    if not (len(p) == 3 and len(c) == 4 and p[0] == c[0]):
        return False
    # Option A: c=(B,nh,S,d) — match S in dim2 and nh*d in last
    if p[1] == c[2] and p[2] == c[1] * c[3]:
        return True
    # Option B: c=(B,S,nh,d) — match S in dim1 and nh*d in last
    if p[1] == c[1] and p[2] == c[2] * c[3]:
        return True
    return False


def _split_heads_transform(
    p: tuple[int, ...], c: tuple[int, ...]
) -> tuple[int, ...]:
    """Always emit (B, nh, S, d). Choose nh/d based on the consumer's
    layout (Option A vs B from ``_split_heads_trigger``)."""
    if p[1] == c[2] and p[2] == c[1] * c[3]:   # Option A
        return (p[0], c[1], p[1], c[3])
    return (p[0], c[2], p[1], c[3])             # Option B


def _transpose_bnsd_trigger(
    p: tuple[int, ...], c: tuple[int, ...]
) -> bool:
    """(B, nh, S, d) <-> (B, S, nh, d) reorder."""
    return (
        len(p) == 4 and len(c) == 4
        and p[0] == c[0] and p[1] == c[2]
        and p[2] == c[1] and p[3] == c[3]
        and p != c
    )


def _linear_bridge_trigger(
    p: tuple[int, ...], c: tuple[int, ...]
) -> bool:
    return (
        len(p) == 3 and len(c) == 3
        and p[0] == c[0] and p[1] == c[1]
        and p[2] != c[2]
    )


ADAPTER_RULES: tuple[AdapterRule, ...] = (
    AdapterRule(
        kind="adapter_merge_heads",
        when=_merge_heads_trigger,
        transform=lambda p, c: (p[0], p[2], p[1] * p[3]),
        description="merge heads: (B,nh,S,d) -> (B,S,nh*d)",
    ),
    AdapterRule(
        kind="adapter_split_heads",
        when=_split_heads_trigger,
        transform=_split_heads_transform,
        description="split heads: (B,S,nh*d) -> (B,nh,S,d)",
    ),
    AdapterRule(
        kind="adapter_transpose_bnsd",
        when=_transpose_bnsd_trigger,
        transform=lambda p, c: (p[0], p[2], p[1], p[3]),
        description="transpose: (B,nh,S,d) -> (B,S,nh,d)",
    ),
    AdapterRule(
        kind="adapter_linear_bridge",
        when=_linear_bridge_trigger,
        transform=lambda p, c: (p[0], p[1], c[2]),
        description="Linear(H_in, H_out) projection bridge",
    ),
)


@dataclass(frozen=True)
class AdapterSuggestion:
    """One step in a planned adapter chain."""

    kind: str
    description: str
    output_shape: tuple[int, ...]


def suggest_adapter_chain(
    producer_shape: tuple[int, ...],
    consumer_shape: tuple[int, ...],
    *,
    max_steps: int = 4,
) -> list[AdapterSuggestion] | None:
    """Return the shortest chain of adapter steps that turns
    ``producer_shape`` into ``consumer_shape``, or None if no chain
    of length ≤ ``max_steps`` exists.

    The walker tries each rule whose ``when`` predicate matches the
    current shape; ties go to the order in :data:`ADAPTER_RULES`. The
    chain is greedy (not exhaustive search) — for the patterns we care
    about (max 2 hops: split_heads then transpose, etc.) this finds the
    right answer.
    """
    if producer_shape == consumer_shape:
        return []
    chain: list[AdapterSuggestion] = []
    current = producer_shape
    for _ in range(max_steps):
        if current == consumer_shape:
            return chain
        progressed = False
        for rule in ADAPTER_RULES:
            if not rule.when(current, consumer_shape):
                continue
            new_shape = rule.transform(current, consumer_shape)
            chain.append(
                AdapterSuggestion(
                    kind=rule.kind,
                    description=rule.description,
                    output_shape=new_shape,
                )
            )
            current = new_shape
            progressed = True
            break
        if not progressed:
            return None
    if current == consumer_shape:
        return chain
    return None


def insert_adapter_chain(
    graph: BrickGraph,
    producer_name: str,
    consumer_name: str,
    suggestions: list[AdapterSuggestion],
    *,
    name_prefix: str | None = None,
) -> BrickGraph:
    """Splice ``suggestions`` into ``graph`` on the edge
    ``producer_name -> consumer_name``. Returns a *new* BrickGraph.

    Each adapter node gets a synthesised unique name
    ``{name_prefix or producer_name}__adapt_{i}_{kind}``.
    """
    if not suggestions:
        return graph
    if (producer_name, consumer_name) not in graph.edges:
        raise KeyError(
            f"edge ({producer_name!r} -> {consumer_name!r}) not in graph"
        )

    prefix = name_prefix or producer_name
    new_nodes: list[BrickNode] = list(graph.nodes)
    new_edges: list[tuple[str, str]] = [
        e for e in graph.edges if e != (producer_name, consumer_name)
    ]
    existing_names = {n.name for n in new_nodes}

    prev = producer_name
    for i, s in enumerate(suggestions):
        candidate = f"{prefix}__adapt_{i}_{s.kind}"
        suffix = 0
        unique = candidate
        while unique in existing_names:
            suffix += 1
            unique = f"{candidate}_{suffix}"
        existing_names.add(unique)
        new_nodes.append(
            BrickNode(kind=s.kind, name=unique, params={}, module=None)
        )
        new_edges.append((prev, unique))
        prev = unique
    new_edges.append((prev, consumer_name))

    # Reinstate original edge ordering: replace the missing edge slot
    # with the new chain at the same position.
    original_idx = list(graph.edges).index((producer_name, consumer_name))
    chain_edges: list[tuple[str, str]] = []
    cur = producer_name
    for s_idx, _ in enumerate(suggestions):
        nxt = new_nodes[len(graph.nodes) + s_idx].name
        chain_edges.append((cur, nxt))
        cur = nxt
    chain_edges.append((cur, consumer_name))
    final_edges = (
        list(graph.edges[:original_idx])
        + chain_edges
        + list(graph.edges[original_idx + 1:])
    )
    return BrickGraph(nodes=tuple(new_nodes), edges=tuple(final_edges))


__all__ = [
    "ADAPTER_RULES",
    "AdapterRule",
    "AdapterSuggestion",
    "insert_adapter_chain",
    "suggest_adapter_chain",
]
