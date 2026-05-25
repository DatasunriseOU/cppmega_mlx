"""BrickGraph — graph IR over composed V4 bricks.

A V4 model is a sequence (or DAG, for MoE routing) of bricks chosen from
``cppmega_v4.models.unified_superblock_v4.BLOCK_BUILDERS``. BrickGraph is
the smallest representation that downstream fusion planners need:

  - what brick (kind + params + instantiated module)
  - producer→consumer dependencies (linear chain or routed fan-out/in)

Two ingest paths:

  ``from_block_specs(specs)`` — JSON-shaped block specs (the same surface
  the future GUI emits). Each spec is ``{"kind": "...", "name": "...",
  "params": {...}}``. Linear chain inferred from list order.

  ``from_mlx_model(model)`` — walk an existing ``nn.Module`` tree and
  identify which submodules are V4 bricks (by class match against
  BLOCK_BUILDERS' construction output). Linear chain inferred from
  attribute iteration order on the parent module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.nn as nn

from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


_ADDITIONAL_FUSION_KINDS = frozenset(
    {
        "mamba3",
        "m2rnn",
        "sparse_mla_fp8",
        "ssm",
    }
)


@dataclass(frozen=True)
class BrickNode:
    """One brick in the graph: kind, name, params, instantiated module."""

    kind: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    module: nn.Module | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("BrickNode.kind must be non-empty")
        if not self.name:
            raise ValueError("BrickNode.name must be non-empty")
        if self.kind not in BLOCK_BUILDERS and self.kind not in _ADDITIONAL_FUSION_KINDS:
            raise ValueError(
                f"BrickNode.kind={self.kind!r} not in BLOCK_BUILDERS "
                f"({sorted(BLOCK_BUILDERS)!r})"
            )


@dataclass(frozen=True)
class BrickGraph:
    """Directed graph of bricks. Edges are ``(producer_name, consumer_name)``.

    A simple linear chain has ``edges = [(n0,n1), (n1,n2), ...]``. MoE
    routing introduces fan-out: one router node with N outbound edges, one
    combine node with N inbound edges.
    """

    nodes: tuple[BrickNode, ...]
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for n in self.nodes:
            if n.name in seen:
                raise ValueError(f"duplicate brick name {n.name!r}")
            seen.add(n.name)
        names = {n.name for n in self.nodes}
        for p, c in self.edges:
            if p not in names:
                raise ValueError(f"edge producer {p!r} not in node set")
            if c not in names:
                raise ValueError(f"edge consumer {c!r} not in node set")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)

    def by_name(self, name: str) -> BrickNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def successors(self, name: str) -> tuple[str, ...]:
        return tuple(c for p, c in self.edges if p == name)

    def predecessors(self, name: str) -> tuple[str, ...]:
        return tuple(p for p, c in self.edges if c == name)


# ---------------------------------------------------------------------------
# Ingest from JSON-shaped block specs
# ---------------------------------------------------------------------------


def from_block_specs(
    specs: Sequence[dict],
    *,
    hidden_size: int,
    instantiate: bool = True,
) -> BrickGraph:
    """Build a BrickGraph from a list of block specs.

    Each spec is either:
      - **leaf**: ``{"kind": "<BLOCK_BUILDERS key>", "name": "...", "params": {...}}``
      - **parallel-block** (GalCov-C): ``{"parallel": [leaf_spec, ...]}`` —
        all branches receive an edge from the preceding spec AND an edge
        to the following spec. Use for Tiny Aya-style ``GQA‖MLP`` patterns
        that BrickGraph DAG already supports but the linear DSL didn't expose.

    Edges:
      - leaf → leaf: linear chain (specs[i] → specs[i+1]).
      - leaf → parallel-block: leaf → every branch's first leaf.
      - parallel-block → leaf: every branch's last leaf → leaf.
      - parallel-block → parallel-block: each prev-block branch tail →
        each next-block branch head (full cross product; conservative
        but rarely used and easy to override later via explicit edges).

    Args:
        specs: ordered list of brick specs (leaf or parallel).
        hidden_size: passed to each ``BLOCK_BUILDERS[kind](hidden_size, params)``.
        instantiate: when True, construct each module via BLOCK_BUILDERS.
    """
    nodes: list[BrickNode] = []
    used_names: set[str] = set()
    # Per-spec entry: tuple (head_names, tail_names) — used to wire edges
    # between adjacent specs (parallel-block heads/tails may be multi-valued).
    spec_endpoints: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    spec_counter = 0

    def _emit_leaf(leaf_spec: dict) -> BrickNode:
        nonlocal spec_counter
        kind = leaf_spec["kind"]
        name = leaf_spec.get("name") or f"{kind}_{spec_counter}"
        spec_counter += 1
        if name in used_names:
            raise ValueError(f"duplicate name {name!r} in specs")
        used_names.add(name)
        params = dict(leaf_spec.get("params") or {})
        module = (
            BLOCK_BUILDERS[kind](hidden_size, params) if instantiate else None
        )
        node = BrickNode(kind=kind, name=name, params=params, module=module)
        nodes.append(node)
        return node

    for spec in specs:
        if "parallel" in spec:
            branches = spec["parallel"]
            if not branches:
                raise ValueError("parallel-block spec must have ≥1 branch")
            heads: list[str] = []
            tails: list[str] = []
            for branch in branches:
                if not isinstance(branch, dict) or "kind" not in branch:
                    raise ValueError(
                        "parallel-block branches must be leaf-specs "
                        "{'kind': ..., 'name': ..., 'params': ...}"
                    )
                node = _emit_leaf(branch)
                heads.append(node.name)
                tails.append(node.name)
            spec_endpoints.append((tuple(heads), tuple(tails)))
        else:
            node = _emit_leaf(spec)
            spec_endpoints.append(((node.name,), (node.name,)))

    edges: list[tuple[str, str]] = []
    for i in range(len(spec_endpoints) - 1):
        _, prev_tails = spec_endpoints[i]
        next_heads, _ = spec_endpoints[i + 1]
        for t in prev_tails:
            for h in next_heads:
                edges.append((t, h))

    return BrickGraph(nodes=tuple(nodes), edges=tuple(edges))


# ---------------------------------------------------------------------------
# Ingest from an existing nn.Module tree
# ---------------------------------------------------------------------------


def _brick_kind_of(module: nn.Module) -> str | None:
    """Return the BLOCK_BUILDERS kind that produced ``module``, or None.

    The check is structural: each builder returns a known class (or wraps
    upstream in our V4 wrapper). We look up by qualified class name.
    """
    cls_name = type(module).__name__
    # Hard-coded map of class-name -> BLOCK_BUILDERS key. Mirrors the
    # builders in cppmega_v4/models/unified_superblock_v4.py. New brick
    # added to BLOCK_BUILDERS must also be added here for graph walking.
    KIND_BY_CLASS: dict[str, str] = {
        "GatedAttentionBlock": "gated_attention",
        "Mistral4MLABlock": "mistral4_mla",
        "DSv4AttentionBlock": "dsv4_attention",
        "BailingLinearAttnBlock": "bailing_linear",
        "BailingMLABlock": "bailing_mla",
        "BailingMoEBlock": "bailing_moe",
        "Gemma4DrafterLayerBlock": "gemma4_drafter",
        "NemotronHMTPBlockWrapper": "nemotron_h_mtp",
        "MLABlock": "mla",
        # Stage D additions
        "GQAWithSlidingWindowBlock": "gqa_sliding",
        "CCAAttentionBlock": "cca_attention",
        "Mamba3ReferenceBlock": "mamba3",
        # GalCov-B additions
        "MLSTMBlock": "mlstm",
        "AbsPosEmbedBlock": "abs_pos_embed",
        "PerLayerEmbedBlock": "per_layer_embed",
        "EmbeddingTableBlock": "embedding_table",
        # gdn / kda / nsa / csa_hca / moe / mlp / engram / attention /
        # lightning_indexer are local closures in unified_superblock_v4.py
        # so their _SelfAttn / _MLP class names aren't unique. Callers
        # that need to walk such pre-built models should attach a
        # ``_v4_brick_kind`` attribute to the module instance, which we
        # honour below.
    }
    if hasattr(module, "_v4_brick_kind"):
        return getattr(module, "_v4_brick_kind")
    return KIND_BY_CLASS.get(cls_name)


def from_mlx_model(
    model: nn.Module,
    *,
    attr_order: Sequence[str] | None = None,
) -> BrickGraph:
    """Walk ``model`` and extract BrickNodes from its direct children.

    Only the IMMEDIATE children of ``model`` are considered (one level
    deep) — deeper recursion is the caller's job. This matches how
    UnifiedSuperblock composes bricks: as flat siblings under one parent.

    Edges are inferred as a linear chain in the order children were
    declared (Python 3.7+ dict ordering). Override via ``attr_order``.
    """
    nodes: list[BrickNode] = []
    children = (
        list(attr_order)
        if attr_order is not None
        else [name for name, val in model.items() if isinstance(val, nn.Module)]
    )
    used_names: set[str] = set()
    for child_name in children:
        child = getattr(model, child_name, None)
        if child is None or not isinstance(child, nn.Module):
            continue
        kind = _brick_kind_of(child)
        if kind is None:
            continue
        unique_name = child_name
        suffix = 0
        while unique_name in used_names:
            suffix += 1
            unique_name = f"{child_name}_{suffix}"
        used_names.add(unique_name)
        nodes.append(
            BrickNode(kind=kind, name=unique_name, params={}, module=child)
        )

    edges = tuple(
        (nodes[i].name, nodes[i + 1].name) for i in range(len(nodes) - 1)
    )
    return BrickGraph(nodes=tuple(nodes), edges=edges)


__all__ = [
    "BrickGraph",
    "BrickNode",
    "from_block_specs",
    "from_mlx_model",
]
