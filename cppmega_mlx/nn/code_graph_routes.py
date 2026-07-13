"""Graph-edge -> fixed routing prior for the DSA dependency indexer.

This module turns the typed code-graph edges carried by a
:class:`~cppmega_mlx.data.graph_packet.GraphPacket` (``call`` edges, and
``type`` edges when present) into a **fixed block-routing prior** ``S_blk`` that
the lightning indexer adds to its learned score.

Design (grounded in the research recipe):

* **TNO "fixed route, learned channel mixing".** The *route* — which block pairs
  are connected — is fixed by the graph structure (it is data, not a parameter).
  The only learnable degrees of freedom here are the per-relation mixing scalars
  ``alpha_r`` (one scalar per relation: ``call``, ``type``, ...). The prior is

      S_blk[t, s] = sum_r  alpha_r * A^r_blk[t, s]

  where ``A^r_blk`` is the block->block adjacency for relation ``r`` obtained by
  block-aggregating the token-level (chunk-level) edges to a fixed ``B`` blocks.

* **Block aggregation** reuses :meth:`GraphPacket.block_aggregate`, which maps
  every node (chunk) index to its block ``node // block_size`` and accumulates an
  integer block->block edge-count matrix. We then normalise the counts so the
  prior magnitude does not scale with how many raw edges happen to land in a
  block pair (``"binary"`` or ``"count"`` modes).

The module exposes two helpers consumed by the indexer / loss:

* :func:`build_attention_bias` -> the dense ``(B, B)`` (or per-relation) prior
  ``S_blk`` used as an additive bias on the indexer score.
* :func:`build_block_candidates` -> the set of *true* destination blocks each
  query block routes to (used for the coverage hinge / BCE supervision and for
  recall@k metrics).

RULE #1: every helper FAILS LOUD with WHERE + WHAT on shape/relation/range
violations. No silent fallback, no clamping, no fabricated edges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cppmega_mlx.data.graph_packet import GraphBatch, GraphPacket

# Relations we know how to route. ``call`` is always primary (callsite->callee);
# ``type`` (template_ref / field access) is folded in when present.
_DEFAULT_RELATIONS: tuple[str, ...] = ("call", "type")
_TOKEN_RELATIONS: tuple[str, ...] = (
    "domain",
    "build",
    "shell",
    "diagnostic",
    "cross_domain",
)
_DEFAULT_TOKEN_RELATION_WEIGHTS: Mapping[str, float] = {
    "call": 1.0,
    "type": 1.0,
    "domain": 1.0,
    "build": 1.0,
    "shell": 1.0,
    "diagnostic": 1.0,
    "cross_domain": 1.0,
}
_NORMALIZE_MODES = ("binary", "count")


def _resolve_num_nodes(packet: GraphPacket, relation: str) -> int:
    """Resolve the node (chunk) count for ``relation``, raising on ambiguity."""

    edge = packet.edge(relation)
    if edge is None:
        raise KeyError(
            f"code_graph_routes._resolve_num_nodes: relation {relation!r} not in "
            f"packet; have {packet.relations}"
        )
    resolved = (
        edge.num_nodes
        if edge.num_nodes is not None
        else packet.num_nodes
    )
    if resolved is None:
        raise ValueError(
            f"code_graph_routes: relation {relation!r} has no num_nodes and the "
            f"GraphPacket carries none; cannot block-aggregate"
        )
    return int(resolved)


def _block_adjacency(
    packet: GraphPacket,
    relation: str,
    *,
    num_blocks: int,
    normalize: str,
) -> mx.array:
    """Return the normalised ``(num_blocks, num_blocks)`` adjacency for ``relation``.

    Aggregates the token/chunk-level edges to exactly ``num_blocks`` blocks by
    choosing ``block_size = ceil(num_nodes / num_blocks)`` and reusing
    :meth:`GraphPacket.block_aggregate` (the deterministic reference). The result
    is padded/validated to be exactly ``(num_blocks, num_blocks)``.
    """

    if normalize not in _NORMALIZE_MODES:
        raise ValueError(
            f"code_graph_routes: normalize must be one of {_NORMALIZE_MODES}, "
            f"got {normalize!r}"
        )
    num_nodes = _resolve_num_nodes(packet, relation)
    if num_blocks < 1:
        raise ValueError(f"code_graph_routes: num_blocks must be >=1, got {num_blocks}")
    # block_size chosen so that ceil(num_nodes / block_size) <= num_blocks.
    block_size = max(1, (num_nodes + num_blocks - 1) // num_blocks)
    counts = packet.block_aggregate(
        relation, block_size=block_size, num_nodes=num_nodes
    )  # (nb_raw, nb_raw) int32
    nb_raw = int(counts.shape[0])
    if nb_raw > num_blocks:
        raise ValueError(
            f"code_graph_routes[{relation}]: block_aggregate produced {nb_raw} "
            f"blocks > requested num_blocks={num_blocks} (num_nodes={num_nodes}, "
            f"block_size={block_size})"
        )
    counts_f = counts.astype(mx.float32)
    if normalize == "binary":
        counts_f = (counts_f > 0).astype(mx.float32)
    if nb_raw < num_blocks:
        pad = num_blocks - nb_raw
        counts_f = mx.pad(counts_f, [(0, pad), (0, pad)])
    return counts_f


@dataclass(frozen=True)
class GraphRouteConfig:
    """Configuration for the fixed graph routing prior."""

    num_blocks: int = 64
    relations: tuple[str, ...] = _DEFAULT_RELATIONS
    normalize: str = "binary"

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >=1, got {self.num_blocks}")
        if not self.relations:
            raise ValueError("relations must be a non-empty tuple")
        if self.normalize not in _NORMALIZE_MODES:
            raise ValueError(
                f"normalize must be one of {_NORMALIZE_MODES}, got {self.normalize!r}"
            )


class CodeGraphRouter(nn.Module):
    """Fixed-route, learned-channel-mixing graph router (TNO-style).

    The *routes* (block adjacencies ``A^r_blk``) are fixed graph data; the only
    parameters are the per-relation mixing scalars ``alpha_r``. Call the module
    with a :class:`GraphPacket` to obtain the additive block prior ``S_blk``.
    """

    def __init__(self, config: GraphRouteConfig | None = None):
        super().__init__()
        self.config = config or GraphRouteConfig()
        # One learnable channel-mixing scalar per relation (init 1.0). These are
        # the ONLY learnable params: the route itself is fixed graph structure.
        self.alpha = mx.ones((len(self.config.relations),), dtype=mx.float32)

    def relation_adjacencies(self, packet: GraphPacket) -> dict[str, mx.array]:
        """Return ``{relation: (B, B) normalised adjacency}`` for present relations."""

        out: dict[str, mx.array] = {}
        for relation in self.config.relations:
            if packet.edge(relation) is None:
                continue
            out[relation] = _block_adjacency(
                packet,
                relation,
                num_blocks=self.config.num_blocks,
                normalize=self.config.normalize,
            )
        if not out:
            raise KeyError(
                "CodeGraphRouter: GraphPacket carries none of the configured "
                f"relations {self.config.relations}; have {packet.relations}"
            )
        return out

    def __call__(self, packet: GraphPacket) -> mx.array:
        """Compute ``S_blk[t,s] = sum_r alpha_r * A^r_blk`` -> ``(B, B)`` fp32."""

        adjacencies = self.relation_adjacencies(packet)
        prior = mx.zeros(
            (self.config.num_blocks, self.config.num_blocks), dtype=mx.float32
        )
        for rel_idx, relation in enumerate(self.config.relations):
            adj = adjacencies.get(relation)
            if adj is None:
                continue
            prior = prior + self.alpha[rel_idx] * adj
        return prior


def build_attention_bias(
    packet: GraphPacket,
    *,
    router: CodeGraphRouter | None = None,
    config: GraphRouteConfig | None = None,
    alpha: Mapping[str, float] | None = None,
) -> mx.array:
    """Build the ``(B, B)`` block routing prior ``S_blk`` for one window.

    Either pass a constructed :class:`CodeGraphRouter` (its learned ``alpha`` is
    used), or pass ``config`` (+ optional fixed ``alpha`` overrides) to build an
    ad-hoc prior — useful for tests / metrics where no module is in scope.

    Raises (RULE #1) when both ``router`` and ``config`` are supplied, or when
    neither is and there is no default-constructible config.
    """

    if router is not None and config is not None:
        raise ValueError(
            "build_attention_bias: pass exactly one of router / config, not both"
        )
    if router is not None:
        if alpha is not None:
            raise ValueError(
                "build_attention_bias: alpha override is only valid with config, "
                "not with a constructed router (the router owns its alpha)"
            )
        return router(packet)

    cfg = config or GraphRouteConfig()
    adjacencies = {
        relation: _block_adjacency(
            packet, relation, num_blocks=cfg.num_blocks, normalize=cfg.normalize
        )
        for relation in cfg.relations
        if packet.edge(relation) is not None
    }
    if not adjacencies:
        raise KeyError(
            "build_attention_bias: GraphPacket carries none of the configured "
            f"relations {cfg.relations}; have {packet.relations}"
        )
    prior = mx.zeros((cfg.num_blocks, cfg.num_blocks), dtype=mx.float32)
    for relation, adj in adjacencies.items():
        weight = 1.0 if alpha is None else float(alpha.get(relation, 1.0))
        prior = prior + weight * adj
    return prior


def build_block_candidates(
    packet: GraphPacket,
    *,
    config: GraphRouteConfig | None = None,
    relations: Sequence[str] | None = None,
) -> list[list[int]]:
    """Return, per query block, the sorted list of true destination blocks.

    ``candidates[t]`` is the set of blocks ``s`` such that *some* configured
    relation has an edge from a node in block ``t`` to a node in block ``s``.
    This is the ground truth for the coverage hinge and recall@k.
    """

    cfg = config or GraphRouteConfig()
    rels = tuple(relations) if relations is not None else cfg.relations
    union = mx.zeros((cfg.num_blocks, cfg.num_blocks), dtype=mx.float32)
    seen = False
    for relation in rels:
        if packet.edge(relation) is None:
            continue
        adj = _block_adjacency(
            packet, relation, num_blocks=cfg.num_blocks, normalize="binary"
        )
        union = mx.maximum(union, adj)
        seen = True
    if not seen:
        raise KeyError(
            "build_block_candidates: GraphPacket carries none of the relations "
            f"{rels}; have {packet.relations}"
        )
    union_np = np.asarray(union)
    return [
        sorted(int(s) for s in np.nonzero(union_np[t])[0])
        for t in range(cfg.num_blocks)
    ]


def build_token_attention_bias(
    graph_batch: GraphBatch,
    *,
    batch_size: int,
    seq_length: int,
    relation_weights: Mapping[str, float] | None = None,
) -> mx.array:
    """Expand a typed graph batch into token-level ``(B,S,S)`` attention bias.

    ``call``/``type`` edges are chunk-index pairs and are expanded through each
    row's ``chunk_starts``/``chunk_ends`` spans. Domain/build/shell/diagnostic
    edges, when present, are token-index pairs and are scattered directly. All
    declared edge endpoints are range-checked; invalid graph data raises instead
    of being clamped, dropped, or converted to a zero bias.
    """

    if not isinstance(graph_batch, GraphBatch):
        raise TypeError(
            "build_token_attention_bias: graph_batch must be a GraphBatch, "
            f"got {type(graph_batch).__name__}"
        )
    if batch_size <= 0 or seq_length <= 0:
        raise ValueError(
            "build_token_attention_bias: batch_size and seq_length must be positive, "
            f"got {batch_size}/{seq_length}"
        )
    if graph_batch.batch_size not in (1, batch_size):
        raise ValueError(
            "build_token_attention_bias: graph batch size must be 1 or "
            f"{batch_size}, got {graph_batch.batch_size}"
        )
    weights = dict(_DEFAULT_TOKEN_RELATION_WEIGHTS)
    if relation_weights is not None:
        unknown = sorted(set(relation_weights) - set(weights))
        if unknown:
            raise KeyError(
                f"build_token_attention_bias: unsupported relation weights {unknown}"
            )
        weights.update({key: float(value) for key, value in relation_weights.items()})
    for relation, weight in weights.items():
        if not math.isfinite(weight):
            raise ValueError(
                f"build_token_attention_bias: {relation} weight must be finite, "
                f"got {weight}"
            )

    rows: list[mx.array] = []
    supported_relations = set(weights)
    for source_row in range(graph_batch.batch_size):
        graph = graph_batch.graphs[source_row]
        if not graph.relations:
            raise KeyError(
                "build_token_attention_bias: graph row "
                f"{source_row} has no route relations"
            )
        unknown_relations = sorted(set(graph.relations) - supported_relations)
        if unknown_relations:
            raise KeyError(
                "build_token_attention_bias: graph contains unsupported relations "
                f"{unknown_relations}"
            )
        starts = graph_batch.chunk_starts[source_row].astype(mx.int32)
        ends = graph_batch.chunk_ends[source_row].astype(mx.int32)
        invalid_end = mx.any(ends > seq_length)
        mx.eval(invalid_end)
        if bool(invalid_end.item()):
            values = np.asarray(ends)
            bad = values[values > seq_length][:8].tolist()
            raise ValueError(
                "build_token_attention_bias: graph chunk ends exceed sequence "
                f"length {seq_length}: {bad}"
            )
        positions = mx.arange(seq_length, dtype=mx.int32)
        membership = (
            (positions[:, None] >= starts[None, :])
            & (positions[:, None] < ends[None, :])
        ).astype(mx.float32)
        row_bias = mx.zeros((seq_length, seq_length), dtype=mx.float32)
        for relation in _DEFAULT_RELATIONS:
            row_bias = row_bias + _chunk_relation_bias(
                graph,
                membership,
                num_chunks=int(starts.shape[0]),
                relation=relation,
                weight=weights[relation],
            )
        for relation in _TOKEN_RELATIONS:
            row_bias = _add_token_relation(
                row_bias,
                graph,
                relation=relation,
                seq_length=seq_length,
                weight=weights[relation],
            )
        rows.append(row_bias)

    stacked = mx.stack(rows, axis=0)
    if graph_batch.batch_size == 1 and batch_size > 1:
        return mx.broadcast_to(stacked, (batch_size, seq_length, seq_length))
    return stacked


def _active_edge_vectors(
    graph: GraphPacket,
    *,
    relation: str,
    upper_bound: int,
    coordinate_name: str,
) -> tuple[mx.array, mx.array] | None:
    edge = graph.edge(relation)
    if edge is None:
        return None
    src = edge.src
    dst = edge.dst
    if edge.mask is not None:
        active = edge.mask > 0
        src = src[active]
        dst = dst[active]
    invalid = mx.any((src < 0) | (dst < 0) | (src >= upper_bound) | (dst >= upper_bound))
    mx.eval(invalid)
    if bool(invalid.item()):
        src_values = np.asarray(src)
        dst_values = np.asarray(dst)
        invalid_values = (
            (src_values < 0)
            | (dst_values < 0)
            | (src_values >= upper_bound)
            | (dst_values >= upper_bound)
        )
        first = int(np.flatnonzero(invalid_values)[0])
        raise ValueError(
            f"build_token_attention_bias: graph {relation} edge "
            f"({int(src_values[first])},{int(dst_values[first])}) out of range for "
            f"{upper_bound} {coordinate_name}"
        )
    return src, dst


def _chunk_relation_bias(
    graph: GraphPacket,
    membership: mx.array,
    *,
    num_chunks: int,
    relation: str,
    weight: float,
) -> mx.array:
    seq_length = int(membership.shape[0])
    vectors = _active_edge_vectors(
        graph,
        relation=relation,
        upper_bound=num_chunks,
        coordinate_name="chunks",
    )
    if vectors is None or weight == 0.0 or num_chunks == 0:
        return mx.zeros((seq_length, seq_length), dtype=mx.float32)
    src, dst = vectors
    adjacency = mx.zeros((num_chunks, num_chunks), dtype=mx.float32)
    adjacency = adjacency.at[src, dst].add(
        mx.full(src.shape, weight, dtype=mx.float32)
    )
    return membership @ adjacency @ membership.T


def _add_token_relation(
    bias: mx.array,
    graph: GraphPacket,
    *,
    relation: str,
    seq_length: int,
    weight: float,
) -> mx.array:
    vectors = _active_edge_vectors(
        graph,
        relation=relation,
        upper_bound=seq_length,
        coordinate_name="tokens",
    )
    if vectors is None or weight == 0.0:
        return bias
    src, dst = vectors
    return bias.at[src, dst].add(mx.full(src.shape, weight, dtype=mx.float32))


__all__ = [
    "GraphRouteConfig",
    "CodeGraphRouter",
    "build_attention_bias",
    "build_block_candidates",
    "build_token_attention_bias",
]
