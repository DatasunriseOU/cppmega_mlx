"""Typed graph-edge containers for code/commit packets.

Edges in the v12/packed parquet schema (``token_call_edges``, ``token_type_edges``)
are stored as window-local *chunk* index pairs ``(from_chunk, to_chunk)`` produced
by ``parquet_dataset._token_graph_edge_windows`` / ``_normalize_edge_pairs``.

This module gives downstream model/loss/indexer code a single typed object —
``EdgeIndex`` — instead of raw ragged lists or padded ``(B, E, 2)`` arrays whose
semantics the caller has to re-derive.  An ``EdgeIndex`` carries the source and
destination index vectors (each shape ``(E,)``) plus an optional presence mask;
``GraphPacket`` groups the per-relation edge indices that belong to one window.

RULE #1: every helper FAILS LOUD with WHERE + WHAT on shape/dtype violations.
No silent truncation, no fabricated edges, no degraded fallbacks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import numpy as np


def _as_int_vector(value: Any, *, where: str) -> mx.array:
    """Coerce ``value`` to a 1-D int32 mx.array, raising with WHERE on failure."""

    if isinstance(value, mx.array):
        arr = value
    elif isinstance(value, np.ndarray):
        arr = mx.array(value)
    elif isinstance(value, (list, tuple)):
        arr = mx.array(np.asarray(value, dtype=np.int32)) if len(value) else mx.array(
            np.zeros((0,), dtype=np.int32)
        )
    else:
        raise TypeError(
            f"EdgeIndex {where}: expected mx.array/np.ndarray/list, got {type(value).__name__}"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"EdgeIndex {where}: expected a 1-D vector, got shape {tuple(arr.shape)}"
        )
    return arr.astype(mx.int32)


@dataclass(frozen=True)
class EdgeIndex:
    """A typed directed edge index for one relation within one window.

    Attributes:
        src: ``(E,)`` int32 source node/chunk indices (window-local).
        dst: ``(E,)`` int32 destination node/chunk indices (window-local).
        mask: optional ``(E,)`` int32 presence mask (1=real edge, 0=padding).
            ``None`` means every listed edge is real.
        relation: relation label, e.g. ``"call"`` / ``"type"`` / ``"def_use"`` / ``"ast"``.
        num_nodes: optional node-count bound used by block-aggregation helpers.
    """

    src: mx.array
    dst: mx.array
    relation: str
    mask: mx.array | None = None
    num_nodes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, str) or not self.relation.strip():
            raise ValueError("EdgeIndex.relation must be a non-empty string")
        object.__setattr__(self, "src", _as_int_vector(self.src, where="src"))
        object.__setattr__(self, "dst", _as_int_vector(self.dst, where="dst"))
        if self.src.shape != self.dst.shape:
            raise ValueError(
                f"EdgeIndex[{self.relation}] src/dst length mismatch: "
                f"src={tuple(self.src.shape)} dst={tuple(self.dst.shape)}"
            )
        if self.mask is not None:
            mask = _as_int_vector(self.mask, where="mask")
            if mask.shape != self.src.shape:
                raise ValueError(
                    f"EdgeIndex[{self.relation}] mask length {tuple(mask.shape)} must "
                    f"match edge count {tuple(self.src.shape)}"
                )
            object.__setattr__(self, "mask", mask)
        if self.num_nodes is not None and int(self.num_nodes) < 0:
            raise ValueError(
                f"EdgeIndex[{self.relation}] num_nodes must be non-negative, "
                f"got {self.num_nodes}"
            )

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[Sequence[int]] | np.ndarray | mx.array,
        *,
        relation: str,
        num_nodes: int | None = None,
    ) -> "EdgeIndex":
        """Build an EdgeIndex from an ``(E, 2)`` list/array of ``(src, dst)`` pairs."""

        if isinstance(pairs, mx.array):
            arr = np.asarray(cast(mx.array, pairs).astype(mx.int32))
        elif isinstance(pairs, np.ndarray):
            arr = pairs
        else:
            arr = np.asarray(list(pairs), dtype=np.int32) if len(pairs) else np.zeros(
                (0, 2), dtype=np.int32
            )
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"EdgeIndex.from_pairs[{relation}]: expected (E, 2) pairs, got shape "
                f"{tuple(arr.shape)}"
            )
        return cls(
            src=mx.array(arr[:, 0].astype(np.int32)),
            dst=mx.array(arr[:, 1].astype(np.int32)),
            relation=relation,
            num_nodes=num_nodes,
        )

    @classmethod
    def from_padded(
        cls,
        values: np.ndarray | mx.array,
        mask: np.ndarray | mx.array | None,
        *,
        relation: str,
        num_nodes: int | None = None,
    ) -> "EdgeIndex":
        """Build an EdgeIndex from a padded ``(E, 2)`` value array + ``(E,)`` mask.

        This matches the ``(B, E, 2)`` / ``(B, E)`` layout that
        ``parquet_dataset._token_graph_edge_windows`` emits once a single batch
        row has been selected.
        """

        vals = np.asarray(values if not isinstance(values, mx.array) else np.asarray(values))
        if vals.ndim != 2 or vals.shape[1] != 2:
            raise ValueError(
                f"EdgeIndex.from_padded[{relation}]: expected (E, 2) values, got "
                f"{tuple(vals.shape)}"
            )
        mask_arr: np.ndarray | None
        if mask is None:
            mask_arr = None
        else:
            mask_arr = np.asarray(mask if not isinstance(mask, mx.array) else np.asarray(mask))
            if mask_arr.shape != (vals.shape[0],):
                raise ValueError(
                    f"EdgeIndex.from_padded[{relation}]: mask shape "
                    f"{tuple(mask_arr.shape)} must be ({vals.shape[0]},)"
                )
        return cls(
            src=mx.array(vals[:, 0].astype(np.int32)),
            dst=mx.array(vals[:, 1].astype(np.int32)),
            relation=relation,
            mask=None if mask_arr is None else mx.array(mask_arr.astype(np.int32)),
            num_nodes=num_nodes,
        )

    def to_pairs(self) -> list[tuple[int, int]]:
        """Return the real (mask>0) edges as a list of ``(src, dst)`` tuples."""

        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        if self.mask is not None:
            keep = np.asarray(self.mask) > 0
            src = src[keep]
            dst = dst[keep]
        return [(int(s), int(d)) for s, d in zip(src, dst)]


@dataclass(frozen=True)
class GraphPacket:
    """A typed bundle of per-relation EdgeIndex objects for one window.

    ``edges`` maps a relation name (``"call"``, ``"type"``, ``"def_use"``,
    ``"ast"``) to its EdgeIndex.  ``num_nodes`` is the shared chunk/node count
    used by block-aggregation (the v12 indexer aggregates over ``B=64`` blocks).
    """

    edges: Mapping[str, EdgeIndex] = field(default_factory=dict)
    num_nodes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.edges, Mapping):
            raise TypeError(
                f"GraphPacket.edges must be a mapping, got {type(self.edges).__name__}"
            )
        normalized: dict[str, EdgeIndex] = {}
        for relation, edge in self.edges.items():
            if not isinstance(edge, EdgeIndex):
                raise TypeError(
                    f"GraphPacket.edges[{relation!r}] must be an EdgeIndex, "
                    f"got {type(edge).__name__}"
                )
            if edge.relation != relation:
                raise ValueError(
                    f"GraphPacket.edges key {relation!r} disagrees with "
                    f"EdgeIndex.relation {edge.relation!r}"
                )
            normalized[relation] = edge
        object.__setattr__(self, "edges", normalized)
        if self.num_nodes is not None and int(self.num_nodes) < 0:
            raise ValueError(
                f"GraphPacket.num_nodes must be non-negative, got {self.num_nodes}"
            )

    @property
    def relations(self) -> tuple[str, ...]:
        return tuple(self.edges)

    def edge(self, relation: str) -> EdgeIndex | None:
        return self.edges.get(relation)

    def is_empty(self) -> bool:
        return all(edge.num_edges == 0 for edge in self.edges.values())

    def block_aggregate(
        self,
        relation: str,
        *,
        block_size: int = 64,
        num_nodes: int | None = None,
    ) -> mx.array:
        """Block-aggregation helper stub: dense block->block edge-count matrix.

        Maps each node index to its block ``node // block_size`` and accumulates
        a ``(num_blocks, num_blocks)`` int32 count of edges between blocks for the
        named relation.  Real edges only (mask>0).  This is the deterministic
        reference the v12 ``B=64`` indexer can validate its fused kernel against.

        Raises (RULE #1) when the node count cannot be resolved or an index is
        out of range — never clamps.
        """

        if block_size < 1:
            raise ValueError(f"block_aggregate block_size must be >=1, got {block_size}")
        edge = self.edges.get(relation)
        if edge is None:
            raise KeyError(
                f"GraphPacket.block_aggregate: relation {relation!r} not present; "
                f"have {self.relations}"
            )
        resolved_nodes = (
            num_nodes
            if num_nodes is not None
            else edge.num_nodes
            if edge.num_nodes is not None
            else self.num_nodes
        )
        pairs = edge.to_pairs()
        if resolved_nodes is None:
            resolved_nodes = (max((max(s, d) for s, d in pairs), default=-1) + 1)
        num_blocks = (int(resolved_nodes) + block_size - 1) // block_size
        result = np.zeros((max(num_blocks, 1), max(num_blocks, 1)), dtype=np.int32)
        for src, dst in pairs:
            if not (0 <= src < resolved_nodes) or not (0 <= dst < resolved_nodes):
                raise ValueError(
                    f"GraphPacket.block_aggregate[{relation}]: edge ({src},{dst}) "
                    f"out of range for num_nodes={resolved_nodes}"
                )
            result[src // block_size, dst // block_size] += 1
        return mx.array(result)


def _as_int_tuple(values: Sequence[Any], *, where: str) -> tuple[mx.array, ...]:
    out: list[mx.array] = []
    for index, value in enumerate(values):
        arr = _as_int_vector(value, where=f"{where}[{index}]")
        out.append(arr)
    return tuple(out)


@dataclass(frozen=True)
class GraphBatch:
    """A typed batch of graph packets plus per-row chunk spans.

    ``GraphPacket`` carries edges; ``GraphBatch`` carries the chunk layout needed
    by model/loss code to expand chunk-index call/type edges into token-level
    attention bias. Empty graphs are explicit and valid. Malformed spans fail
    closed here or at the sequence-length-aware builder.
    """

    graphs: tuple[GraphPacket, ...]
    chunk_starts: tuple[mx.array, ...]
    chunk_ends: tuple[mx.array, ...]
    chunk_kinds: tuple[mx.array, ...] = field(default_factory=tuple)
    chunk_dep_levels: tuple[mx.array, ...] = field(default_factory=tuple)
    edge_kinds: tuple[Mapping[str, mx.array], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.graphs:
            raise ValueError("GraphBatch.graphs must be non-empty")
        graphs = tuple(self.graphs)
        for index, graph in enumerate(graphs):
            if not isinstance(graph, GraphPacket):
                raise TypeError(
                    f"GraphBatch.graphs[{index}] must be a GraphPacket, "
                    f"got {type(graph).__name__}"
                )
        object.__setattr__(self, "graphs", graphs)

        batch_size = len(graphs)
        starts = _as_int_tuple(self.chunk_starts, where="chunk_starts")
        ends = _as_int_tuple(self.chunk_ends, where="chunk_ends")
        kinds = _as_int_tuple(self.chunk_kinds, where="chunk_kinds")
        dep_levels = _as_int_tuple(self.chunk_dep_levels, where="chunk_dep_levels")
        edge_kinds: tuple[dict[str, mx.array], ...] = tuple(
            {
                relation: _as_int_vector(
                    values,
                    where=f"edge_kinds[{row}][{relation!r}]",
                )
                for relation, values in row_kinds.items()
            }
            for row, row_kinds in enumerate(self.edge_kinds)
        )
        if len(starts) != batch_size or len(ends) != batch_size:
            raise ValueError(
                "GraphBatch chunk_starts/chunk_ends length must match graphs "
                f"({batch_size}); got {len(starts)}/{len(ends)}"
            )
        if kinds and len(kinds) != batch_size:
            raise ValueError(
                f"GraphBatch.chunk_kinds length {len(kinds)} must match graphs {batch_size}"
            )
        if dep_levels and len(dep_levels) != batch_size:
            raise ValueError(
                "GraphBatch.chunk_dep_levels length "
                f"{len(dep_levels)} must match graphs {batch_size}"
            )
        if edge_kinds and len(edge_kinds) != batch_size:
            raise ValueError(
                f"GraphBatch.edge_kinds length {len(edge_kinds)} must match "
                f"graphs {batch_size}"
            )
        for row, (row_starts, row_ends) in enumerate(zip(starts, ends)):
            if row_starts.shape != row_ends.shape:
                raise ValueError(
                    f"GraphBatch row {row} chunk_starts shape {tuple(row_starts.shape)} "
                    f"must match chunk_ends {tuple(row_ends.shape)}"
                )
            if kinds and kinds[row].shape != row_starts.shape:
                raise ValueError(
                    f"GraphBatch row {row} chunk_kinds shape {tuple(kinds[row].shape)} "
                    f"must match chunk_starts {tuple(row_starts.shape)}"
                )
            if dep_levels and dep_levels[row].shape != row_starts.shape:
                raise ValueError(
                    "GraphBatch row "
                    f"{row} chunk_dep_levels shape {tuple(dep_levels[row].shape)} "
                    f"must match chunk_starts {tuple(row_starts.shape)}"
                )
            invalid_span = mx.any((row_starts < 0) | (row_ends <= row_starts))
            mx.eval(invalid_span)
            if bool(invalid_span.item()):
                raise ValueError(
                    f"GraphBatch row {row} chunk spans must satisfy 0 <= start < end"
                )
            if int(row_starts.shape[0]) > 1:
                overlaps = mx.any(row_starts[1:] < row_ends[:-1])
                mx.eval(overlaps)
                if bool(overlaps.item()):
                    raise ValueError(
                        f"GraphBatch row {row} chunk spans overlap or are out of order"
                    )
            graph_nodes = graphs[row].num_nodes
            if graph_nodes is not None and int(graph_nodes) != int(row_starts.shape[0]):
                raise ValueError(
                    f"GraphBatch row {row} graph num_nodes {graph_nodes} must match "
                    f"chunk count {int(row_starts.shape[0])}"
                )
            if edge_kinds:
                unknown_relations = sorted(set(edge_kinds[row]) - set(graphs[row].edges))
                if unknown_relations:
                    raise ValueError(
                        f"GraphBatch row {row} edge_kinds contains relations without "
                        f"edges: {unknown_relations}"
                    )
                for relation, relation_kinds in edge_kinds[row].items():
                    edge_count = graphs[row].edges[relation].num_edges
                    if tuple(relation_kinds.shape) != (edge_count,):
                        raise ValueError(
                            f"GraphBatch row {row} edge_kinds[{relation!r}] edge "
                            f"count must be {edge_count}, got "
                            f"{tuple(relation_kinds.shape)}"
                        )
        object.__setattr__(self, "chunk_starts", starts)
        object.__setattr__(self, "chunk_ends", ends)
        object.__setattr__(self, "chunk_kinds", kinds)
        object.__setattr__(self, "chunk_dep_levels", dep_levels)
        object.__setattr__(self, "edge_kinds", edge_kinds)

    @property
    def batch_size(self) -> int:
        return len(self.graphs)

    def is_empty(self) -> bool:
        return all(graph.is_empty() for graph in self.graphs)

    def input_aligned(
        self,
        *,
        source_sequence_length: int,
        input_sequence_length: int,
    ) -> "GraphBatch":
        """Trim graph metadata when next-token training removes the last token.

        Every declared endpoint is validated against the unshifted source first.
        Edges that only touch the target-only suffix are then removed deliberately;
        malformed source edges are never hidden by the alignment.
        """

        source_length = int(source_sequence_length)
        input_length = int(input_sequence_length)
        if source_length <= 0 or input_length <= 0:
            raise ValueError(
                "GraphBatch.input_aligned sequence lengths must be positive, got "
                f"{source_length}/{input_length}"
            )
        if input_length > source_length:
            raise ValueError(
                "GraphBatch.input_aligned input length cannot exceed source length, "
                f"got {input_length}>{source_length}"
            )
        if input_length == source_length:
            return self

        graphs: list[GraphPacket] = []
        starts_out: list[mx.array] = []
        ends_out: list[mx.array] = []
        kinds_out: list[mx.array] = []
        dep_levels_out: list[mx.array] = []
        edge_kinds_out: list[dict[str, mx.array]] = []
        for row, graph in enumerate(self.graphs):
            starts = np.asarray(self.chunk_starts[row], dtype=np.int64)
            ends = np.asarray(self.chunk_ends[row], dtype=np.int64)
            if np.any(ends > source_length):
                bad = ends[ends > source_length][:8].tolist()
                raise ValueError(
                    f"GraphBatch row {row} chunk ends exceed source sequence "
                    f"length {source_length}: {bad}"
                )
            keep_chunks = int(np.count_nonzero(starts < input_length))
            starts_out.append(mx.array(starts[:keep_chunks], dtype=mx.int32))
            ends_out.append(
                mx.array(
                    np.minimum(ends[:keep_chunks], input_length),
                    dtype=mx.int32,
                )
            )
            if self.chunk_kinds:
                kinds_out.append(self.chunk_kinds[row][:keep_chunks])
            if self.chunk_dep_levels:
                dep_levels_out.append(self.chunk_dep_levels[row][:keep_chunks])

            aligned_edges: dict[str, EdgeIndex] = {}
            aligned_edge_kinds: dict[str, mx.array] = {}
            for relation, edge in graph.edges.items():
                if relation in ("call", "type"):
                    source_bound = int(starts.shape[0])
                    input_bound = keep_chunks
                    coordinate_name = "chunks"
                elif relation in (
                    "domain",
                    "build",
                    "shell",
                    "diagnostic",
                    "cross_domain",
                ):
                    source_bound = source_length
                    input_bound = input_length
                    coordinate_name = "tokens"
                else:
                    raise KeyError(
                        "GraphBatch.input_aligned cannot align unsupported relation "
                        f"{relation!r}"
                    )
                aligned_edge, retained_indices = _input_aligned_edge(
                    edge,
                    source_bound=source_bound,
                    input_bound=input_bound,
                    coordinate_name=coordinate_name,
                )
                aligned_edges[relation] = aligned_edge
                if self.edge_kinds and relation in self.edge_kinds[row]:
                    relation_kinds = np.asarray(
                        self.edge_kinds[row][relation], dtype=np.int32
                    )
                    aligned_edge_kinds[relation] = mx.array(
                        relation_kinds[retained_indices], dtype=mx.int32
                    )
            graphs.append(GraphPacket(edges=aligned_edges, num_nodes=keep_chunks))
            edge_kinds_out.append(aligned_edge_kinds)

        return GraphBatch(
            graphs=tuple(graphs),
            chunk_starts=tuple(starts_out),
            chunk_ends=tuple(ends_out),
            chunk_kinds=tuple(kinds_out),
            chunk_dep_levels=tuple(dep_levels_out),
            edge_kinds=tuple(edge_kinds_out) if self.edge_kinds else (),
        )


def _input_aligned_edge(
    edge: EdgeIndex,
    *,
    source_bound: int,
    input_bound: int,
    coordinate_name: str,
) -> tuple[EdgeIndex, np.ndarray]:
    src = np.asarray(edge.src, dtype=np.int64)
    dst = np.asarray(edge.dst, dtype=np.int64)
    retained_indices = np.arange(src.shape[0], dtype=np.int64)
    if edge.mask is not None:
        active = np.asarray(edge.mask) > 0
        src = src[active]
        dst = dst[active]
        retained_indices = retained_indices[active]
    invalid = (src < 0) | (dst < 0) | (src >= source_bound) | (dst >= source_bound)
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"GraphBatch.input_aligned {edge.relation} edge "
            f"({int(src[first])},{int(dst[first])}) out of range for "
            f"{source_bound} source {coordinate_name}"
        )
    keep = (src < input_bound) & (dst < input_bound)
    return (
        EdgeIndex(
            src=mx.array(src[keep], dtype=mx.int32),
            dst=mx.array(dst[keep], dtype=mx.int32),
            relation=edge.relation,
            num_nodes=input_bound,
        ),
        retained_indices[keep],
    )


__all__ = ["EdgeIndex", "GraphBatch", "GraphPacket"]
