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
from typing import Any

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
            arr = np.asarray(pairs.astype(mx.int32))
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


__all__ = ["EdgeIndex", "GraphPacket"]
