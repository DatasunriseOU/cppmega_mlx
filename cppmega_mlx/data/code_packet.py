"""Typed batch contract for code-token training/inference.

``CodePacket`` is the structured replacement for passing raw parquet/dict rows
into model, loss, and indexer code.  It carries:

  * core LM tensors  : token_ids (+ optional target_ids / loss_mask / document_ids)
  * provenance       : repo / filepath / commit_or_ref
  * objective source : typed IFIM instruction token ids
  * structure        : structure_ids / ast_depth / sibling_index / ast_node_type /
                       dep_levels  (token-aligned side channels)
  * token semantics  : symbol_ids / call_targets / type_refs / def_use
                       (token-aligned side channels)
  * graph edges      : call_edges / type_edges as typed ``EdgeIndex`` objects
  * chunk metadata   : chunk_starts / chunk_ends / chunk_kinds / chunk_dep_levels
  * metadata         : free-form Mapping for everything else

All array fields are ``mx.array`` (or ``None`` when the source column was absent).
Absent optional channels are ``None`` — NEVER fabricated.

RULE #1 (fail fast / fail loud): ``__post_init__`` validates that every present
token-aligned channel has the same length as ``token_ids`` and RAISES with
WHERE + WHAT on any mismatch.  No silent truncation, no padding-to-fit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket


# Token-aligned 1-D/2-D channels whose leading length must equal token_ids length.
_STRUCTURE_FIELDS = (
    "structure_ids",
    "ast_depth",
    "sibling_index",
    "ast_node_type",
    "dep_levels",
)
_SEMANTIC_FIELDS = (
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
)
_DOMAIN_TOKEN_FIELDS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "confidence_ids",
)
# Token-aligned channels that must match token_ids length exactly.
_TOKEN_ALIGNED_FIELDS = (
    "target_ids",
    "loss_mask",
    "document_ids",
    *_STRUCTURE_FIELDS,
    *_SEMANTIC_FIELDS,
    *_DOMAIN_TOKEN_FIELDS,
)
# Chunk-aligned channels (all four must agree with each other, not with tokens).
_CHUNK_FIELDS = (
    "chunk_starts",
    "chunk_ends",
    "chunk_kinds",
    "chunk_dep_levels",
)


def _token_len(value: mx.array) -> int:
    if value.ndim < 1:
        raise ValueError(f"expected >=1-D token-aligned array, got scalar {value.shape}")
    return int(value.shape[0])


@dataclass(frozen=True)
class CodePacket:
    """A typed, validated code-token batch element (single window or batch row).

    The leading dimension of ``token_ids`` is the token axis ``S`` (or ``(B, S)``
    if batched — see ``token_axis_len`` which always uses the *last* dim length
    for token alignment when ndim==2).  Every present token-aligned channel must
    share that token-axis length.
    """

    token_ids: mx.array
    target_ids: mx.array | None = None
    loss_mask: mx.array | None = None
    document_ids: mx.array | None = None

    # Provenance.
    repo: str | None = None
    filepath: str | None = None
    commit_or_ref: str | None = None

    # Non-token-aligned objective context from an authoritative source column.
    ifim_instruction_token_ids: mx.array | None = None

    # Structure side-channels (token-aligned).
    structure_ids: mx.array | None = None
    ast_depth: mx.array | None = None
    sibling_index: mx.array | None = None
    ast_node_type: mx.array | None = None
    dep_levels: mx.array | None = None

    # Token-symbol semantics (token-aligned).
    symbol_ids: mx.array | None = None
    call_targets: mx.array | None = None
    type_refs: mx.array | None = None
    def_use: mx.array | None = None

    # Domain-routing side-channels (token-aligned).
    domain_ids: mx.array | None = None
    role_ids: mx.array | None = None
    entity_ids: mx.array | None = None
    scope_ids: mx.array | None = None
    source_doc_ids: mx.array | None = None
    confidence_ids: mx.array | None = None

    # Graph edges (chunk-index space).
    call_edges: EdgeIndex | None = None
    type_edges: EdgeIndex | None = None

    # Domain graph edges (token-index space).
    domain_edges: DomainEdgeIndex | None = None
    build_edges: DomainEdgeIndex | None = None
    shell_edges: DomainEdgeIndex | None = None
    diagnostic_edges: DomainEdgeIndex | None = None
    cross_domain_edges: DomainEdgeIndex | None = None

    # Chunk metadata.
    chunk_starts: mx.array | None = None
    chunk_ends: mx.array | None = None
    chunk_kinds: mx.array | None = None
    chunk_dep_levels: mx.array | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, mx.array):
            raise TypeError(
                f"CodePacket.token_ids must be an mx.array, got "
                f"{type(self.token_ids).__name__}"
            )
        if self.token_ids.ndim not in (1, 2):
            raise ValueError(
                f"CodePacket.token_ids must be 1-D (S,) or 2-D (B, S), got shape "
                f"{tuple(self.token_ids.shape)}"
            )
        token_axis = self.token_ids.ndim - 1
        expected = int(self.token_ids.shape[token_axis])

        for name in _TOKEN_ALIGNED_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, mx.array):
                raise TypeError(
                    f"CodePacket.{name} must be an mx.array or None, got "
                    f"{type(value).__name__}"
                )
            if value.ndim != self.token_ids.ndim:
                raise ValueError(
                    f"CodePacket.{name}: token-aligned channel must have ndim "
                    f"{self.token_ids.ndim} matching token_ids, got shape "
                    f"{tuple(value.shape)}"
                )
            actual = int(value.shape[token_axis])
            if actual != expected:
                raise ValueError(
                    f"CodePacket.{name}: token-aligned length {actual} != token_ids "
                    f"length {expected} (token_ids shape {tuple(self.token_ids.shape)}, "
                    f"{name} shape {tuple(value.shape)})"
                )
            if self.token_ids.ndim == 2 and int(value.shape[0]) != int(
                self.token_ids.shape[0]
            ):
                raise ValueError(
                    f"CodePacket.{name}: batch dimension {int(value.shape[0])} != "
                    f"token_ids batch {int(self.token_ids.shape[0])}"
                )

        # Chunk-metadata channels must agree with each other (chunk axis, not tokens).
        present_chunks = {
            name: getattr(self, name)
            for name in _CHUNK_FIELDS
            if getattr(self, name) is not None
        }
        if present_chunks:
            lengths = {}
            for name, value in present_chunks.items():
                if not isinstance(value, mx.array):
                    raise TypeError(
                        f"CodePacket.{name} must be an mx.array or None, got "
                        f"{type(value).__name__}"
                    )
                lengths[name] = _token_len(value)
            unique = set(lengths.values())
            if len(unique) > 1:
                raise ValueError(
                    f"CodePacket chunk-metadata channels must share length; got "
                    f"{lengths}"
                )

        for name in ("call_edges", "type_edges"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, EdgeIndex):
                raise TypeError(
                    f"CodePacket.{name} must be an EdgeIndex or None, got "
                    f"{type(value).__name__}"
                )

        for name in (
            "domain_edges",
            "build_edges",
            "shell_edges",
            "diagnostic_edges",
            "cross_domain_edges",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, DomainEdgeIndex):
                raise TypeError(
                    f"CodePacket.{name} must be a DomainEdgeIndex or None, got "
                    f"{type(value).__name__}"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"CodePacket.metadata must be a Mapping, got "
                f"{type(self.metadata).__name__}"
            )

        for name in ("repo", "filepath", "commit_or_ref"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"CodePacket.{name} must be a str or None, got "
                    f"{type(value).__name__}"
                )

        instruction_ids = self.ifim_instruction_token_ids
        if instruction_ids is not None:
            if not isinstance(instruction_ids, mx.array):
                raise TypeError(
                    "CodePacket.ifim_instruction_token_ids must be an mx.array "
                    f"or None, got {type(instruction_ids).__name__}"
                )
            if instruction_ids.ndim != 1:
                raise ValueError(
                    "CodePacket.ifim_instruction_token_ids must be 1-D, got "
                    f"shape {tuple(instruction_ids.shape)}"
                )

    @property
    def token_axis_len(self) -> int:
        return int(self.token_ids.shape[-1])

    def structure_fields(self) -> dict[str, mx.array | None]:
        return {name: getattr(self, name) for name in _STRUCTURE_FIELDS}

    def semantic_fields(self) -> dict[str, mx.array | None]:
        return {name: getattr(self, name) for name in _SEMANTIC_FIELDS}

    def domain_token_fields(self) -> dict[str, mx.array | None]:
        return {name: getattr(self, name) for name in _DOMAIN_TOKEN_FIELDS}

    def present_fields(self) -> tuple[str, ...]:
        """Names of optional array/edge fields that are populated (not None)."""

        candidates = (
            "target_ids",
            "loss_mask",
            "document_ids",
            "ifim_instruction_token_ids",
            *_STRUCTURE_FIELDS,
            *_SEMANTIC_FIELDS,
            *_DOMAIN_TOKEN_FIELDS,
            "call_edges",
            "type_edges",
            "domain_edges",
            "build_edges",
            "shell_edges",
            "diagnostic_edges",
            "cross_domain_edges",
            *_CHUNK_FIELDS,
        )
        return tuple(name for name in candidates if getattr(self, name) is not None)

    def graph_packet(self) -> GraphPacket:
        """Bundle present call/type edges into a typed GraphPacket."""

        edges: dict[str, EdgeIndex] = {}
        if self.call_edges is not None:
            edges["call"] = self.call_edges
        if self.type_edges is not None:
            edges["type"] = self.type_edges
        num_nodes = None
        if self.chunk_starts is not None:
            num_nodes = int(self.chunk_starts.shape[0])
        return GraphPacket(edges=edges, num_nodes=num_nodes)


__all__ = ["CodePacket"]
