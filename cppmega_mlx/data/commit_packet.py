"""Typed batch contract for commit/diff (temporal) training elements.

``CommitPacket`` is the temporal-diff counterpart to ``CodePacket``.  It carries
the pre-state and post-state token fields of an edit plus the diff-level
side-channels produced by the v12/packed temporal columns
(``token_change_mask_pre/post``, ``hunk_id_per_token``, ``edit_op_per_token``,
``changed_chunk_ids``, ``changed_chunk_spans``) and optional pre/post
``GraphPacket`` snapshots.

All array fields are ``mx.array`` (or ``None`` when absent).  Absent optional
fields are ``None`` — never fabricated.

RULE #1: ``__post_init__`` validates token-aligned channel lengths against the
relevant (pre vs post) token field and RAISES with WHERE + WHAT on mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from cppmega_mlx.data.graph_packet import GraphPacket


def _check_token_aligned(
    name: str,
    value: mx.array | None,
    reference: mx.array | None,
    *,
    reference_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, mx.array):
        raise TypeError(
            f"CommitPacket.{name} must be an mx.array or None, got "
            f"{type(value).__name__}"
        )
    if reference is None:
        raise ValueError(
            f"CommitPacket.{name} present but its reference token field "
            f"{reference_name!r} is None; cannot validate alignment"
        )
    if value.ndim != reference.ndim:
        raise ValueError(
            f"CommitPacket.{name}: ndim {value.ndim} must match {reference_name} "
            f"ndim {reference.ndim}"
        )
    axis = reference.ndim - 1
    if int(value.shape[axis]) != int(reference.shape[axis]):
        raise ValueError(
            f"CommitPacket.{name}: token-aligned length {int(value.shape[axis])} != "
            f"{reference_name} length {int(reference.shape[axis])} "
            f"({name} shape {tuple(value.shape)}, {reference_name} shape "
            f"{tuple(reference.shape)})"
        )


@dataclass(frozen=True)
class CommitPacket:
    """A typed, validated commit/diff batch element.

    The ``*_pre`` channels align to ``pre_token_ids``; the ``*_post`` channels and
    the token-level diff side-channels (hunk/edit-op) align to ``post_token_ids``.
    ``diff_token_ids`` is an independent representation of the unified diff and is
    self-aligned only.
    """

    pre_token_ids: mx.array | None = None
    post_token_ids: mx.array | None = None
    diff_token_ids: mx.array | None = None
    commit_msg: mx.array | None = None

    # Token-level diff side-channels.
    change_mask_pre: mx.array | None = None
    change_mask_post: mx.array | None = None
    hunk_ids: mx.array | None = None
    edit_ops: mx.array | None = None

    # Changed-chunk metadata.
    changed_chunk_ids: mx.array | None = None
    changed_chunk_spans: mx.array | None = None

    # Pre/post graph snapshots.
    pre_graph: GraphPacket | None = None
    post_graph: GraphPacket | None = None

    # Provenance.
    repo: str | None = None
    filepath: str | None = None
    commit_or_ref: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pre_token_ids", "post_token_ids", "diff_token_ids", "commit_msg"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, mx.array):
                raise TypeError(
                    f"CommitPacket.{name} must be an mx.array or None, got "
                    f"{type(value).__name__}"
                )

        _check_token_aligned(
            "change_mask_pre", self.change_mask_pre, self.pre_token_ids,
            reference_name="pre_token_ids",
        )
        _check_token_aligned(
            "change_mask_post", self.change_mask_post, self.post_token_ids,
            reference_name="post_token_ids",
        )
        _check_token_aligned(
            "hunk_ids", self.hunk_ids, self.post_token_ids,
            reference_name="post_token_ids",
        )
        _check_token_aligned(
            "edit_ops", self.edit_ops, self.post_token_ids,
            reference_name="post_token_ids",
        )

        if (self.changed_chunk_ids is None) != (self.changed_chunk_spans is None):
            raise ValueError(
                "CommitPacket.changed_chunk_ids and changed_chunk_spans must both be "
                "present or both absent"
            )
        if self.changed_chunk_ids is not None:
            ids = self.changed_chunk_ids
            spans = self.changed_chunk_spans
            if not isinstance(ids, mx.array) or not isinstance(spans, mx.array):
                raise TypeError(
                    "CommitPacket.changed_chunk_ids/changed_chunk_spans must be mx.array"
                )
            n_ids = int(ids.shape[0])
            n_spans = int(spans.shape[0])
            if n_ids != n_spans:
                raise ValueError(
                    f"CommitPacket.changed_chunk_ids count {n_ids} != changed_chunk_spans "
                    f"count {n_spans}"
                )
            if spans.ndim != 2 or int(spans.shape[1]) != 2:
                raise ValueError(
                    f"CommitPacket.changed_chunk_spans must be (N, 2), got "
                    f"{tuple(spans.shape)}"
                )

        for name in ("pre_graph", "post_graph"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, GraphPacket):
                raise TypeError(
                    f"CommitPacket.{name} must be a GraphPacket or None, got "
                    f"{type(value).__name__}"
                )

        for name in ("repo", "filepath", "commit_or_ref"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"CommitPacket.{name} must be a str or None, got "
                    f"{type(value).__name__}"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"CommitPacket.metadata must be a Mapping, got "
                f"{type(self.metadata).__name__}"
            )

    def present_fields(self) -> tuple[str, ...]:
        candidates = (
            "pre_token_ids",
            "post_token_ids",
            "diff_token_ids",
            "commit_msg",
            "change_mask_pre",
            "change_mask_post",
            "hunk_ids",
            "edit_ops",
            "changed_chunk_ids",
            "changed_chunk_spans",
            "pre_graph",
            "post_graph",
        )
        return tuple(name for name in candidates if getattr(self, name) is not None)


__all__ = ["CommitPacket"]
