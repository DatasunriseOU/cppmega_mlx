"""Typed trajectory contract for the code-edit world model.

A :class:`TrajectoryPacket` is a temporally-ordered sequence of
:class:`Transition` steps.  Each transition carries a *current observation*
(``obs``) and the *next observation* (``next_obs``) the world model must
predict, plus the token-aligned edit-supervision channels that describe WHICH
tokens changed (``change_mask`` from ``token_change_mask_post``,
``edit_ops`` from ``edit_op_per_token``, ``hunk_ids`` from
``hunk_id_per_token``).

The trajectory is the temporal counterpart of :class:`CommitPacket`: one real
commit transition turns into one :class:`Transition` whose ``obs`` is the
pre-edit token prefix and whose ``next_obs`` is the post-edit token region.

RULE #1 (fail fast / fail loud, NO fabricated labels):

* Real commit transitions have ``reward = None`` and ``done = None`` — they are
  NEVER fabricated.  Reward / done are populated ONLY when a caller explicitly
  supplies a (clearly synthetic) label via :meth:`with_synthetic_control`.
* ``__post_init__`` validates every token-aligned channel length against the
  relevant observation and RAISES with WHERE + WHAT on any mismatch.  No silent
  truncation, no padding-to-fit.

Construction from the REAL ``tests/fixtures/golden_mini/commits`` fixture goes
through :func:`load_golden_mini_transitions`, which pairs the post-edit "chain"
documents (``token_change_mask_post`` has changed tokens) into honest
``(pre, post)`` next-observation transitions.  Documents whose post-change mask
is empty are NOT turned into transitions (they carry no next-observation
signal) — we never invent a transition where the data shows none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# Edit-op codes (mirrors tools/clang_indexer/process_commits.py).
EDIT_OP_UNCHANGED = 0
EDIT_OP_INSERTED = 1
EDIT_OP_MODIFIED = 2
EDIT_OP_CONTEXT = 3


def _as_token_vector(value: Any, *, where: str) -> mx.array:
    """Coerce ``value`` to a 1-D mx.array, raising with WHERE on failure."""

    if isinstance(value, mx.array):
        arr = value
    elif isinstance(value, np.ndarray):
        arr = mx.array(value)
    elif isinstance(value, (list, tuple)):
        arr = mx.array(np.asarray(value))
    else:
        raise TypeError(
            f"{where}: expected mx.array/np.ndarray/list, got {type(value).__name__}"
        )
    if arr.ndim != 1:
        raise ValueError(f"{where}: expected a 1-D (S,) vector, got shape {tuple(arr.shape)}")
    return arr


def _check_aligned(name: str, value: mx.array | None, ref: mx.array, *, ref_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, mx.array):
        raise TypeError(
            f"Transition.{name} must be an mx.array or None, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise ValueError(
            f"Transition.{name} must be 1-D (S,), got shape {tuple(value.shape)}"
        )
    if int(value.shape[0]) != int(ref.shape[0]):
        raise ValueError(
            f"Transition.{name}: token-aligned length {int(value.shape[0])} != "
            f"{ref_name} length {int(ref.shape[0])}"
        )


@dataclass(frozen=True)
class Transition:
    """One world-model step: ``obs`` -> ``next_obs`` with edit supervision.

    ``obs`` / ``next_obs`` are token-id vectors ``(S_obs,)`` / ``(S_next,)``.
    ``change_mask`` / ``edit_ops`` / ``hunk_ids`` align to ``next_obs`` (they
    describe which of the next-observation tokens were inserted/modified by the
    edit).  ``reward`` / ``done`` are ``None`` for real transitions and are only
    set for explicitly-synthetic control labels.
    """

    obs: mx.array
    next_obs: mx.array
    change_mask: mx.array | None = None
    edit_ops: mx.array | None = None
    hunk_ids: mx.array | None = None
    reward: float | None = None
    done: bool | None = None
    is_synthetic: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("obs", "next_obs"):
            value = getattr(self, name)
            if not isinstance(value, mx.array):
                raise TypeError(
                    f"Transition.{name} must be an mx.array, got {type(value).__name__}"
                )
            if value.ndim != 1:
                raise ValueError(
                    f"Transition.{name} must be 1-D (S,), got shape {tuple(value.shape)}"
                )
            if int(value.shape[0]) < 1:
                raise ValueError(f"Transition.{name} must be non-empty")

        _check_aligned("change_mask", self.change_mask, self.next_obs, ref_name="next_obs")
        _check_aligned("edit_ops", self.edit_ops, self.next_obs, ref_name="next_obs")
        _check_aligned("hunk_ids", self.hunk_ids, self.next_obs, ref_name="next_obs")

        # RULE #1: reward/done are labels. They are present ONLY on transitions
        # explicitly flagged synthetic. A real transition carrying a reward/done
        # would be a fabricated label -> RAISE.
        has_label = self.reward is not None or self.done is not None
        if has_label and not self.is_synthetic:
            raise ValueError(
                "Transition carries reward/done but is_synthetic=False; real "
                "transitions must have reward=None and done=None (no fabricated "
                "labels). Use with_synthetic_control() to attach a synthetic label."
            )
        if self.reward is not None and not isinstance(self.reward, (int, float)):
            raise TypeError(
                f"Transition.reward must be a number or None, got {type(self.reward).__name__}"
            )
        if self.done is not None and not isinstance(self.done, bool):
            raise TypeError(
                f"Transition.done must be a bool or None, got {type(self.done).__name__}"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"Transition.metadata must be a Mapping, got {type(self.metadata).__name__}"
            )

    def with_synthetic_control(self, *, reward: float, done: bool) -> "Transition":
        """Return a copy carrying an EXPLICITLY-synthetic reward/done label.

        This is the only sanctioned way to attach control labels — they are
        flagged ``is_synthetic=True`` so the loss code can route them to the
        reward/done heads while real transitions stay label-free.
        """

        meta = dict(self.metadata)
        meta["synthetic_control"] = True
        return Transition(
            obs=self.obs,
            next_obs=self.next_obs,
            change_mask=self.change_mask,
            edit_ops=self.edit_ops,
            hunk_ids=self.hunk_ids,
            reward=float(reward),
            done=bool(done),
            is_synthetic=True,
            metadata=meta,
        )


@dataclass(frozen=True)
class TrajectoryPacket:
    """An ordered sequence of :class:`Transition` steps for the world model."""

    transitions: Sequence[Transition]
    repo: str | None = None
    filepath: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        steps = list(self.transitions)
        if not steps:
            raise ValueError("TrajectoryPacket must contain at least one Transition")
        for i, step in enumerate(steps):
            if not isinstance(step, Transition):
                raise TypeError(
                    f"TrajectoryPacket.transitions[{i}] must be a Transition, got "
                    f"{type(step).__name__}"
                )
        object.__setattr__(self, "transitions", tuple(steps))
        for name in ("repo", "filepath"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"TrajectoryPacket.{name} must be a str or None, got "
                    f"{type(value).__name__}"
                )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"TrajectoryPacket.metadata must be a Mapping, got "
                f"{type(self.metadata).__name__}"
            )

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def horizon(self) -> int:
        """Number of transition steps (the rollout horizon)."""
        return len(self.transitions)

    def has_real_transitions(self) -> bool:
        return any(not t.is_synthetic for t in self.transitions)


# --------------------------------------------------------------------------- #
# Real golden-mini loading
# --------------------------------------------------------------------------- #
def _pre_post_split(
    token_ids: np.ndarray,
    change_mask_post: np.ndarray,
    edit_ops: np.ndarray,
    hunk_ids: np.ndarray,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Split one post-edit document into a (pre-context, post-region) transition.

    The post-change tokens (``change_mask_post == 1``) are the inserted/modified
    region.  Everything strictly before the FIRST changed token is the pre-edit
    *context* observation; the changed region (from the first changed token to
    the end of the final contiguous changed span) is the *next* observation the
    world model must predict.  This is a real, label-free transition extracted
    directly from the commit's edit signal — nothing is fabricated.
    """

    changed = np.nonzero(change_mask_post.astype(np.int64))[0]
    if changed.size == 0:
        raise ValueError("_pre_post_split called on a document with no post-change tokens")
    first = int(changed[0])
    last = int(changed[-1])
    if first < 1:
        # No pre-context tokens before the edit -> not a usable transition.
        raise ValueError(
            f"post-change region starts at token {first}; need >=1 pre-context token"
        )
    pre = token_ids[:first]
    post = token_ids[first : last + 1]
    post_mask = change_mask_post[first : last + 1]
    post_ops = edit_ops[first : last + 1]
    post_hunks = hunk_ids[first : last + 1]
    return (
        mx.array(pre.astype(np.int32)),
        mx.array(post.astype(np.int32)),
        mx.array(post_mask.astype(np.int32)),
        mx.array(post_ops.astype(np.int32)),
        mx.array(post_hunks.astype(np.int32)),
    )


def load_golden_mini_transitions(
    commits_parquet: str | Path,
) -> list[TrajectoryPacket]:
    """Load REAL commit transitions from the golden-mini commits fixture.

    Each row whose ``token_change_mask_post`` has at least one changed token is
    turned into a one-step :class:`TrajectoryPacket` (a ``pre -> post``
    transition).  Rows whose post-change mask is empty are skipped — they carry
    no next-observation signal, and inventing a transition for them would
    fabricate data (RULE #1).

    Reward / done are left ``None`` (real transitions are unlabeled).
    """

    import pyarrow.parquet as pq  # local import keeps pyarrow optional at import time

    path = Path(commits_parquet)
    if not path.exists():
        raise FileNotFoundError(f"golden-mini commits parquet not found at {path}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"golden-mini commits parquet at {path} is empty")

    required = ("token_ids", "token_change_mask_post", "edit_op_per_token", "hunk_id_per_token")
    trajectories: list[TrajectoryPacket] = []
    for idx, row in enumerate(rows):
        for col in required:
            if col not in row:
                raise KeyError(
                    f"golden-mini row {idx} missing required column {col!r}; "
                    f"present columns: {sorted(row.keys())[:8]}..."
                )
        token_ids = np.asarray(row["token_ids"], dtype=np.int64)
        change_mask_post = np.asarray(row["token_change_mask_post"], dtype=np.int64)
        edit_ops = np.asarray(row["edit_op_per_token"], dtype=np.int64)
        hunk_ids = np.asarray(row["hunk_id_per_token"], dtype=np.int64)
        n = token_ids.shape[0]
        for col, arr in (
            ("token_change_mask_post", change_mask_post),
            ("edit_op_per_token", edit_ops),
            ("hunk_id_per_token", hunk_ids),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"golden-mini row {idx}: {col} length {arr.shape[0]} != token_ids "
                    f"length {n}"
                )
        if int(change_mask_post.sum()) == 0:
            # No post-edit region in this document; not a transition.
            continue
        pre, post, post_mask, post_ops, post_hunks = _pre_post_split(
            token_ids, change_mask_post, edit_ops, hunk_ids
        )
        transition = Transition(
            obs=pre,
            next_obs=post,
            change_mask=post_mask,
            edit_ops=post_ops,
            hunk_ids=post_hunks,
            reward=None,
            done=None,
            is_synthetic=False,
            metadata={"row": idx},
        )
        trajectories.append(
            TrajectoryPacket(
                transitions=(transition,),
                repo=row.get("repo"),
                filepath=row.get("filepath"),
                metadata={
                    "commit_hash": row.get("commit_hash"),
                    "source": "golden_mini/commits",
                },
            )
        )

    if not trajectories:
        raise ValueError(
            f"golden-mini commits parquet at {path} produced no real transitions "
            "(no document had a non-empty post-change mask)"
        )
    return trajectories


__all__ = [
    "EDIT_OP_CONTEXT",
    "EDIT_OP_INSERTED",
    "EDIT_OP_MODIFIED",
    "EDIT_OP_UNCHANGED",
    "Transition",
    "TrajectoryPacket",
    "load_golden_mini_transitions",
]
