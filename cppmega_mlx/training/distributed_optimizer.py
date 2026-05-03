"""ZeRO-1 (optimizer state sharding) wrapper for cppmega.mlx training.

Wraps any cppmega.mlx optimizer
(:class:AdamWFP32Moments, :class:LionFP32Moments, :class:MuonAdamWMulti)
and shards optimizer state across mx.distributed ranks.

Status: scaffold + single-rank receipts. The 48 GB Stream F peer is not yet
connected, so this module ships with simulated multi-rank tests but no real
2-node receipt. See :mod:docs/distributed_zero1_smoke_procedure.md for the
hand-off procedure that will produce the multi-node receipt once peer-48 is
online.

Design (mirrors Megatron's DistributedOptimizer):

1. Forward / backward runs the full bf16 model on every rank (each rank holds
   the complete parameter tree); only the optimizer-state half is sharded.
2. apply_gradients:

   a. mx.distributed.all_sum reduces gradients across ranks
      (mean = sum / world_size).
   b. Each rank steps the inner optimizer **only on its parameter shard**
      (params[R::W] round-robin assignment).
   c. mx.distributed.all_gather reconstitutes the full updated parameter
      tree on every rank.
3. State per rank: full bf16 param tree (forward/backward needs them) +
   optimizer state (m, v, momentum) restricted to the rank's shard.

The wrapper deliberately falls back to a no-op (single-rank, behaves
identically to the inner optimizer) when world_size == 1 so existing
single-Mac training paths are untouched.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from cppmega_mlx.training.optimizers import (
    AdamWFP32Moments,
    LionFP32Moments,
    MuonAdamWMulti,
)


SUPPORTED_INNER_CLASSES: tuple[type, ...] = (
    AdamWFP32Moments,
    LionFP32Moments,
    MuonAdamWMulti,
)

ZERO1_STREAM_F_POLICY = (
    "ZeRO-1 wrapper is a scaffold + single-rank receipts; multi-node "
    "receipt pending peer-48 hardware connection per docs/multimac_training.md "
    "Phase 2."
)


def _flatten_param_tree(tree: Any) -> list[tuple[str, mx.array]]:
    """Return [(name, leaf), ...] for every :class:mx.array leaf, sorted
    by name so every rank produces the same ordering deterministically."""

    flat = tree_flatten(tree)
    return sorted(
        ((name, leaf) for name, leaf in flat if isinstance(leaf, mx.array)),
        key=lambda item: item[0],
    )


def _shard_assignment(num_leaves: int, world_size: int) -> list[int]:
    """Return [owner_rank for leaf_index in range(num_leaves)] using a
    round-robin policy (leaf_index % world_size).

    Round-robin balances opt-state bytes when leaf sizes are heterogeneous
    better than a contiguous slice. Megatron's CUDA DistributedOptimizer uses
    a bucketed contiguous slice for nccl efficiency; for the MLX scaffold,
    round-robin is simpler and gives ~equal coverage at small leaf counts.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    return [index % world_size for index in range(num_leaves)]


def _select_owned_subtree(
    tree: Any,
    rank: int,
    world_size: int,
) -> Any:
    """Filter tree to only the leaves owned by rank under round-robin
    sharding. Returns a tree with the same nested structure, restricted to the
    owned leaves only."""

    flat = _flatten_param_tree(tree)
    assignment = _shard_assignment(len(flat), world_size)
    owned_pairs = [
        (name, leaf)
        for index, (name, leaf) in enumerate(flat)
        if assignment[index] == rank
    ]
    return tree_unflatten(owned_pairs) if owned_pairs else {}


def _state_numel_bytes(state: Any) -> int:
    """Sum bytes of every :class:mx.array leaf in an optimizer state tree.

    Used by memory-budget tests to assert that ZeRO-1 at W=2 holds ~half the
    state bytes of W=1.
    """

    total = 0

    def walk(value: Any) -> None:
        nonlocal total
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                walk(item)
            return
        if isinstance(value, mx.array):
            total += value.nbytes

    walk(state)
    return total


class DistributedZeRO1Optimizer:
    """ZeRO-1 wrapper that shards optimizer state across mx.distributed ranks.

    The wrapper holds an inner optimizer and presents the standard MLX
    Optimizer surface (init, apply_gradients, update, state,
    learning_rate). When world_size > 1 the wrapper:

    1. Receives full gradients (one tree per rank from local autograd).
    2. mx.distributed.all_sum reduces gradients across ranks.
    3. Restricts the gradient + parameter trees to the shard owned by this
       rank and runs the inner optimizer on the shard only. The inner
       optimizer's state for non-owned leaves is never instantiated, so each
       rank carries only 1 / world_size of the optimizer-state bytes.
    4. mx.distributed.all_gather reconstitutes the full updated parameter
       tree so every rank sees the complete model after the step.

    When world_size == 1 (default, single-Mac training), the wrapper
    short-circuits to the inner optimizer with no collectives, preserving
    bit-for-bit numerics.

    Construct a wrapper with :func:make_distributed_optimizer for the most
    common path. Use __init__ directly when injecting an existing inner
    optimizer instance.

    Args:
        inner_optimizer: an :class:AdamWFP32Moments, :class:LionFP32Moments
            or :class:MuonAdamWMulti instance. Other optimizers are accepted
            but trigger a TypeError to keep the supported surface explicit.
        world_size: optional override for the rank count. Defaults to
            mx.distributed.init(strict=False).size().
        rank: optional override for this rank's index. Defaults to
            mx.distributed.init(strict=False).rank().
        group: optional :class:mx.distributed.Group to scope collectives to a
            sub-group. Defaults to the global group.
    """

    def __init__(
        self,
        inner_optimizer: Any,
        *,
        world_size: int | None = None,
        rank: int | None = None,
        group: Any = None,
    ) -> None:
        if not isinstance(inner_optimizer, SUPPORTED_INNER_CLASSES):
            allowed = ", ".join(cls.__name__ for cls in SUPPORTED_INNER_CLASSES)
            raise TypeError(
                f"inner_optimizer must be one of ({allowed}); "
                f"got {type(inner_optimizer).__name__}"
            )
        self._inner = inner_optimizer
        self._group = group

        # Resolve world_size / rank with mx.distributed if no overrides are
        # supplied. mx.distributed.init(strict=False) returns a singleton
        # (size=1, rank=0) when no backend is initialized, which is the
        # single-Mac fallback we want.
        if world_size is None or rank is None:
            try:
                resolved_group = group or mx.distributed.init(strict=False)
                resolved_size = int(resolved_group.size())
                resolved_rank = int(resolved_group.rank())
            except Exception:  # pragma: no cover - defensive
                resolved_size = 1
                resolved_rank = 0
            if world_size is None:
                world_size = resolved_size
            if rank is None:
                rank = resolved_rank

        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(
                f"rank must be in [0, world_size); got rank={rank}, "
                f"world_size={world_size}"
            )

        self._world_size = int(world_size)
        self._rank = int(rank)
        self._initialized = False
        self._owned_names: tuple[str, ...] = ()

    @property
    def inner(self) -> Any:
        """The wrapped optimizer instance."""

        return self._inner

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_sharded(self) -> bool:
        """True iff world_size > 1 and the wrapper actively shards state."""

        return self._world_size > 1

    @property
    def owned_param_names(self) -> tuple[str, ...]:
        """Sorted tuple of leaf names this rank holds optimizer state for.

        Empty before :meth:init runs. Useful for receipts and tests.
        """

        return self._owned_names

    def _select_owned(self, tree: Any) -> Any:
        return _select_owned_subtree(tree, self._rank, self._world_size)

    def init(self, parameters: Any) -> None:
        """Initialize the wrapped optimizer's state restricted to this rank's shard.

        Other ranks instantiate state for their own shards independently. After
        :meth:init returns, self.state reflects only the local shard
        (~1 / world_size of total optimizer-state bytes).
        """

        if self.is_sharded:
            owned = self._select_owned(parameters)
            self._inner.init(owned)
            self._owned_names = tuple(name for name, _ in _flatten_param_tree(owned))
        else:
            self._inner.init(parameters)
            self._owned_names = tuple(
                name for name, _ in _flatten_param_tree(parameters)
            )
        self._initialized = True

    def _all_reduce_mean(self, gradients: Any) -> Any:
        """Sum-reduce gradients across the group then divide by world_size to
        produce the mean gradient (matches DDP semantics)."""

        if not self.is_sharded:
            return gradients

        scale = 1.0 / float(self._world_size)
        group = self._group

        def reduce(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: reduce(item) for key, item in value.items()}
            if isinstance(value, list):
                return [reduce(item) for item in value]
            if isinstance(value, tuple):
                return tuple(reduce(item) for item in value)
            if isinstance(value, mx.array):
                summed = mx.distributed.all_sum(value, group=group)
                return summed * scale
            return value

        return reduce(gradients)

    def _gather_full_params(
        self,
        owned_updates: Any,
        full_parameters: Any,
    ) -> Any:
        """Reconstitute the full updated parameter tree from each rank's
        owned-shard update.

        We use mx.distributed.all_sum of a sparse tensor (rank-R contributes
        the actual update for the leaves it owns and zeros for everything else)
        rather than all_gather because round-robin sharding produces
        non-contiguous shapes per rank — all_gather requires uniform shape
        per rank and send/recv would need a hand-rolled scatter. The
        sum-of-sparse-tensors approach is one collective per leaf and produces
        the same final tree on every rank.

        For a contiguous-slice sharding policy the more efficient path would be
        all_gather on the leaves; the round-robin policy used here is
        simpler and the wrapper is a scaffold, not a perf path.
        """

        if not self.is_sharded:
            return owned_updates

        owned_lookup = {
            name: leaf for name, leaf in _flatten_param_tree(owned_updates)
        }
        full_pairs = _flatten_param_tree(full_parameters)
        assignment = _shard_assignment(len(full_pairs), self._world_size)
        group = self._group

        merged: list[tuple[str, mx.array]] = []
        for index, (name, current) in enumerate(full_pairs):
            owner = assignment[index]
            if owner == self._rank:
                contribution = owned_lookup[name].astype(current.dtype)
            else:
                contribution = mx.zeros(current.shape, dtype=current.dtype)
            gathered = mx.distributed.all_sum(contribution, group=group)
            merged.append((name, gathered))

        return tree_unflatten(merged)

    def apply_gradients(self, gradients: Any, parameters: Any) -> Any:
        """Reduce gradients across ranks, step the inner optimizer on the
        local shard, and reconstitute the full updated parameter tree.

        Returns a parameter tree with the same structure as parameters;
        every rank receives identical output (so it is safe to feed the result
        back into model.update).
        """

        if not self._initialized:
            self.init(parameters)

        reduced = self._all_reduce_mean(gradients)
        if not self.is_sharded:
            return self._inner.apply_gradients(reduced, parameters)

        owned_grads = self._select_owned(reduced)
        owned_params = self._select_owned(parameters)
        if not owned_grads:
            # Defensive: every rank must contribute the same number of leaves
            # to the all_sum gather; with round-robin sharding and num_leaves
            # >= world_size, this branch is unreachable. We keep the guard so
            # tiny test models with fewer leaves than ranks still work.
            owned_updates: Any = {}
        else:
            owned_updates = self._inner.apply_gradients(owned_grads, owned_params)
        return self._gather_full_params(owned_updates, parameters)

    def update(self, model: nn.Module, gradients: Any) -> None:
        """Convenience method matching :class:mlx.optimizers.Optimizer."""

        model.update(self.apply_gradients(gradients, model))

    @property
    def state(self) -> Any:
        """The wrapped optimizer's state restricted to the local shard."""

        return self._inner.state

    @state.setter
    def state(self, state: Any) -> None:
        self._inner.state = state
        self._initialized = False  # caller must re-init for owned_names to refresh.

    @property
    def learning_rate(self) -> mx.array:
        return self._inner.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float | mx.array) -> None:
        self._inner.learning_rate = learning_rate

    def shard_owned_state_bytes(self) -> int:
        """Bytes held by this rank's owned optimizer state. Useful for memory
        receipts and the test_zero1_memory_smaller_than_full_replicated
        regression test."""

        return _state_numel_bytes(self._inner.state)


def make_distributed_optimizer(
    inner_optimizer: Any,
    *,
    world_size: int | None = None,
    rank: int | None = None,
    group: Any = None,
) -> DistributedZeRO1Optimizer:
    """Wrap inner_optimizer in a :class:DistributedZeRO1Optimizer.

    Convenience factory that mirrors the make_* style used by the rest of
    :mod:cppmega_mlx.training.optimizers. When world_size is omitted the
    wrapper auto-detects via mx.distributed.init(strict=False) — that
    returns a singleton group on a single Mac, which makes the wrapper a no-op
    by default.
    """

    return DistributedZeRO1Optimizer(
        inner_optimizer,
        world_size=world_size,
        rank=rank,
        group=group,
    )


__all__ = [
    "SUPPORTED_INNER_CLASSES",
    "ZERO1_STREAM_F_POLICY",
    "DistributedZeRO1Optimizer",
    "make_distributed_optimizer",
]
