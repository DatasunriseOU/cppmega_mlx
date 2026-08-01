"""MLX-local paged KV scheduling primitives for bounded inference serving.

This module ports the scheduler-facing subset of nanochat's paged KV serving
surface. It owns MLX block pools, sequence block tables, and continuous-batch
admission/preemption metadata. The model attention path can opt into the
*paged KV compatibility path* via :func:`gather_paged_kv` and
:func:`scatter_paged_kv`, which round-trip paged blocks through contiguous
K/V tensors so that ``mx.fast.scaled_dot_product_attention`` can be reused
until a native paged attention kernel is wired and tested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Sequence
import warnings

import mlx.core as mx
import numpy as np

RequestState = Literal["waiting", "running", "preempted", "completed"]

PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE = (
    "MLX paged KV scheduling is available, but model-integrated paged "
    "attention is not wired yet; use contiguous KV inference or keep this "
    "path fail-closed."
)


@dataclass(frozen=True)
class PagedKVBlockManagerConfig:
    """Shape contract for the MLX paged KV block pool."""

    num_blocks: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: mx.Dtype = mx.float32

    def __post_init__(self) -> None:
        _validate_positive_int("num_blocks", self.num_blocks)
        _validate_positive_int("block_size", self.block_size)
        _validate_positive_int("num_layers", self.num_layers)
        _validate_positive_int("num_kv_heads", self.num_kv_heads)
        _validate_positive_int("head_dim", self.head_dim)


class PagedKVBlockManager:
    """Fixed-size MLX KV block pool with per-sequence allocation metadata."""

    def __init__(
        self,
        config: PagedKVBlockManagerConfig | None = None,
        *,
        num_blocks: int | None = None,
        block_size: int | None = None,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        self.config = _resolve_block_manager_config(
            config=config,
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.num_blocks = self.config.num_blocks
        self.block_size = self.config.block_size
        self.num_layers = self.config.num_layers
        self.num_kv_heads = self.config.num_kv_heads
        self.head_dim = self.config.head_dim

        pool_shape = (
            self.num_blocks,
            self.num_layers,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        self.k_pool = mx.zeros(pool_shape, dtype=self.config.dtype)
        self.v_pool = mx.zeros(pool_shape, dtype=self.config.dtype)
        self._free_blocks = list(range(self.num_blocks))
        self._allocated_blocks: set[int] = set()
        self._seq_blocks: dict[int, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._allocated_blocks)

    def allocate_block(self) -> int:
        """Allocate one physical block, raising if the pool is exhausted."""

        if not self._free_blocks:
            raise RuntimeError(
                f"PagedKVBlockManager: all {self.num_blocks} blocks are allocated"
            )
        block_idx = self._free_blocks.pop()
        self._allocated_blocks.add(block_idx)
        return block_idx

    def free_block(self, block_idx: int) -> None:
        """Return one allocated block to the free pool and clear its MLX storage."""

        self._validate_block_idx(block_idx)
        if block_idx not in self._allocated_blocks:
            raise ValueError(f"block {block_idx} is not allocated")
        self._allocated_blocks.remove(block_idx)
        self._free_blocks.append(block_idx)
        self.k_pool = mx.slice_update(
            self.k_pool,
            mx.zeros_like(self.k_pool[block_idx : block_idx + 1]),
            mx.array([block_idx, 0, 0, 0, 0]),
            axes=(0, 1, 2, 3, 4),
        )
        self.v_pool = mx.slice_update(
            self.v_pool,
            mx.zeros_like(self.v_pool[block_idx : block_idx + 1]),
            mx.array([block_idx, 0, 0, 0, 0]),
            axes=(0, 1, 2, 3, 4),
        )

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate enough blocks to cover ``num_tokens`` for a new sequence."""

        _validate_int("seq_id", seq_id)
        _validate_non_negative_int("num_tokens", num_tokens)
        if seq_id in self._seq_blocks:
            raise ValueError(f"seq_id {seq_id} already has allocated blocks")
        num_needed = self.blocks_for_tokens(num_tokens)
        if num_needed > self.num_free_blocks:
            raise RuntimeError(
                f"Cannot allocate {num_needed} blocks for seq_id {seq_id}: "
                f"only {self.num_free_blocks} free"
            )
        blocks = [self.allocate_block() for _ in range(num_needed)]
        self._seq_blocks[seq_id] = blocks
        return list(blocks)

    def ensure_sequence_capacity(self, seq_id: int, num_tokens: int) -> list[int]:
        """Grow an existing sequence allocation to cover ``num_tokens``."""

        _validate_int("seq_id", seq_id)
        _validate_non_negative_int("num_tokens", num_tokens)
        blocks = self._seq_blocks.get(seq_id)
        if blocks is None:
            raise KeyError(f"seq_id {seq_id} has no allocated blocks")
        num_needed = self.blocks_for_tokens(num_tokens)
        extra = num_needed - len(blocks)
        if extra <= 0:
            return list(blocks)
        if extra > self.num_free_blocks:
            raise RuntimeError(
                f"Cannot grow seq_id {seq_id} by {extra} blocks: "
                f"only {self.num_free_blocks} free"
            )
        for _ in range(extra):
            blocks.append(self.allocate_block())
        return list(blocks)

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks owned by ``seq_id``; unknown sequences are ignored."""

        _validate_int("seq_id", seq_id)
        blocks = self._seq_blocks.pop(seq_id, None)
        if blocks is None:
            return
        for block_idx in reversed(blocks):
            self.free_block(block_idx)

    def get_sequence_blocks(self, seq_id: int) -> list[int]:
        """Return a copy of the ordered physical blocks for ``seq_id``."""

        _validate_int("seq_id", seq_id)
        blocks = self._seq_blocks.get(seq_id)
        if blocks is None:
            raise KeyError(f"seq_id {seq_id} has no allocated blocks")
        return list(blocks)

    def get_block_kv(self, block_idx: int, layer_idx: int) -> tuple[mx.array, mx.array]:
        """Return the K/V storage slices for one physical block and layer."""

        self._validate_block_idx(block_idx)
        _validate_positive_int("layer_idx", layer_idx + 1)
        if layer_idx >= self.num_layers:
            raise IndexError("layer_idx out of range")
        return self.k_pool[block_idx, layer_idx], self.v_pool[block_idx, layer_idx]

    def block_table_for_sequences(
        self,
        seq_ids: Sequence[int],
        *,
        max_blocks_per_seq: int | None = None,
    ) -> mx.array:
        """Build an MLX int32 block table for the provided sequence rows."""

        rows = [self.get_sequence_blocks(seq_id) for seq_id in seq_ids]
        return build_paged_block_table(rows, max_blocks_per_seq=max_blocks_per_seq)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        _validate_non_negative_int("num_tokens", num_tokens)
        if num_tokens == 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def _validate_block_idx(self, block_idx: int) -> None:
        _validate_int("block_idx", block_idx)
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise IndexError("block_idx out of range")


@dataclass
class SequenceRequest:
    """One scheduler-tracked inference request."""

    seq_id: int
    prompt_ids: list[int]
    adapter_key: str = ""
    priority: int = 0
    max_tokens: int = 256
    arrival_order: int = 0
    generated_tokens: int = 0
    state: RequestState = "waiting"
    block_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_int("seq_id", self.seq_id)
        if not isinstance(self.prompt_ids, list):
            raise TypeError("prompt_ids must be a list[int]")
        for token_id in self.prompt_ids:
            _validate_int("prompt_ids item", token_id)
        if not isinstance(self.adapter_key, str):
            raise TypeError("adapter_key must be a str")
        _validate_int("priority", self.priority)
        _validate_positive_int("max_tokens", self.max_tokens)
        _validate_non_negative_int("arrival_order", self.arrival_order)
        _validate_non_negative_int("generated_tokens", self.generated_tokens)
        if self.state not in ("waiting", "running", "preempted", "completed"):
            raise ValueError("state must be waiting, running, preempted, or completed")


@dataclass
class SchedulerOutput:
    """Requests scheduled for one forward step, grouped by adapter key."""

    scheduled: dict[str, list[SequenceRequest]] = field(default_factory=dict)
    preempted: list[SequenceRequest] = field(default_factory=list)
    num_blocks_used: int = 0

    @property
    def total_requests(self) -> int:
        return sum(len(requests) for requests in self.scheduled.values())

    @property
    def is_empty(self) -> bool:
        return self.total_requests == 0


class ContinuousBatchScheduler:
    """Priority/FIFO continuous-batch scheduler backed by paged KV blocks."""

    def __init__(
        self,
        block_manager: PagedKVBlockManager,
        *,
        max_batch_size: int = 32,
    ) -> None:
        if not isinstance(block_manager, PagedKVBlockManager):
            raise TypeError("block_manager must be a PagedKVBlockManager")
        _validate_positive_int("max_batch_size", max_batch_size)
        self._block_manager = block_manager
        self._max_batch_size = max_batch_size
        self._waiting: list[SequenceRequest] = []
        self._waiting_by_seq_id: dict[int, SequenceRequest] = {}
        self._running: dict[int, SequenceRequest] = {}
        self._completed_buffer: list[SequenceRequest] = []
        self._arrival_counter = 0

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def add_request(
        self,
        seq_id: int,
        prompt_ids: list[int],
        adapter_key: str = "",
        priority: int = 0,
        max_tokens: int = 256,
    ) -> SequenceRequest:
        """Add a request to the waiting queue."""

        if seq_id in self._running:
            raise ValueError(f"seq_id {seq_id} is already running")
        if seq_id in self._waiting_by_seq_id:
            raise ValueError(f"seq_id {seq_id} is already waiting")
        req = SequenceRequest(
            seq_id=seq_id,
            prompt_ids=list(prompt_ids),
            adapter_key=adapter_key,
            priority=priority,
            max_tokens=max_tokens,
            arrival_order=self._arrival_counter,
        )
        self._arrival_counter += 1
        self._waiting.append(req)
        self._waiting_by_seq_id[seq_id] = req
        return req

    def schedule_batch(self) -> SchedulerOutput:
        """Plan the next batch, admitting high-priority work when possible."""

        output = SchedulerOutput()
        admitted: list[SequenceRequest] = []
        preempted_for_retry: list[SequenceRequest] = []

        for req in sorted(self._running.values(), key=_request_sort_key):
            if req.seq_id not in self._running:
                continue
            if self._ensure_running_capacity(req, output, preempted_for_retry):
                admitted.append(req)
                continue
            evicted = self._preempt_request(req)
            output.preempted.append(evicted)
            preempted_for_retry.append(evicted)

        waiting_snapshot = list(self._waiting) + preempted_for_retry
        waiting_snapshot.sort(key=_request_sort_key)
        self._replace_waiting([])
        still_waiting: list[SequenceRequest] = []

        for req in waiting_snapshot:
            if req.seq_id in self._running:
                continue

            needed_blocks = self._blocks_needed(req)
            while (
                len(admitted) >= self._max_batch_size
                or self._block_manager.num_free_blocks < needed_blocks
            ):
                victim = self._find_preemption_victim(req.priority)
                if victim is None:
                    break
                evicted = self._preempt_request(victim)
                admitted = [item for item in admitted if item.seq_id != evicted.seq_id]
                output.preempted.append(evicted)
                preempted_for_retry.append(evicted)

            if (
                len(admitted) < self._max_batch_size
                and self._block_manager.num_free_blocks >= needed_blocks
                and self._try_allocate(req)
            ):
                admitted.append(req)
                continue

            still_waiting.append(req)

        self._replace_waiting(
            _dedupe_waiting(
                still_waiting + preempted_for_retry,
                running_seq_ids=set(self._running),
            )
        )
        groups: defaultdict[str, list[SequenceRequest]] = defaultdict(list)
        for req in admitted:
            if req.seq_id in self._running:
                groups[req.adapter_key].append(req)
        output.scheduled = dict(groups)
        output.num_blocks_used = self._block_manager.num_allocated_blocks
        return output

    def step(self) -> SchedulerOutput:
        return self.schedule_batch()

    def record_generated_token(self, seq_id: int) -> None:
        req = self._running.get(seq_id)
        if req is None:
            raise KeyError(f"seq_id {seq_id} is not running")
        req.generated_tokens += 1

    def complete_request(self, seq_id: int) -> None:
        req = self._running.get(seq_id)
        if req is None:
            raise KeyError(f"seq_id {seq_id} is not running")
        self._block_manager.free_sequence(seq_id)
        req.block_indices = []
        req.state = "completed"
        del self._running[seq_id]
        self._completed_buffer.append(req)

    def abort_request(self, seq_id: int) -> bool:
        req = self._running.pop(seq_id, None)
        if req is not None:
            self._block_manager.free_sequence(seq_id)
            req.block_indices = []
            req.state = "completed"
            return True
        waiting_req = self._waiting_by_seq_id.pop(seq_id, None)
        if waiting_req is not None:
            waiting_req.state = "completed"
            self._waiting = [req for req in self._waiting if req.seq_id != seq_id]
            return True
        return False

    def get_request(self, seq_id: int) -> SequenceRequest | None:
        if seq_id in self._running:
            return self._running[seq_id]
        return self._waiting_by_seq_id.get(seq_id)

    def get_completed(self) -> list[SequenceRequest]:
        completed = list(self._completed_buffer)
        self._completed_buffer.clear()
        return completed

    def _blocks_needed(self, req: SequenceRequest) -> int:
        total_tokens = len(req.prompt_ids) + max(req.generated_tokens, 1)
        return self._block_manager.blocks_for_tokens(total_tokens)

    def _try_allocate(self, req: SequenceRequest) -> bool:
        needed_blocks = self._blocks_needed(req)
        if needed_blocks > self._block_manager.num_free_blocks:
            return False
        total_tokens = len(req.prompt_ids) + max(req.generated_tokens, 1)
        req.block_indices = self._block_manager.allocate_sequence(
            req.seq_id,
            total_tokens,
        )
        req.state = "running"
        self._running[req.seq_id] = req
        return True

    def _ensure_running_capacity(
        self,
        req: SequenceRequest,
        output: SchedulerOutput,
        preempted_for_retry: list[SequenceRequest],
    ) -> bool:
        needed_blocks = self._blocks_needed(req)
        if len(req.block_indices) >= needed_blocks:
            return True

        while self._block_manager.num_free_blocks < needed_blocks - len(req.block_indices):
            victim = self._find_preemption_victim(req.priority)
            if victim is None or victim.seq_id == req.seq_id:
                return False
            evicted = self._preempt_request(victim)
            output.preempted.append(evicted)
            preempted_for_retry.append(evicted)

        total_tokens = len(req.prompt_ids) + max(req.generated_tokens, 1)
        req.block_indices = self._block_manager.ensure_sequence_capacity(
            req.seq_id,
            total_tokens,
        )
        return True

    def _find_preemption_victim(self, min_priority: int) -> SequenceRequest | None:
        candidates = [
            req for req in self._running.values() if req.priority < min_priority
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda req: (req.priority, -req.arrival_order))

    def _preempt_request(self, req: SequenceRequest) -> SequenceRequest:
        self._block_manager.free_sequence(req.seq_id)
        req.block_indices = []
        req.state = "preempted"
        del self._running[req.seq_id]
        return req

    def _replace_waiting(self, requests: Sequence[SequenceRequest]) -> None:
        self._waiting = list(requests)
        self._waiting_by_seq_id = {req.seq_id: req for req in self._waiting}


def build_paged_block_table(
    block_indices_by_row: Sequence[Sequence[int]],
    *,
    max_blocks_per_seq: int | None = None,
) -> mx.array:
    """Build a padded ``(batch, max_blocks_per_seq)`` MLX int32 block table."""

    if max_blocks_per_seq is not None:
        _validate_non_negative_int("max_blocks_per_seq", max_blocks_per_seq)
    rows: list[list[int]] = []
    inferred_width = 0
    for row in block_indices_by_row:
        row_blocks = list(row)
        for block_idx in row_blocks:
            _validate_non_negative_int("block index", block_idx)
        inferred_width = max(inferred_width, len(row_blocks))
        rows.append(row_blocks)

    width = inferred_width if max_blocks_per_seq is None else max_blocks_per_seq
    table = np.zeros((len(rows), width), dtype=np.int32)
    if width == 0:
        return mx.array(table)
    for row_idx, row_blocks in enumerate(rows):
        if len(row_blocks) > width:
            raise ValueError("row has more blocks than max_blocks_per_seq")
        if row_blocks:
            table[row_idx, : len(row_blocks)] = row_blocks
    return mx.array(table)


def gather_paged_kv(
    manager: PagedKVBlockManager,
    block_table: mx.array,
    layer_idx: int,
    seq_lengths: mx.array | Sequence[int],
) -> tuple[mx.array, mx.array]:
    """Gather contiguous K/V tensors from a paged block table.

    The returned tensors have shape ``(batch, kv_heads, max_seq_len, head_dim)``
    and are suitable for ``mx.fast.scaled_dot_product_attention``. Positions
    beyond each sequence's ``seq_lengths`` are zeroed so that a causal mask
    keeps them out of the attention computation.

    Args:
        manager: paged KV block manager that owns the physical pool.
        block_table: int32 array of shape ``(batch, max_blocks_per_seq)``.
        layer_idx: layer whose K/V blocks should be read.
        seq_lengths: int32 array or list of length ``batch`` with actual token
            counts. May be shorter than ``max_blocks_per_seq * block_size``.

    Returns:
        Contiguous ``(k, v)`` tensors in MLX attention layout.
    """

    if not isinstance(manager, PagedKVBlockManager):
        raise TypeError("manager must be a PagedKVBlockManager")
    _validate_int("layer_idx", layer_idx)
    if layer_idx < 0 or layer_idx >= manager.num_layers:
        raise IndexError("layer_idx out of range")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D, got {block_table.ndim}")
    batch = int(block_table.shape[0])
    max_blocks = int(block_table.shape[1])
    if isinstance(seq_lengths, mx.array):
        seq_lengths_arr = seq_lengths.astype(mx.int32)
    else:
        seq_lengths_arr = mx.array(list(seq_lengths), dtype=mx.int32)
    if seq_lengths_arr.shape != (batch,):
        raise ValueError(
            f"seq_lengths must have shape ({batch},), got {seq_lengths_arr.shape}"
        )

    k_layer = manager.k_pool[:, layer_idx, :, :, :]
    v_layer = manager.v_pool[:, layer_idx, :, :, :]
    gathered_k = mx.take(k_layer, block_table, axis=0)
    gathered_v = mx.take(v_layer, block_table, axis=0)
    gathered_k = gathered_k.reshape(
        batch, max_blocks * manager.block_size, manager.num_kv_heads, manager.head_dim
    )
    gathered_v = gathered_v.reshape(
        batch, max_blocks * manager.block_size, manager.num_kv_heads, manager.head_dim
    )
    gathered_k = mx.transpose(gathered_k, (0, 2, 1, 3))
    gathered_v = mx.transpose(gathered_v, (0, 2, 1, 3))

    max_seq_len = int(gathered_k.shape[2])
    actual_max = int(mx.max(seq_lengths_arr).item())
    if actual_max < max_seq_len:
        pos = mx.arange(max_seq_len, dtype=mx.int32)[None, None, :, None]
        mask = pos < seq_lengths_arr[:, None, None, None]
        mask = mask.astype(manager.k_pool.dtype)
        gathered_k = gathered_k * mask
        gathered_v = gathered_v * mask
    return gathered_k, gathered_v


def scatter_paged_kv(
    manager: PagedKVBlockManager,
    block_table: mx.array,
    layer_idx: int,
    k: mx.array,
    v: mx.array,
    seq_lengths: mx.array | Sequence[int],
) -> None:
    """Write contiguous K/V tensors into the paged pool via a block table.

    This is the write side of the paged KV compatibility path. The caller is
    responsible for ensuring that ``block_table`` entries are unique across the
    batch (an invariant enforced by ``PagedKVBlockManager``).

    Args:
        manager: paged KV block manager that owns the physical pool.
        block_table: int32 array of shape ``(batch, max_blocks_per_seq)``.
        layer_idx: layer whose K/V blocks should be written.
        k: tensor of shape ``(batch, kv_heads, seq_len, head_dim)``.
        v: tensor of shape ``(batch, kv_heads, seq_len, head_dim)``.
        seq_lengths: int32 array or list of length ``batch`` with actual token
            counts. May be shorter than ``max_blocks_per_seq * block_size``.
    """

    if not isinstance(manager, PagedKVBlockManager):
        raise TypeError("manager must be a PagedKVBlockManager")
    _validate_int("layer_idx", layer_idx)
    if layer_idx < 0 or layer_idx >= manager.num_layers:
        raise IndexError("layer_idx out of range")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D, got {block_table.ndim}")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if k.ndim != 4:
        raise ValueError(f"k must be 4-D, got {k.ndim}")
    batch = int(block_table.shape[0])
    max_blocks = int(block_table.shape[1])
    if k.shape[0] != batch:
        raise ValueError(f"k batch dim {k.shape[0]} must match block_table {batch}")
    if k.shape[1] != manager.num_kv_heads:
        raise ValueError(
            f"k kv_heads dim {k.shape[1]} must match manager {manager.num_kv_heads}"
        )
    if k.shape[3] != manager.head_dim:
        raise ValueError(
            f"k head_dim {k.shape[3]} must match manager {manager.head_dim}"
        )
    if isinstance(seq_lengths, mx.array):
        seq_lengths_arr = seq_lengths.astype(mx.int32)
    else:
        seq_lengths_arr = mx.array(list(seq_lengths), dtype=mx.int32)
    if seq_lengths_arr.shape != (batch,):
        raise ValueError(
            f"seq_lengths must have shape ({batch},), got {seq_lengths_arr.shape}"
        )

    max_seq_len = max_blocks * manager.block_size
    if k.shape[2] > max_seq_len:
        raise ValueError(
            f"k sequence dim {k.shape[2]} exceeds block_table capacity {max_seq_len}"
        )

    # Pad/truncate to (batch, kv_heads, max_seq_len, head_dim), then reshape to
    # (batch, max_blocks, block_size, kv_heads, head_dim).
    k_pad = _pad_or_trim_kv(k, max_seq_len)
    v_pad = _pad_or_trim_kv(v, max_seq_len)
    k_blocks = mx.transpose(
        k_pad.reshape(batch, max_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim),
        (0, 1, 2, 3, 4),
    )
    v_blocks = mx.transpose(
        v_pad.reshape(batch, max_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim),
        (0, 1, 2, 3, 4),
    )

    # Scatter each batch row into its own (num_blocks, block_size, kv_heads, head_dim)
    # work tensor, then sum over the batch dimension. Conflicts are an allocator bug;
    # the sum would corrupt data, so we rely on PagedKVBlockManager invariants.
    work_k = mx.zeros(
        (batch, manager.num_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim),
        dtype=k.dtype,
    )
    work_v = mx.zeros(
        (batch, manager.num_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim),
        dtype=v.dtype,
    )
    indices = block_table[:, :, None, None, None]
    work_k = mx.put_along_axis(work_k, indices, k_blocks, axis=1)
    work_v = mx.put_along_axis(work_v, indices, v_blocks, axis=1)
    pooled_k = mx.sum(work_k, axis=0)
    pooled_v = mx.sum(work_v, axis=0)

    # Mask out positions beyond seq_lengths so that partially-filled trailing
    # blocks do not leak stale zeros into the physical pool.
    total_len = manager.block_size * manager.num_blocks
    pos = mx.arange(total_len, dtype=mx.int32).reshape(
        manager.num_blocks, manager.block_size
    )
    live_mask = pos[None, :, :] < seq_lengths_arr[:, None, None]
    live_mask = live_mask.reshape(batch, manager.num_blocks, manager.block_size, 1, 1)
    work_k_masked = work_k * live_mask.astype(k.dtype)
    work_v_masked = work_v * live_mask.astype(v.dtype)
    pooled_k = mx.sum(work_k_masked, axis=0)
    pooled_v = mx.sum(work_v_masked, axis=0)

    # Write the pooled layer slice back into the physical pool.
    update_k = pooled_k[None, ...]
    update_k = mx.transpose(update_k, (1, 0, 2, 3, 4))
    update_v = pooled_v[None, ...]
    update_v = mx.transpose(update_v, (1, 0, 2, 3, 4))
    manager.k_pool = mx.slice_update(
        manager.k_pool,
        update_k,
        mx.array([0, layer_idx, 0, 0, 0]),
        axes=(0, 1, 2, 3, 4),
    )
    manager.v_pool = mx.slice_update(
        manager.v_pool,
        update_v,
        mx.array([0, layer_idx, 0, 0, 0]),
        axes=(0, 1, 2, 3, 4),
    )


def _pad_or_trim_kv(x: mx.array, target_len: int) -> mx.array:
    """Pad or trim a (batch, heads, seq, dim) tensor along the sequence axis."""

    seq_len = int(x.shape[2])
    if seq_len == target_len:
        return x
    if seq_len > target_len:
        return x[:, :, :target_len, :]
    pad_shape = list(x.shape)
    pad_shape[2] = target_len - seq_len
    pad = mx.zeros(pad_shape, dtype=x.dtype)
    return mx.concatenate([x, pad], axis=2)


def require_model_integrated_paged_attention() -> None:
    """Deprecated sentinel; the paged KV compatibility path is now available.

    Callers should pass ``paged_kv_manager`` and ``paged_block_table`` to
    :class:`CausalSelfAttention` instead. This function is kept for backward
    compatibility and will be removed once the native paged attention kernel
    lands.
    """

    warnings.warn(
        (
            "require_model_integrated_paged_attention() is deprecated. "
            f"{PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE} "
            "Use CausalSelfAttention(..., paged_kv_manager=..., paged_block_table=...) "
            "for the compatibility path."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _resolve_block_manager_config(
    *,
    config: PagedKVBlockManagerConfig | None,
    num_blocks: int | None,
    block_size: int | None,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    dtype: mx.Dtype,
) -> PagedKVBlockManagerConfig:
    if config is not None:
        if any(
            value is not None
            for value in (num_blocks, block_size, num_layers, num_kv_heads, head_dim)
        ):
            raise ValueError("pass either config or shape kwargs, not both")
        return config
    if (
        num_blocks is None
        or block_size is None
        or num_layers is None
        or num_kv_heads is None
        or head_dim is None
    ):
        raise ValueError(
            "num_blocks, block_size, num_layers, num_kv_heads, and head_dim are required"
        )
    return PagedKVBlockManagerConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )


def _request_sort_key(req: SequenceRequest) -> tuple[int, int]:
    return (-req.priority, req.arrival_order)


def _dedupe_waiting(
    requests: Sequence[SequenceRequest],
    *,
    running_seq_ids: set[int],
) -> list[SequenceRequest]:
    waiting: list[SequenceRequest] = []
    seen: set[int] = set()
    for req in requests:
        if req.seq_id in running_seq_ids or req.seq_id in seen:
            continue
        waiting.append(req)
        seen.add(req.seq_id)
    return waiting


def _validate_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def _validate_positive_int(name: str, value: int) -> None:
    _validate_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_int(name: str, value: int) -> None:
    _validate_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


__all__ = [
    "ContinuousBatchScheduler",
    "PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE",
    "PagedKVBlockManager",
    "PagedKVBlockManagerConfig",
    "SchedulerOutput",
    "SequenceRequest",
    "build_paged_block_table",
    "gather_paged_kv",
    "require_model_integrated_paged_attention",
    "scatter_paged_kv",
]
