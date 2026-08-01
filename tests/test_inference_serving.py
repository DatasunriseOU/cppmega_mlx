from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    ContinuousBatchScheduler,
    PagedKVBlockManager,
    PagedKVBlockManagerConfig,
    build_paged_block_table,
    gather_paged_kv,
    require_model_integrated_paged_attention,
    scatter_paged_kv,
    scatter_paged_kv_offsets,
)


def _as_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def _manager(num_blocks: int = 8, block_size: int = 4) -> PagedKVBlockManager:
    return PagedKVBlockManager(
        PagedKVBlockManagerConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=mx.float32,
        )
    )


def _flatten_scheduled(output) -> list[int]:
    return [
        req.seq_id
        for adapter_reqs in output.scheduled.values()
        for req in adapter_reqs
    ]


def test_paged_kv_block_manager_initializes_mlx_pools() -> None:
    manager = _manager()

    assert manager.num_free_blocks == 8
    assert manager.num_allocated_blocks == 0
    assert manager.k_pool.shape == (8, 2, 4, 2, 8)
    assert manager.v_pool.shape == (8, 2, 4, 2, 8)
    assert manager.k_pool.dtype == mx.float32


def test_sequence_allocation_ceil_growth_and_lifo_reuse() -> None:
    manager = _manager(num_blocks=4, block_size=4)

    blocks = manager.allocate_sequence(seq_id=11, num_tokens=5)
    assert len(blocks) == 2
    assert manager.num_free_blocks == 2

    grown = manager.ensure_sequence_capacity(seq_id=11, num_tokens=9)
    assert len(grown) == 3
    assert grown[:2] == blocks
    assert manager.num_free_blocks == 1

    manager.free_sequence(11)
    assert manager.num_free_blocks == 4
    assert manager.allocate_block() == grown[0]


def test_paged_kv_block_manager_fails_closed_on_invalid_allocation() -> None:
    manager = _manager(num_blocks=2, block_size=4)

    manager.allocate_sequence(seq_id=1, num_tokens=4)
    with pytest.raises(ValueError, match="already"):
        manager.allocate_sequence(seq_id=1, num_tokens=4)
    with pytest.raises(RuntimeError, match="Cannot allocate"):
        manager.allocate_sequence(seq_id=2, num_tokens=8)
    with pytest.raises(KeyError, match="no allocated blocks"):
        manager.ensure_sequence_capacity(seq_id=99, num_tokens=1)


def test_build_paged_block_table_uses_mlx_int32_padding() -> None:
    table = build_paged_block_table([[7, 3], [2]], max_blocks_per_seq=3)

    assert table.shape == (2, 3)
    assert table.dtype == mx.int32
    np.testing.assert_array_equal(
        _as_numpy(table),
        np.array([[7, 3, 0], [2, 0, 0]], dtype=np.int32),
    )


def test_manager_builds_block_table_for_sequences() -> None:
    manager = _manager(num_blocks=6, block_size=4)
    blocks_1 = manager.allocate_sequence(seq_id=1, num_tokens=5)
    blocks_2 = manager.allocate_sequence(seq_id=2, num_tokens=1)

    table = manager.block_table_for_sequences([1, 2], max_blocks_per_seq=2)

    np.testing.assert_array_equal(
        _as_numpy(table),
        np.array([blocks_1, [blocks_2[0], 0]], dtype=np.int32),
    )


def test_scheduler_admits_by_priority_fifo_and_groups_by_adapter() -> None:
    scheduler = ContinuousBatchScheduler(_manager(num_blocks=8, block_size=4))
    scheduler.add_request(1, [1, 2], adapter_key="base", priority=1)
    scheduler.add_request(2, [3, 4], adapter_key="adapter-a", priority=3)
    scheduler.add_request(3, [5, 6], adapter_key="base", priority=3)

    output = scheduler.schedule_batch()

    assert output.total_requests == 3
    assert output.num_blocks_used == 3
    assert [req.seq_id for req in output.scheduled["adapter-a"]] == [2]
    assert [req.seq_id for req in output.scheduled["base"]] == [3, 1]
    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 3


def test_scheduler_rejects_duplicate_waiting_request() -> None:
    scheduler = ContinuousBatchScheduler(_manager())
    first = scheduler.add_request(1, [1, 2])

    assert scheduler.get_request(1) is first
    with pytest.raises(ValueError, match="already waiting"):
        scheduler.add_request(1, [3, 4])


def test_scheduler_abort_waiting_request_updates_lookup() -> None:
    scheduler = ContinuousBatchScheduler(_manager())
    first = scheduler.add_request(1, [1, 2])

    assert scheduler.abort_request(1) is True
    assert first.state == "completed"
    assert scheduler.get_request(1) is None

    replacement = scheduler.add_request(1, [3, 4])
    assert replacement.prompt_ids == [3, 4]


def test_scheduler_keeps_waiting_lookup_after_partial_schedule() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(num_blocks=1, block_size=4),
        max_batch_size=1,
    )
    scheduler.add_request(1, [1, 2], priority=0)
    waiting = scheduler.add_request(2, [3, 4], priority=0)

    output = scheduler.schedule_batch()

    assert _flatten_scheduled(output) == [1]
    assert scheduler.get_request(2) is waiting
    with pytest.raises(ValueError, match="already waiting"):
        scheduler.add_request(2, [5, 6])


def test_scheduler_preempts_lower_priority_running_request() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(num_blocks=2, block_size=4),
        max_batch_size=2,
    )
    scheduler.add_request(1, [1, 2, 3], priority=0)
    scheduler.add_request(2, [4, 5, 6], priority=0)
    first = scheduler.schedule_batch()
    assert set(_flatten_scheduled(first)) == {1, 2}

    scheduler.add_request(3, [7, 8, 9], priority=10)
    second = scheduler.schedule_batch()

    assert [req.seq_id for req in second.preempted] == [2]
    assert set(_flatten_scheduled(second)) == {1, 3}
    assert scheduler.get_request(2) is not None
    assert scheduler.get_request(2).state == "preempted"  # type: ignore[union-attr]


def test_scheduler_grows_running_capacity_after_generated_token() -> None:
    scheduler = ContinuousBatchScheduler(_manager(num_blocks=3, block_size=4))
    scheduler.add_request(1, [1, 2, 3], priority=1)
    first = scheduler.schedule_batch()
    req = first.scheduled[""][0]
    assert len(req.block_indices) == 1

    scheduler.record_generated_token(1)
    scheduler.record_generated_token(1)
    second = scheduler.schedule_batch()

    req = second.scheduled[""][0]
    assert len(req.block_indices) == 2
    assert req.generated_tokens == 2


def test_scheduler_complete_frees_blocks_and_buffers_completed() -> None:
    scheduler = ContinuousBatchScheduler(_manager(num_blocks=2, block_size=4))
    scheduler.add_request(1, [1, 2, 3], priority=1)
    scheduler.schedule_batch()

    scheduler.complete_request(1)

    assert scheduler.num_running == 0
    assert scheduler.get_completed()[0].seq_id == 1
    assert scheduler.get_completed() == []


def test_streaming_generation_is_not_paged_attention_integration() -> None:
    assert inference.stream_generate_tokens is not require_model_integrated_paged_attention
    assert (
        "model-integrated paged attention is not wired yet"
        in inference.PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE
    )
    assert "contiguous KV inference" in inference.PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE


def test_paged_attention_model_integration_is_deprecated_warning() -> None:
    with pytest.warns(DeprecationWarning, match="not wired yet"):
        require_model_integrated_paged_attention()


def test_gather_paged_kv_returns_contiguous_masked_tensors() -> None:
    manager = _manager(num_blocks=4, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=8)
    manager.allocate_sequence(seq_id=2, num_tokens=1)
    table = manager.block_table_for_sequences([1, 2], max_blocks_per_seq=2)

    # Seed distinct values into the physical pool for layer 0.
    manager.k_pool = manager.k_pool + 1.0
    manager.v_pool = manager.v_pool + 2.0

    k, v = gather_paged_kv(manager, table, layer_idx=0, seq_lengths=[8, 1])

    assert k.shape == (2, manager.num_kv_heads, 8, manager.head_dim)
    assert v.shape == k.shape
    # A full first row must not disable masking for the shorter second row.
    np.testing.assert_array_equal(
        _as_numpy(mx.sum(k[0], axis=(0, 2)) != 0),
        np.ones(8, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _as_numpy(mx.sum(k[1, :, 1:, :], axis=(0, 2)) == 0),
        np.ones(7, dtype=np.float32),
    )


def test_scatter_paged_kv_round_trip_matches_source() -> None:
    manager = _manager(num_blocks=4, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=5)
    manager.allocate_sequence(seq_id=2, num_tokens=1)
    untouched_block = manager.allocate_sequence(seq_id=3, num_tokens=1)[0]
    table = manager.block_table_for_sequences([1, 2], max_blocks_per_seq=2)
    manager.k_pool = manager.k_pool + 7.0
    manager.v_pool = manager.v_pool + 9.0

    batch, kv_heads, seq_len, head_dim = 2, manager.num_kv_heads, 8, manager.head_dim
    k = mx.arange(batch * kv_heads * seq_len * head_dim, dtype=mx.float32).reshape(
        batch, kv_heads, seq_len, head_dim
    )
    v = k * 10.0

    scatter_paged_kv(manager, table, layer_idx=0, k=k, v=v, seq_lengths=[5, 1])
    k_out, v_out = gather_paged_kv(manager, table, layer_idx=0, seq_lengths=[5, 1])

    # First 5 positions for seq 1 and first 1 position for seq 2 must match.
    np.testing.assert_allclose(_as_numpy(k_out[0, :, :5, :]), _as_numpy(k[0, :, :5, :]))
    np.testing.assert_allclose(_as_numpy(k_out[1, :, :1, :]), _as_numpy(k[1, :, :1, :]))
    np.testing.assert_allclose(_as_numpy(v_out[0, :, :5, :]), _as_numpy(v[0, :, :5, :]))
    np.testing.assert_allclose(_as_numpy(v_out[1, :, :1, :]), _as_numpy(v[1, :, :1, :]))
    # Padding beyond seq_lengths is zeroed.
    np.testing.assert_array_equal(
        _as_numpy(mx.sum(k_out[1, :, 1:, :])), 0.0
    )
    untouched_k, untouched_v = manager.get_block_kv(untouched_block, layer_idx=0)
    np.testing.assert_array_equal(_as_numpy(untouched_k), 7.0)
    np.testing.assert_array_equal(_as_numpy(untouched_v), 9.0)


@pytest.mark.parametrize("seq_length", [-1, 9])
def test_gather_paged_kv_rejects_out_of_capacity_length(seq_length: int) -> None:
    manager = _manager(num_blocks=2, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=8)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)

    with pytest.raises(ValueError, match=r"within \[0, 8\]"):
        gather_paged_kv(
            manager,
            table,
            layer_idx=0,
            seq_lengths=[seq_length],
        )


def test_scatter_paged_kv_rejects_shape_mismatch() -> None:
    manager = _manager(num_blocks=4, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=5)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)
    k = mx.zeros((1, manager.num_kv_heads, 20, manager.head_dim))
    v = mx.zeros((1, manager.num_kv_heads, 20, manager.head_dim))

    with pytest.raises(ValueError, match="exceeds block_table capacity"):
        scatter_paged_kv(manager, table, layer_idx=0, k=k, v=v, seq_lengths=[5])


def test_scatter_paged_kv_offsets_appends_at_absolute_positions() -> None:
    manager = _manager(num_blocks=4, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=6)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)

    batch, kv_heads, head_dim = 1, manager.num_kv_heads, manager.head_dim
    past = mx.arange(kv_heads * 4 * head_dim, dtype=mx.float32).reshape(
        batch, kv_heads, 4, head_dim
    )
    scatter_paged_kv(manager, table, layer_idx=0, k=past, v=past, seq_lengths=[4])

    new_tokens = mx.full((batch, kv_heads, 2, head_dim), 100.0)
    scatter_paged_kv_offsets(
        manager, table, layer_idx=0, k=new_tokens, v=new_tokens, write_offsets=[4]
    )
    k_out, _ = gather_paged_kv(manager, table, layer_idx=0, seq_lengths=[6])

    np.testing.assert_allclose(_as_numpy(k_out[0, :, :4, :]), _as_numpy(past[0]))
    np.testing.assert_allclose(_as_numpy(k_out[0, :, 4:6, :]), 100.0)


@pytest.mark.parametrize(
    "write_offsets, new_tokens, match",
    [
        ([7], 2, "exceed block_table capacity"),
        ([-1], 1, r"within \[0, 8\]"),
    ],
)
def test_scatter_paged_kv_offsets_rejects_out_of_capacity(
    write_offsets, new_tokens, match
) -> None:
    manager = _manager(num_blocks=2, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=8)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)
    k = mx.zeros((1, manager.num_kv_heads, new_tokens, manager.head_dim))

    with pytest.raises(ValueError, match=match):
        scatter_paged_kv_offsets(
            manager, table, layer_idx=0, k=k, v=k, write_offsets=write_offsets
        )


def test_scatter_paged_kv_offsets_rejects_missing_live_block() -> None:
    manager = _manager(num_blocks=4, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=2)
    # Wider table than the allocation covers: second logical block is padding.
    table = mx.array([[manager.block_table_for_sequences([1], max_blocks_per_seq=1)[0, 0], -1]], dtype=mx.int32)
    k = mx.zeros((1, manager.num_kv_heads, 3, manager.head_dim))

    with pytest.raises(ValueError, match="missing a live physical block"):
        scatter_paged_kv_offsets(
            manager, table, layer_idx=0, k=k, v=k, write_offsets=[2]
        )


@pytest.mark.parametrize(
    "table, match",
    [
        ([[-1, 0]], "missing a live physical block"),
        ([[0, 0]], "duplicate physical block"),
    ],
)
def test_scatter_paged_kv_offsets_preflights_live_prefix(table, match) -> None:
    manager = _manager(num_blocks=2, block_size=4)
    k = mx.zeros((1, manager.num_kv_heads, 1, manager.head_dim))

    with pytest.raises(ValueError, match=match):
        scatter_paged_kv_offsets(
            manager,
            mx.array(table, dtype=mx.int32),
            layer_idx=0,
            k=k,
            v=k,
            write_offsets=[4],
        )


def test_scatter_paged_kv_offsets_tilelang_matches_mlx_backend() -> None:
    from cppmega_mlx.nn._tilelang.paged_kv_tilelang import paged_kv_native_status

    status = paged_kv_native_status()
    if not status.available:
        pytest.skip(f"native paged KV scatter unavailable: {status.reason}")

    def run(backend: str) -> tuple[np.ndarray, np.ndarray]:
        manager = _manager(num_blocks=4, block_size=4)
        manager.allocate_sequence(seq_id=1, num_tokens=6)
        manager.allocate_sequence(seq_id=2, num_tokens=7)
        table = manager.block_table_for_sequences([1, 2], max_blocks_per_seq=2)
        batch, kv_heads, head_dim = 2, manager.num_kv_heads, manager.head_dim
        past = mx.arange(
            batch * kv_heads * 4 * head_dim, dtype=mx.float32
        ).reshape(batch, kv_heads, 4, head_dim)
        scatter_paged_kv(
            manager, table, layer_idx=0, k=past, v=past, seq_lengths=[4, 4]
        )
        shape = (batch, kv_heads, 2, head_dim)
        new_tokens = 50.0 + mx.arange(
            int(np.prod(shape)), dtype=mx.float32
        ).reshape(shape)
        scatter_paged_kv_offsets(
            manager,
            table,
            layer_idx=0,
            k=new_tokens,
            v=new_tokens + 1000.0,
            write_offsets=[4, 5],
            backend=backend,
        )
        k_out, v_out = gather_paged_kv(
            manager, table, layer_idx=0, seq_lengths=[6, 7]
        )
        return _as_numpy(k_out), _as_numpy(v_out)

    tilelang_k, tilelang_v = run("tilelang")
    mlx_k, mlx_v = run("mlx")
    np.testing.assert_allclose(tilelang_k, mlx_k, atol=0, rtol=0)
    np.testing.assert_allclose(tilelang_v, mlx_v, atol=0, rtol=0)


def test_scatter_paged_kv_offsets_tilelang_fails_closed_on_unsupported_dtype() -> None:
    from cppmega_mlx.nn._tilelang.paged_kv_tilelang import (
        PagedKvNativeUnavailable,
        paged_kv_native_status,
    )

    status = paged_kv_native_status()
    if not status.available:
        pytest.skip(f"native paged KV scatter unavailable: {status.reason}")

    manager = PagedKVBlockManager(
        PagedKVBlockManagerConfig(
            num_blocks=2,
            block_size=4,
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            dtype=mx.int32,
        )
    )
    manager.allocate_sequence(seq_id=1, num_tokens=8)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)
    k = mx.zeros((1, manager.num_kv_heads, 1, manager.head_dim), dtype=mx.int32)

    with pytest.raises(PagedKvNativeUnavailable, match="unsupported pool dtype"):
        scatter_paged_kv_offsets(
            manager,
            table,
            layer_idx=0,
            k=k,
            v=k,
            write_offsets=[0],
            backend="tilelang",
        )


def test_scatter_paged_kv_offsets_rejects_unknown_backend() -> None:
    manager = _manager(num_blocks=2, block_size=4)
    manager.allocate_sequence(seq_id=1, num_tokens=8)
    table = manager.block_table_for_sequences([1], max_blocks_per_seq=2)
    k = mx.zeros((1, manager.num_kv_heads, 1, manager.head_dim))

    with pytest.raises(ValueError, match="unknown backend"):
        scatter_paged_kv_offsets(
            manager,
            table,
            layer_idx=0,
            k=k,
            v=k,
            write_offsets=[0],
            backend="cuda",  # type: ignore[arg-type]
        )


def test_inference_root_exports_serving_primitives() -> None:
    assert inference.PagedKVBlockManager is PagedKVBlockManager
    assert inference.ContinuousBatchScheduler is ContinuousBatchScheduler
    assert inference.gather_paged_kv is gather_paged_kv
    assert inference.scatter_paged_kv is scatter_paged_kv
    assert inference.scatter_paged_kv_offsets is scatter_paged_kv_offsets
    assert "PagedKVBlockManager" in inference.__all__
    assert "gather_paged_kv" in inference.__all__
    assert "scatter_paged_kv" in inference.__all__
    assert "scatter_paged_kv_offsets" in inference.__all__
    assert "require_model_integrated_paged_attention" in inference.__all__
