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
    require_model_integrated_paged_attention,
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


def test_paged_attention_model_integration_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="not wired yet"):
        require_model_integrated_paged_attention()


def test_inference_root_exports_serving_primitives() -> None:
    assert inference.PagedKVBlockManager is PagedKVBlockManager
    assert inference.ContinuousBatchScheduler is ContinuousBatchScheduler
    assert "PagedKVBlockManager" in inference.__all__
    assert "require_model_integrated_paged_attention" in inference.__all__
