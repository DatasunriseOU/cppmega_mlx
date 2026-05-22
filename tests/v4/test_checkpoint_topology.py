"""V7-C05: resume across world-size change uses V7-C02 sharded loader."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.checkpoint_shard import (
    load_sharded, load_sharded_for_rank, save_sharded,
)


def _state(seed: int = 0) -> dict[str, mx.array]:
    return {
        f"layer{i}.w": mx.random.normal(shape=(8, 16),
                                          key=mx.random.key(seed + i))
        for i in range(8)
    }


@pytest.mark.parametrize("save_world,load_world",
                          [(8, 1), (1, 8), (4, 2), (2, 4)])
def test_v7_c05_resume_across_world_size_recovers_full_state(
    tmp_path, save_world, load_world,
):
    """Save N → Load M reconstructs the full param tree.

    `save_sharded` with num_shards=save_world stores
    weight_map per key. On load, calling load_sharded (single rank
    reads everything) reconstructs bitwise-equal full state
    regardless of original shard count.

    For load_world>1 verify rank-N-of-M slices combine back to
    the full set across ranks."""
    state = _state()
    save_sharded(state, tmp_path, num_shards=save_world)
    full = load_sharded(tmp_path / "model.index.json")
    assert set(full.keys()) == set(state.keys())
    for k in state:
        assert mx.allclose(full[k], state[k], atol=0.0)

    # Per-rank load on the new topology — collect across all ranks
    # in the load_world.
    if load_world > 1:
        combined: dict[str, mx.array] = {}
        for r in range(load_world):
            combined.update(load_sharded_for_rank(
                tmp_path / "model.index.json",
                rank=r, world_size=load_world,
            ))
        assert set(combined.keys()) == set(state.keys()), (
            f"world_size={load_world} round-robin missed keys: "
            f"{set(state) - set(combined)}"
        )


def test_v7_c05_invalid_rank_rejected(tmp_path):
    save_sharded(_state(), tmp_path, num_shards=2)
    with pytest.raises(ValueError):
        load_sharded_for_rank(
            tmp_path / "model.index.json",
            rank=5, world_size=2,
        )
