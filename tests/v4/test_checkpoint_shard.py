"""V7-C02: sharded checkpoint save + index.json + load round-trip."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from cppmega_v4.runtime.checkpoint_shard import (
    load_sharded, load_sharded_for_rank, save_sharded,
)


def _state(seed: int = 0) -> dict[str, mx.array]:
    return {
        "layer0.w": mx.random.normal(shape=(8, 16),
                                       key=mx.random.key(seed)),
        "layer0.b": mx.random.normal(shape=(16,),
                                       key=mx.random.key(seed + 1)),
        "layer1.w": mx.random.normal(shape=(16, 8),
                                       key=mx.random.key(seed + 2)),
        "layer1.b": mx.random.normal(shape=(8,),
                                       key=mx.random.key(seed + 3)),
        "head.w":   mx.random.normal(shape=(8, 4),
                                       key=mx.random.key(seed + 4)),
    }


def test_v7_c02_round_trip_4_shards(tmp_path):
    state = _state()
    save_sharded(state, tmp_path, num_shards=4)
    index = json.loads((tmp_path / "model.index.json").read_text())
    assert index["metadata"]["num_shards"] == 4
    assert set(index["weight_map"].keys()) == set(state.keys())

    loaded = load_sharded(tmp_path / "model.index.json")
    assert set(loaded.keys()) == set(state.keys())
    for k in state:
        assert mx.allclose(loaded[k], state[k], atol=0.0)


def test_v7_c02_index_json_has_weight_map_and_total_size(tmp_path):
    state = _state()
    save_sharded(state, tmp_path, num_shards=2)
    idx = json.loads((tmp_path / "model.index.json").read_text())
    assert "metadata" in idx and "weight_map" in idx
    assert idx["metadata"]["total_size"] > 0
    # Every param key maps to a shard filename.
    for k in state:
        assert idx["weight_map"][k].endswith(".safetensors")


def test_v7_c02_load_sharded_for_rank_round_robin(tmp_path):
    state = _state()
    save_sharded(state, tmp_path, num_shards=4)
    # 4 shards, world_size=2 → rank 0 owns shards {0,2}; rank 1 owns {1,3}
    r0 = load_sharded_for_rank(
        tmp_path / "model.index.json", rank=0, world_size=2)
    r1 = load_sharded_for_rank(
        tmp_path / "model.index.json", rank=1, world_size=2)
    # Combined coverage equals full state.
    combined = {**r0, **r1}
    assert set(combined.keys()) == set(state.keys())


def test_v7_c02_num_shards_validation(tmp_path):
    with pytest.raises(ValueError):
        save_sharded({"k": mx.zeros((2,))}, tmp_path, num_shards=0)
