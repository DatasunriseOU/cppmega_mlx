"""V7-C02: sharded checkpoint save + index.json + load round-trip."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

import mlx.core as mx
import safetensors.mlx as st

from cppmega_v4.runtime.checkpoint_shard import (
    load_sharded, load_sharded_for_rank,
    load_with_backward_compat, save_sharded,
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


def test_v7_c02_load_sharded_for_rank_opens_owned_only_once(tmp_path):
    """AC#2: rank R must open ONLY its own shards, each exactly once.

    Previously the impl re-opened the same shard once per parameter
    (buggy `out.get('__loaded__')` check) — opened count scaled with
    num_params instead of num_owned_shards."""
    save_sharded(_state(), tmp_path, num_shards=4)
    opened: list[str] = []
    out = load_sharded_for_rank(
        tmp_path / "model.index.json",
        rank=0, world_size=2, _opened=opened)
    # rank 0 / world 2 owns shards 0 and 2 (i % 2 == 0).
    assert len(opened) == 2, opened
    assert all("shard-00001-of-00004" in p or "shard-00003-of-00004" in p
               for p in opened)
    # Other shards (1, 3) must NOT have been opened.
    assert not any("shard-00002-of-00004" in p for p in opened)
    assert not any("shard-00004-of-00004" in p for p in opened)
    # Coverage still right for rank's owned subset.
    assert len(out) > 0


def test_v7_c02_load_sharded_opens_each_shard_exactly_once(tmp_path):
    save_sharded(_state(), tmp_path, num_shards=3)
    opened: list[str] = []
    load_sharded(tmp_path / "model.index.json", _opened=opened)
    assert len(opened) == 3
    # No duplicates.
    assert len(set(opened)) == 3


def test_v7_c02_backward_compat_loads_legacy_single_file(tmp_path):
    """AC#3: single-file safetensors checkpoints saved before the
    sharded format still load via load_with_backward_compat."""
    state = _state()
    legacy = tmp_path / "legacy.safetensors"
    st.save_file(state, str(legacy))
    opened: list[str] = []
    out = load_with_backward_compat(legacy, _opened=opened)
    assert set(out.keys()) == set(state.keys())
    for k in state:
        assert mx.allclose(out[k], state[k], atol=0.0)
    assert opened == [str(legacy)]


def test_v7_c02_backward_compat_loads_sharded_via_index(tmp_path):
    save_sharded(_state(), tmp_path, num_shards=2)
    opened: list[str] = []
    out = load_with_backward_compat(
        tmp_path / "model.index.json", _opened=opened)
    assert set(out.keys()) == set(_state().keys())
    # 2 shards opened, no duplicates.
    assert len(opened) == 2
    assert len(set(opened)) == 2


def test_v7_c02_backward_compat_directory_picks_index(tmp_path):
    save_sharded(_state(), tmp_path, num_shards=2)
    out = load_with_backward_compat(tmp_path)
    assert set(out.keys()) == set(_state().keys())


def test_v7_c02_backward_compat_missing_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_with_backward_compat(empty)
