"""V7-C05: tensor-axis re-shard for topology change resume.

Covers the AC gaps the round-robin shard test missed:
  AC#1: save N -> load M reconstructs identical model outputs to
        within fp32 tolerance.
  AC#2: opt.state (Adam moments) re-sharded along same axis as params.
  AC#3: ValueError when target_world is incompatible with arch.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.checkpoint_reshard import (
    ensure_axis_divisible, gather_full_from_axis_shards,
    load_resharded, reshard_along_axis, save_resharded,
)


def _state(seed: int = 0) -> dict[str, mx.array]:
    """Tiny attention-style state: column-parallel out_proj (axis 0),
    row-parallel mlp w1 (axis 1), un-sharded norm scale (None)."""
    return {
        "out_proj.w":   mx.random.normal(shape=(32, 16),
                                          key=mx.random.key(seed)),
        "mlp.w1":       mx.random.normal(shape=(16, 64),
                                          key=mx.random.key(seed + 1)),
        "norm.scale":   mx.random.normal(shape=(16,),
                                          key=mx.random.key(seed + 2)),
    }


def _axis_map() -> dict[str, int]:
    return {
        "out_proj.w": 0,  # column-parallel: split along axis 0
        "mlp.w1":     1,  # row-parallel: split along axis 1
        # norm.scale absent → replicated
    }


def _adam_moments(state: dict[str, mx.array]) -> dict[str, mx.array]:
    """Synthetic AdamW first-moment table — same shape as params, so
    must re-shard along the same axes."""
    return {f"{k}.m": v * 0.5 + 1.0 for k, v in state.items()}


# ---------------------------------------------------------------------------
# AC#1: identical model outputs across topology change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("save_world,load_world",
                          [(8, 1), (1, 8), (4, 2), (2, 4)])
def test_v7_c05_axis_reshard_round_trip_identical(
    save_world, load_world,
):
    state = _state()
    axis = _axis_map()
    saved = reshard_along_axis(state, axis, save_world)
    # Save → gather → re-split into load_world.
    full_after_save = gather_full_from_axis_shards(saved, axis)
    loaded = reshard_along_axis(full_after_save, axis, load_world)
    full_after_load = gather_full_from_axis_shards(loaded, axis)
    # Bit-exact reconstruction of every tensor.
    for k in state:
        assert mx.array_equal(state[k], full_after_load[k]), (
            f"key {k} mismatched after {save_world}->{load_world} reshard")


def test_v7_c05_axis_reshard_disk_round_trip(tmp_path):
    state = _state()
    axis = _axis_map()
    save_resharded(state, axis, tmp_path, target_world=4)
    rank_states = load_resharded(
        tmp_path / "model.axis-index.json", load_world=2)
    assert len(rank_states) == 2
    full = gather_full_from_axis_shards(rank_states, axis)
    for k in state:
        assert mx.array_equal(state[k], full[k])


# ---------------------------------------------------------------------------
# AC#2: opt.state Adam moments re-sharded along same axis as params.
# ---------------------------------------------------------------------------


def test_v7_c05_adam_moments_reshard_along_same_axis():
    state = _state()
    moms = _adam_moments(state)
    axis = _axis_map()
    # Adam moment keys mirror param keys (with .m suffix); axis map for
    # moments uses the same per-suffix policy.
    mom_axis = {f"{k}.m": v for k, v in axis.items()}
    saved = reshard_along_axis(moms, mom_axis, 4)
    full = gather_full_from_axis_shards(saved, mom_axis)
    for k in moms:
        assert mx.array_equal(moms[k], full[k]), (
            f"adam moment {k} corrupted after 4-way reshard")


# ---------------------------------------------------------------------------
# AC#3: clear error when target_world is incompatible.
# ---------------------------------------------------------------------------


def test_v7_c05_incompatible_world_raises():
    state = _state()
    axis = _axis_map()
    # out_proj.w has dim 32 on axis 0 — 32 % 5 != 0 → must raise.
    with pytest.raises(ValueError) as exc:
        ensure_axis_divisible(state, axis, target_world=5)
    msg = str(exc.value)
    assert "out_proj.w" in msg
    assert "not divisible" in msg


def test_v7_c05_zero_world_raises():
    with pytest.raises(ValueError):
        ensure_axis_divisible(_state(), _axis_map(), target_world=0)


def test_v7_c05_reshard_propagates_divisibility_error():
    with pytest.raises(ValueError):
        reshard_along_axis(_state(), _axis_map(), target_world=5)


# ---------------------------------------------------------------------------
# Replicated-key path (no axis entry → every rank gets a copy).
# ---------------------------------------------------------------------------


def test_v7_c05_unmapped_keys_are_replicated():
    state = {"norm.scale": mx.array([1.0, 2.0, 3.0])}
    shards = reshard_along_axis(state, {}, target_world=3)
    assert len(shards) == 3
    for rs in shards:
        assert mx.array_equal(rs["norm.scale"], state["norm.scale"])


def test_v7_c05_index_records_axis_metadata(tmp_path):
    import json
    save_resharded(_state(), _axis_map(), tmp_path, target_world=2)
    idx = json.loads(
        (tmp_path / "model.axis-index.json").read_text())
    assert idx["format"] == "axis-shard-v1"
    assert idx["target_world"] == 2
    assert idx["axis_for_key"]["out_proj.w"] == 0
    assert idx["axis_for_key"]["mlp.w1"] == 1


def test_v7_c05_load_resharded_rejects_unknown_format(tmp_path):
    import json
    bad = tmp_path / "bad.axis-index.json"
    bad.write_text(json.dumps({
        "format": "garbage", "target_world": 1,
        "axis_for_key": {}, "weight_map": {},
        "shard_template": "x",
    }))
    with pytest.raises(ValueError):
        load_resharded(bad, load_world=1)
