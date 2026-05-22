"""V7-C04: streaming safetensors load equality + sharded composition."""

from __future__ import annotations

import mlx.core as mx
import pytest
import safetensors.mlx as st

from cppmega_v4.runtime.checkpoint_streaming import (
    streaming_load, streaming_load_all, streaming_load_sharded,
)


def _write(tmp_path, tensors):
    p = str(tmp_path / "weights.safetensors")
    st.save_file(tensors, p)
    return p


def test_v7_c04_streaming_load_matches_bulk_load(tmp_path):
    tensors = {
        "layer0.w": mx.random.normal(shape=(8, 16),
                                       key=mx.random.key(0)),
        "layer1.w": mx.random.normal(shape=(16, 8),
                                       key=mx.random.key(1)),
    }
    path = _write(tmp_path, tensors)
    bulk = st.load_file(path)
    stream: dict[str, mx.array] = {}
    for k, t in streaming_load(path):
        stream[k] = t
    assert set(bulk.keys()) == set(stream.keys())
    for k in bulk:
        assert mx.allclose(bulk[k], stream[k], atol=0.0)


def test_v7_c04_streaming_load_all_progress_callback(tmp_path):
    tensors = {f"t{i}": mx.zeros((2, 2)) for i in range(20)}
    path = _write(tmp_path, tensors)
    seen: list[tuple[int, int]] = []
    out = streaming_load_all(
        path, progress_cb=lambda done, tot: seen.append((done, tot)))
    assert len(out) == 20
    # At least one progress callback fired and totals match.
    assert seen
    assert all(t == 20 for _, t in seen)


def test_v7_c04_streaming_load_sharded(tmp_path):
    t1 = {"a.w": mx.array([[1.0, 2.0]])}
    t2 = {"b.w": mx.array([[3.0, 4.0]])}
    p1 = str(tmp_path / "s0.safetensors")
    p2 = str(tmp_path / "s1.safetensors")
    st.save_file(t1, p1)
    st.save_file(t2, p2)
    merged = dict(streaming_load_sharded([p1, p2]))
    assert set(merged.keys()) == {"a.w", "b.w"}
    assert mx.allclose(merged["a.w"], t1["a.w"], atol=0.0)
    assert mx.allclose(merged["b.w"], t2["b.w"], atol=0.0)
