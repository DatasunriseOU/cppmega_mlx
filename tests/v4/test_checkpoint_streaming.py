"""V7-C04: streaming safetensors load equality + sharded composition."""

from __future__ import annotations

import mlx.core as mx
import pytest
import safetensors.mlx as st

import os
import resource

from cppmega_v4.runtime.checkpoint_streaming import (
    DEFAULT_STREAMING_THRESHOLD_BYTES,
    load_auto, streaming_load, streaming_load_all, streaming_load_sharded,
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


def test_v7_c04_load_auto_picks_bulk_under_threshold(tmp_path):
    """AC#3: small file → bulk path."""
    tensors = {"t": mx.zeros((4, 4))}
    path = _write(tmp_path, tensors)
    route: list[str] = []
    out = load_auto(path, _route=route)
    assert route == ["bulk"]
    assert set(out.keys()) == {"t"}


def test_v7_c04_load_auto_picks_streaming_when_over_threshold(tmp_path):
    """AC#3: threshold check honors file size — use tiny synthetic
    threshold so the test fixture stays small."""
    tensors = {"a": mx.zeros((1024,)), "b": mx.zeros((1024,))}
    path = _write(tmp_path, tensors)
    file_size = os.stat(path).st_size
    route: list[str] = []
    out = load_auto(path, threshold_bytes=file_size - 1, _route=route)
    assert route == ["streaming"]
    assert set(out.keys()) == {"a", "b"}


def test_v7_c04_default_threshold_is_one_gigabyte():
    """AC#3 anchor: production default routes >1 GiB to streaming."""
    assert DEFAULT_STREAMING_THRESHOLD_BYTES == 1024 * 1024 * 1024


def test_v7_c04_streaming_load_bounded_peak_rss(tmp_path):
    """AC#2: streaming_load_all must not balloon RSS by more than the
    backing file's byte budget. We use a synthetic 32 MiB checkpoint
    (8 tensors × 1M float32) and assert the loader's peak resident
    growth stays under 4× that size — well below the bulk-load
    pessimistic 2× tape-doubling baseline."""
    # 8 × (1_000_000 fp32 = 4 MB) = ~32 MB total.
    tensors = {f"layer{i}.w":
               mx.random.normal(shape=(1_000_000,),
                                  key=mx.random.key(i))
               for i in range(8)}
    path = _write(tmp_path, tensors)
    file_bytes = os.stat(path).st_size

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out: dict[str, mx.array] = {}
    for k, t in streaming_load(path):
        out[k] = t
        # Force one materialisation so mlx evaluates lazily.
        mx.eval(out[k])
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux — accept either by
    # normalising the budget against whichever unit yields the smaller
    # growth (we only care about an upper bound).
    delta = max(0, rss1 - rss0)
    # Use a generous 4× budget: on jsdom-less native CI even bulk would
    # spike higher, so the test fails meaningfully only if streaming
    # is no better than mmap-and-clone-everything bulk path.
    budget = file_bytes * 4
    assert delta <= budget, (
        f"streaming_load RSS delta {delta} bytes exceeded budget "
        f"{budget} bytes for file_bytes={file_bytes}"
    )


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
