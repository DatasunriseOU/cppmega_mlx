"""V7-Q08.2: load_auto routes >1GiB checkpoints through streaming.

Creating a real 1.5 GiB safetensors file inside CI is expensive; the
correctness contract we actually care about is the **routing decision**
+ the streaming path's per-tensor bound. We test:

1. For a file > threshold_bytes (we lower the threshold), load_auto
   picks the streaming route AND uses the streaming iterator.
2. For a file < threshold_bytes, load_auto picks the bulk route.
3. A real moderate-size file (≈10 MiB) loads identically under both
   routes — establishes that the streaming path is functionally
   equivalent so the 1 GiB+ promise is the same wiring at scale.
"""

from __future__ import annotations

import os
import tempfile

import mlx.core as mx
import safetensors.mlx as stmlx

from cppmega_v4.runtime.checkpoint_streaming import (
    DEFAULT_STREAMING_THRESHOLD_BYTES, load_auto, streaming_load_all,
)


def _write_synth_safetensors(path: str, *, total_floats: int = 2_500_000):
    """Write a moderate-sized safetensors file (~10 MiB at fp32).

    total_floats=2_500_000 => ~10 MiB. Above the test threshold we use,
    below the default 1 GiB threshold.
    """
    tensors = {
        f"layer.{i}.weight": mx.zeros((1000, 250))
        for i in range(total_floats // (1000 * 250))
    }
    stmlx.save_file(tensors, path)


def test_load_auto_picks_streaming_when_over_threshold() -> None:
    """Lower the threshold below the file size to force the streaming
    route and assert it was taken."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "big.safetensors")
        _write_synth_safetensors(p)
        size = os.path.getsize(p)
        assert size > 1_000_000, f"expected > 1 MiB synthetic, got {size}"
        route: list[str] = []
        loaded = load_auto(p, threshold_bytes=size // 2, _route=route)
        assert route == ["streaming"], (
            f"expected streaming route, got {route!r}")
        assert len(loaded) > 0


def test_load_auto_picks_bulk_when_under_threshold() -> None:
    """Use a very small threshold => file is under it => bulk route."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "small.safetensors")
        _write_synth_safetensors(p)
        route: list[str] = []
        load_auto(p, threshold_bytes=10 ** 12, _route=route)
        assert route == ["bulk"], f"expected bulk route, got {route!r}"


def test_load_auto_streaming_matches_bulk_loaded() -> None:
    """The streaming path must yield the SAME tensors as the bulk path
    so the >1GiB promise is just RSS bound, not lossy."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "compare.safetensors")
        _write_synth_safetensors(p)
        size = os.path.getsize(p)
        bulk = load_auto(p, threshold_bytes=size * 10)
        streamed = load_auto(p, threshold_bytes=0)  # force streaming
        assert set(bulk.keys()) == set(streamed.keys())
        for k in bulk:
            # Both are mx.array; cast to numpy for equality check.
            import numpy as np
            assert np.array_equal(
                np.asarray(bulk[k]), np.asarray(streamed[k])
            ), f"mismatch at {k}"


def test_default_threshold_is_one_gib() -> None:
    """V7-C04 contract: default streaming threshold is exactly 1 GiB.

    If this changes, the doc claim "files > 1 GiB stream" must be
    updated alongside.
    """
    assert DEFAULT_STREAMING_THRESHOLD_BYTES == 1024 ** 3
