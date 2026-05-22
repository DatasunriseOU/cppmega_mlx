"""V7-F02: KVCache structural tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.kv_cache import KVCache


def test_v7_f02_empty_initial_length():
    c = KVCache(num_layers=2)
    assert c.length(0) == 0
    assert c.length(1) == 0
    assert c.total_bytes() == 0


def test_v7_f02_append_grows_seq_axis():
    c = KVCache(num_layers=1)
    k1 = mx.zeros((1, 1, 16))
    v1 = mx.zeros((1, 1, 16))
    c.append(0, k1, v1)
    assert c.length(0) == 1
    k2 = mx.zeros((1, 1, 16))
    v2 = mx.zeros((1, 1, 16))
    c.append(0, k2, v2)
    assert c.length(0) == 2
    full_k, full_v = c.get(0)
    assert full_k.shape == (1, 2, 16)
    assert full_v.shape == (1, 2, 16)


def test_v7_f02_per_layer_independence():
    c = KVCache(num_layers=3)
    c.append(0, mx.zeros((1, 1, 8)), mx.zeros((1, 1, 8)))
    c.append(2, mx.zeros((1, 1, 8)), mx.zeros((1, 1, 8)))
    c.append(2, mx.zeros((1, 1, 8)), mx.zeros((1, 1, 8)))
    assert c.length(0) == 1
    assert c.length(1) == 0
    assert c.length(2) == 2


def test_v7_f02_reset_clears_all_layers():
    c = KVCache(num_layers=2)
    c.append(0, mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)))
    c.append(1, mx.zeros((1, 3, 4)), mx.zeros((1, 3, 4)))
    assert c.length(0) == 2
    c.reset()
    assert c.length(0) == 0
    assert c.length(1) == 0


def test_v7_f02_total_bytes_grows_with_appends():
    c = KVCache(num_layers=1)
    assert c.total_bytes() == 0
    c.append(0, mx.zeros((1, 1, 16), dtype=mx.float32),
              mx.zeros((1, 1, 16), dtype=mx.float32))
    # 1*1*16*4 (fp32) × 2 (k+v) = 128
    assert c.total_bytes() == 128
    c.append(0, mx.zeros((1, 1, 16), dtype=mx.float32),
              mx.zeros((1, 1, 16), dtype=mx.float32))
    assert c.total_bytes() == 256


def test_v7_f02_get_empty_layer_raises():
    c = KVCache(num_layers=1)
    with pytest.raises(ValueError):
        c.get(0)


def test_v7_f02_mismatched_seq_dims_raises():
    c = KVCache(num_layers=1)
    with pytest.raises(ValueError):
        c.append(0, mx.zeros((1, 2, 4)), mx.zeros((1, 3, 4)))


def test_v7_f02_num_layers_validation():
    with pytest.raises(ValueError):
        KVCache(num_layers=0)
