"""Path B real-kernel parity tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.linear_attention_path_b import gdn_forward_path_b
from cppmega_v4._tilelang.linear_attention_paths import _path_b_status
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


def test_path_b_status_available_when_metal_present():
    st = _path_b_status()
    # On Apple Silicon with MLX, metal_kernel is available.
    if hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"):
        assert st.available
    else:
        assert not st.available


def test_path_b_parity_with_path_a():
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, T, H, D = 1, 5, 2, 4
    rng = np.random.default_rng(101)
    q = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    o_b, _ = gdn_forward_path_b(q, k, v, beta, g)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    mx.eval(o_b, o_a)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)


def test_path_b_falls_back_when_kv_dim_mismatch():
    """When head_v_dim != head_k_dim, Path B delegates to Path A correctly."""
    B, T, H, K, V = 1, 3, 2, 4, 6
    rng = np.random.default_rng(2)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    o_b, _ = gdn_forward_path_b(q, k, v, beta, g)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-5)


def test_path_b_output_final_state():
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, T, H, D = 1, 3, 2, 4
    rng = np.random.default_rng(3)
    q = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    _, sf = gdn_forward_path_b(q, k, v, beta, g, output_final_state=True)
    assert sf is not None
    assert sf.shape == (B, H, D, D)
