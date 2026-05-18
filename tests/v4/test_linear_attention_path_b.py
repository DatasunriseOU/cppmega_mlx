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


def test_path_b_parity_kv_dim_mismatch():
    """Path B Metal kernel supports head_k_dim != head_v_dim natively."""
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, T, H, K, V = 1, 3, 2, 4, 6
    rng = np.random.default_rng(2)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    o_b, sf_b = gdn_forward_path_b(q, k, v, beta, g, output_final_state=True)
    o_a, sf_a = naive_recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
    mx.eval(o_b, o_a, sf_b, sf_a)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)
    assert sf_b.shape == (B, H, K, V)
    np.testing.assert_allclose(np.array(sf_b), np.array(sf_a), atol=1e-4, rtol=1e-4)


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


def test_path_b_initial_state_parity():
    """Streaming-decode style: feeding a nonzero h0 must match Path A."""
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, T, H, K, V = 1, 4, 2, 6, 8
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    h0 = mx.array(rng.standard_normal((B, H, K, V)).astype(np.float32) * 0.5)
    o_b, sf_b = gdn_forward_path_b(
        q, k, v, beta, g, initial_state=h0, output_final_state=True
    )
    o_a, sf_a = naive_recurrent_gated_delta_rule(
        q, k, v, beta, g, initial_state=h0, output_final_state=True
    )
    mx.eval(o_b, o_a, sf_b, sf_a)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(sf_b), np.array(sf_a), atol=1e-4, rtol=1e-4)


def test_path_b_custom_scale_parity():
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, T, H, K, V = 1, 4, 2, 8, 8
    rng = np.random.default_rng(13)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    custom_scale = 0.25
    o_b, _ = gdn_forward_path_b(q, k, v, beta, g, scale=custom_scale)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=custom_scale)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)


def test_path_b_streaming_chunks_match_full_run():
    """Two halves with h0=mid-state must equal one full run."""
    if not _path_b_status().available:
        pytest.skip("Path B Metal kernel not available on this host")
    B, H, K, V = 1, 2, 6, 6
    T1, T2 = 3, 4
    T = T1 + T2
    rng = np.random.default_rng(17)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)

    o_full, sf_full = gdn_forward_path_b(q, k, v, beta, g, output_final_state=True)

    o1, sf_mid = gdn_forward_path_b(
        q[:, :T1], k[:, :T1], v[:, :T1], beta[:, :T1], g[:, :T1],
        output_final_state=True,
    )
    o2, sf_end = gdn_forward_path_b(
        q[:, T1:], k[:, T1:], v[:, T1:], beta[:, T1:], g[:, T1:],
        initial_state=sf_mid, output_final_state=True,
    )
    o_stream = mx.concatenate([o1, o2], axis=1)
    mx.eval(o_full, o_stream, sf_full, sf_end)
    np.testing.assert_allclose(np.array(o_stream), np.array(o_full), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(sf_end), np.array(sf_full), atol=1e-4, rtol=1e-4)
