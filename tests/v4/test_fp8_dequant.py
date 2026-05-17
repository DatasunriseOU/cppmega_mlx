"""Tests for vendored mlx-lm PR #1224 FP8 block-dequant utility."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external._mlx_lm_fp8_dequant_vendored import (
    dequant_block_fp8,
    sanitize_fp8_weights,
)


def _make_fp8_weight(m: int, n: int, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Build a (weight_fp8, scale_inv) pair from a bf16 reference tensor."""
    rng = np.random.default_rng(seed)
    bs = 128
    blocks_m = (m + bs - 1) // bs
    blocks_n = (n + bs - 1) // bs
    scale_inv = mx.array(rng.uniform(0.5, 1.5, (blocks_m, blocks_n)).astype(np.float32))
    bf16_ref = mx.array(rng.standard_normal((m, n)).astype(np.float32) * 0.1).astype(
        mx.bfloat16
    )
    # Simulate fp8 quantization: divide by per-block scale, cast to fp8.
    pad_bottom = (-m) % bs
    pad_side = (-n) % bs
    padded = mx.pad(bf16_ref.astype(mx.float32), ((0, pad_bottom), (0, pad_side)))
    blocks = padded.reshape(blocks_m, bs, blocks_n, bs)
    scaled = (blocks / scale_inv[:, None, :, None]).reshape(
        m + pad_bottom, n + pad_side
    )[:m, :n]
    fp8 = mx.to_fp8(scaled)  # uint8 storage
    return fp8, scale_inv


def test_dequant_block_fp8_round_trip_shape():
    m, n = 256, 384
    fp8, scale_inv = _make_fp8_weight(m, n)
    out = dequant_block_fp8(fp8, scale_inv)
    assert out.shape == (m, n)
    assert out.dtype == mx.bfloat16


def test_dequant_block_fp8_recovers_signal():
    """Quant→dequant must preserve signal within fp8 precision (~5% rel error)."""
    m, n = 128, 128
    fp8, scale_inv = _make_fp8_weight(m, n, seed=7)
    out = dequant_block_fp8(fp8, scale_inv)
    # Build the expected value: scale_inv * fp8_as_fp32
    expected = (
        mx.from_fp8(fp8, dtype=mx.bfloat16).astype(mx.float32)
        * scale_inv[0, 0, None, None]
    ).astype(mx.bfloat16)
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=1e-3,
    )


def test_dequant_block_fp8_handles_non_block_aligned_shapes():
    """Shapes not divisible by 128 must still produce correct (m, n) output."""
    m, n = 100, 200  # both < 128 multiples, edge of pad path
    fp8, scale_inv = _make_fp8_weight(m, n)
    out = dequant_block_fp8(fp8, scale_inv)
    assert out.shape == (m, n)


def test_sanitize_fp8_weights_replaces_paired_keys():
    fp8_a, scale_a = _make_fp8_weight(64, 64, seed=1)
    fp8_b, scale_b = _make_fp8_weight(128, 128, seed=2)
    other = mx.array(np.random.randn(8, 8).astype(np.float32))
    weights = {
        "model.layers.0.mlp.gate_proj.weight": fp8_a,
        "model.layers.0.mlp.gate_proj.weight_scale_inv": scale_a,
        "model.layers.1.mlp.gate_proj.weight": fp8_b,
        "model.layers.1.mlp.gate_proj.weight_scale_inv": scale_b,
        "model.norm.weight": other,
        # PTQ artefact that must be dropped:
        "model.layers.0.mlp.gate_proj.activation_scale": mx.array([1.0]),
    }
    out = sanitize_fp8_weights(weights)
    assert "model.layers.0.mlp.gate_proj.weight" in out
    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in out
    assert "model.layers.0.mlp.gate_proj.activation_scale" not in out
    assert out["model.layers.0.mlp.gate_proj.weight"].dtype == mx.bfloat16
    assert out["model.norm.weight"] is other  # untouched


def test_sanitize_fp8_weights_no_op_when_no_fp8():
    """Without any weight_scale_inv key, return a dict copy unchanged."""
    weights = {
        "model.norm.weight": mx.array([1.0, 2.0]),
        "model.embed_tokens.weight": mx.array([[0.0, 1.0]]),
    }
    out = sanitize_fp8_weights(weights)
    assert out == dict(weights)
    # Different dict object (defensive copy), same values
    assert out is not weights
