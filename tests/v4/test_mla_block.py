"""Tests for the real V3-style MLA block (LoRA Q + LoRA KV + RoPE + absorb)."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.mla_block import MLABlock, MLABlockConfig


def _cfg(**overrides):
    base = dict(
        hidden_size=128, num_heads=4,
        qk_nope_head_dim=16, qk_rope_head_dim=8,
        v_head_dim=16, q_lora_rank=32, kv_lora_rank=24,
    )
    base.update(overrides)
    return MLABlockConfig(**base)


def test_config_validation():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        MLABlockConfig(hidden_size=0, num_heads=4)
    with pytest.raises(ValueError, match="q_lora_rank must be positive"):
        MLABlockConfig(hidden_size=64, num_heads=4, q_lora_rank=0)


def test_block_forward_prefill_shape():
    cfg = _cfg()
    blk = MLABlock(cfg)
    B, S = 1, 8
    x = mx.random.normal((B, S, cfg.hidden_size))
    out = blk(x)
    assert out.shape == x.shape
    assert not bool(mx.any(mx.isnan(out)).item())


def test_block_residual_at_init():
    """o-proj zero-init means initial output should equal input (residual passthrough)."""
    cfg = _cfg()
    blk = MLABlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = blk(x)
    # wo.weight = 0 → block delta = 0 → out == x.
    np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-5)


def test_block_perturbed_changes_output():
    """After perturbing wkv_b / wo, output should differ from residual."""
    cfg = _cfg()
    blk = MLABlock(cfg)
    rng = np.random.default_rng(0)
    blk.wo.weight = mx.array(
        rng.standard_normal(blk.wo.weight.shape).astype(np.float32) * 0.05
    )
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = blk(x)
    assert not np.allclose(np.array(out), np.array(x), atol=1e-5)


def test_block_decode_path_uses_absorb():
    """At S=1 with use_absorb=True, internal absorbed weights should be built."""
    cfg = _cfg(use_absorb=True)
    blk = MLABlock(cfg)
    # Perturb wkv_b so absorbed weights are non-trivial.
    rng = np.random.default_rng(1)
    blk.wkv_b.weight = mx.array(
        rng.standard_normal(blk.wkv_b.weight.shape).astype(np.float32) * 0.05
    )
    blk.wo.weight = mx.array(
        rng.standard_normal(blk.wo.weight.shape).astype(np.float32) * 0.05
    )
    assert blk._w_uk_abs is None
    x = mx.random.normal((1, 1, cfg.hidden_size))
    out = blk(x)
    assert out.shape == (1, 1, cfg.hidden_size)
    # Absorbed weights cached.
    assert blk._w_uk_abs is not None
    assert blk._w_uv_w_o is not None
    assert blk._w_uk_abs.shape == (cfg.num_heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank)


def test_block_grads_propagate():
    cfg = _cfg()
    blk = MLABlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    cot = mx.random.normal(x.shape)

    def loss(x_):
        return (blk(x_) * cot).sum()

    g = mx.grad(loss)(x)
    assert g.shape == x.shape
    assert np.all(np.isfinite(np.array(g)))


def test_lora_rank_compresses_params():
    """W_KV LoRA bottleneck saves params vs full W_KV."""
    cfg = _cfg(kv_lora_rank=16)
    blk = MLABlock(cfg)
    # wkv_a: [D, r_kv + rope_pe]; wkv_b: [r_kv, H*(nope+v)].
    # Combined LoRA params vs full W_KV (= D * H * (nope + v)):
    full = cfg.hidden_size * cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
    lora = (cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
            + cfg.kv_lora_rank * cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim))
    assert lora < full, (
        f"LoRA KV ({lora} params) should be smaller than full W_KV ({full})"
    )
