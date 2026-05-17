"""Tests for cppmega_v4.nn.kimi_delta_attention — Path A (FLA naive port to MLX)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda
from cppmega_v4.nn.kimi_delta_attention import (
    KimiDeltaAttentionBlock,
    KimiDeltaAttentionConfig,
)


def _tiny_cfg(**overrides) -> KimiDeltaAttentionConfig:
    base = dict(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        expand_v=1.0,
        use_short_conv=False,
    )
    base.update(overrides)
    return KimiDeltaAttentionConfig(**base)


# ----- config validation -----


@pytest.mark.parametrize("field,bad_value", [
    ("hidden_size", 0), ("num_heads", 0), ("head_dim", 0),
    ("expand_v", 0), ("conv_size", -1), ("norm_eps", 0.0),
])
def test_config_rejects_invalid(field, bad_value):
    kwargs = dict(hidden_size=8, num_heads=2, head_dim=4)
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        KimiDeltaAttentionConfig(**kwargs)


def test_num_v_heads_must_be_multiple_of_num_heads():
    with pytest.raises(ValueError, match="multiple of num_heads"):
        KimiDeltaAttentionConfig(
            hidden_size=8, num_heads=2, head_dim=4, num_v_heads=3
        )


def test_derived_dims_with_gqa():
    cfg = KimiDeltaAttentionConfig(
        hidden_size=16, num_heads=2, head_dim=8, num_v_heads=4
    )
    assert cfg.key_dim == 16   # 2 * 8
    assert cfg.value_dim == 32  # 4 * 8
    assert cfg.gate_dim == 32   # 4 * 8


# ----- naive_recurrent_kda shape contract -----


def test_naive_recurrent_kda_shape():
    B, T, H, K, HV, V = 1, 5, 2, 4, 4, 6
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = mx.random.normal((B, T, HV, K))
    beta = mx.random.normal((B, T, HV))
    o, S = naive_recurrent_kda(q, k, v, g, beta)
    assert o.shape == (B, T, HV, V)
    assert S is None


def test_naive_recurrent_kda_output_final_state():
    B, T, H, K, HV, V = 1, 3, 2, 4, 2, 4
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = mx.random.normal((B, T, HV, K))
    beta = mx.random.normal((B, T, HV))
    o, S = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    assert S is not None
    assert S.shape == (B, HV, K, V)


def test_naive_recurrent_kda_initial_state_used():
    B, T, H, K, HV, V = 1, 2, 1, 2, 1, 2
    q = mx.zeros((B, T, H, K))
    k = mx.zeros((B, T, H, K))
    v = mx.zeros((B, T, HV, V))
    g = mx.zeros((B, T, HV, K))   # exp(0) = 1, no decay
    beta = mx.zeros((B, T, HV))
    init = mx.array([[[[5.0, 0.0], [0.0, 0.0]]]])  # [B,HV,K,V]
    o, S = naive_recurrent_kda(
        q, k, v, g, beta, initial_state=init, output_final_state=True
    )
    # q=0 -> output is 0; state stays = initial.
    np.testing.assert_allclose(np.array(o), np.zeros((B, T, HV, V)), atol=1e-6)
    np.testing.assert_allclose(np.array(S), np.array(init), atol=1e-6)


# ----- parity vs PyTorch FLA reference -----


@pytest.fixture(scope="module")
def fla_naive_kda_torch():
    torch = pytest.importorskip("torch")
    repo = Path("/Volumes/external/sources/rent_kernels/flash-linear-attention")
    if not repo.exists():
        pytest.skip("FLA repo not present at expected path")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from fla.ops.kda.naive import naive_recurrent_kda as fla_fn
    except Exception as exc:
        pytest.skip(f"could not import FLA KDA naive: {exc}")
    return torch, fla_fn


def test_parity_with_fla_torch_kda(fla_naive_kda_torch):
    torch, fla_fn = fla_naive_kda_torch
    B, T, H, K, HV, V = 1, 6, 2, 4, 4, 4
    rng = np.random.default_rng(13)
    q_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
    k_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
    v_np = rng.standard_normal((B, T, HV, V)).astype(np.float32)
    g_np = rng.standard_normal((B, T, HV, K)).astype(np.float32) * 0.05  # small
    beta_np = rng.standard_normal((B, T, HV)).astype(np.float32)

    o_t, _ = fla_fn(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        torch.from_numpy(g_np),
        torch.from_numpy(beta_np),
    )
    o_t_np = o_t.detach().cpu().numpy()
    o_m, _ = naive_recurrent_kda(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        mx.array(g_np),
        mx.array(beta_np),
    )
    o_m_np = np.array(o_m)
    np.testing.assert_allclose(o_m_np, o_t_np, rtol=1e-4, atol=1e-5)


# ----- KimiDeltaAttentionBlock -----


def test_block_forward_shape():
    cfg = _tiny_cfg()
    block = KimiDeltaAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    assert out.shape == x.shape


def test_block_rejects_wrong_input():
    block = KimiDeltaAttentionBlock(_tiny_cfg())
    with pytest.raises(ValueError, match="must be shaped"):
        block(mx.zeros((4, 16)))
    with pytest.raises(ValueError, match="last dim must be"):
        block(mx.zeros((1, 4, 17)))


def test_block_is_identity_at_init():
    cfg = _tiny_cfg()
    block = KimiDeltaAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    np.testing.assert_allclose(np.array(out), np.zeros_like(np.array(x)), atol=1e-5)


def test_block_with_short_conv():
    cfg = _tiny_cfg(use_short_conv=True, conv_size=3)
    block = KimiDeltaAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    assert out.shape == x.shape


def test_block_with_gqa():
    cfg = _tiny_cfg(num_heads=2, num_v_heads=4)
    block = KimiDeltaAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    assert out.shape == x.shape


def test_block_doc_ids_changes_output():
    cfg = _tiny_cfg()
    block = KimiDeltaAttentionBlock(cfg)
    block.o_proj.weight = mx.random.normal(block.o_proj.weight.shape) * 0.1
    x = mx.random.normal((1, 6, cfg.hidden_size))
    out_no = block(x)
    out_yes = block(x, doc_ids=mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32))
    np.testing.assert_allclose(
        np.array(out_no[:, :3, :]), np.array(out_yes[:, :3, :]), atol=1e-5
    )
    assert np.abs(np.array(out_no[:, 3:, :]) - np.array(out_yes[:, 3:, :])).max() > 1e-5


def test_block_gradient_flows():
    cfg = _tiny_cfg()
    block = KimiDeltaAttentionBlock(cfg)
    block.o_proj.weight = mx.random.normal(block.o_proj.weight.shape) * 0.1
    x = mx.random.normal((1, 4, cfg.hidden_size))

    def loss_fn(params):
        block.update(params)
        return mx.mean(mx.square(block(x)))

    grads = mx.grad(loss_fn)(block.trainable_parameters())
    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_proj_1", "f_proj_2", "o_proj"):
        g = grads[name]["weight"]
        assert float(mx.max(mx.abs(g)).item()) > 0.0, f"{name} zero grad"
