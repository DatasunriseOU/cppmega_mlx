"""Tests for cppmega_v4.nn.linear_attention — Path A (FLA naive port to MLX).

Includes a parity test against the original PyTorch ``naive_recurrent_gated_delta_rule``
from fla-org/flash-linear-attention when torch is importable; otherwise the
parity test is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)
from cppmega_v4.nn.linear_attention import LinearAttentionBlock, LinearAttentionConfig


def _tiny_cfg(**overrides) -> LinearAttentionConfig:
    base = dict(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        expand_v=1.0,
        use_short_conv=False,
        use_gate=False,
    )
    base.update(overrides)
    return LinearAttentionConfig(**base)


# ----- config validation -----


@pytest.mark.parametrize("field,bad_value", [
    ("hidden_size", 0), ("num_heads", 0), ("head_dim", 0),
    ("expand_v", 0), ("conv_size", -1), ("norm_eps", 0.0),
])
def test_config_rejects_invalid(field, bad_value):
    kwargs = dict(hidden_size=8, num_heads=2, head_dim=4)
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        LinearAttentionConfig(**kwargs)


def test_config_derived_dims():
    cfg = LinearAttentionConfig(
        hidden_size=16, num_heads=4, head_dim=8, expand_v=1.5
    )
    assert cfg.head_k_dim == 8
    assert cfg.head_v_dim == 12  # 8 * 1.5
    assert cfg.key_dim == 32      # 4 * 8
    assert cfg.value_dim == 48    # 4 * 12


def test_num_v_heads_divisibility_validated():
    with pytest.raises(ValueError, match="divisible"):
        LinearAttentionConfig(
            hidden_size=8, num_heads=2, head_dim=4, num_v_heads=3  # 3 not divisible by 2
        )


# ----- naive_recurrent_gated_delta_rule shape contract -----


def test_naive_recurrent_shape():
    B, T, H, K, V = 2, 5, 3, 4, 6
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.random.normal((B, T, H))
    g = mx.random.normal((B, T, H))
    o, h = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    assert o.shape == (B, T, H, V)
    assert h is None


def test_naive_recurrent_output_final_state():
    B, T, H, K, V = 1, 3, 2, 4, 4
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.random.normal((B, T, H))
    g = mx.random.normal((B, T, H))
    o, h = naive_recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
    assert h is not None
    assert h.shape == (B, H, K, V)


def test_naive_recurrent_initial_state_carries():
    B, T, H, K, V = 1, 2, 1, 2, 2
    q = mx.zeros((B, T, H, K))
    k = mx.zeros((B, T, H, K))
    v = mx.zeros((B, T, H, V))
    beta = mx.zeros((B, T, H))
    g = mx.zeros((B, T, H))  # alpha = 1 — no decay
    init = mx.array([[[[3.0, 0.0], [0.0, 0.0]]]])  # [B,H,K,V] = [1,1,2,2]
    # With q=k=v=beta=0 and alpha=1, state stays = initial_state; output
    # = state^T q = 0 since q=0. Test that final state equals initial.
    o, h = naive_recurrent_gated_delta_rule(
        q, k, v, beta, g, initial_state=init, output_final_state=True
    )
    np.testing.assert_allclose(np.array(o), np.zeros((B, T, H, V)), atol=1e-6)
    np.testing.assert_allclose(np.array(h), np.array(init), atol=1e-6)


# ----- parity vs original PyTorch FLA reference -----


@pytest.fixture(scope="module")
def fla_naive():
    """Load fla-org's PyTorch naive_recurrent_gated_delta_rule for parity."""
    torch = pytest.importorskip("torch")
    repo = Path("/Volumes/external/sources/rent_kernels/flash-linear-attention")
    if not repo.exists():
        pytest.skip("FLA repo not present at expected path")
    sys.path.insert(0, str(repo))
    try:
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule as fla_fn
    except Exception as exc:
        pytest.skip(f"could not import FLA naive: {exc}")
    finally:
        # Keep on sys.path so subsequent tests in the same session can find it.
        pass
    return torch, fla_fn


def test_parity_with_fla_torch_naive(fla_naive):
    """MLX port must produce numerically equal output to the PyTorch reference."""
    torch, fla_fn = fla_naive
    B, T, H, K, V = 1, 6, 2, 4, 4

    rng = np.random.default_rng(7)
    q_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
    k_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
    v_np = rng.standard_normal((B, T, H, V)).astype(np.float32)
    beta_np = rng.standard_normal((B, T, H)).astype(np.float32)
    g_np = rng.standard_normal((B, T, H)).astype(np.float32) * 0.1  # small g

    q_t = torch.from_numpy(q_np)
    k_t = torch.from_numpy(k_np)
    v_t = torch.from_numpy(v_np)
    beta_t = torch.from_numpy(beta_np)
    g_t = torch.from_numpy(g_np)
    o_t, _ = fla_fn(q_t, k_t, v_t, beta_t, g_t)
    o_t_np = o_t.detach().cpu().numpy()

    q_m = mx.array(q_np)
    k_m = mx.array(k_np)
    v_m = mx.array(v_np)
    beta_m = mx.array(beta_np)
    g_m = mx.array(g_np)
    o_m, _ = naive_recurrent_gated_delta_rule(q_m, k_m, v_m, beta_m, g_m)
    o_m_np = np.array(o_m)

    np.testing.assert_allclose(o_m_np, o_t_np, rtol=1e-4, atol=1e-5)


# ----- LinearAttentionBlock -----


def test_block_forward_shape():
    cfg = _tiny_cfg()
    block = LinearAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    assert out.shape == x.shape


def test_block_rejects_wrong_rank_or_dim():
    block = LinearAttentionBlock(_tiny_cfg())
    with pytest.raises(ValueError, match="must be shaped"):
        block(mx.zeros((4, 16)))
    with pytest.raises(ValueError, match="last dim must be"):
        block(mx.zeros((1, 4, 17)))


def test_block_is_identity_at_init():
    """o_proj zero-init => output is zero (RMSNorm of zero stays zero)."""
    cfg = _tiny_cfg()
    block = LinearAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    np.testing.assert_allclose(np.array(out), np.zeros_like(np.array(x)), atol=1e-5)


def test_block_short_conv_runs():
    cfg = _tiny_cfg(use_short_conv=True, conv_size=3)
    block = LinearAttentionBlock(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = block(x)
    assert out.shape == x.shape


def test_block_doc_ids_changes_output():
    cfg = _tiny_cfg()
    block = LinearAttentionBlock(cfg)
    # Give o_proj a small non-zero weight so we see a diff.
    block.o_proj.weight = mx.random.normal(block.o_proj.weight.shape) * 0.1
    x = mx.random.normal((1, 6, cfg.hidden_size))
    out_no_doc = block(x)
    out_with_doc = block(x, doc_ids=mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32))
    # First doc: matches.
    np.testing.assert_allclose(
        np.array(out_no_doc[:, :3, :]), np.array(out_with_doc[:, :3, :]), atol=1e-5
    )
    # Second doc: differs (state was reset).
    assert np.abs(
        np.array(out_no_doc[:, 3:, :]) - np.array(out_with_doc[:, 3:, :])
    ).max() > 1e-5


def test_block_doc_ids_shape_validation():
    block = LinearAttentionBlock(_tiny_cfg())
    x = mx.random.normal((1, 6, 16))
    with pytest.raises(ValueError, match="doc_ids"):
        block(x, doc_ids=mx.array([0, 0, 0, 1, 1, 1], dtype=mx.int32))  # not 2D


def test_block_gradient_flows():
    cfg = _tiny_cfg()
    block = LinearAttentionBlock(cfg)
    block.o_proj.weight = mx.random.normal(block.o_proj.weight.shape) * 0.1
    x = mx.random.normal((1, 4, cfg.hidden_size))

    def loss_fn(params):
        block.update(params)
        return mx.mean(mx.square(block(x)))

    grads = mx.grad(loss_fn)(block.trainable_parameters())
    for name in ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "o_proj"):
        g = grads[name]["weight"]
        assert float(mx.max(mx.abs(g)).item()) > 0.0, f"{name} got zero grad"
