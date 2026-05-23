"""V7-D04: HybridLM.set_dtype('hybrid') works end-to-end."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.runtime.hybrid_lm import HybridLM


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


def test_default_mode_fp32():
    lm = HybridLM(_TinyLM())
    assert lm.mode == "fp32"
    assert lm.master_dtype == mx.float32
    assert lm.fwd_dtype == mx.float32


def test_set_dtype_hybrid_keeps_master_fp32_fwd_bf16():
    lm = HybridLM(_TinyLM())
    lm.set_dtype("hybrid")
    assert lm.mode == "hybrid"
    assert lm.master_dtype == mx.float32
    assert lm.fwd_dtype == mx.bfloat16
    # Inner params still fp32 (master copy preserved).
    for _, v in nn.utils.tree_flatten(lm.inner.parameters()):
        assert v.dtype == mx.float32


def test_set_dtype_bf16_casts_inner_params():
    lm = HybridLM(_TinyLM())
    lm.set_dtype("bf16")
    for _, v in nn.utils.tree_flatten(lm.inner.parameters()):
        assert v.dtype == mx.bfloat16


def test_set_dtype_rejects_unknown_mode():
    lm = HybridLM(_TinyLM())
    try:
        lm.set_dtype("int8")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "unsupported HybridLM mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_forward_returns_real_output():
    lm = HybridLM(_TinyLM())
    lm.set_dtype("hybrid")
    x = mx.ones((1, 8), dtype=mx.float32)
    out = lm(x)
    assert out.shape == (1, 8)
    # After forward, master params restored to fp32.
    for _, v in nn.utils.tree_flatten(lm.inner.parameters()):
        assert v.dtype == mx.float32


def test_hybrid_value_and_grad_returns_master_dtype_grads():
    lm = HybridLM(_TinyLM())
    lm.set_dtype("hybrid")

    def _loss(model: nn.Module, x: mx.array) -> mx.array:
        return mx.mean(model(x) ** 2)

    lvg = lm.value_and_grad(_loss)
    x = mx.ones((1, 8), dtype=mx.float32)
    loss, grads = lvg(x)
    mx.eval(loss, grads)
    assert loss.dtype == mx.float32 or loss.dtype == mx.bfloat16
    for _, g in nn.utils.tree_flatten(grads):
        # Master dtype (fp32) per the cast_grads_to_master pass.
        assert g.dtype == mx.float32
