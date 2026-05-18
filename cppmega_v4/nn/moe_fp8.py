"""FP8 expert layer + V4MoE checkpoint converter.

Wraps each MoE expert's gate_proj / up_proj / down_proj in an FP8 path
that uses the fused dequant+GEMM Metal kernel from
``cppmega_v4/_tilelang/fused_fp8_gemm.py``.

Two surfaces:
  - ``FP8FeedForwardExpert`` — drop-in nn.Module that replaces
    ``cppmega_mlx.nn.moe.FeedForwardExpert``. Weights are stored as
    uint8 fp8 + per-128-block fp32 scale_inv per projection.
  - ``convert_v4moe_to_fp8(moe)`` — in-place converter: replaces every
    ``moe.experts[i]`` with an ``FP8FeedForwardExpert`` whose weights
    are quantized from the original bf16/fp32 linears via
    ``quantize_linear_to_fp8``.

Numeric tolerance: fp8 e4m3 round-trip ≈ 2% on per-row activations,
so quant→fp8→dequant→matmul matches the original bf16 matmul to
atol ~5e-2 on random inputs. Training-time pattern is the standard
"bf16 master weight + fp8 inference" loop: keep the master in bf16,
re-quantize to fp8 each step for the forward, and take gradients
through the bf16 master (covered by
``tests/v4/test_moe_fp8.py::test_fp8_fwd_with_bf16_master_weight_training_step``).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4._tilelang.fused_fp8_gemm import fused_fp8_gemm
from cppmega_v4.nn._external._mlx_lm_fp8_dequant_vendored import dequant_block_fp8

_BLOCK = 128


def quantize_linear_to_fp8(weight: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize a [out, in] fp32/bf16 weight to fp8 + per-block scale_inv.

    Returns (w_fp8: uint8 [out, in], scale_inv: fp32 [ceil(out/128), ceil(in/128)]).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out, in]; got {weight.shape}")
    w = weight.astype(mx.float32)
    out_dim, in_dim = w.shape
    bs = _BLOCK
    pad_o = (-out_dim) % bs
    pad_i = (-in_dim) % bs
    blocks_o = (out_dim + pad_o) // bs
    blocks_i = (in_dim + pad_i) // bs
    padded = mx.pad(w, ((0, pad_o), (0, pad_i)))
    blocks = padded.reshape(blocks_o, bs, blocks_i, bs)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=(1, 3), keepdims=False), 1e-6)
    scale_inv = (amax / 448.0).astype(mx.float32)
    scaled = (blocks / scale_inv[:, None, :, None]).reshape(
        out_dim + pad_o, in_dim + pad_i,
    )[:out_dim, :in_dim]
    fp8 = mx.to_fp8(scaled.astype(mx.bfloat16))
    return fp8, scale_inv


class FP8Linear(nn.Module):
    """nn.Linear analogue with fp8-stored weight, fused dequant+GEMM forward."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialise with zero weights — caller must load via load_fp8 or
        # use ``from_linear`` to quantize an existing nn.Linear.
        self.weight_fp8 = mx.zeros((out_features, in_features), dtype=mx.uint8)
        blocks_o = (out_features + _BLOCK - 1) // _BLOCK
        blocks_i = (in_features + _BLOCK - 1) // _BLOCK
        self.weight_scale_inv = mx.ones((blocks_o, blocks_i), dtype=mx.float32)
        self.bias = (
            mx.zeros((out_features,), dtype=mx.bfloat16)
            if bias else None
        )

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "FP8Linear":
        out_dim, in_dim = lin.weight.shape
        has_bias = getattr(lin, "bias", None) is not None
        mod = cls(in_dim, out_dim, bias=has_bias)
        mod.weight_fp8, mod.weight_scale_inv = quantize_linear_to_fp8(lin.weight)
        if has_bias:
            mod.bias = lin.bias.astype(mx.bfloat16)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        out = fused_fp8_gemm(self.weight_fp8, self.weight_scale_inv, x)
        if self.bias is not None:
            out = out + self.bias.astype(out.dtype)
        return out


class FP8FeedForwardExpert(nn.Module):
    """Drop-in for ``FeedForwardExpert`` with fp8 weights + fused GEMM."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        *,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.gate_proj = FP8Linear(d_model, hidden_size, bias=bias)
        self.up_proj = (
            FP8Linear(d_model, hidden_size, bias=bias)
            if activation == "swiglu"
            else None
        )
        self.down_proj = FP8Linear(hidden_size, d_model, bias=bias)

    @classmethod
    def from_fp32_expert(cls, expert: nn.Module) -> "FP8FeedForwardExpert":
        """Quantize an existing FeedForwardExpert in-place style (returns new module)."""
        d_in = expert.gate_proj.weight.shape[1]
        hidden = expert.gate_proj.weight.shape[0]
        has_up = getattr(expert, "up_proj", None) is not None
        activation = getattr(expert, "activation", "swiglu" if has_up else "gelu")
        bias = expert.gate_proj.bias is not None if hasattr(expert.gate_proj, "bias") else False
        mod = cls(d_in, hidden, activation=activation, bias=bias)
        mod.gate_proj = FP8Linear.from_linear(expert.gate_proj)
        if has_up:
            mod.up_proj = FP8Linear.from_linear(expert.up_proj)
        mod.down_proj = FP8Linear.from_linear(expert.down_proj)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_proj(x)
        if self.activation == "swiglu":
            assert self.up_proj is not None
            h = nn.silu(h) * self.up_proj(x)
        elif self.activation == "relu2":
            h = mx.square(mx.maximum(h, mx.array(0.0, dtype=h.dtype)))
        else:
            h = nn.gelu_approx(h)
        return self.down_proj(h)


def convert_v4moe_to_fp8(moe) -> None:
    """In-place convert every expert in a V4MoE to FP8FeedForwardExpert.

    The shared_expert (if any) is also converted.
    """
    new_experts = [FP8FeedForwardExpert.from_fp32_expert(e) for e in moe.experts]
    moe.experts = new_experts
    if getattr(moe, "shared_expert", None) is not None:
        moe.shared_expert = FP8FeedForwardExpert.from_fp32_expert(moe.shared_expert)


__all__ = [
    "FP8FeedForwardExpert",
    "FP8Linear",
    "convert_v4moe_to_fp8",
    "quantize_linear_to_fp8",
]
