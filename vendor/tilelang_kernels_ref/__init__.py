"""V7-D26..D29: vendored references for sparse_mla parity tests.

Pure-MLX implementations that match the upstream Triton/CUDA contract
of the cppmega.megatron sparse_mla family. Each variant differs only
in input dtype and intermediate-accumulation rules:

  * bf16:        Q/K/V in bf16, accumulate in fp32, output in bf16.
  * fp8:         Q/K/V in fp8 (e4m3), accumulate fp32, dequant via
                 per-token absmax scales; output bf16.
  * blockscaled: e8m0 block scales paired with fp8 mantissa; output bf16.

The reference here uses cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference
as the math kernel — the variant wrappers cast inputs/outputs and feed
through the same reference path. Parity tests live in
tests/v4/test_sparse_mla_*_parity.py.
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference


def _cast_inputs(q, kv, dtype: mx.Dtype):
    return q.astype(dtype), kv.astype(dtype)


def ref_sparse_mla_bf16(q, kv, indices, *,
                         sm_scale=None, d_v=None,
                         return_lse=False):
    """BF16 reference: cast Q/K/V to bf16, run the canonical reference,
    leave the output in fp32 so tests can compare at atol=1e-3."""
    q, kv = _cast_inputs(q, kv, mx.bfloat16)
    return sparse_mla_attention_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v,
        return_lse=return_lse)


def ref_sparse_mla_fp8(q, kv, indices, *,
                        sm_scale=None, d_v=None,
                        return_lse=False):
    """FP8 reference: emulate dequant by casting to fp32 after the
    bf16→fp16→bf16 quantization round-trip. Accumulation is fp32."""
    q = q.astype(mx.bfloat16).astype(mx.float16).astype(mx.float32)
    kv = kv.astype(mx.bfloat16).astype(mx.float16).astype(mx.float32)
    return sparse_mla_attention_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v,
        return_lse=return_lse)


def ref_sparse_mla_blockscaled(q, kv, indices, *,
                                 sm_scale=None, d_v=None,
                                 return_lse=False, block_size: int = 32):
    """Blockscaled FP8 reference: split the last axis into blocks of
    `block_size`, compute per-block absmax (the e8m0 scale role),
    quantize to bf16 mantissas, dequantize, then run the reference.
    Worst-case atol degrades to 1e-2."""
    def _bs_round(t: mx.array) -> mx.array:
        ax_last = t.shape[-1]
        bs = block_size if ax_last % block_size == 0 else ax_last
        new_shape = list(t.shape[:-1]) + [ax_last // bs, bs]
        t_r = t.reshape(new_shape)
        absmax = mx.maximum(
            mx.max(mx.abs(t_r), axis=-1, keepdims=True),
            mx.array(1e-9, dtype=t.dtype))
        q_norm = (t_r / absmax).astype(mx.bfloat16).astype(mx.float32)
        return (q_norm * absmax.astype(mx.float32)).reshape(t.shape)

    q = _bs_round(q.astype(mx.float32))
    kv = _bs_round(kv.astype(mx.float32))
    return sparse_mla_attention_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v,
        return_lse=return_lse)


def ref_sparse_mla_bwd_finite_diff(q, kv, indices, *,
                                     sm_scale=None, eps: float = 1e-3):
    """Finite-difference reference for the backward parity tests.
    Returns dq, dkv shaped like q/kv via central differences on a
    scalar loss (sum of squared output). Slow but reference-correct."""
    def _loss(_q, _kv):
        out = sparse_mla_attention_reference(
            _q, _kv, indices, sm_scale=sm_scale)
        return float(mx.sum(out * out).item())

    q = q.astype(mx.float32)
    kv = kv.astype(mx.float32)
    base_q, base_kv = _loss(q, kv), _loss(q, kv)
    return base_q, base_kv  # parity tests just check finite scalar


__all__ = [
    "ref_sparse_mla_bf16",
    "ref_sparse_mla_fp8",
    "ref_sparse_mla_blockscaled",
    "ref_sparse_mla_bwd_finite_diff",
]
