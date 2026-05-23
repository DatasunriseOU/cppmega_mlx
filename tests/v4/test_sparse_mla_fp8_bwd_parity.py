"""V7-D28 (za1.6): sparse_mla FP8 backward grad finite + bounded."""

from __future__ import annotations

import mlx.core as mx

from vendor.tilelang_kernels_ref import ref_sparse_mla_fp8


def test_sparse_mla_fp8_bwd_grad_finite():
    B, S, Skv, H, G, D, topk = 1, 4, 8, 2, 1, 16, 4
    q = mx.random.normal(shape=(B, S, H, D), key=mx.random.key(0)).astype(
        mx.float32)
    kv = mx.random.normal(shape=(B, Skv, G, D), key=mx.random.key(1)).astype(
        mx.float32)
    indices = mx.random.randint(0, Skv, shape=(B, S, G, topk),
                                  key=mx.random.key(2))

    def loss_fn(q_arg):
        out = ref_sparse_mla_fp8(q_arg, kv, indices)
        return mx.sum(out * out)

    dq = mx.grad(loss_fn)(q)
    mx.eval(dq)
    g_max = float(mx.max(mx.abs(dq)).item())
    assert g_max > 0.0 and g_max < 1e6
