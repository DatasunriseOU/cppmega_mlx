"""V7-D29 (za1.8): sparse_mla blockscaled FP8 backward parity."""

from __future__ import annotations

import mlx.core as mx

from vendor.tilelang_kernels_ref import ref_sparse_mla_blockscaled


def test_sparse_mla_blockscaled_bwd_grad_finite():
    q = mx.random.normal(shape=(1, 4, 2, 32), key=mx.random.key(0)).astype(
        mx.float32)
    kv = mx.random.normal(shape=(1, 8, 1, 32), key=mx.random.key(1)).astype(
        mx.float32)
    indices = mx.random.randint(0, 8, shape=(1, 4, 1, 4),
                                  key=mx.random.key(2))

    def loss_fn(q_arg):
        out = ref_sparse_mla_blockscaled(q_arg, kv, indices, block_size=16)
        return mx.sum(out * out)

    dq = mx.grad(loss_fn)(q)
    mx.eval(dq)
    g_max = float(mx.max(mx.abs(dq)).item())
    assert g_max > 0.0 and g_max < 1e6
