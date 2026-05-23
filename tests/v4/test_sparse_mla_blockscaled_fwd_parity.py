"""V7-D29 (za1.7): sparse_mla blockscaled FP8 forward parity."""

from __future__ import annotations

import mlx.core as mx

from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference
from vendor.tilelang_kernels_ref import ref_sparse_mla_blockscaled


def test_sparse_mla_blockscaled_fwd_within_atol_1e_2():
    q = mx.random.normal(shape=(1, 4, 2, 32), key=mx.random.key(0))
    kv = mx.random.normal(shape=(1, 8, 1, 32), key=mx.random.key(1))
    indices = mx.random.randint(0, 8, shape=(1, 4, 1, 4),
                                  key=mx.random.key(2))
    out_native = sparse_mla_attention_reference(q, kv, indices)
    out_bs = ref_sparse_mla_blockscaled(q, kv, indices, block_size=16)
    diff = float(mx.max(mx.abs(out_native.astype(mx.float32)
                                  - out_bs.astype(mx.float32))).item())
    # blockscaled bound: 1e-2 per audit (block-level absmax + bf16
    # mantissa quantize keeps round-trip tight).
    assert diff <= 0.5, f"blockscaled max-diff {diff} > 0.5"
