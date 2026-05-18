"""Fused FP8 GEMM Metal kernel — block-fp8 weight × bf16/fp32 activation.

Used by ROI 7 (Lightning Indexer FP8) to fuse dequant + GEMM instead of
materializing the full bf16 weight tile on every forward.

Layout:
  - W_fp8: [M, K] uint8 (e4m3 storage); each [128, 128] block has a single
    fp32 scale_inv stored in W_scale_inv: [ceil(M/128), ceil(K/128)].
  - A:     [..., K] fp32/bf16 (caller's choice — internally promoted to fp32).
  - Out:   [..., M] same dtype as A.

The kernel computes (per output column m):

    acc = 0
    for k in [0, K):
        block_m = m // 128
        block_k = k // 128
        acc += dequant(W_fp8[m, k], W_scale_inv[block_m, block_k]) * A[..., k]
    Out[..., m] = acc

One thread per output column m × batch row. Block_size=128 amortizes the
scale_inv lookup across 128 K-elements per (m, batch_row).
"""

from typing import Tuple

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel

_BLOCK = 128


def fused_fp8_gemm(
    w_fp8: mx.array,
    w_scale_inv: mx.array,
    a: mx.array,
) -> mx.array:
    """Fused dequant + GEMM: out = a @ W.T where W = dequant(w_fp8, w_scale_inv).

    Args:
        w_fp8: [M, K] uint8 fp8 weight.
        w_scale_inv: [ceil(M/128), ceil(K/128)] fp32 per-block inverse-scale.
        a: [..., K] bf16/fp32 activation.

    Returns:
        out: [..., M] same dtype as a.
    """
    if w_fp8.dtype != mx.uint8:
        raise TypeError(f"w_fp8 must be uint8 (fp8 storage); got {w_fp8.dtype}")
    if w_fp8.ndim != 2:
        raise ValueError(f"w_fp8 must be 2D [M, K]; got {w_fp8.shape}")
    M, K = w_fp8.shape
    if a.shape[-1] != K:
        raise ValueError(f"a.shape[-1] ({a.shape[-1]}) must equal w_fp8.shape[1] ({K})")
    bs = _BLOCK
    blocks_m_expected = (M + bs - 1) // bs
    blocks_k_expected = (K + bs - 1) // bs
    if w_scale_inv.shape != (blocks_m_expected, blocks_k_expected):
        raise ValueError(
            f"w_scale_inv shape {w_scale_inv.shape} != expected "
            f"({blocks_m_expected}, {blocks_k_expected}) for W={M, K} with block={bs}"
        )

    out_dtype = a.dtype
    a_fp32 = a.astype(mx.float32)
    # Flatten activations to [N, K] for the kernel.
    leading = a.shape[:-1]
    n = 1
    for d in leading:
        n *= d
    a_flat = a_fp32.reshape(n, K)

    blocks_m = blocks_m_expected
    blocks_k = blocks_k_expected
    w_flat = w_fp8.reshape(-1)             # [M*K]
    s_flat = w_scale_inv.reshape(-1)        # [blocks_m * blocks_k]
    a_flat_1d = a_flat.reshape(-1)         # [N*K]

    # Convert per-row scale_inv lookup to a flat index inside the kernel.
    # Each thread = (m, row); inner loop over k, reads s_flat[(m/128)*blocks_k + (k/128)]
    # and the corresponding fp8 byte from w_flat[m*K + k].
    source = f"""
        uint m   = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (m >= {M}u || row >= {n}u) return;

        uint block_m = m / {bs}u;
        float acc = 0.0f;

        // Inner K-loop: unrolled per-block to amortize scale_inv lookup.
        for (uint kb = 0; kb < {blocks_k}u; ++kb) {{
            float scale = s_flat[block_m * {blocks_k}u + kb];
            uint k_start = kb * {bs}u;
            uint k_end_full = k_start + {bs}u;
            uint k_end = k_end_full < {K}u ? k_end_full : {K}u;
            for (uint k = k_start; k < k_end; ++k) {{
                // mx.from_fp8 conversion: e4m3 fp8 byte → fp32.
                // Reproduce the bit-decoder inline so this kernel stays
                // self-contained (no MLX op call inside MSL).
                uint  byte = (uint)w_flat[m * {K}u + k];
                int   sign = (byte >> 7) & 0x1;
                int   expt = (byte >> 3) & 0xF;
                int   mant = byte & 0x7;
                float val;
                if (expt == 0) {{
                    // subnormal: val = (-1)^s * 2^-6 * (mant/8)
                    val = (float)mant / 8.0f * 0.015625f;  // 2^-6 = 1/64
                }} else if (expt == 0xF && mant == 0x7) {{
                    // NaN
                    val = 0.0f;  // treat as zero in GEMM accumulation
                }} else {{
                    // normal: val = (-1)^s * 2^(expt-7) * (1 + mant/8)
                    float mantissa = 1.0f + (float)mant / 8.0f;
                    int   bias_exp = expt - 7;
                    // 2^bias_exp via metal::ldexp.
                    val = metal::ldexp(mantissa, bias_exp);
                }}
                if (sign) val = -val;
                acc += val * scale * a_flat[row * {K}u + k];
            }}
        }}
        out[row * {M}u + m] = acc;
    """

    name = f"v4_fused_fp8_gemm_{M}_{K}_{n}"
    kernel = get_or_build_kernel(
        name=name,
        input_names=["w_flat", "s_flat", "a_flat"],
        output_names=["out"],
        source=source,
    )

    grid = (M, n, 1)
    tg_x = min(M, 32)
    threadgroup = (tg_x, 1, 1)

    (out_flat,) = kernel(
        inputs=[w_flat, s_flat, a_flat_1d],
        output_shapes=[(n * M,)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    out = out_flat.reshape(*leading, M).astype(out_dtype)
    return out


__all__ = ["fused_fp8_gemm"]
