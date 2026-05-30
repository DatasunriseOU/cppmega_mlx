# pyright: reportMissingImports=false
"""MXFP4 (e2m1 OCP MX block-16) Path C: real dequant + cooperative GEMM.

Tier-3 Path C scaffold #10. MXFP4 is the OCP MX 4-bit float codec: a
canonical ``e2m1`` 4-bit mantissa (codebook ``±{0,.5,1,1.5,2,3,4,6}``,
sign in the top nibble bit) plus one per-block-16 scale. Storage is
packed ``uint8`` (two nibbles per byte) + a per-block scale tensor.

There is **no native Apple FP4 ALU** and no end-to-end ``float4_e2m1``
TVM dtype usable through the MLX/tvm-ffi ABI (only ``float4_e2m1fn``
exists, and it has no Metal codegen decode helper). So -- exactly like
the dense FP8 matmul2d surface dequants ``e4m3 -> half`` into the
cooperative-input shared buffer before ``T.gemm`` -- MXFP4 dequants the
e2m1 block payload to ``half`` **at the MLX boundary** and feeds the
*already numerically-correct* Metal-4 cooperative-tensor ``matmul2d``
GEMM (``make_sparse_mla_grouped_gemm_matmul2d_kernel``, committed and
verified exact on M4). No new codegen intrinsic is needed; this dodges
the in-kernel ``T.cast`` miscompile the same way ``fused_fp8_gemm`` did.

The dequant here is a **vectorized MLX gather** (LUT index + per-block
scale broadcast), not the per-element Python loop in
``cppmega_mlx.quant.mxfp4_metal`` -- that reference codec stays the
parity oracle; this is the GPU-feedable fast path.

RULE #1 -- no silent fallback. Unsupported shapes/dtypes RAISE with
where + what. A GEMM shape that admits no legal cooperative tile RAISES
(it is NOT silently routed to a degraded path); the OCP block size is
fixed at 16 and any other block size RAISES.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.quant.mxfp4_metal import (
    MXFP4_BLOCK_SIZE,
    MXFP4_LUT,
)


__all__ = [
    "MXFP4MatmulPathCError",
    "mxfp4_dequant_to_half",
    "mxfp4_dequant_blockwise_2d",
    "mxfp4_matmul_path_c",
    "mxfp4_matmul_path_c_status",
]


class MXFP4MatmulPathCError(RuntimeError):
    """Raised when the MXFP4 Path C route cannot run (shape/dtype/Metal)."""


# The 16-entry e2m1 codebook as an MLX fp32 constant, so the nibble->value
# decode is a single vectorized ``mx`` gather instead of a numpy loop.
_MXFP4_LUT_MX = mx.array(MXFP4_LUT.astype("float32"))


def _unpack_nibbles_last_axis(qdata: mx.array, *, n_along_axis: int) -> mx.array:
    """Unpack a packed-nibble ``uint8`` payload into ``uint8`` nibble indices.

    ``qdata`` packs two e2m1 nibbles per byte (low nibble first), matching
    ``cppmega_mlx.quant.mxfp4_metal._encode_block``. The last axis of
    ``qdata`` is ``ceil(n_along_axis / 2)`` bytes; this returns an array
    whose last axis is ``n_along_axis`` nibble indices in ``[0, 15]``.
    """

    if qdata.dtype != mx.uint8:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: packed payload must be mx.uint8; got {qdata.dtype}"
        )
    low = mx.bitwise_and(qdata, mx.array(0x0F, dtype=mx.uint8))
    high = mx.bitwise_and(
        mx.right_shift(qdata, mx.array(4, dtype=mx.uint8)),
        mx.array(0x0F, dtype=mx.uint8),
    )
    # Interleave low,high along a new trailing axis then flatten: [..., bytes, 2].
    interleaved = mx.stack([low, high], axis=-1)
    flat = interleaved.reshape(*interleaved.shape[:-2], -1)
    if flat.shape[-1] < n_along_axis:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: packed payload has {flat.shape[-1]} nibbles on the "
            f"last axis but {n_along_axis} were requested (truncated payload)"
        )
    # Trim the padding nibble when n_along_axis is odd.
    return flat[..., :n_along_axis]


def mxfp4_dequant_to_half(
    nibbles: mx.array,
    scales: mx.array,
    *,
    block_size: int = MXFP4_BLOCK_SIZE,
    out_dtype: Any = mx.float16,
) -> mx.array:
    """Decode already-unpacked e2m1 nibble indices + per-block scales -> dtype.

    ``nibbles`` is an integer array of LUT indices in ``[0, 15]`` whose last
    axis is a multiple of ``block_size``. ``scales`` carries one scale per
    block; its last axis is ``nibbles.shape[-1] // block_size`` and its
    leading axes broadcast against ``nibbles``' leading axes.
    """

    if block_size != MXFP4_BLOCK_SIZE:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: block_size={block_size} unsupported; OCP MX fixes "
            f"the block at {MXFP4_BLOCK_SIZE}"
        )
    n_last = int(nibbles.shape[-1])
    if n_last % block_size != 0:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: last axis {n_last} is not a multiple of the "
            f"block size {block_size}"
        )
    n_blocks = n_last // block_size
    if int(scales.shape[-1]) != n_blocks:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: scales last axis {int(scales.shape[-1])} != "
            f"n_blocks {n_blocks} (= {n_last}/{block_size})"
        )
    # LUT gather: nibble index -> signed codebook magnitude.
    idx = nibbles.astype(mx.int32)
    decoded = _MXFP4_LUT_MX[idx]  # fp32, same shape as nibbles
    # Broadcast the per-block scale across the 16 elements of each block.
    lead = tuple(decoded.shape[:-1])
    decoded_b = decoded.reshape(*lead, n_blocks, block_size)
    scale_b = scales.astype(mx.float32).reshape(*lead, n_blocks, 1)
    scaled = (decoded_b * scale_b).reshape(*lead, n_last)
    return scaled.astype(out_dtype)


def mxfp4_dequant_blockwise_2d(
    qdata: mx.array,
    scales: mx.array,
    *,
    rows: int,
    cols: int,
    block_size: int = MXFP4_BLOCK_SIZE,
    out_dtype: Any = mx.float16,
) -> mx.array:
    """Dequant a packed 2D MXFP4 matrix ``(rows, cols)`` to ``out_dtype``.

    Blocks tile the **last** axis (the GEMM contraction axis ``K``), so
    ``cols`` MUST be a multiple of ``block_size``. ``qdata`` is the packed
    ``uint8`` payload reshaped to ``(rows, cols // 2)`` (two nibbles/byte);
    ``scales`` is ``(rows, cols // block_size)``. Returns ``(rows, cols)``.
    """

    if block_size != MXFP4_BLOCK_SIZE:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: block_size={block_size} unsupported; OCP MX fixes "
            f"the block at {MXFP4_BLOCK_SIZE}"
        )
    if cols % block_size != 0:
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: contraction dim cols={cols} must be a multiple of "
            f"the block size {block_size} (blocks tile the K axis)"
        )
    bytes_per_row = (cols + 1) // 2
    if tuple(qdata.shape) != (rows, bytes_per_row):
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: packed qdata shape {tuple(qdata.shape)} != expected "
            f"({rows}, {bytes_per_row}) for a ({rows}, {cols}) matrix"
        )
    n_blocks = cols // block_size
    if tuple(scales.shape) != (rows, n_blocks):
        raise MXFP4MatmulPathCError(
            f"mxfp4 dequant: scales shape {tuple(scales.shape)} != expected "
            f"({rows}, {n_blocks})"
        )
    nibbles = _unpack_nibbles_last_axis(qdata, n_along_axis=cols)
    return mxfp4_dequant_to_half(
        nibbles, scales, block_size=block_size, out_dtype=out_dtype
    )


def mxfp4_matmul_path_c(
    A_qdata: mx.array,
    A_scales: mx.array,
    B_qdata: mx.array,
    B_scales: mx.array,
    *,
    M: int,
    N: int,
    K: int,
    out: mx.array,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> mx.array:
    """MXFP4 ``A(M,K) @ B(N,K).T -> C(M,N)`` through the Path C cooperative GEMM.

    ``A_qdata``/``B_qdata`` are packed ``uint8`` e2m1 payloads (rows ``M``/``N``,
    each row ``ceil(K/2)`` bytes); ``A_scales``/``B_scales`` are the per-block-16
    fp32 scales (``(M, K//16)`` / ``(N, K//16)``). ``out`` is the caller-owned
    fp32 ``(M, N)`` output.

    The e2m1 blocks are dequantized to ``half`` at the MLX boundary and fed to
    the proven Metal-4 cooperative-tensor ``matmul2d`` GEMM
    (``sparse_mla_grouped_qk_path_c``, ``transpose_B=True``). Shape dispatch is
    explicit: ``M%16==0, N%32==0, K%16==0`` and a legal cooperative tile MUST
    exist, else RAISE (RULE #1 -- no silent fallback to a degraded path).
    """

    if not _msl_transform.can_run_metal():
        raise MXFP4MatmulPathCError("mxfp4_matmul_path_c: MLX Metal unavailable")
    if block_size != MXFP4_BLOCK_SIZE:
        raise MXFP4MatmulPathCError(
            f"mxfp4_matmul_path_c: block_size={block_size} unsupported; OCP MX "
            f"fixes the block at {MXFP4_BLOCK_SIZE}"
        )
    if A_qdata.dtype != mx.uint8 or B_qdata.dtype != mx.uint8:
        raise MXFP4MatmulPathCError(
            "mxfp4_matmul_path_c: packed payloads must be mx.uint8; got "
            f"A={A_qdata.dtype} B={B_qdata.dtype}"
        )
    if tuple(out.shape) != (M, N) or out.dtype != mx.float32:
        raise MXFP4MatmulPathCError(
            f"mxfp4_matmul_path_c: out must be fp32 shape ({M}, {N}); got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    # Explicit cooperative-tile dispatch BEFORE any dequant work -- a sub-tile
    # shape RAISES here, it is never silently routed to a slow scalar path.
    from cppmega_mlx.nn._tilelang.fp8_matmul_path_c import _coop_tile_for

    coop = _coop_tile_for(int(M), int(N), int(K))
    if coop is None:
        raise MXFP4MatmulPathCError(
            f"mxfp4_matmul_path_c: GEMM shape M={M} N={N} K={K} admits no legal "
            "Metal-4 cooperative tile (needs M%16==0, N%32==0, K%16==0). MXFP4 "
            "has no native Apple FP4 ALU; sub-tile shapes are NOT served by a "
            "degraded path -- dequant via mxfp4_dequant_blockwise_2d and use the "
            "pure-MLX reference matmul, or pad to a cooperative-tileable shape."
        )

    A_half = mxfp4_dequant_blockwise_2d(
        A_qdata, A_scales, rows=int(M), cols=int(K),
        block_size=block_size, out_dtype=mx.float16,
    )
    B_half = mxfp4_dequant_blockwise_2d(
        B_qdata, B_scales, rows=int(N), cols=int(K),
        block_size=block_size, out_dtype=mx.float16,
    )
    mx.eval(A_half, B_half)

    # Reuse the committed, numerically-exact cooperative-tensor matmul2d GEMM
    # (transpose_B=True: A(M,K) @ B(N,K).T -> C(M,N)). It owns the same
    # _coop_tile_for dispatch and RAISES on sub-tile shapes.
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import (
        sparse_mla_grouped_qk_path_c,
    )

    try:
        return sparse_mla_grouped_qk_path_c(A_half, B_half, out)
    except Exception as exc:
        raise MXFP4MatmulPathCError(
            f"mxfp4_matmul_path_c: cooperative matmul2d GEMM failed for "
            f"M={M} N={N} K={K} ({type(exc).__name__}: {exc})"
        ) from exc


def mxfp4_matmul_path_c_status() -> dict[str, Any]:
    """Lightweight availability probe (mirrors the FP8 status surface)."""

    if not _msl_transform.can_run_metal():
        return {"available": False, "reason": "MLX Metal unavailable"}
    try:
        import tilelang  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"tilelang import failed: {exc}"}
    return {
        "available": True,
        "reason": "MXFP4 e2m1 dequant + cooperative matmul2d GEMM dispatchable",
        "codec": "mxfp4_e2m1_v1",
        "block_size": MXFP4_BLOCK_SIZE,
        "gemm": "sparse_mla_grouped matmul2d (transpose_B=True)",
        "needs_codegen_intrinsic": False,
    }
