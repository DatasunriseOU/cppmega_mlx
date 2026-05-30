"""Fused FP8 GEMM Metal kernel — block-fp8 weight × bf16/fp32 activation.

Used by ROI 7 (Lightning Indexer FP8) and the V4 FP8 MoE experts to fuse
dequant + GEMM instead of materializing the full bf16 weight tile on every
forward.

Layout:
  - W_fp8: [M, K] uint8 (e4m3 storage); each [128, 128] block has a single
    fp32 scale_inv stored in W_scale_inv: [ceil(M/128), ceil(K/128)].
  - A:     [..., K] fp32/bf16 (caller's choice — internally promoted to half).
  - Out:   [..., M] same dtype as A.

The kernel computes (per output column m):

    acc = 0
    for k in [0, K):
        block_m = m // 128
        block_k = k // 128
        acc += dequant(W_fp8[m, k], W_scale_inv[block_m, block_k]) * A[..., k]
    Out[..., m] = acc

i.e. ``Out = A @ W.T`` where ``W = dequant(W_fp8, W_scale_inv)``.

Two implementations share the public ``fused_fp8_gemm`` entry point:

  * **TileLang cooperative-tensor path** (preferred).  A real ``@T.prim_func``
    GEMM (C in *shared* scope) feeding the cooperative-tensor ``T.gemm``
    (Metal simdgroup ``simdgroup_multiply_accumulate`` on M1-M4,
    ``mpp::tensor_ops::matmul2d`` cooperative tensor on M5+) — the same
    keystone GEMM validated in ``tilelang/tileop/gemm/gemm_metal.py``.  The
    block-FP8 weight is decoded e4m3 -> half and per-128-block scaled (via
    MLX ``mx.from_fp8``) into the half weight that populates the cooperative
    input tensor; the decode is intentionally done at the MLX boundary rather
    than inside the kernel because TileLang's Metal lowering compiles an
    in-kernel ``T.cast(float8_e4m3 -> float16)`` to an *integer* widen of the
    storage byte (not the signed e4m3 decode), which silently corrupts the
    result.  The compiled MSL is driven through MLX's native
    ``mx.fast.metal_kernel`` stream via the host ``wrap_tilelang_metal_kernel``
    adapter, so it shares MLX command buffers (the tvm-ffi / DLPack
    owner-output route is not used because this MLX build does not expose
    ``mx.metal._current_command_buffer``, so its kernels read uncommitted
    buffers — even the shipped ``fp8_matmul_path_c`` returns zeros here).

  * **Scalar hand-MSL fallback** (legacy).  One thread per output column ×
    batch row, used when TileLang is unimportable, the Metal compile callback
    is unavailable, or the prim fails to compile/run for a given shape.  This
    keeps the op fail-safe and bit-for-bit identical to the original scaffold.

Both paths use the identical e4m3 dequant arithmetic, so they agree to FP8
tolerance (~1e-2 absolute).  The callable signature is unchanged, so existing
callers (``moe_fp8.FP8Linear`` / ``FP8FeedForwardExpert`` and the future
``lightning_indexer_fp8`` fused projection) work without modification.
"""


import os
import threading

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel

_BLOCK = 128

# Cooperative-tensor GEMM tile.  32x32x32 is the size at which TileLang's
# Metal lowering emits ``simdgroup_multiply_accumulate`` (the M1-M4 cooperative
# simdgroup matmul) rather than the M5-only ``MetalPerformancePrimitives``
# ``matmul2d`` include — which keeps the kernel compilable through
# ``mx.fast.metal_kernel`` on current Apple silicon while staying numerically
# identical to the cooperative-tensor contract.
_TILE_M = 32  # over output features (M / weight rows)
_TILE_N = 32  # over batch rows (N)
_TILE_K = 32
_THREADS = 128

# Force the scalar fallback (debugging / hosts without a working TileLang
# Metal stack).  ``CPPMEGA_V4_FUSED_FP8_GEMM_SCALAR=1``.
_FORCE_SCALAR_ENV = "CPPMEGA_V4_FUSED_FP8_GEMM_SCALAR"

# Compiled-adapter cache keyed by padded (N, M, K) and output dtype token.
_TL_ADAPTER_CACHE: dict[tuple, object] = {}
_TL_ADAPTER_CACHE_LOCK = threading.RLock()
# One-shot probe of whether the TileLang cooperative path is usable here.
_TL_STATUS: tuple[bool, str] | None = None
_TL_STATUS_LOCK = threading.RLock()


def _force_scalar() -> bool:
    return os.environ.get(_FORCE_SCALAR_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _tilelang_status() -> tuple[bool, str]:
    """Probe (once) whether the TileLang cooperative-tensor path can run.

    Requires: tilelang importable, the default Metal compile callback
    registerable, and the host ``wrap_tilelang_metal_kernel`` MSL->mx.fast
    bridge reachable.
    """
    global _TL_STATUS
    if _TL_STATUS is not None:
        return _TL_STATUS
    with _TL_STATUS_LOCK:
        if _TL_STATUS is not None:
            return _TL_STATUS
        try:
            import tilelang  # noqa: F401
            import tilelang.language as _T  # noqa: F401
            from tilelang.engine.callback import (
                register_default_metal_compile_callback,
            )
            from cppmega_mlx.nn._tilelang._mlx_runtime import (
                wrap_tilelang_metal_kernel,  # noqa: F401
            )
        except Exception as exc:  # noqa: BLE001
            _TL_STATUS = (
                False,
                f"TileLang cooperative path unavailable: "
                f"{exc.__class__.__name__}: {exc}",
            )
            return _TL_STATUS
        try:
            # Idempotent; wires the MSL4 / cooperative-tensor compile callback
            # that makes the keystone GEMM numerically correct.
            register_default_metal_compile_callback(override=True)
        except Exception as exc:  # noqa: BLE001
            _TL_STATUS = (
                False,
                f"register_default_metal_compile_callback failed: "
                f"{exc.__class__.__name__}: {exc}",
            )
            return _TL_STATUS
        _TL_STATUS = (True, "TileLang cooperative-tensor FP8 GEMM available")
        return _TL_STATUS


def fused_fp8_gemm_status() -> tuple[bool, str]:
    """Public status hook: is the cooperative-tensor prim path live?"""
    if _force_scalar():
        return False, f"{_FORCE_SCALAR_ENV} set; scalar fallback forced"
    return _tilelang_status()


# ---------------------------------------------------------------------------
# TileLang cooperative-tensor prim
# ---------------------------------------------------------------------------


def _build_fp8_gemm_prim(N: int, M: int, K: int, c_dtype: str):
    """Build the cooperative-tensor GEMM ``@T.prim_func`` (half inputs).

    Computes ``C[N,M] = A[N,K] @ W[M,K].T`` with C in *shared* scope, feeding
    the cooperative-tensor ``T.gemm`` (Metal ``simdgroup_multiply_accumulate``
    on M1-M4, ``matmul2d`` cooperative tensor on M5+ — same keystone path as
    ``tilelang/tileop/gemm/gemm_metal.py``).

    The block-FP8 weight is decoded e4m3 -> half *and* block-scaled into ``W``
    by :func:`_dequant_weight_to_half` (via MLX ``mx.from_fp8``) before launch,
    so the dequantized half weight directly populates the cooperative input
    tensor.  An earlier attempt to dequant the e4m3 byte *inside* the kernel
    via ``T.cast(float8_e4m3 -> float16)`` is intentionally avoided: TileLang's
    Metal lowering compiles that cast to an integer ``(half4)uchar4`` widen of
    the storage byte rather than the signed e4m3 decode, which silently
    corrupts the result.  Driving the decode through ``mx.from_fp8`` keeps the
    arithmetic bit-identical to the scalar reference.

    Shapes are the *padded* dims: ``N`` (batch rows), ``M`` (output features)
    and ``K`` are all multiples of the GEMM tile.
    """
    import tilelang  # noqa: F401
    import tilelang.language as T

    TILE_M, TILE_N, TILE_K = _TILE_M, _TILE_N, _TILE_K

    @T.prim_func
    def fp8_gemm(
        A: T.Tensor((N, K), "float16"),
        W: T.Tensor((M, K), "float16"),
        C: T.Tensor((N, M), c_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, TILE_M), T.ceildiv(N, TILE_N), threads=_THREADS
        ) as (bx, by):
            A_sh = T.alloc_shared((TILE_N, TILE_K), "float16", scope="shared")
            W_sh = T.alloc_shared((TILE_M, TILE_K), "float16", scope="shared")
            C_sh = T.alloc_shared((TILE_N, TILE_M), "float32", scope="shared")
            T.clear(C_sh)
            for ko in T.Pipelined(T.ceildiv(K, TILE_K), num_stages=0):
                # Stage the activation and dequantized half weight into SMEM,
                # feeding the cooperative input tensors.
                T.copy(A[by * TILE_N, ko * TILE_K], A_sh)
                T.copy(W[bx * TILE_M, ko * TILE_K], W_sh)
                # Cooperative-tensor matmul: C += A @ W.T (C in shared scope).
                T.gemm(A_sh, W_sh, C_sh, transpose_B=True)
            T.copy(C_sh, C[by * TILE_N, bx * TILE_M])

    return fp8_gemm


def _dequant_weight_to_half(
    w_fp8: mx.array,
    w_scale_inv: mx.array,
    *,
    M: int,
    K: int,
    Mp: int,
    Kp: int,
) -> mx.array:
    """Decode e4m3 + per-128-block scale -> padded half weight ``[Mp, Kp]``.

    ``mx.from_fp8`` performs the exact e4m3 decode (bit-identical to the
    scalar kernel's inline decoder), then each ``[128,128]`` block is scaled
    by its ``scale_inv``.  Padding bytes / scale blocks are zero, so padded
    rows/cols contribute nothing to the accumulation.
    """
    bs = _BLOCK
    blocks_m = Mp // bs
    blocks_k = Kp // bs
    w_dec = mx.from_fp8(w_fp8.reshape(M, K), dtype=mx.float16)
    if Mp != M or Kp != K:
        w_dec = mx.pad(w_dec, ((0, Mp - M), (0, Kp - K)))
    s = w_scale_inv.astype(mx.float16)
    pad_sm = blocks_m - w_scale_inv.shape[0]
    pad_sk = blocks_k - w_scale_inv.shape[1]
    if pad_sm or pad_sk:
        s = mx.pad(s, ((0, pad_sm), (0, pad_sk)))
    w_blocked = w_dec.reshape(blocks_m, bs, blocks_k, bs)
    return (w_blocked * s[:, None, :, None]).reshape(Mp, Kp)


def _output_dtype_token(dtype) -> str:
    """Map an MLX output dtype to the TileLang/Metal dtype token.

    Metal codegen cannot emit ``bfloat`` outputs through this ABI, so bf16 is
    handled by computing in float32 and casting at the MLX boundary.
    """
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    # bf16 (and anything else) -> compute fp32, cast outside.
    return "float32"


def _get_tilelang_adapter(N: int, M: int, K: int, c_dtype: str):
    key = (int(N), int(M), int(K), str(c_dtype))
    with _TL_ADAPTER_CACHE_LOCK:
        cached = _TL_ADAPTER_CACHE.get(key)
        if cached is not None:
            return cached
    import tilelang
    from cppmega_mlx.nn._tilelang._mlx_runtime import wrap_tilelang_metal_kernel

    prim = _build_fp8_gemm_prim(N, M, K, c_dtype)
    artifact = tilelang.lower(prim, target="metal")
    adapter = wrap_tilelang_metal_kernel(
        artifact,
        input_count=2,
        output_count=1,
        # Explicit mapping: the emitted Metal signature interleaves the output
        # buffer (``C``) ahead of some inputs, so a positional split is wrong.
        input_buffer_names=("A", "W"),
        output_buffer_names=("C",),
        allow_mx_fast_metal_kernel=True,
    )
    with _TL_ADAPTER_CACHE_LOCK:
        _TL_ADAPTER_CACHE[key] = adapter
    return adapter


def _fused_fp8_gemm_tilelang(
    w_fp8: mx.array,
    w_scale_inv: mx.array,
    a: mx.array,
    *,
    M: int,
    K: int,
    n: int,
    leading: tuple,
    out_dtype,
) -> mx.array:
    """Cooperative-tensor implementation of ``out = a @ W.T``.

    Pads N (batch rows) up to the GEMM tile and M/K up to 128 so a single
    block scale covers each weight sub-tile, decodes the e4m3 weight + block
    scale into a half ``[Mp, Kp]`` tensor, runs the cooperative-tensor GEMM,
    then slices the padding off.
    """
    bs = _BLOCK
    # Padded dims.
    Np = ((n + _TILE_N - 1) // _TILE_N) * _TILE_N
    Mp = ((M + bs - 1) // bs) * bs
    Kp = ((K + bs - 1) // bs) * bs

    c_dtype = _output_dtype_token(out_dtype)

    # Activation: [n, K] -> half, padded to [Np, Kp].
    a_flat = a.astype(mx.float16).reshape(n, K)
    if Np != n or Kp != K:
        a_flat = mx.pad(a_flat, ((0, Np - n), (0, Kp - K)))

    # Decode e4m3 + per-128-block scale -> padded half weight [Mp, Kp].
    w_half = _dequant_weight_to_half(
        w_fp8, w_scale_inv, M=M, K=K, Mp=Mp, Kp=Kp
    )

    adapter = _get_tilelang_adapter(Np, Mp, Kp, c_dtype)

    grid_x = Mp // _TILE_M
    grid_y = Np // _TILE_N
    out_mx_dtype = mx.float32 if c_dtype == "float32" else mx.float16

    (out_flat,) = adapter(
        inputs=[a_flat, w_half],
        output_shapes=[(Np, Mp)],
        output_dtypes=[out_mx_dtype],
        grid=(grid_x * _THREADS, grid_y, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    # Slice off padding and restore leading dims + output dtype.
    out = out_flat[:n, :M]
    return out.reshape(*leading, M).astype(out_dtype)


# ---------------------------------------------------------------------------
# Scalar hand-MSL fallback (original scaffold)
# ---------------------------------------------------------------------------


def _fused_fp8_gemm_scalar(
    w_fp8: mx.array,
    w_scale_inv: mx.array,
    a: mx.array,
    *,
    M: int,
    K: int,
    n: int,
    leading: tuple,
    out_dtype,
    blocks_k: int,
) -> mx.array:
    """Original one-thread-per-output-column scalar MSL kernel."""
    bs = _BLOCK
    a_fp32 = a.astype(mx.float32)
    a_flat = a_fp32.reshape(n, K)

    w_flat = w_fp8.reshape(-1)
    s_flat = w_scale_inv.reshape(-1)
    a_flat_1d = a_flat.reshape(-1)

    source = f"""
        uint m   = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (m >= {M}u || row >= {n}u) return;

        uint block_m = m / {bs}u;
        float acc = 0.0f;

        for (uint kb = 0; kb < {blocks_k}u; ++kb) {{
            float scale = s_flat[block_m * {blocks_k}u + kb];
            uint k_start = kb * {bs}u;
            uint k_end_full = k_start + {bs}u;
            uint k_end = k_end_full < {K}u ? k_end_full : {K}u;
            for (uint k = k_start; k < k_end; ++k) {{
                uint  byte = (uint)w_flat[m * {K}u + k];
                int   sign = (byte >> 7) & 0x1;
                int   expt = (byte >> 3) & 0xF;
                int   mant = byte & 0x7;
                float val;
                if (expt == 0) {{
                    val = (float)mant / 8.0f * 0.015625f;
                }} else if (expt == 0xF && mant == 0x7) {{
                    val = 0.0f;
                }} else {{
                    float mantissa = 1.0f + (float)mant / 8.0f;
                    int   bias_exp = expt - 7;
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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

    Uses the TileLang cooperative-tensor block-FP8 GEMM when available and
    falls back to the scalar hand-MSL kernel otherwise (or when
    ``CPPMEGA_V4_FUSED_FP8_GEMM_SCALAR`` is set).
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
    leading = a.shape[:-1]
    n = 1
    for d in leading:
        n *= d

    use_tilelang = (not _force_scalar()) and _tilelang_status()[0]
    if use_tilelang:
        try:
            return _fused_fp8_gemm_tilelang(
                w_fp8,
                w_scale_inv,
                a,
                M=M,
                K=K,
                n=n,
                leading=leading,
                out_dtype=out_dtype,
            )
        except Exception:  # noqa: BLE001 -- never let the fused path break the op
            # Fall through to the always-available scalar kernel.
            pass

    return _fused_fp8_gemm_scalar(
        w_fp8,
        w_scale_inv,
        a,
        M=M,
        K=K,
        n=n,
        leading=leading,
        out_dtype=out_dtype,
        blocks_k=blocks_k_expected,
    )


__all__ = ["fused_fp8_gemm", "fused_fp8_gemm_status"]
