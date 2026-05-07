"""TileLang ports of the three Mamba3 backward Triton helpers.

Source attribution
------------------

The three helpers ported here cover the same mathematical contract as the
upstream Triton kernels in
``mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py`` (state-spaces/mamba):

  * ``compute_dacs_segsum_triton``  - segment cumulative-sum reduction over the
    Mamba3 time axis. The cppmega rewrite at
    ``cppmega/megatron/tilelang_mimo_autograd.py`` carries the same name; the
    pure-MLX sibling (``_mamba3_helpers.compute_dacs_segsum``) keeps the
    cppmega ``(B, T, H)`` shape contract.
  * ``bwd_dadt_fused_triton``       - fused dA/ddt computation.
  * ``bwd_dtrap_ddt_triton``        - fused ddt/dtrap from the trapezoidal
    parameterisation.

All three are translated to TileLang ``@T.prim_func`` definitions and lowered
through TileLang's ``metal`` target. The emitted MSL body is stitched into an
``mx.fast.metal_kernel`` source so the three helpers ride the same Path B
codegen path as the main Mamba3 forward/backward kernels.

API parity
----------

The helpers below mirror the *signatures* of the pure-MLX siblings in
``_mamba3_helpers.py`` so the two implementations are drop-in swappable for
parity benches and tests. The math intentionally matches the cppmega Mamba3
reference (``cppmega_mlx/nn/mamba3.py``) rather than the upstream Triton's
``(B, H, S, ...)`` chunked layout: that translation already happens in the
sibling and is the parity oracle for this module too.

fp16 carrier
------------

The Triton originals run in fp32. tilelang 0.1.9's bf16 simdgroup MSL path has
known issues (cubecl#1202), so this Path B port forces the *carrier* dtype to
fp16 while keeping the per-step accumulator in fp32. Inputs in bf16 are
round-tripped through fp32 to avoid mantissa loss; fp32 inputs pass through
unchanged.

Per-helper accuracy budget vs the pure-MLX sibling:
  * ``compute_dacs_segsum``  - rtol=1e-4 / atol=1e-3 (fp16 boundary).
  * ``bwd_dadt_fused``       - rtol=1e-4 / atol=1e-3 (fp16 boundary).
  * ``bwd_dtrap_ddt``        - rtol=1e-4 / atol=1e-3 (fp16 boundary).

Fallback
--------

When tilelang is not importable, the Metal target rejects an op, or the
emitted kernel diverges beyond tolerance, callers should route through the
pure-MLX sibling. Each public helper here exposes a status object so callers
can pre-flight the dispatch decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _mamba3_helpers as _pure_helpers
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
    lower_tilelang_to_msl_inline,
)


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileLangHelperStatus:
    """Runtime status of one of the Path B Mamba3 helper kernels."""

    available: bool
    reason: str
    fp16_carrier: bool = True


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - macOS without tilelang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def helpers_metal_status() -> TileLangHelperStatus:
    """Return whether the Path B Mamba3 helpers can dispatch on this host."""

    if not can_run_metal():
        return TileLangHelperStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return TileLangHelperStatus(available=False, reason=reason)
    return TileLangHelperStatus(available=True, reason="Path B helpers ready")


# ---------------------------------------------------------------------------
# fp16 carrier helper
# ---------------------------------------------------------------------------


def _to_fp16_carrier(x: mx.array) -> mx.array:
    if x.dtype == mx.float16:
        return x
    if x.dtype == mx.bfloat16:
        return x.astype(mx.float32).astype(mx.float16)
    return x.astype(mx.float16)


def _to_fp32(x: mx.array) -> mx.array:
    if x.dtype == mx.float32:
        return x
    if x.dtype == mx.bfloat16:
        return x.astype(mx.float32)
    return x.astype(mx.float32)


# ---------------------------------------------------------------------------
# Kernel factories (cached) — each returns (kernel, lowering metadata)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _segsum_kernel_for(BH: int, T_: int, K: int, BLOCK_K: int):
    """Build & cache a TileLang segsum kernel for (BH, T, K, BLOCK_K)."""

    import tilelang.language as T

    accum_dtype = "float32"
    carrier_dtype = "float16"

    @T.prim_func
    def segsum(
        A: T.Tensor((BH, T_), accum_dtype),
        dt: T.Tensor((BH, T_), accum_dtype),
        dh: T.Tensor((BH, T_, K), carrier_dtype),
        out: T.Tensor((BH, T_, K), carrier_dtype),
    ):
        with T.Kernel(BH, T.ceildiv(K, BLOCK_K), threads=BLOCK_K) as (bh, bk):
            tk = T.get_thread_binding(0)
            k_index = bk * BLOCK_K + tk
            weights = T.alloc_shared((T_,), accum_dtype, scope="shared")
            if tk == 0:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0
                # rev[t] = sum_{u=t+1..T-1} A[u]*dt[u]
                # The last position has no later contribution: rev[T-1] = 0,
                # so weight[T-1] = exp(0) = 1.
                weights[T_ - 1] = T.exp(acc[0])
                for r in T.serial(T_ - 1):
                    t = T_ - 2 - r  # iterate T-2 down to 0
                    acc[0] = acc[0] + A[bh, t + 1] * dt[bh, t + 1]
                    weights[t] = T.exp(acc[0])
            T.sync_threads()
            if k_index < K:
                for t in T.serial(T_):
                    out[bh, t, k_index] = T.cast(
                        T.cast(dh[bh, t, k_index], accum_dtype) * weights[t],
                        carrier_dtype,
                    )

    artifact = dispatch_lower(segsum, target="metal")
    if hasattr(artifact, "_tilelang_engine_target"):
        return artifact, None
    lowering = artifact
    kernel = mx.fast.metal_kernel(
        name=f"tlmamba3_segsum_{BH}_{T_}_{K}_{BLOCK_K}",
        # Buffer order in the lowered MSL is alphabetic: A, dh, dt, out.
        input_names=["A", "dh", "dt"],
        output_names=["out"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _dadt_kernel_for(BH: int, T_: int, K: int, BLOCK_T: int):
    """TileLang kernel for bwd_dadt_fused.

    Computes d_decay[bh, t] = sum_k dY[bh, t, k] * h[bh, t, k] in fp32, then
    dA = d_decay * dt and ddt = d_decay * A.

    Grid: (BH, ceil(T, BLOCK_T)). Each program handles one BH row and one T
    block, reducing K serially per (bh, t) using one thread per t in the block.
    """

    import tilelang.language as T

    accum_dtype = "float32"
    carrier_dtype = "float16"

    @T.prim_func
    def dadt(
        A: T.Tensor((BH, T_), accum_dtype),
        dY: T.Tensor((BH, T_, K), carrier_dtype),
        dt: T.Tensor((BH, T_), accum_dtype),
        h: T.Tensor((BH, T_, K), carrier_dtype),
        dA: T.Tensor((BH, T_), accum_dtype),
        ddt: T.Tensor((BH, T_), accum_dtype),
    ):
        with T.Kernel(BH, T.ceildiv(T_, BLOCK_T), threads=BLOCK_T) as (bh, btile):
            tt = T.get_thread_binding(0)
            t_index = btile * BLOCK_T + tt
            if t_index < T_:
                # Reference carrier_dtype inside the body so the closure captures it.
                _scratch = T.alloc_local((1,), carrier_dtype)
                _scratch[0] = T.cast(0.0, carrier_dtype)
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0
                for k in T.serial(K):
                    acc[0] = acc[0] + (
                        T.cast(dY[bh, t_index, k], accum_dtype)
                        * T.cast(h[bh, t_index, k], accum_dtype)
                    )
                dA[bh, t_index] = acc[0] * dt[bh, t_index]
                ddt[bh, t_index] = acc[0] * A[bh, t_index]

    artifact = dispatch_lower(dadt, target="metal")
    if hasattr(artifact, "_tilelang_engine_target"):
        return artifact, None
    lowering = artifact
    kernel = mx.fast.metal_kernel(
        name=f"tlmamba3_dadt_{BH}_{T_}_{K}_{BLOCK_T}",
        # Buffer order alphabetic: A, dA, dY, ddt, dh, dt -> after sort:
        # actually MSL emits in alphabetic order which is:
        # A, dA, dY, ddt, dt, h
        input_names=["A", "dY", "dt", "h"],
        output_names=["dA", "ddt"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _dtrap_kernel_for(BH: int, T_: int, BLOCK_T: int):
    """TileLang kernel for bwd_dtrap_ddt.

    Pure element-wise / 1-token-shift kernel in (BH, T). We launch one thread
    per element. For each (bh, t):
      s          = sigmoid(trap[bh, t])
      s_shift    = sigmoid(trap[bh, t+1]) if t+1<T else 0.5
      dt_shift   = dt[bh, t+1] if t+1<T else 0
      d_scale    = dB_scaled[bh, t]
      d_scale_lp = dB_scaled[bh, t-1] if t>0 else 0
      s_shift_lp = sigmoid(trap[bh, t]) if t>0 else 0    (i.e. s)
      dt_lp      = dt[bh, t]            if t>0 else 0    (i.e. dt[t])
      ddt[bh,t]  = d_scale * s + (d_scale_lp * (1 - s_shift_lp))    [for t>0]
                 = d_scale * s                                       [for t=0]
      dtrap[bh,t] = (d_scale * dt - d_scale_lp * dt_lp) * s * (1 - s)
    """

    import tilelang.language as T

    accum_dtype = "float32"
    carrier_dtype = "float16"

    @T.prim_func
    def dtrap(
        dB_scaled: T.Tensor((BH, T_), carrier_dtype),
        dt: T.Tensor((BH, T_), carrier_dtype),
        trap: T.Tensor((BH, T_), carrier_dtype),
        ddt_out: T.Tensor((BH, T_), carrier_dtype),
        dtrap_out: T.Tensor((BH, T_), carrier_dtype),
    ):
        with T.Kernel(BH, T.ceildiv(T_, BLOCK_T), threads=BLOCK_T) as (bh, btile):
            tt = T.get_thread_binding(0)
            t_index = btile * BLOCK_T + tt
            if t_index < T_:
                trap_val = T.cast(trap[bh, t_index], accum_dtype)
                one = T.cast(1.0, accum_dtype)
                # Stable sigmoid via 1 / (1 + exp(-x)).
                s = one / (one + T.exp(-trap_val))
                d_scale = T.cast(dB_scaled[bh, t_index], accum_dtype)
                dt_val = T.cast(dt[bh, t_index], accum_dtype)

                # Use mutable locals to avoid the immutable-rebind warning.
                ddt_v = T.alloc_local((1,), accum_dtype)
                dtrap_v = T.alloc_local((1,), accum_dtype)

                # ddt: for t==0 -> d_scale*s
                #      for t>0  -> d_scale*s + d_scale[t-1]*(1 - s)
                #                  (because s_shift_at_t-1 = sigmoid(trap[t]) = s)
                # dtrap: contrib_from_s + contrib_from_s_shift_full
                #        contrib_from_s = d_scale * dt
                #        contrib_from_s_shift_full[t] = -d_scale[t-1] * dt[t]   (t > 0)
                #        contrib_from_s_shift_full[0] = 0
                #        dtrap = (contrib...) * s * (1-s)
                if t_index > 0:
                    d_scale_lp = T.cast(dB_scaled[bh, t_index - 1], accum_dtype)
                    ddt_v[0] = d_scale * s + d_scale_lp * (one - s)
                    dtrap_v[0] = (d_scale * dt_val - d_scale_lp * dt_val) * s * (one - s)
                else:
                    ddt_v[0] = d_scale * s
                    dtrap_v[0] = (d_scale * dt_val) * s * (one - s)
                ddt_out[bh, t_index] = T.cast(ddt_v[0], carrier_dtype)
                dtrap_out[bh, t_index] = T.cast(dtrap_v[0], carrier_dtype)

    artifact = dispatch_lower(dtrap, target="metal")
    if hasattr(artifact, "_tilelang_engine_target"):
        return artifact, None
    lowering = artifact
    kernel = mx.fast.metal_kernel(
        name=f"tlmamba3_dtrap_{BH}_{T_}_{BLOCK_T}",
        # Alphabetic order from MSL: dB_scaled, ddt_out, dt, dtrap_out, trap
        input_names=["dB_scaled", "dt", "trap"],
        output_names=["ddt_out", "dtrap_out"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


# ---------------------------------------------------------------------------
# Public helpers (drop-in API parity with _mamba3_helpers.py)
# ---------------------------------------------------------------------------


def _flatten_bh_kt(A: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
    """Reshape ``(B, T, H)`` -> ``(BH, T)`` and return shape metadata."""

    B, T_, H = A.shape
    BH = B * H
    flat = A.transpose(0, 2, 1).reshape(BH, T_)
    return flat, (B, T_, H)


def _flatten_dh(dh: mx.array, A_shape: tuple[int, int, int]) -> tuple[mx.array, int]:
    """Reshape dh ``(B, T, H, ...)`` to ``(BH, T, K)`` where K=prod trailing."""

    B, T_, H = A_shape
    if dh.shape[:3] != (B, T_, H):
        raise ValueError(f"dh leading dims must be {(B, T_, H)}, got {dh.shape[:3]}")
    trailing = dh.shape[3:]
    K = 1
    for d in trailing:
        K *= int(d)
    if K == 0:
        return dh, 0
    # Bring H next to B: (B, T, H, ...) -> (B, H, T, ...) -> (B*H, T, K)
    flat = dh.transpose(0, 2, 1, *range(3, dh.ndim)).reshape(B * H, T_, K)
    return flat, K


def _unflatten_dh(out: mx.array, dh_shape: tuple[int, ...]) -> mx.array:
    """Inverse of ``_flatten_dh``: ``(BH, T, K)`` -> ``(B, T, H, ...)``."""

    B = dh_shape[0]
    T_ = dh_shape[1]
    H = dh_shape[2]
    trailing = dh_shape[3:]
    if not trailing:
        return out.reshape(B, H, T_).transpose(0, 2, 1)
    return out.reshape(B, H, T_, *trailing).transpose(0, 2, 1, *range(3, len(dh_shape)))


def _ceil_pow2(n: int) -> int:
    """Round up to a power of two (used for cache-friendly TileLang kernels)."""

    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _validate_segsum_inputs(A: mx.array, dt: mx.array, dh: mx.array) -> None:
    if A.ndim != 3:
        raise ValueError(f"A must be (B,T,H), got {A.shape}")
    if dt.shape != A.shape:
        raise ValueError(f"dt must match A {A.shape}, got {dt.shape}")
    if dh.shape[:3] != A.shape:
        raise ValueError(
            f"dh leading dims must match A {A.shape}, got {dh.shape[:3]}"
        )


def compute_dacs_segsum(
    A: mx.array,
    dt: mx.array,
    dh: mx.array,
    *,
    accumulate_in_fp32: bool = True,
    force_fallback: bool = False,
) -> mx.array:
    """TileLang/Metal compute_dacs_segsum with pure-MLX fallback.

    See ``_mamba3_helpers.compute_dacs_segsum`` for the contract.
    """

    _validate_segsum_inputs(A, dt, dh)

    if dh.size == 0 or dh.shape[-1] == 0:
        return _pure_helpers.compute_dacs_segsum(
            A, dt, dh, accumulate_in_fp32=accumulate_in_fp32
        )

    status = helpers_metal_status()
    if force_fallback or not status.available:
        return _pure_helpers.compute_dacs_segsum(
            A, dt, dh, accumulate_in_fp32=accumulate_in_fp32
        )

    B, T_, H = A.shape
    BH = B * H
    A_flat, _ = _flatten_bh_kt(A)
    dt_flat, _ = _flatten_bh_kt(dt)
    dh_flat, K = _flatten_dh(dh, A.shape)

    A_f32 = _to_fp32(A_flat)
    dt_f32 = _to_fp32(dt_flat)
    dh_f16 = _to_fp16_carrier(dh_flat)

    BLOCK_K = min(_ceil_pow2(K) if K <= 256 else 64, 64)
    if BLOCK_K < 1:
        BLOCK_K = 1
    K_padded = _round_up(K, BLOCK_K)
    if K_padded != K:
        # Pad the trailing dim so the static TileLang shape matches.
        pad_zeros = mx.zeros((BH, T_, K_padded - K), dtype=mx.float16)
        dh_f16 = mx.concatenate([dh_f16, pad_zeros], axis=-1)

    try:
        kernel, lowering = _segsum_kernel_for(BH, T_, K_padded, BLOCK_K)
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        # Lowering failed -- fall back transparently.
        _ = exc
        return _pure_helpers.compute_dacs_segsum(
            A, dt, dh, accumulate_in_fp32=accumulate_in_fp32
        )

    if lowering is None:
        # Engine path: runtime call signature for the engine artifact is not
        # yet wired through the mlx fast-kernel dispatch (mx.fast.metal_kernel
        # expects mx.array buffers, the engine artifact takes raw prim args).
        # Fall back to pure-MLX for parity until Phase-4 plumbs the engine
        # artifact through the fast-kernel runtime.
        return _pure_helpers.compute_dacs_segsum(
            A, dt, dh, accumulate_in_fp32=accumulate_in_fp32
        )

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        out_list = kernel(
            inputs=[A_f32, dh_f16, dt_f32],
            output_shapes=[(BH, T_, K_padded)],
            output_dtypes=[mx.float16],
            grid=grid,
            threadgroup=lowering.threadgroup,
        )
    except Exception as exc:
        _ = exc
        return _pure_helpers.compute_dacs_segsum(
            A, dt, dh, accumulate_in_fp32=accumulate_in_fp32
        )
    out_padded = out_list[0]
    if K_padded != K:
        out_padded = out_padded[:, :, :K]
    out_full = _unflatten_dh(out_padded, dh.shape)
    return out_full.astype(dh.dtype)


def bwd_dadt_fused(
    dY: mx.array,
    A: mx.array,
    dt: mx.array,
    h: mx.array,
    *,
    accumulate_in_fp32: bool = True,
    force_fallback: bool = False,
) -> Tuple[mx.array, mx.array]:
    """TileLang/Metal bwd_dadt_fused with pure-MLX fallback."""

    if A.ndim != 3:
        raise ValueError(f"A must be (B,T,H), got {A.shape}")
    if dt.shape != A.shape:
        raise ValueError(f"dt must match A {A.shape}, got {dt.shape}")
    if dY.shape != h.shape:
        raise ValueError(f"dY must match h {h.shape}, got {dY.shape}")
    if dY.shape[:3] != A.shape:
        raise ValueError(
            f"dY leading dims must match A {A.shape}, got {dY.shape[:3]}"
        )

    if dY.size == 0 or h.size == 0:
        return _pure_helpers.bwd_dadt_fused(
            dY, A, dt, h, accumulate_in_fp32=accumulate_in_fp32
        )

    status = helpers_metal_status()
    if force_fallback or not status.available:
        return _pure_helpers.bwd_dadt_fused(
            dY, A, dt, h, accumulate_in_fp32=accumulate_in_fp32
        )

    B, T_, H = A.shape
    BH = B * H
    A_flat, _ = _flatten_bh_kt(A)
    dt_flat, _ = _flatten_bh_kt(dt)
    dY_flat, K = _flatten_dh(dY, A.shape)
    h_flat, _ = _flatten_dh(h, A.shape)

    A_f32 = _to_fp32(A_flat)
    dt_f32 = _to_fp32(dt_flat)
    dY_f16 = _to_fp16_carrier(dY_flat)
    h_f16 = _to_fp16_carrier(h_flat)

    BLOCK_T = min(64, max(1, _ceil_pow2(T_)))
    BLOCK_T = min(BLOCK_T, 256)

    try:
        kernel, lowering = _dadt_kernel_for(BH, T_, K, BLOCK_T)
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return _pure_helpers.bwd_dadt_fused(
            dY, A, dt, h, accumulate_in_fp32=accumulate_in_fp32
        )

    if lowering is None:
        # Engine path: not yet wired through fast-kernel runtime — fall back.
        return _pure_helpers.bwd_dadt_fused(
            dY, A, dt, h, accumulate_in_fp32=accumulate_in_fp32
        )

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        out_list = kernel(
            inputs=[A_f32, dY_f16, dt_f32, h_f16],
            output_shapes=[(BH, T_), (BH, T_)],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=lowering.threadgroup,
        )
    except Exception:
        return _pure_helpers.bwd_dadt_fused(
            dY, A, dt, h, accumulate_in_fp32=accumulate_in_fp32
        )
    dA_flat, ddt_flat = out_list
    dA = dA_flat.reshape(B, H, T_).transpose(0, 2, 1).astype(A.dtype)
    ddt = ddt_flat.reshape(B, H, T_).transpose(0, 2, 1).astype(dt.dtype)
    return dA, ddt


def bwd_dtrap_ddt(
    dB_scaled: mx.array,
    dt: mx.array,
    trap: mx.array,
    *,
    accumulate_in_fp32: bool = True,
    force_fallback: bool = False,
) -> Tuple[mx.array, mx.array]:
    """TileLang/Metal bwd_dtrap_ddt with pure-MLX fallback."""

    if dB_scaled.ndim != 3:
        raise ValueError(f"dB_scaled must be (B,T,H), got {dB_scaled.shape}")
    if dt.shape != dB_scaled.shape:
        raise ValueError(f"dt must match dB_scaled {dB_scaled.shape}, got {dt.shape}")
    if trap.shape != dB_scaled.shape:
        raise ValueError(f"trap must match dB_scaled {dB_scaled.shape}, got {trap.shape}")
    if dB_scaled.size == 0 or dB_scaled.shape[1] == 0:
        return _pure_helpers.bwd_dtrap_ddt(
            dB_scaled, dt, trap, accumulate_in_fp32=accumulate_in_fp32
        )

    status = helpers_metal_status()
    if force_fallback or not status.available:
        return _pure_helpers.bwd_dtrap_ddt(
            dB_scaled, dt, trap, accumulate_in_fp32=accumulate_in_fp32
        )

    B, T_, H = dB_scaled.shape
    BH = B * H

    dB_flat, _ = _flatten_bh_kt(dB_scaled)
    dt_flat, _ = _flatten_bh_kt(dt)
    trap_flat, _ = _flatten_bh_kt(trap)

    dB_f16 = _to_fp16_carrier(dB_flat)
    dt_f16 = _to_fp16_carrier(dt_flat)
    trap_f16 = _to_fp16_carrier(trap_flat)

    BLOCK_T = min(64, max(1, _ceil_pow2(T_)))
    BLOCK_T = min(BLOCK_T, 256)

    try:
        kernel, lowering = _dtrap_kernel_for(BH, T_, BLOCK_T)
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return _pure_helpers.bwd_dtrap_ddt(
            dB_scaled, dt, trap, accumulate_in_fp32=accumulate_in_fp32
        )

    if lowering is None:
        # Engine path: not yet wired through fast-kernel runtime — fall back.
        return _pure_helpers.bwd_dtrap_ddt(
            dB_scaled, dt, trap, accumulate_in_fp32=accumulate_in_fp32
        )

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        out_list = kernel(
            inputs=[dB_f16, dt_f16, trap_f16],
            output_shapes=[(BH, T_), (BH, T_)],
            output_dtypes=[mx.float16, mx.float16],
            grid=grid,
            threadgroup=lowering.threadgroup,
        )
    except Exception:
        return _pure_helpers.bwd_dtrap_ddt(
            dB_scaled, dt, trap, accumulate_in_fp32=accumulate_in_fp32
        )
    ddt_flat, dtrap_flat = out_list
    ddt = ddt_flat.reshape(B, H, T_).transpose(0, 2, 1).astype(dt.dtype)
    dtrap = dtrap_flat.reshape(B, H, T_).transpose(0, 2, 1).astype(trap.dtype)
    return ddt, dtrap


__all__ = [
    "TileLangHelperStatus",
    "bwd_dadt_fused",
    "bwd_dtrap_ddt",
    "compute_dacs_segsum",
    "helpers_metal_status",
]
