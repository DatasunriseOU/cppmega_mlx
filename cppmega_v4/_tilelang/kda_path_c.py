"""KDA Path C — TileLang DSL ``@T.prim_func`` lowered to Metal via tvm_ffi.

Mirrors ``linear_attention_path_c.py`` but for the KDA recurrence:

    q, k: [B, T, H, K]    (q pre-scaled by 1/sqrt(K), then both repeat to HV)
    v:    [B, T, HV, V]
    g:    [B, T, HV, K]   per-K vectorized log-gate
    beta: [B, T, HV]
    S:    [B, HV, K, V]

Per-lane scan: one lane per (b, hv, v_idx). State held in registers as K
floats per lane. Inner loop: per-K decay + KS reduction (interleaved),
δ correction, rank-1 outer add + q·S projection.

Plugin invariant: imports ``dispatch_lower`` / ``_msl_transform`` from the
host ``cppmega_mlx`` package read-only — never modifies them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx


def _tilelang_importable() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        import tilelang.language as _T  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
    except Exception as exc:
        return False, f"tilelang import failed: {exc.__class__.__name__}: {exc}"
    try:
        from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower  # noqa: F401
        from cppmega_mlx.nn._tilelang import _msl_transform  # noqa: F401
    except Exception as exc:
        return False, f"host TileLang→MSL infra not reachable: {exc}"
    return True, "tilelang + host TileLang→MSL infra reachable"


def _threads_for(lanes: int) -> int:
    for tg in (256, 192, 128, 96, 64, 32):
        if lanes % tg == 0:
            return tg
    return 32


@lru_cache(maxsize=64)
def _kda_fwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HV: int,
    HEADDIM_K: int,
    HEADDIM_V: int,
    q_dtype: str = "float32",
    k_dtype: str = "float32",
    v_dtype: str = "float32",
    g_dtype: str = "float32",
    beta_dtype: str = "float32",
    h0_dtype: str = "float32",
    y_dtype: str = "float32",
    h_last_dtype: str = "float32",
) -> tuple[Any, Any]:
    """Build (and cache) the KDA fwd Path C kernel for this shape/dtype tuple."""
    import tilelang
    import tilelang.language as T
    from cppmega_mlx.nn._tilelang import _msl_transform
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    assert HV % HEADS == 0, f"HV ({HV}) must be divisible by HEADS ({HEADS})"
    GROUP = HV // HEADS
    LANES = BATCH * HV * HEADDIM_V
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        q: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_K), q_dtype),
        k: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_K), k_dtype),
        v: T.Tensor((BATCH, SEQ, HV, HEADDIM_V), v_dtype),
        g: T.Tensor((BATCH, SEQ, HV, HEADDIM_K), g_dtype),
        beta: T.Tensor((BATCH, SEQ, HV), beta_dtype),
        h0: T.Tensor((BATCH, HV, HEADDIM_K, HEADDIM_V), h0_dtype),
        y: T.Tensor((BATCH, SEQ, HV, HEADDIM_V), y_dtype),
        h_last: T.Tensor((BATCH, HV, HEADDIM_K, HEADDIM_V), h_last_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((HEADDIM_K,), accum_dtype)
            if global_lane < LANES:
                vj = global_lane % HEADDIM_V
                hv_idx = (global_lane // HEADDIM_V) % HV
                bb = global_lane // (HEADDIM_V * HV)
                h_idx = hv_idx // GROUP
                for i in T.serial(HEADDIM_K):
                    h_state[i] = T.cast(h0[bb, hv_idx, i, vj], accum_dtype)
                for t in T.serial(SEQ):
                    beta_val = T.cast(beta[bb, t, hv_idx], accum_dtype)
                    v_j = T.cast(v[bb, t, hv_idx, vj], accum_dtype)
                    # Phase 1: per-K decay + KS reduction (interleaved).
                    kth_S_j = T.alloc_var(T.float32, init=0.0)
                    for i in T.serial(HEADDIM_K):
                        decay_i = T.exp(T.cast(g[bb, t, hv_idx, i], accum_dtype))
                        h_state[i] = h_state[i] * decay_i
                        kth_S_j += T.cast(k[bb, t, h_idx, i], accum_dtype) * h_state[i]
                    inner_j = v_j - kth_S_j
                    # Phase 2: rank-1 outer add + q·h projection.
                    out = T.alloc_var(T.float32, init=0.0)
                    for i in T.serial(HEADDIM_K):
                        k_i = T.cast(k[bb, t, h_idx, i], accum_dtype)
                        q_i = T.cast(q[bb, t, h_idx, i], accum_dtype)
                        h_state[i] = h_state[i] + beta_val * k_i * inner_j
                        out += q_i * h_state[i]
                    y[bb, t, hv_idx, vj] = T.cast(out, y_dtype)
                for i in T.serial(HEADDIM_K):
                    h_last[bb, hv_idx, i, vj] = T.cast(h_state[i], h_last_dtype)

    artifact = dispatch_lower(fwd, target="metal", return_msl=True)
    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=[6, 7],  # y, h_last
    )
    return kernel, artifact


def _path_c_runtime_status() -> tuple[bool, str]:
    return _tilelang_importable()


def _kda_fwd_path_c_call(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    scale: float | None = None,
    initial_state: mx.array | None = None,
    output_final_state: bool = False,
):
    """KDA Path C entry — same signature as ``naive_recurrent_kda``."""
    import math

    B, T_, H, K_dim = q.shape
    HV, V_dim = v.shape[2], v.shape[-1]
    fla_scale = scale if scale is not None else 1.0 / math.sqrt(K_dim)
    q_scaled = (q.astype(mx.float32) * fla_scale).astype(q.dtype)

    h0 = initial_state
    if h0 is None:
        h0 = mx.zeros((B, HV, K_dim, V_dim), dtype=mx.float32)

    kernel, _ = _kda_fwd_kernel_for(
        B, T_, H, HV, K_dim, V_dim,
        q_dtype=str(q.dtype).rsplit(".", 1)[-1],
        k_dtype=str(k.dtype).rsplit(".", 1)[-1],
        v_dtype=str(v.dtype).rsplit(".", 1)[-1],
        g_dtype=str(g.dtype).rsplit(".", 1)[-1],
        beta_dtype=str(beta.dtype).rsplit(".", 1)[-1],
    )
    y, h_last = kernel(q_scaled, k, v, g, beta, h0)
    return y, (h_last if output_final_state else None)


__all__ = [
    "_kda_fwd_kernel_for",
    "_kda_fwd_path_c_call",
    "_path_c_runtime_status",
    "_tilelang_importable",
]
