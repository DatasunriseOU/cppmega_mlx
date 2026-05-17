"""GDN Path C — TileLang DSL ``@T.prim_func`` lowered to Metal via tvm_ffi.

Mirrors the structure of ``cppmega_mlx/nn/_tilelang/mamba3_path_c.py`` (which
ports Mamba3 MIMO fwd to TileLang): we declare the GDN recurrence as a
per-lane scan in TileLang IR, run it through ``dispatch_lower(..., target='metal',
return_msl=True)`` for MSL extraction, then compile with
``tilelang.compile(..., target=_as_metal_target('metal'),
execution_backend='tvm_ffi', out_idx=[...])`` into caller-owned MLX buffers.

This file is plugin-isolated: it imports ``dispatch_lower`` /
``_msl_transform`` from the host ``cppmega_mlx`` package (read-only use of
the existing TileLang→MSL infrastructure) but never modifies them.

When tilelang isn't importable in the current env (no torch, etc.), the
status check returns ``available=False`` with a precise reason, and any
direct call falls back to Path A's pure-MLX reference. Once tilelang lands
in the env, the kernel will compile on first invocation, then be cached via
``functools.lru_cache``.

Layout / lane mapping
---------------------
Following mamba3 fwd's per-lane recurrence, we use ``LANES = B * H * V``
threads. Each lane owns one ``(b, h, v_idx)`` slice and walks the time
dimension serially while reducing over K for the kv-state inner product and
the q-state output projection. State is held in registers as ``K`` floats
per lane (size of the K dimension).

Recurrence per step ``t``:
    decay = exp(g[b, t, h])
    h[i] *= decay             # for i in K (alpha decay)
    kth_S = sum_i k[i] * h[i]
    v_eff = beta[b, t, h] * (v[v_idx] - kth_S)
    h[i] += k[i] * v_eff
    out  = sum_i q[i] * h[i]
    y[b, t, h, v_idx] = out
"""

from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx

# Deliberately NOT using `from __future__ import annotations` here:
# tilelang's @T.prim_func builder calls get_type_hints which evaluates
# the T.Tensor((BATCH, SEQ, ...)) annotations against globals — under
# PEP 563 the closure locals (BATCH/SEQ/...) become unresolvable strings.


def _tilelang_importable() -> tuple[bool, str]:
    """Probe whether the full tilelang stack is loadable.

    Mirrors ``cppmega_mlx.nn._tilelang.mamba3_path_c._tilelang_available`` but
    inlined here so this module can be imported even when the host package's
    deeper helpers aren't reachable (e.g. partial install).
    """
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
    """Pick a thread-group size that evenly tiles ``lanes`` (mirror mamba3)."""
    for tg in (256, 192, 128, 96, 64, 32):
        if lanes % tg == 0:
            return tg
    return 32


@lru_cache(maxsize=64)
def _fwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM_K: int,
    HEADDIM_V: int,
    q_dtype: str = "float32",
    k_dtype: str = "float32",
    v_dtype: str = "float32",
    beta_dtype: str = "float32",
    g_dtype: str = "float32",
    h0_dtype: str = "float32",
    y_dtype: str = "float32",
    h_last_dtype: str = "float32",
) -> tuple[Any, Any]:
    """Build (and cache) the GDN fwd Path C kernel for this shape/dtype tuple.

    Raises ``ImportError`` if tilelang isn't importable — callers should
    wrap in a try/except and fall back to Path A.
    """
    import tilelang
    import tilelang.language as T
    from cppmega_mlx.nn._tilelang import _msl_transform
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    LANES = BATCH * HEADS * HEADDIM_V
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        q: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_K), q_dtype),
        k: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_K), k_dtype),
        v: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_V), v_dtype),
        beta: T.Tensor((BATCH, SEQ, HEADS), beta_dtype),
        g: T.Tensor((BATCH, SEQ, HEADS), g_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM_K, HEADDIM_V), h0_dtype),
        y: T.Tensor((BATCH, SEQ, HEADS, HEADDIM_V), y_dtype),
        h_last: T.Tensor((BATCH, HEADS, HEADDIM_K, HEADDIM_V), h_last_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid_in_block = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid_in_block
            # Per-lane register state of size K (one column of the H matrix).
            h_state = T.alloc_local((HEADDIM_K,), accum_dtype)
            if global_lane < LANES:
                vj = global_lane % HEADDIM_V
                head = (global_lane // HEADDIM_V) % HEADS
                bb = global_lane // (HEADDIM_V * HEADS)
                # Init from h0[bb, head, :, vj].
                for i in T.serial(HEADDIM_K):
                    h_state[i] = T.cast(h0[bb, head, i, vj], accum_dtype)
                for t in T.serial(SEQ):
                    g_val = T.cast(g[bb, t, head], accum_dtype)
                    beta_val = T.cast(beta[bb, t, head], accum_dtype)
                    decay = T.exp(g_val)
                    v_j = T.cast(v[bb, t, head, vj], accum_dtype)
                    # Phase 1: alpha decay and KS reduction (interleaved).
                    kth_S_j = T.alloc_var(T.float32, init=0.0)
                    for i in T.serial(HEADDIM_K):
                        h_state[i] = h_state[i] * decay
                        kth_S_j += T.cast(k[bb, t, head, i], accum_dtype) * h_state[i]
                    # Phase 2: delta correction.
                    v_eff = beta_val * (v_j - kth_S_j)
                    # Phase 3: rank-1 outer add + output projection along K.
                    out = T.alloc_var(T.float32, init=0.0)
                    for i in T.serial(HEADDIM_K):
                        k_i = T.cast(k[bb, t, head, i], accum_dtype)
                        q_i = T.cast(q[bb, t, head, i], accum_dtype)
                        h_state[i] = h_state[i] + k_i * v_eff
                        out += q_i * h_state[i]
                    y[bb, t, head, vj] = T.cast(out, y_dtype)
                for i in T.serial(HEADDIM_K):
                    h_last[bb, head, i, vj] = T.cast(h_state[i], h_last_dtype)

    artifact = dispatch_lower(fwd, target="metal", return_msl=True)
    lowering = artifact  # TileLangMSLLowering when MSL extraction succeeds
    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=[6, 7],  # y, h_last
    )
    return kernel, lowering


def _path_c_runtime_status() -> tuple[bool, str]:
    """Status visible to the dispatch layer."""
    return _tilelang_importable()


def _gdn_fwd_path_c_call(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[mx.array] = None,
    output_final_state: bool = False,
):
    """Path C entry — same signature as ``naive_recurrent_gated_delta_rule``.

    On any failure (tilelang missing, compile error, runtime error) raises
    ``RuntimeError`` so the caller can fall back to Path A.
    """
    import math

    B, T_, H, K_dim = q.shape
    V_dim = v.shape[-1]
    # FLA applies q *= 1/sqrt(K) inside; we pre-scale to match.
    fla_scale = scale if scale is not None else 1.0 / math.sqrt(K_dim)
    q_scaled = (q.astype(mx.float32) * fla_scale).astype(q.dtype)

    h0 = initial_state
    if h0 is None:
        h0 = mx.zeros((B, H, K_dim, V_dim), dtype=mx.float32)

    kernel, _lowering = _fwd_kernel_for(
        B, T_, H, K_dim, V_dim,
        q_dtype=str(q.dtype).rsplit(".", 1)[-1],
        k_dtype=str(k.dtype).rsplit(".", 1)[-1],
        v_dtype=str(v.dtype).rsplit(".", 1)[-1],
        beta_dtype=str(beta.dtype).rsplit(".", 1)[-1],
        g_dtype=str(g.dtype).rsplit(".", 1)[-1],
    )
    y, h_last = kernel(q_scaled, k, v, beta, g, h0)
    return y, (h_last if output_final_state else None)


__all__ = [
    "_fwd_kernel_for",
    "_gdn_fwd_path_c_call",
    "_path_c_runtime_status",
    "_tilelang_importable",
]
