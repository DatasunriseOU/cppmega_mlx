"""GDN Path B forward + backward — fwd via fast Metal kernel, bwd via
hand-MSL Metal kernel (real, fused recurrent backward).

The previous revision routed VJP through ``mx.grad`` over the FLA naive
reference. That kept training correct but threw away the entire forward
speedup on every step. This revision replaces the backward with a real
``mx.fast.metal_kernel`` that fuses:

  - forward replay to recover the final state,
  - reverse-time scan walking S backwards via cached snapshots,
  - per-thread ``dS`` register array,
  - ``simd_sum`` across the j-axis for grads that contract over j
    (``dq[i]``, ``dk[i]``, ``dbeta``, ``dg``),
  - threadgroup-shared-memory cross-simdgroup reductions for
    ``max(K, V) > 32``.

Layout matches the forward kernel: one thread per (b, h, vj) where vj
indexes the V-axis (reduction source). Per-thread register array
``state[K]`` walks the K-axis serially. Threadgroup is padded to a
``32 * ceil(max(K, V) / 32)``-lane block so simd_sum covers a full
32-lane simdgroup and the cross-simdgroup write loop can address every
K-output (even when K > tg_size, by looping the reduce-store).

Supports:
    - ``head_k_dim != head_v_dim`` (separate K and V dims).
    - ``max(K, V) <= 256``  (cap to keep per-thread register pressure sane).

Fallback path: when constraints are violated (max(K,V) > 256 or shape
mismatch) the wrapper falls back to ``mx.grad`` through
``naive_recurrent_gated_delta_rule`` so callers never get
silently-wrong numerics.

Backward algebra (derived from the forward in
``linear_attention_path_b.py``):

    Forward (with q' = q * scale, scale = 1/sqrt(K)):
      alpha_t          = exp(g_t)
      S_decayed[i,j]   = alpha_t * S_{t-1}[i,j]                (i in K, j in V)
      kth[j]           = sum_i k_i * S_decayed[i,j]
      v_eff[j]         = beta_t * (v_t[j] - kth[j])
      S_t[i,j]         = S_decayed[i,j] + k_i * v_eff[j]
      o_t[j]           = sum_i q'_i * S_t[i,j]

    Backward (reverse t, with incoming dS from later steps):
      dq'_i        += sum_j dO[j] * S_t[i,j]
      dS_t[i,j]    += dO[j] * q'_i                   (carry into delta-deriv block)
      dv_eff[j]     = sum_i dS_t[i,j] * k_i
      dk_i (delta) += sum_j dS_t[i,j] * v_eff[j]
      dbeta_t      += sum_j dv_eff[j] * (v_t[j] - kth[j])   = sum_j dv_eff[j] * v_eff[j] / beta_t
      dv_t[j]       = dv_eff[j] * beta_t
      dkth[j]       = -dv_eff[j] * beta_t
      dk_i (kth)   += sum_j dkth[j] * S_decayed[i,j]
      dS_decayed[i,j] = dS_t[i,j] + dkth[j] * k_i
      d_alpha (scalar) = sum_{i,j} dS_decayed[i,j] * S_{t-1}[i,j]
      dS_{t-1}[i,j] = dS_decayed[i,j] * alpha_t
      dg_t          = d_alpha * alpha_t

API:
    gdn_apply_path_b(q, k, v, beta, g) -> y   (autograd-traced via custom_function)
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel
from cppmega_v4._tilelang.linear_attention_path_b import gdn_forward_path_b
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


_SIMD_WIDTH = 32
_MAX_DIM    = 256  # cap on max(K, V) for the real bwd kernel


def _gdn_backward_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    dy: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Real Metal backward for the GDN recurrence.

    Returns float32 grads ``(dq, dk, dv, dbeta, dg)`` in the original input
    shapes. ``dq/dk`` are ``[B, T, H, K]``; ``dv`` is ``[B, T, H, V]``;
    ``dbeta/dg`` are ``[B, T, H]``.
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(
            f"q/k must match shape [B, T, H, K]; got q={q.shape}, k={k.shape}"
        )
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must be [B, T, H, V] with matching B,T,H; got v={v.shape}, q={q.shape}"
        )
    if dy.shape != v.shape:
        raise ValueError(f"dy must match v shape; got dy={dy.shape}, v={v.shape}")
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError(
            f"beta/g must be [B, T, H]; got beta={beta.shape}, g={g.shape}"
        )

    b, t, h, kdim = q.shape
    vdim = v.shape[-1]
    max_dim = max(kdim, vdim)
    if max_dim > _MAX_DIM:
        raise ValueError(
            f"Real-MSL GDN bwd currently requires max(K, V)<={_MAX_DIM} "
            f"(got K={kdim}, V={vdim}); caller should fall back to Path A grad path"
        )

    # Threadgroup sized to cover max(K, V) — V threads do reduction-source
    # work, K threads do the reduce-store. Pad to a 32-multiple so simd_sum
    # always covers a full simdgroup.
    use_shared = max_dim > _SIMD_WIDTH
    tg_size = ((max_dim + _SIMD_WIDTH - 1) // _SIMD_WIDTH) * _SIMD_WIDTH
    n_simd = tg_size // _SIMD_WIDTH

    # Shared-memory budget: vector tile reused across the three vector
    # reductions (dq, dk, propagation) sized for K outputs; two tiny
    # [n_simd] tiles for the scalar reductions (dbeta, d_alpha). Worst
    # case K=V=256, n_simd=8 → 256*8*4 = 8192 bytes + 32+32 = 8256 bytes
    # total per threadgroup, well below the 32KB M-series limit.
    _ = (kdim * n_simd + 2 * n_simd) * 4 if use_shared else 0  # informational

    scale = kdim ** -0.5
    q_f = q.astype(mx.float32).reshape(-1)
    k_f = k.astype(mx.float32).reshape(-1)
    v_f = v.astype(mx.float32).reshape(-1)
    beta_f = beta.astype(mx.float32).reshape(-1)
    g_f = g.astype(mx.float32).reshape(-1)
    dy_f = dy.astype(mx.float32).reshape(-1)

    # Per-step state snapshots replace algebraic peel-off (state[i] =
    # (S_t[i,j] - k_i*v_eff_t) / alpha_t). Peel-off is numerically
    # catastrophic for our recurrence: alpha < 1 so 1/alpha amplifies,
    # and after T~64 steps tiny rounding errors blow up to inf / NaN.
    shared_decls = (
        f"""
        threadgroup float tg_vec[{kdim * n_simd}];   // batched per-K partials
        threadgroup float tg_scalar0[{n_simd}];      // scalar reduction A
        threadgroup float tg_scalar1[{n_simd}];      // scalar reduction B
        """ if use_shared else ""
    )

    source = f"""
        uint tid_in_tg = thread_position_in_threadgroup.x;
        uint bh        = threadgroup_position_in_grid.x;
        uint vj        = tid_in_tg;
        bool active_v  = (vj < {vdim}u) && (bh < {b * h}u);
        uint simd_id   = tid_in_tg / 32u;
        uint lane      = tid_in_tg & 31u;

        uint bb   = bh / {h}u;
        uint head = bh % {h}u;
        {shared_decls}
        float state[{kdim}];
        float dS[{kdim}];
        float v_eff_hist[{max(t, 1)}];
        for (int i = 0; i < {kdim}; i++) {{ state[i] = 0.0f; dS[i] = 0.0f; }}

        // state_hist layout: [bh, ti, i, vj]; per (bh, ti) plane is K*V.
        int hist_bh_stride = {(t + 1) * kdim * vdim};
        int hist_ti_stride = {kdim * vdim};

        // Snapshot t=0 = zero state for this column.
        for (int i = 0; i < {kdim}; i++) {{
            int idx = bh * hist_bh_stride + 0 * hist_ti_stride + i * {vdim} + (int)vj;
            if (active_v) state_hist[idx] = 0.0f;
        }}

        // -------------------------------------------------------------
        // Forward replay: rebuild final state, capture per-step v_eff[vj],
        // snapshot S_t[:,vj] into device memory at every step.
        // -------------------------------------------------------------
        for (int ti = 0; ti < {t}; ti++) {{
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int qk_base = ((bb * {t} + ti) * {h} + head) * {kdim};
            int v_base  = ((bb * {t} + ti) * {h} + head) * {vdim};
            float v_j   = active_v ? v[v_base + vj] : 0.0f;

            for (int i = 0; i < {kdim}; i++) state[i] *= alpha_t;

            // kth[vj] = sum_i k_i * state[i, vj]  (Kahan-compensated)
            float kth_S_j = 0.0f;
            float kth_c   = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float k_i  = k[qk_base + i];
                float p    = k_i * state[i];
                float a    = p - kth_c;
                float bsum = kth_S_j + a;
                kth_c      = (bsum - kth_S_j) - a;
                kth_S_j    = bsum;
            }}

            float v_eff_j = beta_t * (v_j - kth_S_j);
            v_eff_hist[ti] = v_eff_j;
            for (int i = 0; i < {kdim}; i++) {{
                float k_i = k[qk_base + i];
                state[i] += k_i * v_eff_j;
            }}

            // Snapshot S_t for this column at slot (ti+1).
            for (int i = 0; i < {kdim}; i++) {{
                int idx = bh * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {vdim} + (int)vj;
                if (active_v) state_hist[idx] = state[i];
            }}
        }}

        // -------------------------------------------------------------
        // Reverse-time backward scan.
        // -------------------------------------------------------------
        for (int rr = 0; rr < {t}; rr++) {{
            int ti = {t} - 1 - rr;
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int qk_base = ((bb * {t} + ti) * {h} + head) * {kdim};
            int v_base  = ((bb * {t} + ti) * {h} + head) * {vdim};
            float v_j   = active_v ? v[v_base + vj]  : 0.0f;
            float dY_j  = active_v ? dy[v_base + vj] : 0.0f;

            float v_eff_t = v_eff_hist[ti];

            float S_t_col[{kdim}];
            float S_prev[{kdim}];
            for (int i = 0; i < {kdim}; i++) {{
                int idx_t   = bh * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {vdim} + (int)vj;
                int idx_tm1 = bh * hist_bh_stride + ti       * hist_ti_stride + i * {vdim} + (int)vj;
                S_t_col[i] = active_v ? state_hist[idx_t]   : 0.0f;
                S_prev[i]  = active_v ? state_hist[idx_tm1] : 0.0f;
            }}
            float S_decayed[{kdim}];
            for (int i = 0; i < {kdim}; i++) {{
                S_decayed[i] = alpha_t * S_prev[i];
            }}

            // kth_t[vj] = v_j - v_eff_t / beta_t  (algebraic identity).
            float kth_t = v_j - v_eff_t / beta_t;

            // ---- (1) o_t[vj] = sum_i (q_i*scale) * S_t[i,vj] ----
            //   dq_i = scale * sum_j dY[j] * S_t[i,j]
            //   dS_t[i,j] += dY[j] * (q_i * scale)
            for (int i = 0; i < {kdim}; i++) {{
                float q_i_scaled = q[qk_base + i] * {scale}f;
                float contrib = active_v ? (dY_j * S_t_col[i]) : 0.0f;
                float dq_i_sum = simd_sum(contrib);
                {(
                    f'''if (lane == 0u) tg_vec[i * {n_simd} + simd_id] = dq_i_sum;'''
                    if use_shared else
                    f'''if (active_v && vj == 0u) {{
                        dq[qk_base + i] = dq_i_sum * {scale}f;
                    }}'''
                )}
                dS[i] += dY_j * q_i_scaled;
            }}
            {(
                f'''threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            // Reduce-store: write K outputs. Loop in case tg_size < K.
            for (int oi_base = 0; oi_base < {kdim}; oi_base += {tg_size}) {{
                int oi = oi_base + (int)tid_in_tg;
                if (oi < {kdim}) {{
                    float total = 0.0f;
                    for (int s = 0; s < {n_simd}; s++) total += tg_vec[oi * {n_simd} + s];
                    dq[qk_base + oi] = total * {scale}f;
                }}
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else ""
            )}

            // ---- (2) S_t[i,vj] = S_decayed[i,vj] + k_i * v_eff_t ----
            //   dv_eff[vj] = sum_i dS_t[i,vj] * k_i        (per-vj scalar)
            //   dk_i (delta) = sum_j dS_t[i,j] * v_eff_t   (reduce over j)
            float dv_eff_j = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float k_i = k[qk_base + i];
                dv_eff_j += dS[i] * k_i;
            }}

            // ---- (3) v_eff[vj] = beta * (v - kth) ----
            //   dbeta_t += sum_j dv_eff[j] * (v - kth)
            //   dv[vj]   = dv_eff[vj] * beta
            //   dkth[vj] = -dv_eff[vj] * beta
            float dv_j   = dv_eff_j * beta_t;
            float dkth_j = -dv_eff_j * beta_t;
            float dbeta_contrib = active_v ? (dv_eff_j * (v_j - kth_t)) : 0.0f;
            float dbeta_simd = simd_sum(dbeta_contrib);
            if (active_v) dv[v_base + vj] = dv_j;
            {(
                f'''if (lane == 0u) tg_scalar0[simd_id] = dbeta_simd;
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (tid_in_tg == 0u) {{
                float total = 0.0f;
                for (int s = 0; s < {n_simd}; s++) total += tg_scalar0[s];
                dbeta[g_idx] = total;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else
                f'''if (active_v && vj == 0u) dbeta[g_idx] = dbeta_simd;'''
            )}

            // ---- (4) kth[vj] = sum_i k_i * S_decayed[i,vj] ----
            //   dk_i (kth) += sum_j dkth[j] * S_decayed[i,j]
            // Combined dk reduction over j:
            for (int i = 0; i < {kdim}; i++) {{
                float dk_delta = active_v ? (dS[i] * v_eff_t)       : 0.0f;
                float dk_kth   = active_v ? (dkth_j * S_decayed[i]) : 0.0f;
                float dk_i_sum = simd_sum(dk_delta + dk_kth);
                {(
                    f'''if (lane == 0u) tg_vec[i * {n_simd} + simd_id] = dk_i_sum;'''
                    if use_shared else
                    f'''if (active_v && vj == 0u) dk[qk_base + i] = dk_i_sum;'''
                )}
            }}
            {(
                f'''threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            for (int oi_base = 0; oi_base < {kdim}; oi_base += {tg_size}) {{
                int oi = oi_base + (int)tid_in_tg;
                if (oi < {kdim}) {{
                    float total = 0.0f;
                    for (int s = 0; s < {n_simd}; s++) total += tg_vec[oi * {n_simd} + s];
                    dk[qk_base + oi] = total;
                }}
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else ""
            )}

            // dS_decayed[i,vj] = dS_t[i,vj] + dkth[vj] * k_i
            for (int i = 0; i < {kdim}; i++) {{
                float k_i = k[qk_base + i];
                dS[i] = dS[i] + dkth_j * k_i;
            }}

            // ---- (5) S_decayed = alpha * S_{{t-1}} ----
            //   d_alpha (scalar) = sum_{{i,j}} dS_decayed[i,j] * S_{{t-1}}[i,j]
            //   dS_{{t-1}}[i,j]  = dS_decayed[i,j] * alpha
            //   dg_t             = d_alpha * alpha
            float d_alpha_partial = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                d_alpha_partial += dS[i] * S_prev[i];
            }}
            float d_alpha_simd = simd_sum(active_v ? d_alpha_partial : 0.0f);
            {(
                f'''if (lane == 0u) tg_scalar1[simd_id] = d_alpha_simd;
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (tid_in_tg == 0u) {{
                float total = 0.0f;
                for (int s = 0; s < {n_simd}; s++) total += tg_scalar1[s];
                dg[g_idx] = total * alpha_t;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else
                f'''if (active_v && vj == 0u) dg[g_idx] = d_alpha_simd * alpha_t;'''
            )}

            // Carry dS to next (earlier) timestep.
            for (int i = 0; i < {kdim}; i++) {{
                dS[i] = dS[i] * alpha_t;
            }}
        }}
    """

    name = f"v4_gdn_bwd_{b}_{t}_{h}_{kdim}_{vdim}"
    kernel = get_or_build_kernel(
        name=name,
        input_names=["q", "k", "v", "beta", "g", "dy"],
        output_names=["dq", "dk", "dv", "dbeta", "dg", "state_hist"],
        source=source,
    )

    grid = (tg_size * b * h, 1, 1)
    threadgroup = (tg_size, 1, 1)

    dq_flat, dk_flat, dv_flat, dbeta_flat, dg_flat, _state_hist = kernel(
        inputs=[q_f, k_f, v_f, beta_f, g_f, dy_f],
        output_shapes=[
            (b * t * h * kdim,),
            (b * t * h * kdim,),
            (b * t * h * vdim,),
            (b * t * h,),
            (b * t * h,),
            (b * h * (t + 1) * kdim * vdim,),
        ],
        output_dtypes=[mx.float32] * 6,
        grid=grid,
        threadgroup=threadgroup,
        init_value=0.0,
    )
    dq_ = dq_flat.reshape(b, t, h, kdim)
    dk_ = dk_flat.reshape(b, t, h, kdim)
    dv_ = dv_flat.reshape(b, t, h, vdim)
    dbeta_ = dbeta_flat.reshape(b, t, h)
    dg_ = dg_flat.reshape(b, t, h)
    return dq_, dk_, dv_, dbeta_, dg_


def _path_a_grad_fallback(primals, cotangent):
    q, k, v, beta, g = primals

    def _loss(q_, k_, v_, beta_, g_):
        y, _ = naive_recurrent_gated_delta_rule(q_, k_, v_, beta_, g_)
        return (y * cotangent).sum()

    return mx.grad(_loss, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, g)


@mx.custom_function
def gdn_apply_path_b(
    q: mx.array, k: mx.array, v: mx.array, beta: mx.array, g: mx.array,
) -> mx.array:
    """Forward-only call returning y (no state). Differentiable via custom VJP.

    Forward uses the fast Path B Metal kernel; backward uses the matching
    real Metal backward kernel below. For shape configurations the bwd
    kernel does not yet support (``max(K, V) > 256``), the VJP falls back
    to the Path A reference grad.
    """
    y, _ = gdn_forward_path_b(q, k, v, beta, g, output_final_state=False)
    return y


@gdn_apply_path_b.vjp
def _gdn_apply_path_b_vjp(
    primals: tuple,
    cotangent: mx.array,
    output: mx.array,
) -> tuple:
    del output
    q, k, v, beta, g = primals
    kdim = q.shape[-1]
    vdim = v.shape[-1]
    bwd_ok = (
        k.shape == q.shape
        and v.shape[:3] == q.shape[:3]
        and max(kdim, vdim) <= _MAX_DIM
    )
    if not bwd_ok:
        return _path_a_grad_fallback(primals, cotangent)
    return _gdn_backward_kernel(q, k, v, beta, g, cotangent)


__all__ = ["gdn_apply_path_b"]
