"""GDN Path B forward + backward — fwd via fast Metal kernel, bwd via
hand-MSL Metal kernel (real, fused recurrent backward).

The previous revision routed VJP through ``mx.grad`` over the FLA naive
reference. That kept training correct but threw away the entire forward
speedup on every step. This revision replaces the backward with a real
``mx.fast.metal_kernel`` that fuses:

  - forward replay to recover the final state,
  - reverse-time scan with peel-off recompute of ``S_{t-1}``,
  - per-thread ``dS`` register array,
  - ``simd_sum`` across the j-axis for grads that contract over j
    (``dq[i]``, ``dk[i]``, ``dbeta``, ``dg``).

Layout matches the forward kernel: one thread per (b, h, j), one
threadgroup per (b, h) sized to a full 32-lane simdgroup so simd_sum
covers every j in one instruction (for ``head_k_dim <= 32``, which
covers every shape currently exercised by Path B; larger ``Dh`` would
need a multi-simdgroup tile and is not yet implemented).

Fallback path: when constraints are violated (``head_k_dim != head_v_dim``,
``initial_state`` provided, or ``Dh > 32``) the wrapper falls back to
``mx.grad`` through ``naive_recurrent_gated_delta_rule`` so callers never
get silently-wrong numerics.

Backward algebra (derived from the forward in
``linear_attention_path_b.py``):

    Forward (with q' = q * scale, scale = 1/sqrt(Dh)):
      alpha_t          = exp(g_t)
      S_decayed[i,j]   = alpha_t * S_{t-1}[i,j]
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

    Peel-off recompute (to walk S backwards):
      S_decayed[i,j] = S_t[i,j] - k_i * v_eff[j]
      S_{t-1}[i,j]   = S_decayed[i,j] / alpha_t

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
    shapes. ``dq/dk/dv`` are ``[B, T, H, Dh]``; ``dbeta/dg`` are ``[B, T, H]``.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q/k/v must match shape [B, T, H, Dh]; got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if dy.shape != q.shape:
        raise ValueError(f"dy must match q shape; got dy={dy.shape}, q={q.shape}")
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError(
            f"beta/g must be [B, T, H]; got beta={beta.shape}, g={g.shape}"
        )

    b, t, h, dh = q.shape
    if dh > _SIMD_WIDTH * 8:
        # Cap at 8 simdgroups (Dh up to 256) — beyond that the per-thread
        # register pressure (state[Dh], dS[Dh], S_t_col[Dh], S_prev[Dh],
        # S_decayed[Dh]) gets prohibitive.
        raise ValueError(
            f"Real-MSL GDN bwd currently requires head_dim<=256 (got {dh}); "
            f"caller should fall back to Path A grad path"
        )
    # Multi-simdgroup path when dh > 32: pad threadgroup to a 32-multiple
    # so simd_sum still works (32 lanes per simdgroup). Cross-simdgroup
    # reductions go through a threadgroup-shared-memory tile instead of
    # atomic_fetch_add_explicit (atomics serialised badly at Dh>=64 and
    # killed the multi-simdgroup speedup).
    use_shared = dh > _SIMD_WIDTH
    tg_size = ((dh + _SIMD_WIDTH - 1) // _SIMD_WIDTH) * _SIMD_WIDTH
    n_simd = tg_size // _SIMD_WIDTH  # number of simdgroups in a threadgroup
    # Shared-memory budget: one batched "[dh, n_simd]" tile reused across the
    # three vector reductions (dq, dk, propagation) plus tiny [n_simd] tiles
    # for the two scalar reductions (dbeta, d_alpha). Worst case Dh=256 with
    # 8 simdgroups → 256*8*4 = 8192 bytes vector tile + 32+32 = 8256 bytes
    # total per threadgroup, well below the 32KB M-series limit.
    shared_bytes = (dh * n_simd + 2 * n_simd) * 4 if use_shared else 0

    scale = dh ** -0.5
    q_f = q.astype(mx.float32).reshape(-1)
    k_f = k.astype(mx.float32).reshape(-1)
    v_f = v.astype(mx.float32).reshape(-1)
    beta_f = beta.astype(mx.float32).reshape(-1)
    g_f = g.astype(mx.float32).reshape(-1)
    dy_f = dy.astype(mx.float32).reshape(-1)

    # Source: each thread owns one j-column. Threadgroup is one 32-lane
    # simdgroup so simd_sum reduces across j in one instruction. Inactive
    # lanes (j >= dh) contribute zero to every simd_sum and skip stores.
    #
    # Memory plan:
    #   per-thread registers:
    #     state[Dh], dS[Dh], v_eff_hist[T]
    #   device-memory workspace (allocated as an extra kernel output):
    #     state_hist[B, H, T+1, Dh, Dh]
    #   Per-step state snapshots replace algebraic peel-off (state[i] =
    #   (S_t[i,j] - k_i*v_eff_t) / alpha_t). Peel-off is numerically
    #   catastrophic for our recurrence: alpha < 1 so 1/alpha amplifies,
    #   and after T~64 steps tiny rounding errors blow up to inf / NaN.
    #   Mirroring mamba3_path_c's switch from inverse-walk to cached
    #   snapshots (see _bwd_state_snapshots_kernel_for).
    # Shared-mem declarations only emitted in the multi-simdgroup path.
    shared_decls = (
        f"""
        threadgroup float tg_vec[{dh * n_simd}];   // batched per-i partials
        threadgroup float tg_scalar0[{n_simd}];    // scalar reduction A
        threadgroup float tg_scalar1[{n_simd}];    // scalar reduction B
        """ if use_shared else ""
    )

    source = f"""
        uint tid_in_tg = thread_position_in_threadgroup.x;
        uint bh        = threadgroup_position_in_grid.x;
        uint j         = tid_in_tg;
        bool active    = (j < {dh}u) && (bh < {b * h}u);
        uint simd_id   = tid_in_tg / 32u;
        uint lane      = tid_in_tg & 31u;

        uint bb   = bh / {h}u;
        uint head = bh % {h}u;
        {shared_decls}
        float state[{dh}];
        float dS[{dh}];
        float v_eff_hist[{max(t, 1)}];
        for (int i = 0; i < {dh}; i++) {{ state[i] = 0.0f; dS[i] = 0.0f; }}

        // state_hist layout: [bh, ti, i, j]; per (bh, ti) plane is dh*dh.
        int hist_bh_stride = {(t + 1) * dh * dh};
        int hist_ti_stride = {dh * dh};

        // Snapshot t=0 = zero state for this column.
        for (int i = 0; i < {dh}; i++) {{
            int idx = bh * hist_bh_stride + 0 * hist_ti_stride + i * {dh} + (int)j;
            if (active) state_hist[idx] = 0.0f;
        }}

        // -------------------------------------------------------------
        // Forward replay: rebuild final state, capture per-step v_eff[j],
        // and snapshot S_t[:,j] into device memory at every step (so the
        // backward pass can read S_{{t-1}} directly instead of dividing
        // by alpha — alpha<1 makes inverse-walk numerically unstable).
        // -------------------------------------------------------------
        for (int ti = 0; ti < {t}; ti++) {{
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int kv_base = (bb * {t} + ti) * {h * dh} + head * {dh};
            float v_j   = active ? v[kv_base + j] : 0.0f;

            for (int i = 0; i < {dh}; i++) state[i] *= alpha_t;

            float kth_S_j = 0.0f;
            float kth_c   = 0.0f;
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                float p   = k_i * state[i];
                float a   = p - kth_c;
                float bsum = kth_S_j + a;
                kth_c     = (bsum - kth_S_j) - a;
                kth_S_j   = bsum;
            }}

            float v_eff_j = beta_t * (v_j - kth_S_j);
            v_eff_hist[ti] = v_eff_j;
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                state[i] += k_i * v_eff_j;
            }}

            // Snapshot S_t for this column at slot (ti+1).
            for (int i = 0; i < {dh}; i++) {{
                int idx = bh * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {dh} + (int)j;
                if (active) state_hist[idx] = state[i];
            }}
        }}

        // -------------------------------------------------------------
        // Reverse-time backward scan. Read S_t[:,j] and S_{{t-1}}[:,j]
        // directly from the snapshot buffer; no division by alpha.
        // -------------------------------------------------------------
        for (int rr = 0; rr < {t}; rr++) {{
            int ti = {t} - 1 - rr;
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int kv_base = (bb * {t} + ti) * {h * dh} + head * {dh};
            float v_j   = active ? v[kv_base + j] : 0.0f;
            float dY_j  = active ? dy[kv_base + j] : 0.0f;

            float v_eff_t = v_eff_hist[ti];

            // Read S_t (after step ti) and S_{{t-1}} (after step ti-1) for
            // this thread's column. S_decayed[i,j] = alpha * S_{{t-1}}[i,j].
            float S_t_col[{dh}];
            float S_prev[{dh}];
            for (int i = 0; i < {dh}; i++) {{
                int idx_t   = bh * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {dh} + (int)j;
                int idx_tm1 = bh * hist_bh_stride + ti       * hist_ti_stride + i * {dh} + (int)j;
                S_t_col[i] = active ? state_hist[idx_t]   : 0.0f;
                S_prev[i]  = active ? state_hist[idx_tm1] : 0.0f;
            }}
            float S_decayed[{dh}];
            for (int i = 0; i < {dh}; i++) {{
                S_decayed[i] = alpha_t * S_prev[i];
            }}

            // kth_t[j] = v_j - v_eff_t / beta_t  (algebraic identity from
            // v_eff = beta*(v - kth); used only for dbeta_contrib).
            float kth_t = v_j - v_eff_t / beta_t;

            // ---- (1) o_t[j] = sum_i (q_i*scale) * S_t[i,j] ----
            //   dq_i = scale * sum_j dY[j] * S_t[i,j]   (reduce over j)
            //   dS_t[i,j] += dY[j] * (q_i * scale)
            for (int i = 0; i < {dh}; i++) {{
                float q_i_scaled = q[kv_base + i] * {scale}f;
                float contrib = active ? (dY_j * S_t_col[i]) : 0.0f;
                float dq_i_sum = simd_sum(contrib);
                {(
                    f'''// Stash this simdgroup's partial; one barrier amortises
                    // all dh writes (no atomic contention).
                    if (lane == 0u) tg_vec[i * {n_simd} + simd_id] = dq_i_sum;'''
                    if use_shared else
                    f'''if (active && j == 0u) {{
                        dq[kv_base + i] = dq_i_sum * {scale}f;
                    }}'''
                )}
                dS[i] += dY_j * q_i_scaled;
            }}
            {(
                f'''threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            // Reduce: one thread per output cell; lanes >= dh idle.
            if (tid_in_tg < {dh}u) {{
                int oi = (int)tid_in_tg;
                float total = 0.0f;
                for (int s = 0; s < {n_simd}; s++) total += tg_vec[oi * {n_simd} + s];
                dq[kv_base + oi] = total * {scale}f;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else ""
            )}

            // ---- (2) S_t[i,j] = S_decayed[i,j] + k_i * v_eff_t ----
            //   dv_eff[j]      = sum_i dS_t[i,j] * k_i        (per-j scalar)
            //   dk_i (delta)   = sum_j dS_t[i,j] * v_eff_t    (reduce over j)
            float dv_eff_j = 0.0f;
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                dv_eff_j += dS[i] * k_i;
            }}

            // ---- (3) v_eff[j] = beta * (v - kth) ----
            //   dbeta_t += sum_j dv_eff[j] * (v - kth)
            //   dv[j]    = dv_eff[j] * beta
            //   dkth[j]  = -dv_eff[j] * beta
            float dv_j   = dv_eff_j * beta_t;
            float dkth_j = -dv_eff_j * beta_t;
            float dbeta_contrib = active ? (dv_eff_j * (v_j - kth_t)) : 0.0f;
            float dbeta_simd = simd_sum(dbeta_contrib);
            if (active) dv[kv_base + j] = dv_j;
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
                f'''if (active && j == 0u) dbeta[g_idx] = dbeta_simd;'''
            )}

            // ---- (4) kth[j] = sum_i k_i * S_decayed[i,j] ----
            //   dk_i (kth) += sum_j dkth[j] * S_decayed[i,j]
            // Combined dk reduction over j:
            for (int i = 0; i < {dh}; i++) {{
                float dk_delta = active ? (dS[i] * v_eff_t)        : 0.0f;
                float dk_kth   = active ? (dkth_j * S_decayed[i])  : 0.0f;
                float dk_i_sum = simd_sum(dk_delta + dk_kth);
                {(
                    f'''if (lane == 0u) tg_vec[i * {n_simd} + simd_id] = dk_i_sum;'''
                    if use_shared else
                    f'''if (active && j == 0u) dk[kv_base + i] = dk_i_sum;'''
                )}
            }}
            {(
                f'''threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (tid_in_tg < {dh}u) {{
                int oi = (int)tid_in_tg;
                float total = 0.0f;
                for (int s = 0; s < {n_simd}; s++) total += tg_vec[oi * {n_simd} + s];
                dk[kv_base + oi] = total;
            }}
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);'''
                if use_shared else ""
            )}

            // dS_decayed[i,j] = dS_t[i,j] + dkth[j] * k_i
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                dS[i] = dS[i] + dkth_j * k_i;
            }}

            // ---- (5) S_decayed = alpha * S_{{t-1}} ----
            //   d_alpha (scalar) = sum_{{i,j}} dS_decayed[i,j] * S_{{t-1}}[i,j]
            //   dS_{{t-1}}[i,j]  = dS_decayed[i,j] * alpha
            //   dg_t             = d_alpha * alpha
            float d_alpha_partial = 0.0f;
            for (int i = 0; i < {dh}; i++) {{
                d_alpha_partial += dS[i] * S_prev[i];
            }}
            float d_alpha_simd = simd_sum(active ? d_alpha_partial : 0.0f);
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
                f'''if (active && j == 0u) dg[g_idx] = d_alpha_simd * alpha_t;'''
            )}

            // Carry dS to next (earlier) timestep.
            for (int i = 0; i < {dh}; i++) {{
                dS[i] = dS[i] * alpha_t;
            }}
        }}
    """

    name = f"v4_gdn_bwd_{b}_{t}_{h}_{dh}"
    kernel = get_or_build_kernel(
        name=name,
        input_names=["q", "k", "v", "beta", "g", "dy"],
        output_names=["dq", "dk", "dv", "dbeta", "dg", "state_hist"],
        source=source,
    )

    grid = (tg_size * b * h, 1, 1)
    threadgroup = (tg_size, 1, 1)

    # state_hist workspace: [B*H, T+1, Dh, Dh] flat — written by fwd replay,
    # consumed by reverse pass. Discarded after the kernel returns.
    dq_flat, dk_flat, dv_flat, dbeta_flat, dg_flat, _state_hist = kernel(
        inputs=[q_f, k_f, v_f, beta_f, g_f, dy_f],
        output_shapes=[
            (b * t * h * dh,),
            (b * t * h * dh,),
            (b * t * h * dh,),
            (b * t * h,),
            (b * t * h,),
            (b * h * (t + 1) * dh * dh,),
        ],
        output_dtypes=[mx.float32] * 6,
        grid=grid,
        threadgroup=threadgroup,
        init_value=0.0,
    )
    dq_ = dq_flat.reshape(b, t, h, dh)
    dk_ = dk_flat.reshape(b, t, h, dh)
    dv_ = dv_flat.reshape(b, t, h, dh)
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
    kernel does not yet support (``head_k_dim != head_v_dim`` or
    ``head_dim > 32``), the VJP falls back to the Path A reference grad.
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
    # Constraints for the real-MSL bwd kernel: shapes match + Dh <= 256
    # (multi-simdgroup path, threadgroup-shared-memory accumulation across
    # simdgroups — replaced atomic_fetch_add to remove serialisation).
    dh = q.shape[-1]
    bwd_ok = (
        k.shape == q.shape
        and v.shape == q.shape
        and dh <= _SIMD_WIDTH * 8
    )
    if not bwd_ok:
        return _path_a_grad_fallback(primals, cotangent)
    return _gdn_backward_kernel(q, k, v, beta, g, cotangent)


__all__ = ["gdn_apply_path_b"]
