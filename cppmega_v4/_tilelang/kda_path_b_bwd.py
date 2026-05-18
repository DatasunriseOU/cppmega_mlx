"""KDA Path B forward + backward — fwd via fast Metal kernel, bwd via
hand-MSL Metal kernel (real, fused recurrent backward).

Mirrors the GDN Path B bwd pattern in ``linear_attention_path_b_bwd.py``:
forward replay snapshots S_t per j-column into a device-memory workspace
(``state_hist[B*HV, T+1, K, V]``), and the reverse-time scan reads
``S_t`` / ``S_{t-1}`` directly — never divides by ``decay = exp(g)``
(``decay`` ≤ 1 per-K makes inverse-walk numerically catastrophic; this
mirrors the mamba3_path_c switch from inverse-walk to cached snapshots).

KDA-specific bits vs GDN:
  - Gate ``g`` is per-K vector ``[B, T, HV, K]``, so ``dg`` is per-K.
  - ``v`` and the output share the V axis (not K); each thread owns one
    ``vj`` column. Constraint ``V <= 32`` (= simd width) so ``simd_sum``
    over j fits one instruction.
  - ``q``/``k`` are ``[B, T, H, K]`` with HV groups expanded; head
    indexing is ``h_idx = hv_idx // (HV/H)``.
  - ``q`` is pre-scaled by ``1/sqrt(K)`` (FLA convention).

Backward algebra (derived from the forward in ``kda_path_b.py``):

    Forward (with q' = q * scale, scale = 1/sqrt(K)):
      decay_t[i]       = exp(g_t[i])
      S_decayed[i,j]   = decay_t[i] * S_{t-1}[i,j]
      kth_t[j]         = sum_i k_t[i] * S_decayed[i,j]
      inner_t[j]       = v_t[j] - kth_t[j]
      S_t[i,j]         = S_decayed[i,j] + beta_t * k_t[i] * inner_t[j]
      o_t[j]           = sum_i q'_t[i] * S_t[i,j]

    Backward (reverse t):
      dq'_i        += sum_j dO[j] * S_t[i,j]
      dS_t[i,j]    += dO[j] * q'_i
      dv[j]         = dinner[j]
      dkth[j]       = -dinner[j]
      dinner[j]     = beta_t * sum_i dS_t[i,j] * k_t[i]
      dk_i (delta) += sum_j dS_t[i,j] * (beta_t * inner_t[j])
      dk_i (kth)   += sum_j dkth[j] * S_decayed[i,j]
      dbeta_t      += sum_{i,j} dS_t[i,j] * k_t[i] * inner_t[j]
                    = sum_i k_t[i] * (sum_j dS_t[i,j] * inner_t[j])
      dS_decayed[i,j] = dS_t[i,j] + dkth[j] * k_t[i]
      ddecay[i]     = sum_j dS_decayed[i,j] * S_{t-1}[i,j]   (per-K)
      dS_{t-1}[i,j] = dS_decayed[i,j] * decay_t[i]
      dg_t[i]       = ddecay[i] * decay_t[i]                  (per-K)

Falls back to ``mx.grad`` through ``naive_recurrent_kda`` for:
  - ``V > 32`` (multi-simdgroup not yet implemented)
  - ``initial_state`` provided
  - ``HV % H != 0``
  - any future shape outside the kernel's domain.
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel
from cppmega_v4._tilelang.kda_path_b import kda_forward_path_b
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda


_SIMD_WIDTH = 32


def _kda_backward_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    dy: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Real Metal backward for the KDA recurrence.

    Returns float32 grads ``(dq, dk, dv, dg, dbeta)``. Shapes match inputs:
        dq/dk: [B, T, H, K]
        dv:    [B, T, HV, V]
        dg:    [B, T, HV, K]
        dbeta: [B, T, HV]
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(
            f"q/k must match shape [B, T, H, K]; got q={q.shape}, k={k.shape}"
        )
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"v must be [B, T, HV, V]; got v={v.shape}")
    if g.shape != (*v.shape[:3], k.shape[-1]):
        raise ValueError(f"g must be [B, T, HV, K]; got g={g.shape}")
    if beta.shape != v.shape[:3]:
        raise ValueError(f"beta must be [B, T, HV]; got beta={beta.shape}")
    if dy.shape != v.shape:
        raise ValueError(f"dy must match v shape; got dy={dy.shape}, v={v.shape}")

    b, t, h, kdim = q.shape
    hv, vdim = v.shape[2], v.shape[-1]
    if hv % h != 0:
        raise ValueError(f"HV ({hv}) must be divisible by H ({h})")
    if vdim > _SIMD_WIDTH:
        raise ValueError(
            f"Real-MSL KDA bwd currently requires V<=32 (got {vdim}); "
            f"caller should fall back to Path A grad path"
        )
    group = hv // h

    scale = kdim ** -0.5
    q_f = q.astype(mx.float32).reshape(-1)
    k_f = k.astype(mx.float32).reshape(-1)
    v_f = v.astype(mx.float32).reshape(-1)
    g_f = g.astype(mx.float32).reshape(-1)
    beta_f = beta.astype(mx.float32).reshape(-1)
    dy_f = dy.astype(mx.float32).reshape(-1)

    source = f"""
        uint tid_in_tg = thread_position_in_threadgroup.x;
        uint bhv       = threadgroup_position_in_grid.x;
        uint vj        = tid_in_tg;
        bool active    = (vj < {vdim}u) && (bhv < {b * hv}u);

        uint bb     = bhv / {hv}u;
        uint hv_idx = bhv % {hv}u;
        uint h_idx  = hv_idx / {group}u;

        // Per-thread registers:
        //   state[K], dS[K], inner_hist[T]
        // Device-memory workspace (extra kernel output):
        //   state_hist[B*HV, T+1, K, V]
        float state[{kdim}];
        float dS[{kdim}];
        float inner_hist[{max(t, 1)}];
        for (int i = 0; i < {kdim}; i++) {{ state[i] = 0.0f; dS[i] = 0.0f; }}

        int hist_bh_stride = {(t + 1) * kdim * vdim};
        int hist_ti_stride = {kdim * vdim};

        // Snapshot t=0 = zero state for this column.
        for (int i = 0; i < {kdim}; i++) {{
            int idx = bhv * hist_bh_stride + 0 * hist_ti_stride + i * {vdim} + (int)vj;
            if (active) state_hist[idx] = 0.0f;
        }}

        // ============================================================
        // Forward replay: rebuild final state, capture inner[t] per
        // (vj), snapshot S_t[:,vj] into device memory at every step.
        // ============================================================
        for (int ti = 0; ti < {t}; ti++) {{
            int g_base   = ((bb * {t} + ti) * {hv} + hv_idx) * {kdim};
            int beta_idx = (bb * {t} + ti) * {hv} + hv_idx;
            int qk_base  = ((bb * {t} + ti) * {h} + h_idx) * {kdim};
            int v_idx    = ((bb * {t} + ti) * {hv} + hv_idx) * {vdim} + (int)vj;

            float beta_t = beta[beta_idx];
            float v_j    = active ? v[v_idx] : 0.0f;

            // Per-K decay + interleaved KS reduction.
            float kth_j = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float decay_i = exp(g[g_base + i]);
                state[i] *= decay_i;
                kth_j += k[qk_base + i] * state[i];
            }}
            float inner_j = v_j - kth_j;
            inner_hist[ti] = inner_j;

            // Rank-1 outer add: S[i, vj] += beta * k[i] * inner_j.
            for (int i = 0; i < {kdim}; i++) {{
                state[i] += beta_t * k[qk_base + i] * inner_j;
            }}

            // Snapshot S_t at slot (ti+1) for this thread's column.
            for (int i = 0; i < {kdim}; i++) {{
                int idx = bhv * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {vdim} + (int)vj;
                if (active) state_hist[idx] = state[i];
            }}
        }}

        // ============================================================
        // Reverse-time backward scan.
        // ============================================================
        for (int rr = 0; rr < {t}; rr++) {{
            int ti = {t} - 1 - rr;
            int g_base   = ((bb * {t} + ti) * {hv} + hv_idx) * {kdim};
            int beta_idx = (bb * {t} + ti) * {hv} + hv_idx;
            int qk_base  = ((bb * {t} + ti) * {h} + h_idx) * {kdim};
            int v_idx    = ((bb * {t} + ti) * {hv} + hv_idx) * {vdim} + (int)vj;

            float beta_t = beta[beta_idx];
            float v_j    = active ? v[v_idx] : 0.0f;
            float dY_j   = active ? dy[v_idx] : 0.0f;
            float inner_t = inner_hist[ti];

            // Read S_t (after step ti) and S_{{t-1}} (after step ti-1).
            float S_t_col[{kdim}];
            float S_prev[{kdim}];
            for (int i = 0; i < {kdim}; i++) {{
                int idx_t   = bhv * hist_bh_stride + (ti + 1) * hist_ti_stride + i * {vdim} + (int)vj;
                int idx_tm1 = bhv * hist_bh_stride + ti       * hist_ti_stride + i * {vdim} + (int)vj;
                S_t_col[i] = active ? state_hist[idx_t]   : 0.0f;
                S_prev[i]  = active ? state_hist[idx_tm1] : 0.0f;
            }}

            // S_decayed[i, vj] = decay_t[i] * S_prev[i, vj]
            float decay_arr[{kdim}];
            float S_decayed[{kdim}];
            for (int i = 0; i < {kdim}; i++) {{
                decay_arr[i] = exp(g[g_base + i]);
                S_decayed[i] = decay_arr[i] * S_prev[i];
            }}

            // ---- (1) o_t[j] = sum_i q'_i * S_t[i, j] ----
            //   dq'_i = sum_j dY[j] * S_t[i, j]   (reduce over j == vj axis)
            //   dS_t[i, j] += dY[j] * q'_i
            // dq writes per-K to (b, t, h_idx, i); groups inside HV share q/k,
            // so multiple hv lanes in the same (b, t, h_idx) would race.
            // We restrict the dq write to the first hv in each group via
            // (hv_idx % group == 0) and multiply by group_size? No — we need
            // the SUM of grads from all hv in the group. Use atomic_fetch_add.
            for (int i = 0; i < {kdim}; i++) {{
                float q_i_scaled = q[qk_base + i] * {scale}f;
                float contrib = active ? (dY_j * S_t_col[i]) : 0.0f;
                float dq_i_sum = simd_sum(contrib);
                if (active && vj == 0u) {{
                    atomic_fetch_add_explicit(
                        (device atomic_float*)&dq[qk_base + i],
                        dq_i_sum * {scale}f,
                        memory_order_relaxed
                    );
                }}
                dS[i] += dY_j * q_i_scaled;
            }}

            // ---- (2) S_t = S_decayed + beta * k * inner ----
            //   dinner[j] = beta_t * sum_i dS_t[i, j] * k_i   (per-j scalar)
            //   dk_i (delta) += sum_j dS_t[i, j] * (beta_t * inner_t)
            //   dbeta_t      += sum_{{i,j}} dS_t[i, j] * k_i * inner_t
            //                 = sum_i k_i * (sum_j dS_t[i, j] * inner_t)
            float sum_k_dS = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                sum_k_dS += k[qk_base + i] * dS[i];
            }}
            float dinner_j = beta_t * sum_k_dS;

            // dv[j] = dinner_j
            if (active) {{
                atomic_fetch_add_explicit(
                    (device atomic_float*)&dv[v_idx],
                    dinner_j,
                    memory_order_relaxed
                );
            }}

            // dbeta_t = sum_i k_i * (sum_j dS_t[i,j] * inner_t)
            float dbeta_partial = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float term = active ? (dS[i] * inner_t) : 0.0f;
                float term_sum = simd_sum(term);  // sum over j
                if (vj == 0u) {{
                    dbeta_partial += k[qk_base + i] * term_sum;
                }}
            }}
            if (active && vj == 0u) {{
                atomic_fetch_add_explicit(
                    (device atomic_float*)&dbeta[beta_idx],
                    dbeta_partial,
                    memory_order_relaxed
                );
            }}

            // dk_i (delta) += sum_j dS_t[i, j] * (beta_t * inner_t)
            //   Note inner_t is a per-j scalar; cannot factor it out of simd_sum.
            // dk_i (kth)   += sum_j dkth[j] * S_decayed[i, j]; dkth[j] = -dinner_j
            float dkth_j = -dinner_j;
            for (int i = 0; i < {kdim}; i++) {{
                float dk_delta = active ? (dS[i] * beta_t * inner_t) : 0.0f;
                float dk_kth   = active ? (dkth_j * S_decayed[i])    : 0.0f;
                float dk_i_sum = simd_sum(dk_delta + dk_kth);
                if (active && vj == 0u) {{
                    atomic_fetch_add_explicit(
                        (device atomic_float*)&dk[qk_base + i],
                        dk_i_sum,
                        memory_order_relaxed
                    );
                }}
            }}

            // ---- (3) dS_decayed[i, j] = dS_t[i, j] + dkth[j] * k_i ----
            for (int i = 0; i < {kdim}; i++) {{
                dS[i] = dS[i] + dkth_j * k[qk_base + i];
            }}

            // ---- (4) S_decayed[i, j] = decay[i] * S_{{t-1}}[i, j] ----
            //   ddecay[i] = sum_j dS_decayed[i, j] * S_{{t-1}}[i, j]   (per-i)
            //   dg_t[i]   = ddecay[i] * decay[i]                       (per-K)
            //   dS_{{t-1}}[i, j] = dS_decayed[i, j] * decay[i]
            // dg[b, t, hv_idx, i] is the per-K dg write at this step.
            for (int i = 0; i < {kdim}; i++) {{
                float contrib = active ? (dS[i] * S_prev[i]) : 0.0f;
                float ddecay_i = simd_sum(contrib);
                if (active && vj == 0u) {{
                    dg[g_base + i] = ddecay_i * decay_arr[i];
                }}
                dS[i] = dS[i] * decay_arr[i];
            }}
        }}
    """

    name = f"v4_kda_bwd_{b}_{t}_{h}_{hv}_{kdim}_{vdim}"
    kernel = get_or_build_kernel(
        name=name,
        input_names=["q", "k", "v", "g", "beta", "dy"],
        output_names=["dq", "dk", "dv", "dg", "dbeta", "state_hist"],
        source=source,
    )

    grid = (_SIMD_WIDTH * b * hv, 1, 1)
    threadgroup = (_SIMD_WIDTH, 1, 1)

    dq_flat, dk_flat, dv_flat, dg_flat, dbeta_flat, _ = kernel(
        inputs=[q_f, k_f, v_f, g_f, beta_f, dy_f],
        output_shapes=[
            (b * t * h * kdim,),
            (b * t * h * kdim,),
            (b * t * hv * vdim,),
            (b * t * hv * kdim,),
            (b * t * hv,),
            (b * hv * (t + 1) * kdim * vdim,),
        ],
        output_dtypes=[mx.float32] * 6,
        grid=grid,
        threadgroup=threadgroup,
        init_value=0.0,
    )
    return (
        dq_flat.reshape(b, t, h, kdim),
        dk_flat.reshape(b, t, h, kdim),
        dv_flat.reshape(b, t, hv, vdim),
        dg_flat.reshape(b, t, hv, kdim),
        dbeta_flat.reshape(b, t, hv),
    )


def _path_a_grad_fallback(primals, cotangent):
    q, k, v, g, beta = primals

    def _loss(q_, k_, v_, g_, beta_):
        y, _ = naive_recurrent_kda(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    return mx.grad(_loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)


@mx.custom_function
def kda_apply_path_b(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
) -> mx.array:
    """Forward via fast Path B Metal kernel; backward via real Metal kernel.

    Falls back to ``mx.grad`` through ``naive_recurrent_kda`` for shapes
    outside the kernel's domain (``V > 32``, ``HV % H != 0``, etc.).
    """
    y, _ = kda_forward_path_b(q, k, v, g, beta, output_final_state=False)
    return y


@kda_apply_path_b.vjp
def _kda_apply_path_b_vjp(primals, cotangent, output):
    del output
    q, k, v, g, beta = primals
    vdim = v.shape[-1]
    hv = v.shape[2]
    h = q.shape[2]
    bwd_ok = (
        v.ndim == 4
        and g.shape == (*v.shape[:3], k.shape[-1])
        and beta.shape == v.shape[:3]
        and hv % h == 0
        and vdim <= _SIMD_WIDTH
    )
    if not bwd_ok:
        return _path_a_grad_fallback(primals, cotangent)
    return _kda_backward_kernel(q, k, v, g, beta, cotangent)


__all__ = ["kda_apply_path_b"]
