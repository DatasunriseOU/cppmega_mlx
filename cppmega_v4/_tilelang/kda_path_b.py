"""KDA Path B — hand-MSL recurrent forward via mx.fast.metal_kernel.

Mirrors ``linear_attention_path_b.py`` but for the KDA recurrence (FLA naive):

    q, k: [B, T, H, K]         (q scaled by 1/sqrt(K), then both repeat to HV)
    v:    [B, T, HV, V]
    g:    [B, T, HV, K]        (per-K vectorized log-gate)
    beta: [B, T, HV]
    S:    [B, HV, K, V]

    for t in [0, T):
        S       *= exp(g_t)               # per-K decay, broadcast over V
        inner    = v_t - sum_k(k_t * S)
        S       += (beta_t * k_t)[:, :, None] * inner[:, None, :]
        o_t      = sum_k(q_t * S)

Per-thread layout: one thread per (batch, hv_head, v_index). Each thread
owns one column of the K x V state for one (b, hv) and walks time
serially while reducing over K for state-derived quantities.

Group expansion (HV / H): each ``hv`` maps to ``h = hv // G`` for indexing
into q and k. The kernel performs that mapping inline.

Supports:
    - ``initial_state`` of shape [B, HV, K, V] (streaming decode).
    - Custom ``scale`` (defaults to 1/sqrt(K)).
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel


def _kda_forward_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    scale: float | None = None,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal forward for KDA recurrence.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, HV, V]
        g:    [B, T, HV, K]   per-K vectorized log-gate
        beta: [B, T, HV]
        scale: optional float — defaults to 1/sqrt(K)
        h0:   optional [B, HV, K, V] initial state

    Returns:
        o: [B, T, HV, V]
        S_final: [B, HV, K, V]
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(f"q/k must be [B,T,H,K]; got q={q.shape}, k={k.shape}")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"v must be [B,T,HV,V]; got v={v.shape}")
    if g.shape != (*v.shape[:3], k.shape[-1]):
        raise ValueError(f"g must be [B,T,HV,K]; got g={g.shape}")
    if beta.shape != v.shape[:3]:
        raise ValueError(f"beta must be [B,T,HV]; got beta={beta.shape}")

    b, t, h, kdim = q.shape
    hv, vdim = v.shape[2], v.shape[-1]
    if hv % h != 0:
        raise ValueError(f"HV ({hv}) must be divisible by H ({h})")
    group = hv // h

    if h0 is not None and tuple(h0.shape) != (b, hv, kdim, vdim):
        raise ValueError(
            f"initial_state must be [B={b}, HV={hv}, K={kdim}, V={vdim}]; got {h0.shape}"
        )

    sc = float(scale) if scale is not None else (kdim ** -0.5)
    q = q.astype(mx.float32) * sc
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)

    q_flat = q.reshape(-1)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    g_flat = g.reshape(-1)
    beta_flat = beta.reshape(-1)

    has_h0 = h0 is not None
    if has_h0:
        h0_flat = h0.astype(mx.float32).reshape(-1)

    init_state_block = (
        f"""
        // Init from h0[bb, hv_idx, i, vj]
        int h0_base = (bb * {hv} + hv_idx) * {kdim * vdim} + vj;
        for (int i = 0; i < {kdim}; i++) {{
            state[i] = h0[h0_base + i * {vdim}];
        }}
        """
        if has_h0
        else f"""
        for (int i = 0; i < {kdim}; i++) state[i] = 0.0f;
        """
    )

    source = f"""
        uint vj  = thread_position_in_grid.x;
        uint bhv = thread_position_in_grid.y;

        if (vj >= {vdim}u || bhv >= {b * hv}u) return;

        uint bb     = bhv / {hv}u;
        uint hv_idx = bhv % {hv}u;
        uint h_idx  = hv_idx / {group}u;

        // Per-thread state column: S[bb, hv_idx, :, vj]  size K
        float state[{kdim}];
        {init_state_block}

        for (int ti = 0; ti < {t}; ti++) {{
            // g_base: g[bb, ti, hv_idx, 0]
            int g_base    = ((bb * {t} + ti) * {hv} + hv_idx) * {kdim};
            // beta scalar: beta[bb, ti, hv_idx]
            int beta_idx  = (bb * {t} + ti) * {hv} + hv_idx;
            float beta_t  = beta[beta_idx];

            // k/q base for this (b, ti, h_idx)
            int qk_base   = ((bb * {t} + ti) * {h} + h_idx) * {kdim};
            // v scalar: v[bb, ti, hv_idx, vj]
            int v_idx     = ((bb * {t} + ti) * {hv} + hv_idx) * {vdim} + vj;
            float v_j     = v[v_idx];

            // Phase 1: per-K decay  AND  KS reduction along K (interleaved)
            float kth_S_j = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float decay_i = exp(g[g_base + i]);
                state[i] *= decay_i;
                kth_S_j += k[qk_base + i] * state[i];
            }}

            // Phase 2: delta correction
            float inner_j = v_j - kth_S_j;

            // Phase 3: rank-1 outer add  S[i,j] += beta * k[i] * inner  AND  o[j] = sum_i q[i] * S[i,j]
            float o_j = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float k_i = k[qk_base + i];
                float q_i = q[qk_base + i];
                state[i] += beta_t * k_i * inner_j;
                o_j += q_i * state[i];
            }}

            // output[bb, ti, hv_idx, vj]
            int o_idx = ((bb * {t} + ti) * {hv} + hv_idx) * {vdim} + vj;
            output[o_idx] = o_j;
        }}

        // Final state column: S_final[bb, hv_idx, :, vj]
        int sf_base = (bb * {hv} + hv_idx) * {kdim * vdim} + vj;
        for (int i = 0; i < {kdim}; i++) {{
            state_final[sf_base + i * {vdim}] = state[i];
        }}
    """

    h0_tag = "h0" if has_h0 else "noh0"
    name = f"v4_kda_fwd_{b}_{t}_{h}_{hv}_{kdim}_{vdim}_{h0_tag}"
    input_names = ["q", "k", "v", "g", "beta"] + (["h0"] if has_h0 else [])
    kernel = get_or_build_kernel(
        name=name,
        input_names=input_names,
        output_names=["output", "state_final"],
        source=source,
    )

    grid = (vdim, b * hv, 1)
    tg_x = min(vdim, 64)
    threadgroup = (tg_x, 1, 1)

    inputs = [q_flat, k_flat, v_flat, g_flat, beta_flat]
    if has_h0:
        inputs.append(h0_flat)

    out, sf = kernel(
        inputs=inputs,
        output_shapes=[
            (b * t * hv * vdim,),
            (b * hv * kdim * vdim,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    return out.reshape(b, t, hv, vdim), sf.reshape(b, hv, kdim, vdim)


def kda_forward_path_b(
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
    """KDA Path B forward, signature matching ``naive_recurrent_kda``.

    Falls back to Path A only when HV is not divisible by H (architectural
    mismatch). Supports ``initial_state`` and custom ``scale``.
    """
    if v.shape[2] % q.shape[2] != 0:
        from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda
        return naive_recurrent_kda(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
        )
    o, sf = _kda_forward_kernel(q, k, v, g, beta, scale=scale, h0=initial_state)
    return o, (sf if output_final_state else None)


__all__ = ["kda_forward_path_b"]
