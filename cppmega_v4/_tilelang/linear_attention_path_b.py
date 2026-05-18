"""GDN Path B — hand-MSL recurrent forward via mx.fast.metal_kernel.

Adapted from ``mlx-recurrence/mlx_recurrence/gla_scan.py`` (MIT, D-CSIL):
the GLA recurrence
    h[i] = gate * h[i] + k[i] * v[j]
is extended to the GDN recurrence (FLA naive form)
    h          *= exp(g)                      # alpha decay
    v_eff[j]    = beta * (v[j] - sum_i k[i] * h[i,j])
    h[i]       += k[i] * v_eff[j]
    o[j]        = sum_i q[i] * h[i,j]

Per-thread layout (j fixed, i varies in registers): each thread owns
column j of the K x V state for one (batch, head). K is the key dim
(loop length in registers), V is the value dim (grid x-axis).

Supports:
    - ``head_k_dim != head_v_dim`` (separate K and V dimensions).
    - ``initial_state`` of shape [B, H, K, V] for streaming decode.
    - ``scale`` parameter (defaults to 1/sqrt(K)).

Backward pass: not implemented in this revision — calls fall back to the
Path A reference for autograd. Forward kernel can still be used for
inference benchmarking via the dispatch table.

API matches ``naive_recurrent_gated_delta_rule`` (returns ``(o, final_state)``).
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4._tilelang._kernel_cache import get_or_build_kernel


def _gdn_forward_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    *,
    scale: float | None = None,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal forward for GDN recurrence.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        beta: [B, T, H]
        g:    [B, T, H]
        scale: optional float — defaults to 1/sqrt(K)
        h0:   optional [B, H, K, V] initial state

    Returns:
        o:       [B, T, H, V]
        h_final: [B, H, K, V]   (final timestep's state)
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(
            f"q/k must match shape [B, T, H, K]; got q={q.shape}, k={k.shape}"
        )
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must be [B, T, H, V] with matching B,T,H; got v={v.shape}, q={q.shape}"
        )
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError(
            f"beta/g must be [B, T, H]; got beta={beta.shape}, g={g.shape}"
        )

    b, t, h, kdim = q.shape
    vdim = v.shape[-1]

    if h0 is not None and tuple(h0.shape) != (b, h, kdim, vdim):
        raise ValueError(
            f"initial_state must be [B={b}, H={h}, K={kdim}, V={vdim}]; got {h0.shape}"
        )

    sc = float(scale) if scale is not None else (kdim ** -0.5)
    q = q.astype(mx.float32) * sc
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    beta = beta.astype(mx.float32)
    g = g.astype(mx.float32)

    q_flat = q.reshape(-1)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    beta_flat = beta.reshape(-1)
    g_flat = g.reshape(-1)

    has_h0 = h0 is not None
    if has_h0:
        h0_flat = h0.astype(mx.float32).reshape(-1)

    init_state_block = (
        f"""
        // Init from h0[bb, head, i, vj]
        int h0_base = (bb * {h} + head) * {kdim * vdim} + vj;
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
        uint vj = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;

        if (vj >= {vdim}u || bh >= {b * h}u) return;

        uint bb   = bh / {h}u;
        uint head = bh % {h}u;

        // Thread-local state: one column of the K x V state matrix (length K)
        float state[{kdim}];
        {init_state_block}

        for (int ti = 0; ti < {t}; ti++) {{
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int qk_base   = ((bb * {t} + ti) * {h} + head) * {kdim};
            int v_base    = ((bb * {t} + ti) * {h} + head) * {vdim};
            float v_j     = v[v_base + vj];

            // Phase 1: alpha decay (per FLA naive: applied BEFORE delta)
            for (int i = 0; i < {kdim}; i++) state[i] *= alpha_t;

            // Phase 2: kth_S_j = sum_i k[i] * state[i, j]  (Kahan-compensated)
            float kth_S_j = 0.0f;
            float kth_c   = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float p = k[qk_base + i] * state[i];
                float a = p - kth_c;
                float b = kth_S_j + a;
                kth_c   = (b - kth_S_j) - a;
                kth_S_j = b;
            }}

            // Phase 3: v_eff = beta * (v - kth_S)
            float v_eff_j = beta_t * (v_j - kth_S_j);

            // Phase 4: state[i] += k[i] * v_eff_j  and  o[j] = sum_i q[i] * state[i]
            float o_j = 0.0f;
            for (int i = 0; i < {kdim}; i++) {{
                float k_i = k[qk_base + i];
                float q_i = q[qk_base + i];
                state[i] += k_i * v_eff_j;
                o_j += q_i * state[i];
            }}

            output[v_base + vj] = o_j;
        }}

        // Write final state column: state_final[bb, head, :, vj]
        int sf_base = (bb * {h} + head) * {kdim * vdim} + vj;
        for (int i = 0; i < {kdim}; i++) {{
            state_final[sf_base + i * {vdim}] = state[i];
        }}
    """

    h0_tag = "h0" if has_h0 else "noh0"
    kernel_name = f"v4_gdn_fwd_{b}_{t}_{h}_{kdim}_{vdim}_{h0_tag}"
    input_names = ["q", "k", "v", "beta", "g"] + (["h0"] if has_h0 else [])
    kernel = get_or_build_kernel(
        name=kernel_name,
        input_names=input_names,
        output_names=["output", "state_final"],
        source=source,
    )

    grid = (vdim, b * h, 1)
    tg_x = min(vdim, 64)
    threadgroup = (tg_x, 1, 1)

    inputs = [q_flat, k_flat, v_flat, beta_flat, g_flat]
    if has_h0:
        inputs.append(h0_flat)

    results = kernel(
        inputs=inputs,
        output_shapes=[
            (b * t * h * vdim,),
            (b * h * kdim * vdim,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    o = results[0].reshape(b, t, h, vdim)
    state_final = results[1].reshape(b, h, kdim, vdim)
    return o, state_final


def gdn_forward_path_b(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    *,
    scale: float | None = None,
    initial_state: mx.array | None = None,
    output_final_state: bool = False,
):
    """Path B forward, signature matching ``naive_recurrent_gated_delta_rule``.

    Supports:
        - ``initial_state`` shape [B, H, K, V] (streaming decode).
        - Custom ``scale`` (defaults to 1/sqrt(K) per FLA convention).
        - ``head_k_dim != head_v_dim``.
    """
    o, sf = _gdn_forward_kernel(q, k, v, beta, g, scale=scale, h0=initial_state)
    return o, (sf if output_final_state else None)


__all__ = ["gdn_forward_path_b"]
