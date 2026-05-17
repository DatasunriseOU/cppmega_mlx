"""GDN Path B — hand-MSL recurrent forward via mx.fast.metal_kernel.

Adapted from ``mlx-recurrence/mlx_recurrence/gla_scan.py`` (MIT, D-CSIL):
the GLA recurrence
    h[i] = gate * h[i] + k[i] * v[j]
is extended to the GDN recurrence (FLA naive form)
    h          *= exp(g)                      # alpha decay
    v_eff[j]    = beta * (v[j] - sum_i k[i] * h[i,j])
    h[i]       += k[i] * v_eff[j]
    o[j]        = sum_i q[i] * h[i,j]

Per-thread layout (j fixed, i varies in registers) matches the GLA kernel:
each thread owns column j of the H_k x H_v state for one (batch, head).

Backward pass: not implemented in this revision — calls fall back to the
Path A reference for autograd. Forward kernel can still be used for
inference benchmarking via the dispatch table.

API matches ``naive_recurrent_gated_delta_rule`` (returns ``(o, final_state)``).
Constraints (relaxed in future versions):
    - head_k_dim must equal head_v_dim (same shape per head — matches the
      GLA scan assumption inherited from mlx-recurrence).
    - dtype: all inputs cast to float32 internally (matches FLA naive).
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
) -> tuple[mx.array, mx.array]:
    """Metal forward for GDN recurrence.

    Args:
        q, k: [B, T, H, Dh]
        v:    [B, T, H, Dh]   — must match k's head dim for this kernel
        beta: [B, T, H]
        g:    [B, T, H]       — gate-decay logit (alpha = exp(g))

    Returns:
        o: [B, T, H, Dh]
        h_final: [B, H, Dh, Dh]   (only the last timestep's state)
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q/k/v must match shape [B, T, H, Dh]; got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError(
            f"beta/g must be [B, T, H]; got beta={beta.shape}, g={g.shape}"
        )

    b, t, h, dh = q.shape
    # Match FLA naive: cast to float32, apply 1/sqrt(Dh) scale to q.
    q = q.astype(mx.float32) * (dh ** -0.5)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    beta = beta.astype(mx.float32)
    g = g.astype(mx.float32)

    q_flat = q.reshape(-1)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    beta_flat = beta.reshape(-1)
    g_flat = g.reshape(-1)

    source = f"""
        uint j  = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;

        if (j >= {dh}u || bh >= {b * h}u) return;

        uint bb   = bh / {h}u;
        uint head = bh % {h}u;

        // Thread-local state: one column of the Dh x Dh state matrix
        float state[{dh}];
        for (int i = 0; i < {dh}; i++) state[i] = 0.0f;

        for (int ti = 0; ti < {t}; ti++) {{
            int g_idx     = bb * {t * h} + ti * {h} + head;
            float alpha_t = exp(g[g_idx]);
            float beta_t  = beta[g_idx];

            int kv_base = (bb * {t} + ti) * {h * dh} + head * {dh};
            float v_j   = v[kv_base + j];

            // Phase 1: alpha decay (per FLA naive: applied BEFORE delta)
            for (int i = 0; i < {dh}; i++) state[i] *= alpha_t;

            // Phase 2: kth_S_j = sum_i k[i] * state[i, j]  (own column)
            // mlx-lm PR #1066: Kahan-compensated summation. Without this,
            // long-T runs accumulate bf16 rounding into kv_mem and the
            // delta-corrected state drifts. Compensation costs 3 FLOPs/iter.
            float kth_S_j = 0.0f;
            float kth_c   = 0.0f;
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                float p   = k_i * state[i];
                float a   = p - kth_c;
                float b   = kth_S_j + a;
                kth_c     = (b - kth_S_j) - a;
                kth_S_j   = b;
            }}

            // Phase 3: v_eff = beta * (v - kth_S)
            float v_eff_j = beta_t * (v_j - kth_S_j);

            // Phase 4: state[i] += k[i] * v_eff_j  and  o[j] = sum_i q[i] * state[i]
            float o_j = 0.0f;
            for (int i = 0; i < {dh}; i++) {{
                float k_i = k[kv_base + i];
                float q_i = q[kv_base + i];
                state[i] += k_i * v_eff_j;
                o_j += q_i * state[i];
            }}

            output[kv_base + j] = o_j;
        }}

        // Write final state column for this thread (if requested by caller)
        int sf_base = (bb * {h} + head) * {dh * dh} + j;
        for (int i = 0; i < {dh}; i++) {{
            state_final[sf_base + i * {dh}] = state[i];
        }}
    """

    kernel_name = f"v4_gdn_fwd_{b}_{t}_{h}_{dh}"
    kernel = get_or_build_kernel(
        name=kernel_name,
        input_names=["q", "k", "v", "beta", "g"],
        output_names=["output", "state_final"],
        source=source,
    )

    grid = (dh, b * h, 1)
    tg_x = min(dh, 64)
    threadgroup = (tg_x, 1, 1)

    results = kernel(
        inputs=[q_flat, k_flat, v_flat, beta_flat, g_flat],
        output_shapes=[
            (b * t * h * dh,),
            (b * h * dh * dh,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    o = results[0].reshape(b, t, h, dh)
    state_final = results[1].reshape(b, h, dh, dh)
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

    Constraints:
        - ``initial_state`` not supported in this revision (must be None);
          falls back to zero-init inside the kernel.
        - ``scale`` not supported (uses 1/sqrt(Dh) per FLA convention).
        - ``head_k_dim == head_v_dim`` required.

    On unsupported configs returns the Path A reference output instead so
    the dispatch never silently produces wrong numerics.
    """
    if initial_state is not None or scale is not None or v.shape[-1] != k.shape[-1]:
        # Fall through to Path A for cases this kernel doesn't yet cover.
        from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
            naive_recurrent_gated_delta_rule,
        )
        return naive_recurrent_gated_delta_rule(
            q, k, v, beta, g,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
        )
    o, sf = _gdn_forward_kernel(q, k, v, beta, g)
    return o, (sf if output_final_state else None)


__all__ = ["gdn_forward_path_b"]
