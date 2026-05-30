# Derived from ml-explore/mlx-lm PR #1217 (mlx_lm/models/gated_delta.py).
# Upstream license: MIT, Copyright © 2023 Apple Inc.
#
# Public surface used by cppmega_v4 Path E:
#   - gated_delta_update(...) -> (y, state)
#   - gated_delta_kernel(...) — low-level Metal kernel entry
#   - gated_delta_ops(...) — pure-MLX reference for prefill
#   - compute_g(A_log, a, dt_bias) — gate transform
# cppmega_v4/nn/path_e_adapter.py wraps our (g, beta) API to this module's
# (a, b, A_log, dt_bias) API.
#
# CPPMEGA DELTA over the PR #1217 snapshot (see VENDORED_MANIFEST.json):
#   In-MSL Dk remainder-mask so the fast Metal kernel is CORRECT for ANY Dk,
#   not just Dk % 32 == 0. Upstream uses `n_per_t = Dk / 32` (integer floor),
#   so the trailing Dk % 32 keys owned past lane 31's slice were NEVER loaded
#   -> silent truncation / wrong answer for non-multiple-of-32 Dk. We switch
#   to a ceil tiling `n_per_t = (Dk + 31) / 32` and guard every per-`i` load
#   with `s_idx < Dk`, substituting additive/multiplicative identities for the
#   out-of-range tail (0 for k/q/state contributions; the VECTORIZED gate uses
#   decay = exp(0) = 1, i.e. g_access expands to 0.0 on the tail so that the
#   pre-reduction simd_sum is never poisoned). The scalar GDN gate is per-(b,t,
#   hv) broadcast and so unaffected. Everything else is the verbatim PR #1217
#   snapshot (incl. the PR #1066 Kahan-compensated kv_mem + 4-way unroll).

from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized.
    # Vectorized g is [B,T,Hv,Dk] and is read per-key (s_idx). On the Dk
    # remainder tail (s_idx >= Dk) the load would be OOB, so we mask it and
    # substitute decay = 1.0 (== exp(0)) for that lane's tail entry. Using 1.0
    # leaves state[i] unchanged; since the tail state[i] is itself 0 (masked
    # load) and the tail k_/q_ contributions are 0, the tail never poisons the
    # pre-reduction simd_sum(kv_mem)/simd_sum(out).
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "(s_idx < Dk ? static_cast<float>(g_[s_idx]) : 1.0f)"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        // cppmega delta: ceil tiling so the trailing Dk % 32 keys are owned by
        // a lane and loaded (upstream used floor `Dk / 32`, dropping the tail).
        constexpr int n_per_t = (Dk + 31) / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          // Dk remainder-mask: tail lanes hold 0 (additive identity) so they
          // contribute nothing to any simd_sum reduction.
          state[i] = (s_idx < Dk) ? static_cast<float>(i_state[s_idx]) : 0.0f;
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        // mlx-lm PR #1066: Kahan-compensated kv_mem accumulation +
        // 4-way time-loop unroll. Fixes loss-of-precision on long T where
        // bf16 state values cause kv_mem to drift away from the true sum.
        #define BODY() {{ \
            float kv_mem = 0.0f, kv_c = 0.0f; \
            for (int i = 0; i < n_per_t; ++i) {{ \
              auto s_idx = n_per_t * dk_idx + i; \
              if (s_idx >= Dk) continue; \
              state[i] *= {g_access}; \
              auto p = state[i] * k_[s_idx]; \
              auto a = p - kv_c; \
              auto b = kv_mem + a; \
              kv_c = (b - kv_mem) - a; \
              kv_mem = b; \
            }} \
            kv_mem = simd_sum(kv_mem); \
            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx]; \
            float out = 0.0f; \
            for (int i = 0; i < n_per_t; ++i) {{ \
              auto s_idx = n_per_t * dk_idx + i; \
              if (s_idx >= Dk) continue; \
              state[i] += k_[s_idx] * delta; \
              out += state[i] * q_[s_idx]; \
            }} \
            out = simd_sum(out); \
            if (thread_index_in_simdgroup == 0) \
              y[dv_idx] = static_cast<InT>(out); \
        }}
        #define ADV() q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y += Hv * Dv; {g_advance} beta_ += Hv;

        int t = 0;
        for (; t + 3 < T; t += 4) {{
          if ({mask_source}) BODY() ADV()
          if ({"mask[b_idx * T + t + 1]" if has_mask else "true"}) BODY() ADV()
          if ({"mask[b_idx * T + t + 2]" if has_mask else "true"}) BODY() ADV()
          if ({"mask[b_idx * T + t + 3]" if has_mask else "true"}) BODY() ADV()
        }}
        for (; t < T; ++t) {{
          if ({mask_source}) BODY()
          ADV()
        }}
        #undef BODY
        #undef ADV
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          if (s_idx < Dk) o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    training: bool = False,
) -> Tuple[mx.array, mx.array]:
    if training:
        # Chunked VJP path with O(T/chunk) autodiff graph — fits T≥2048
        # on 36 GB Apple Silicon where the Python-ops path OOMs.
        # Metal backward kernel is 8–11× faster than the Python reference. It
        # now handles ANY Dk (in-MSL remainder-mask), but still needs Dv%4==0
        # for the four-SIMD-group cooperative reduction (ragged Dv backward
        # deferred). The unmasked GPU path is also required.
        Dv = v.shape[-1]
        can_use_metal = (
            mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and mask is None
            and Dv % 4 == 0
        )
        if can_use_metal:
            try:
                from ._mlx_lm_gated_delta_vjp_metal_vendored import (
                    gated_delta_update_vjp_metal,
                )
                return gated_delta_update_vjp_metal(
                    q, k, v, a, b, A_log, dt_bias, state, mask
                )
            except ImportError:
                pass
        from ._mlx_lm_gated_delta_vjp_vendored import gated_delta_update_vjp
        return gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, state, mask)

    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
