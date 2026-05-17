"""ROI 5 — FlashMLA absorb trick (W_UK · W_O folded into QK GEMM).

This is the algebraic identity from
``~/sources/rent_kernels/FlashMLA/docs/20250422-new-kernel-deep-dive.md``.
There is no PyTorch reference in the FlashMLA repo (the trick lives inside
the CUDA/CUTLASS kernel), so this module ports the math directly:

    Standard MLA decode forward:
        Q       = X · W_Q                              # [B, T, H, D_q]
        K       = C_KV · W_UK                          # [B, T_kv, H, D_k]
        V       = C_KV · W_UV                          # [B, T_kv, H, D_v]
        S       = softmax(Q · K^T / sqrt(D_k))         # [B, T, H, T_kv]
        O_h     = S · V                                # [B, T, H, D_v]
        out     = concat(O_h) · W_O                    # [B, T, D_model]

    FlashMLA absorb: fold W_UK^T into Q and W_O^T into V so that during
    decode (T == 1), the inner attention is computed in the *latent*
    space directly against the compressed KV cache ``C_KV`` without ever
    materializing K and V:

        Q_abs    = Q · W_UK^T                          # [B, T, H, D_kv]
        S_abs    = softmax(Q_abs · C_KV^T / sqrt(D_k)) # [B, T, H, T_kv]
        V_abs    = S_abs · C_KV                        # [B, T, H, D_kv]
        out      = concat(V_abs) · (W_UV · W_O)        # [B, T, D_model]

    The fused matrices ``W_UK_abs = W_UK^T`` (one per head) and
    ``W_UV_W_O = W_UV · W_O`` are computed once at weight-load time. The
    runtime per-step cost drops because K and V are never built.

This module exposes the algebraic transform and the absorbed forward in
pure MLX so a future Path B/C/D kernel can match its numerical contract.
"""

from __future__ import annotations

import mlx.core as mx


def absorb_weights(
    w_uk: mx.array,
    w_uv: mx.array,
    w_o: mx.array,
) -> tuple[mx.array, mx.array]:
    """Precompute the absorbed weight matrices at load time.

    Args:
        w_uk: ``[H, D_kv, D_k]`` — per-head up-projection for K.
        w_uv: ``[H, D_kv, D_v]`` — per-head up-projection for V.
        w_o:  ``[H * D_v, D_model]`` — output projection (concat-over-heads).

    Returns:
        w_uk_abs:  ``[H, D_k, D_kv]`` — ``W_UK^T`` per head.
        w_uv_w_o:  ``[H, D_kv, D_model]`` — ``W_UV @ W_O[h]`` per head.
    """
    if w_uk.ndim != 3:
        raise ValueError(f"w_uk must be [H, D_kv, D_k], got {w_uk.shape}")
    if w_uv.ndim != 3 or w_uv.shape[0] != w_uk.shape[0] or w_uv.shape[1] != w_uk.shape[1]:
        raise ValueError(
            f"w_uv shape {w_uv.shape} inconsistent with w_uk {w_uk.shape}"
        )
    h, d_kv, d_v = w_uv.shape
    if w_o.ndim != 2 or w_o.shape[0] != h * d_v:
        raise ValueError(
            f"w_o shape {w_o.shape} must be (H*D_v={h * d_v}, D_model)"
        )
    d_model = w_o.shape[1]

    w_uk_abs = mx.transpose(w_uk, (0, 2, 1))  # [H, D_k, D_kv]
    # W_O per head: [H, D_v, D_model]
    w_o_per_head = w_o.reshape(h, d_v, d_model)
    # W_UV @ W_O_per_head: [H, D_kv, D_v] @ [H, D_v, D_model] -> [H, D_kv, D_model]
    w_uv_w_o = mx.matmul(w_uv, w_o_per_head)
    return w_uk_abs, w_uv_w_o


def absorbed_mla_decode(
    q: mx.array,
    c_kv: mx.array,
    w_uk_abs: mx.array,
    w_uv_w_o: mx.array,
    *,
    sm_scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """Run an MLA decode step in the absorbed (latent) form.

    Args:
        q: ``[B, T, H, D_k]`` — query (post W_Q projection).
        c_kv: ``[B, T_kv, D_kv]`` — compressed/latent KV cache (shared across heads).
        w_uk_abs: ``[H, D_k, D_kv]`` — from ``absorb_weights``.
        w_uv_w_o: ``[H, D_kv, D_model]`` — from ``absorb_weights``.
        sm_scale: softmax scale; defaults to ``1 / sqrt(D_k)``.
        mask: optional additive mask broadcastable to ``[B, T, H, T_kv]``.

    Returns:
        ``[B, T, D_model]`` — final output (sum over heads of the absorbed-V
        outputs, mathematically equivalent to ``concat(V_h) @ W_O`` in the
        standard form, but computed without materializing K or V).
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B, T, H, D_k], got {q.shape}")
    if c_kv.ndim != 3:
        raise ValueError(f"c_kv must be [B, T_kv, D_kv], got {c_kv.shape}")
    if w_uk_abs.ndim != 3 or w_uv_w_o.ndim != 3:
        raise ValueError("absorbed weights must be 3D (H, ..., ...)")
    h = q.shape[2]
    d_k = q.shape[3]
    d_kv = c_kv.shape[-1]
    if w_uk_abs.shape != (h, d_k, d_kv):
        raise ValueError(
            f"w_uk_abs must be (H={h}, D_k={d_k}, D_kv={d_kv}), got {w_uk_abs.shape}"
        )
    if w_uv_w_o.shape[0] != h or w_uv_w_o.shape[1] != d_kv:
        raise ValueError(
            f"w_uv_w_o must be (H={h}, D_kv={d_kv}, D_model), got {w_uv_w_o.shape}"
        )
    if sm_scale is None:
        sm_scale = d_k ** -0.5

    # Q_abs: [B, T, H, D_kv] — multiply per head: q[b, t, h, :] @ w_uk_abs[h]
    # einsum: 'bthk,hkv->bthv' with v = D_kv
    q_abs = mx.einsum("bthk,hkv->bthv", q, w_uk_abs)

    # Logits: [B, T, H, T_kv] — q_abs[b, t, h, :] . c_kv[b, t_kv, :]
    # einsum: 'bthv,bkv->bthk' where k = T_kv
    logits = mx.einsum("bthv,bkv->bthk", q_abs, c_kv) * sm_scale
    if mask is not None:
        logits = logits + mask

    weights = mx.softmax(logits, axis=-1)
    # V_abs: [B, T, H, D_kv] — sum over T_kv
    # einsum: 'bthk,bkv->bthv'
    v_abs = mx.einsum("bthk,bkv->bthv", weights, c_kv)

    # Output: sum over heads of v_abs @ w_uv_w_o[h] -> [B, T, D_model]
    # einsum: 'bthv,hvm->btm'
    out = mx.einsum("bthv,hvm->btm", v_abs, w_uv_w_o)
    return out


def standard_mla_decode(
    q: mx.array,
    c_kv: mx.array,
    w_uk: mx.array,
    w_uv: mx.array,
    w_o: mx.array,
    *,
    sm_scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """Standard (non-absorbed) MLA decode — used as the numerical oracle."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    # K = C_KV @ W_UK^T per head: [B, T_kv, H, D_k]
    # w_uk: [H, D_kv, D_k]. c_kv: [B, T_kv, D_kv].
    # einsum: 'bkv,hvd->bkhd' where d = D_k
    k = mx.einsum("bkv,hvd->bkhd", c_kv, w_uk)
    # V = C_KV @ W_UV^T per head: [B, T_kv, H, D_v]
    v = mx.einsum("bkv,hvd->bkhd", c_kv, w_uv)
    # Logits: [B, T, H, T_kv]
    logits = mx.einsum("bthk,bnhk->bthn", q, k) * sm_scale
    if mask is not None:
        logits = logits + mask
    weights = mx.softmax(logits, axis=-1)
    # O_h = S @ V: [B, T, H, D_v]
    o_h = mx.einsum("bthn,bnhv->bthv", weights, v)
    # Concat heads and project: [B, T, H*D_v] @ W_O -> [B, T, D_model]
    b, t, h, d_v = o_h.shape
    return mx.matmul(o_h.reshape(b, t, h * d_v), w_o)


__all__ = ["absorb_weights", "absorbed_mla_decode", "standard_mla_decode"]
