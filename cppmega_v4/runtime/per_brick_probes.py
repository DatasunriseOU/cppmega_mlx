"""V7-H07: per-brick grad-norm + attention-mean probes.

Extracts the per-brick gradient L2 norms (for the canvas grad overlay)
and per-attention-head attention-weight means (for the heatmap
overlay) from a model + grads pair. Pure helpers — wire into
stage_train extras separately.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def per_brick_grad_norms(model: Any, grads: Any) -> dict[str, float]:
    """Return {brick_path: L2 norm of all grads under that path}.

    The brick path is the top-level key in tree_flatten — e.g.
    'layers.0', 'layers.1', 'shared_expert'.
    """
    flat = dict(nn.utils.tree_flatten(grads))
    by_brick: dict[str, float] = {}
    for k, g in flat.items():
        if not hasattr(g, "shape"):
            continue
        top = k.split(".", 1)[0]
        # Try the "layers.<i>" composite first for layer-indexed paths.
        parts = k.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            top = f"layers.{parts[1]}"
        n = float(mx.sum(g.astype(mx.float32) * g.astype(mx.float32)
                          ).item())
        by_brick[top] = by_brick.get(top, 0.0) + n
    return {k: float(v ** 0.5) for k, v in by_brick.items()}


def attn_head_means(model: Any, x: mx.array) -> dict[str, list[float]]:
    """Return {attn_brick_path: per-head mean attention weight}.

    For each attention module found by attribute walking, run forward
    on `x` and average the softmax(Q·Kᵀ) attention map per head.
    """
    out: dict[str, list[float]] = {}

    def _is_attn(m) -> bool:
        cls = type(m).__name__
        return cls in {"_SelfAttn"} or ("Attn" in cls
                                        and hasattr(m, "q_proj"))

    def _walk(prefix, mod):
        if _is_attn(mod):
            try:
                B, S, _ = x.shape
                q = mod.q_proj(x).reshape(B, S, -1)
                k = mod.k_proj(x).reshape(B, S, -1)
                head_dim = q.shape[-1] // max(
                    1, len(getattr(mod, "_heads", [1])) or 1)
                # Fallback: treat as a single head if num_heads unknown.
                num_heads = max(1, q.shape[-1] // max(1, head_dim))
                head_d = q.shape[-1] // num_heads
                qh = q.reshape(B, S, num_heads, head_d)
                kh = k.reshape(B, S, num_heads, head_d)
                qh = mx.transpose(qh, (0, 2, 1, 3))
                kh = mx.transpose(kh, (0, 2, 1, 3))
                scores = mx.matmul(qh, mx.transpose(
                    kh, (0, 1, 3, 2))) * (head_d ** -0.5)
                attn = mx.softmax(scores, axis=-1)
                per_head = [
                    float(mx.mean(attn[:, h, :, :]).item())
                    for h in range(num_heads)
                ]
                out[prefix or "_root"] = per_head
            except Exception:
                pass
        # Recurse via .children / .modules.
        if hasattr(mod, "children"):
            try:
                for name, child in mod.children().items():
                    if isinstance(child, list):
                        for i, c in enumerate(child):
                            _walk(
                                f"{prefix}.{name}.{i}" if prefix
                                else f"{name}.{i}", c)
                    else:
                        _walk(
                            f"{prefix}.{name}" if prefix else name,
                            child)
            except Exception:
                pass

    _walk("", model)
    return out


__all__ = ["per_brick_grad_norms", "attn_head_means"]
