"""xLSTM matrix-LSTM block — recurrent decoder, no self-attention.

Used by gallery entry #50 xLSTM 7B. Matrix-memory LSTM cell: stores a
``(H, H)`` covariance state per layer instead of the vanilla LSTM
hidden vector. Forward retains rank-2 state across token positions.

Thin wrapper — keeps the brick first-class but the actual recurrence is
a single Linear + sigmoid gates + matmul update (not optimised; the
GUI / sizing planner uses the brick's category for routing decisions).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MLSTMConfig:
    hidden_size: int
    head_dim: int = 64
    rms_norm_eps: float = 1e-6


class MLSTMBlock(nn.Module):
    """Matrix-memory LSTM cell."""

    def __init__(self, cfg: MLSTMConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        d = cfg.head_dim
        self.norm = nn.RMSNorm(H, eps=cfg.rms_norm_eps)
        self.q_proj = nn.Linear(H, d, bias=False)
        self.k_proj = nn.Linear(H, d, bias=False)
        self.v_proj = nn.Linear(H, d, bias=False)
        self.i_proj = nn.Linear(H, d, bias=False)   # input gate
        self.f_proj = nn.Linear(H, d, bias=False)   # forget gate
        self.o_proj = nn.Linear(H, d, bias=False)   # output gate
        self.out_proj = nn.Linear(d, H, bias=False)
        # Identity at init.
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        i_gate = mx.sigmoid(self.i_proj(h))
        f_gate = mx.sigmoid(self.f_proj(h))
        o_gate = mx.sigmoid(self.o_proj(h))
        # Simplified scan: per-token gated mix of (k * v) accumulated
        # in matrix C of shape (B, d, d). We approximate via running
        # sum (no actual recurrence kernel — placeholder for GUI sizing).
        c = mx.zeros((B, self.cfg.head_dim, self.cfg.head_dim), dtype=h.dtype)
        outs = []
        for t in range(S):
            update = mx.expand_dims(k[:, t, :], 2) @ mx.expand_dims(v[:, t, :], 1)
            c = mx.expand_dims(f_gate[:, t, :], 2) * c + \
                mx.expand_dims(i_gate[:, t, :], 2) * update
            yt = (mx.expand_dims(q[:, t, :], 1) @ c).squeeze(1)
            outs.append(o_gate[:, t, :] * yt)
        out = mx.stack(outs, axis=1)
        return self.out_proj(out)


__all__ = ["MLSTMBlock", "MLSTMConfig"]
