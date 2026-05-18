"""Kimi Delta Attention (KDA) block plugin (Path A — FLA naive port to MLX).

Minimal block wrapper around the FLA ``naive_recurrent_kda`` math primitive
(vendored in ``cppmega_v4/nn/_external/`` with MIT attribution). Field names
are kept identical to ``fla.layers.kda.KimiDeltaAttention`` (``q_proj``,
``k_proj``, ``v_proj``, ``b_proj``, ``f_proj``, ``g_proj``, ``o_proj``,
``o_norm``) so a future swap to fused versions is a name-match.

What this wrapper DELIBERATELY does NOT do yet (deferred to Path B/C/D):
    - FusedRMSNormGated (FLA Triton fusion) — we use plain RMSNorm.
    - ShortConvolution (FLA Triton kernel) — we use the same tiny pure-MLX
      causal depthwise conv used by the GDN block.
    - dt_bias / A_log learnable scalars (FLA training quality knob).
    - Allow-negative-eigval / safe_gate / lower_bound branches.
    - cu_seqlens packed-batch path (we expose ``doc_ids`` instead).

KDA recurrence differs from GDN in two ways:
    1. Per-dimension decay gates ``g[B, T, HV, K]`` (GDN uses per-head scalar).
    2. Optional GQA: ``num_v_heads >= num_heads`` with key/query broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda
from cppmega_v4.nn.linear_attention import _causal_short_conv


@dataclass(frozen=True)
class KimiDeltaAttentionConfig:
    """Static-shape config — field names mirror ``fla.layers.kda.KimiDeltaAttention``."""

    hidden_size: int
    num_heads: int = 4
    head_dim: int = 16
    expand_v: float = 1.0
    num_v_heads: int | None = None
    use_short_conv: bool = True
    conv_size: int = 4
    use_gate: bool = False
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.expand_v <= 0:
            raise ValueError("expand_v must be positive")
        nv = self.num_heads if self.num_v_heads is None else self.num_v_heads
        if nv <= 0 or nv % self.num_heads != 0:
            raise ValueError(
                "num_v_heads must be a positive multiple of num_heads"
            )
        if self.conv_size < 0:
            raise ValueError("conv_size must be non-negative")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    @property
    def head_k_dim(self) -> int:
        return self.head_dim

    @property
    def head_v_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def _num_v_heads(self) -> int:
        return self.num_heads if self.num_v_heads is None else self.num_v_heads

    @property
    def value_dim(self) -> int:
        return self._num_v_heads * self.head_v_dim

    @property
    def gate_dim(self) -> int:
        # FLA: gate is per value-head, per key-dim.
        return self._num_v_heads * self.head_k_dim


class KimiDeltaAttentionBlock(nn.Module):
    """KDA block. Forward: ``(B, S, hidden_size) -> (B, S, hidden_size)``.

    Backend: Path A (FLA naive recurrent KDA). Path B/C/D follow.
    """

    def __init__(self, config: KimiDeltaAttentionConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.q_proj = nn.Linear(d, config.key_dim, bias=False)
        self.k_proj = nn.Linear(d, config.key_dim, bias=False)
        self.v_proj = nn.Linear(d, config.value_dim, bias=False)
        self.b_proj = nn.Linear(d, config._num_v_heads, bias=False)
        # FLA f_proj: small bottleneck (Linear -> Linear). Mirrors layer file.
        self.f_proj_1 = nn.Linear(d, config.head_v_dim, bias=False)
        self.f_proj_2 = nn.Linear(config.head_v_dim, config.gate_dim, bias=False)
        if config.use_short_conv and config.conv_size > 0:
            self.q_conv_weight = self._init_id_conv_weight(config.key_dim, config.conv_size)
            self.k_conv_weight = self._init_id_conv_weight(config.key_dim, config.conv_size)
            self.v_conv_weight = self._init_id_conv_weight(config.value_dim, config.conv_size)
        self.o_proj = nn.Linear(config.value_dim, d, bias=False)
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)  # identity at init
        self.o_norm = nn.RMSNorm(d, eps=config.norm_eps)

    @staticmethod
    def _init_id_conv_weight(channels: int, kernel: int) -> mx.array:
        center = kernel - 1
        w_np = [[[0.0] for _ in range(kernel)] for _ in range(channels)]
        for c in range(channels):
            w_np[c][center][0] = 1.0
        return mx.array(w_np, dtype=mx.float32)

    def __call__(self, x: mx.array, *, doc_ids: mx.array | None = None) -> mx.array:
        if x.ndim != 3:
            raise ValueError(f"x must be shaped (B, S, D), got {x.shape}")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"x last dim must be {self.config.hidden_size}, got {x.shape[-1]}"
            )
        cfg = self.config
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if cfg.use_short_conv and cfg.conv_size > 0:
            q = _causal_short_conv(q, self.q_conv_weight.astype(q.dtype))
            k = _causal_short_conv(k, self.k_conv_weight.astype(k.dtype))
            v = _causal_short_conv(v, self.v_conv_weight.astype(v.dtype))

        beta = mx.sigmoid(self.b_proj(x))                            # [B, S, HV]
        g = self.f_proj_2(self.f_proj_1(x))                          # [B, S, HV*K]

        # Reshape: q,k -> [B, S, H, K], v -> [B, S, HV, V], g -> [B, S, HV, K]
        q = q.reshape(batch, seq_len, cfg.num_heads, cfg.head_k_dim)
        k = k.reshape(batch, seq_len, cfg.num_heads, cfg.head_k_dim)
        v = v.reshape(batch, seq_len, cfg._num_v_heads, cfg.head_v_dim)
        g = g.reshape(batch, seq_len, cfg._num_v_heads, cfg.head_k_dim)

        if doc_ids is None:
            from cppmega_v4._tilelang.kda_paths import kda_recurrent_dispatch
            o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
        else:
            o = self._recurrent_with_doc_reset(q, k, v, g, beta, doc_ids)

        o = o.reshape(batch, seq_len, cfg.value_dim)
        o = self.o_proj(o)
        return self.o_norm(o)

    @staticmethod
    def _recurrent_with_doc_reset(q, k, v, g, beta, doc_ids: mx.array) -> mx.array:
        if doc_ids.ndim != 2:
            raise ValueError(f"doc_ids must be 2D (B, T), got {doc_ids.shape}")
        batch, seq_len = doc_ids.shape
        if q.shape[0] != batch or q.shape[1] != seq_len:
            raise ValueError("doc_ids shape must match q's (B, T)")
        ids_np = doc_ids.tolist()
        per_batch: list[mx.array] = []
        for b in range(batch):
            row = ids_np[b]
            runs: list[tuple[int, int]] = []
            start = 0
            for t in range(1, seq_len):
                if row[t] != row[start]:
                    runs.append((start, t))
                    start = t
            runs.append((start, seq_len))
            outs = []
            for s, e in runs:
                from cppmega_v4._tilelang.kda_paths import kda_recurrent_dispatch
                ob, _ = kda_recurrent_dispatch(
                    q[b:b + 1, s:e],
                    k[b:b + 1, s:e],
                    v[b:b + 1, s:e],
                    g[b:b + 1, s:e],
                    beta[b:b + 1, s:e],
                )
                outs.append(ob)
            per_batch.append(mx.concatenate(outs, axis=1))
        return mx.concatenate(per_batch, axis=0)


__all__ = [
    "KimiDeltaAttentionConfig",
    "KimiDeltaAttentionBlock",
    "naive_recurrent_kda",
]
