"""GatedDeltaNet block plugin (Path A entry — naive recurrent kernel).

Minimal block wrapper around the FLA ``naive_recurrent_gated_delta_rule``
math primitive (vendored in ``cppmega_v4/nn/_external/`` with MIT
attribution). Field names are kept identical to FLA's
``fla.layers.gated_deltanet.GatedDeltaNet`` (``q_proj``, ``k_proj``,
``v_proj``, ``a_proj``, ``b_proj``, ``o_proj``, ``o_norm``) so a future
swap to the fused-kernel layer is a name-match drop-in.

What this wrapper DELIBERATELY does NOT do yet (deferred to Path B/C/D/E):
    - FusedRMSNormGated (FLA Triton fusion) — we use plain RMSNorm here.
    - ShortConvolution (FLA Triton kernel) — we use a tiny pure-MLX
      causal depthwise conv copied from cppmega_mlx.nn.engram (same idiom).
    - dt_bias / A_log learnable scalars (FLA training-only quality knob).
    - Allow-negative-eigval branch.
    - cu_seqlens packed-batch path (we expose ``doc_ids`` instead).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


@dataclass(frozen=True)
class LinearAttentionConfig:
    """Static-shape config for the GDN linear-attention block.

    Field names mirror ``fla.layers.gated_deltanet.GatedDeltaNet.__init__``
    where possible (``hidden_size``, ``num_heads``, ``head_dim``,
    ``expand_v``, ``use_short_conv``, ``conv_size``, ``use_gate``,
    ``norm_eps``).
    """

    hidden_size: int
    num_heads: int = 4
    head_dim: int = 32
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
        if nv <= 0:
            raise ValueError("num_v_heads must be positive when set")
        if nv > self.num_heads and nv % self.num_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_heads")
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


def _causal_short_conv(x: mx.array, weight: mx.array) -> mx.array:
    """Depthwise causal conv1d via shift-and-add.

    x: (B, S, C); weight: (C, K, 1). Matches the idiom already in
    ``cppmega_mlx.nn.engram.causal_depthwise_silu_conv1d`` for consistency.
    """
    kernel = weight.shape[1]
    if kernel == 1:
        return x * weight[:, 0, 0].reshape(1, 1, -1)
    seq_len = x.shape[1]
    out = mx.zeros_like(x)
    for shift in range(kernel):
        if shift == 0:
            shifted = x
        elif shift < seq_len:
            shifted = mx.pad(x[:, :-shift, :], [(0, 0), (shift, 0), (0, 0)])
        else:
            shifted = mx.zeros_like(x)
        tap = weight[:, kernel - 1 - shift, 0].reshape(1, 1, -1)
        out = out + shifted * tap
    return out


class LinearAttentionBlock(nn.Module):
    """GDN block. Forward: ``(B, S, hidden_size) -> (B, S, hidden_size)``.

    Backend: Path A (FLA naive recurrent). Path B/C/D/E land under
    ``cppmega_v4/_tilelang/`` and dispatch via env override in a follow-up ROI;
    until then this block always runs Path A.
    """

    def __init__(self, config: LinearAttentionConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        # Field names match FLA: q_proj, k_proj, v_proj, a_proj, b_proj, o_proj.
        self.q_proj = nn.Linear(d, config.key_dim, bias=False)
        self.k_proj = nn.Linear(d, config.key_dim, bias=False)
        self.v_proj = nn.Linear(d, config.value_dim, bias=False)
        self.a_proj = nn.Linear(d, config._num_v_heads, bias=False)
        self.b_proj = nn.Linear(d, config._num_v_heads, bias=False)
        if config.use_gate:
            self.g_proj = nn.Linear(d, config.value_dim, bias=False)
        if config.use_short_conv and config.conv_size > 0:
            # Identity-initialized depthwise short conv (center tap = 1).
            self.q_conv_weight = self._init_id_conv_weight(config.key_dim, config.conv_size)
            self.k_conv_weight = self._init_id_conv_weight(config.key_dim, config.conv_size)
            self.v_conv_weight = self._init_id_conv_weight(config.value_dim, config.conv_size)
        self.o_proj = nn.Linear(config.value_dim, d, bias=False)
        # Zero-init out projection so block is identity at init.
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)
        self.o_norm = nn.RMSNorm(d, eps=config.norm_eps)

    @staticmethod
    def _init_id_conv_weight(channels: int, kernel: int) -> mx.array:
        center = kernel - 1  # causal: last tap is "current"
        w = mx.zeros((channels, kernel, 1), dtype=mx.float32)
        w_np = w.tolist()
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

        # FLA layout: q/k/v ∈ [B, T, H, K|V]
        q = q.reshape(batch, seq_len, cfg.num_heads, cfg.head_k_dim)
        k = k.reshape(batch, seq_len, cfg.num_heads, cfg.head_k_dim)
        v = v.reshape(batch, seq_len, cfg._num_v_heads, cfg.head_v_dim)

        # a → gate decay logit; b → beta (learning rate). Sigmoid per FLA layer.
        beta = mx.sigmoid(self.b_proj(x))   # (B, T, num_v_heads)
        # FLA uses ``-softplus(a_proj + dt_bias)`` for gate (with dt_bias). We
        # match the simpler "g = log(sigmoid(a_proj))" until the dt_bias path
        # lands as part of a fused upgrade.
        g = mx.log(mx.sigmoid(self.a_proj(x)) + cfg.norm_eps)  # (B, T, num_v_heads)

        if doc_ids is None:
            # Dispatch through the path system: env-overridable via
            # CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION, auto-mode picks the
            # fastest available backend (Path B/C/E if Metal/tilelang/mlx-lm
            # vendored op are reachable; falls back to Path A).
            from cppmega_v4._tilelang.linear_attention_paths import (
                gated_delta_recurrent_dispatch,
            )
            o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
        else:
            # Document-boundary state reset: split into runs of contiguous
            # doc_id, dispatch each run through the path system, concat.
            o = self._recurrent_with_doc_reset(q, k, v, beta, g, doc_ids)

        # o has shape [B, T, H_v, V_dim]; flatten head axis for o_proj.
        o = o.reshape(batch, seq_len, cfg.value_dim)
        o = self.o_proj(o)
        return self.o_norm(o)

    @staticmethod
    def _recurrent_with_doc_reset(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        beta: mx.array,
        g: mx.array,
        doc_ids: mx.array,
    ) -> mx.array:
        """Apply naive recurrent kernel per document segment.

        We loop over batch entries and over contiguous doc-id runs. This is
        slow (O(B * num_segments) Python iterations) but correct; a fused
        cu_seqlens-style API ports later when Path C lands.
        """
        if doc_ids.ndim != 2:
            raise ValueError(f"doc_ids must be 2D (B, T), got {doc_ids.shape}")
        batch, seq_len = doc_ids.shape
        if q.shape[0] != batch or q.shape[1] != seq_len:
            raise ValueError("doc_ids shape must match q's (B, T)")
        # Iterate batch entries explicitly; build a list of [1, run_len, H, *]
        # slices per run and concatenate along T.
        ids_np = doc_ids.tolist()
        per_batch_outputs: list[mx.array] = []
        for b in range(batch):
            row = ids_np[b]
            runs: list[tuple[int, int]] = []
            start = 0
            for t in range(1, seq_len):
                if row[t] != row[start]:
                    runs.append((start, t))
                    start = t
            runs.append((start, seq_len))
            run_outputs: list[mx.array] = []
            for s, e in runs:
                qb = q[b:b + 1, s:e]
                kb = k[b:b + 1, s:e]
                vb = v[b:b + 1, s:e]
                bb = beta[b:b + 1, s:e]
                gb = g[b:b + 1, s:e]
                from cppmega_v4._tilelang.linear_attention_paths import (
                    gated_delta_recurrent_dispatch,
                )
                ob, _ = gated_delta_recurrent_dispatch(qb, kb, vb, bb, gb)
                run_outputs.append(ob)
            per_batch_outputs.append(mx.concatenate(run_outputs, axis=1))
        return mx.concatenate(per_batch_outputs, axis=0)


__all__ = [
    "LinearAttentionConfig",
    "LinearAttentionBlock",
    "naive_recurrent_gated_delta_rule",
]
