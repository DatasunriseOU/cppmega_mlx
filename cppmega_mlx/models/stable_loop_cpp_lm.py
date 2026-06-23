"""Tiny FPRM stable-looped LM (arXiv:2606.18206), MLX, correctness-first.

Architecture (smoke scale):

    embedding (+ positions)
      -> prelude: a few dense RMSNorm + SwiGLU-MLP blocks
      -> StableFixedPointLoop over a SHARED 2-layer route core (2L = 4
         sublayers: layer1 {attn, FFN}, layer2 {attn, FFN})
      -> coda: one RMSNorm + SwiGLU-MLP block
      -> final RMSNorm -> LM head

The single weight-tied route core is swept ``2L`` times per loop iteration and
the loop iterates either a fixed number of times (training) or to a fixed point
with FPOPT damping (inference). RMSNorm / SwiGLU conventions mirror
``cppmega_mlx.models.hybrid_lm`` (HybridTinyBlock / RMSNorm / nn.Linear bias
choices).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.stable_loop import StableFixedPointLoop


@dataclass(frozen=True)
class StableLoopCppLMConfig:
    """Tiny config for the FPRM stable-looped LM (smoke scale)."""

    vocab_size: int = 64
    hidden_size: int = 32
    num_heads: int = 4
    ffn_hidden_size: int = 64
    max_seq_length: int = 32
    # The route core has ``route_layers`` weight-tied layers, each contributing
    # an {attn, FFN} pair, so n_sublayers (== 2L) == 2 * route_layers.
    route_layers: int = 2
    prelude_blocks: int = 2
    # FPRM loop hyper-parameters (paper defaults).
    a1_init: float = 0.75
    a2_init: float = 0.25
    tau: float = 0.1
    max_loops: int = 32
    train_loops: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.route_layers < 1:
            raise ValueError("route_layers must be positive")
        if self.prelude_blocks < 0:
            raise ValueError("prelude_blocks must be non-negative")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.train_loops < 1:
            raise ValueError("train_loops must be positive")

    @property
    def n_sublayers(self) -> int:
        return 2 * self.route_layers

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block (gate/up/down), bias-free like hybrid_lm."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class DenseBlock(nn.Module):
    """Pre-norm attention + SwiGLU-MLP residual block (prelude/coda)."""

    def __init__(self, cfg: StableLoopCppLMConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size)
        self.attn = nn.MultiHeadAttention(cfg.hidden_size, cfg.num_heads, bias=False)
        self.ffn_norm = nn.RMSNorm(cfg.hidden_size)
        self.ffn = SwiGLUMLP(cfg.hidden_size, cfg.ffn_hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        a = self.attn_norm(x)
        x = x + self.attn(a, a, a, mask)
        return x + self.ffn(self.ffn_norm(x))


class RouteCore(nn.Module):
    """Weight-tied route core exposing ``2L`` sublayers + pre-norms.

    ``route_layers`` layers each contribute two sublayers in order:
    ``[attn_1, ffn_1, attn_2, ffn_2, ...]``. The loop owns the residual scaling
    (a1/b1); each sublayer here only returns its pre-residual delta
    ``f^l(Norm(z))``.
    """

    def __init__(self, cfg: StableLoopCppLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.attns = [
            nn.MultiHeadAttention(cfg.hidden_size, cfg.num_heads, bias=False)
            for _ in range(cfg.route_layers)
        ]
        self.ffns = [
            SwiGLUMLP(cfg.hidden_size, cfg.ffn_hidden_size)
            for _ in range(cfg.route_layers)
        ]
        # 2L pre-norms, one per sublayer.
        self.sublayer_norms = [
            nn.RMSNorm(cfg.hidden_size) for _ in range(cfg.n_sublayers)
        ]
        # The additive causal mask depends on sequence length; the loop passes
        # it via ``ctx`` so the same core works for any S without rebuilding.

    @property
    def norms(self) -> list[nn.Module]:
        return self.sublayer_norms

    @property
    def sublayers(self):
        """Ordered list of ``2L`` ``(norm, x, ctx) -> delta`` callables."""

        fns = []
        for layer_idx in range(self.cfg.route_layers):
            attn = self.attns[layer_idx]
            ffn = self.ffns[layer_idx]

            def attn_sublayer(norm, z, ctx, _attn=attn):
                a = norm(z)
                mask = ctx if ctx is not None else None
                return _attn(a, a, a, mask)

            def ffn_sublayer(norm, z, ctx, _ffn=ffn):
                return _ffn(norm(z))

            fns.append(attn_sublayer)
            fns.append(ffn_sublayer)
        return fns


class StableLoopCppLM(nn.Module):
    """Tiny decoder-only LM built around an FPRM stable looped route core."""

    def __init__(self, config: StableLoopCppLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or StableLoopCppLMConfig()
        cfg = self.config

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)

        self.prelude = [DenseBlock(cfg) for _ in range(cfg.prelude_blocks)]

        self.route_core = RouteCore(cfg)
        self.loop = StableFixedPointLoop(
            self.route_core,
            d_model=cfg.hidden_size,
            n_sublayers=cfg.n_sublayers,
            a1_init=cfg.a1_init,
            a2_init=cfg.a2_init,
            tau=cfg.tau,
            max_loops=cfg.max_loops,
        )

        self.coda = DenseBlock(cfg)
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def embed(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be shaped (B, S), got {input_ids.shape}"
            )
        seq_length = input_ids.shape[1]
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq_length} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )
        positions = mx.arange(seq_length)[None, :]
        return self.token_embedding(input_ids) + self.position_embedding(positions)

    def _causal_mask(self, seq_length: int, dtype: mx.Dtype) -> mx.array:
        return nn.MultiHeadAttention.create_additive_causal_mask(
            seq_length, dtype=dtype
        )

    def run_prelude(self, hidden: mx.array, mask: mx.array) -> mx.array:
        for block in self.prelude:
            hidden = block(hidden, mask)
        return hidden

    def head(self, z: mx.array, mask: mx.array) -> mx.array:
        """Coda + final norm + LM head on a loop state ``z``."""
        z = self.coda(z, mask)
        return self.lm_head(self.norm(z))

    def __call__(
        self,
        input_ids: mx.array,
        *,
        training_loops: int | None = None,
        return_state: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Full forward returning logits ``(B, S, vocab)``.

        ``training_loops`` overrides the loop schedule. When ``None`` the model
        runs ``config.train_loops`` fixed iterations (fully differentiable).
        Pass ``training_loops=-1`` to request inference fixed-point + FPOPT
        halting instead.
        """
        seq_length = input_ids.shape[1]
        hidden = self.embed(input_ids)
        mask = self._causal_mask(seq_length, hidden.dtype)
        hidden = self.run_prelude(hidden, mask)

        # x is the injected input to the iteration mix; z0 starts at x.
        x = hidden
        if training_loops is not None and training_loops < 0:
            z = self.loop.forward(x, x, mask, training_loops=None)
        else:
            loops = training_loops if training_loops is not None else self.config.train_loops
            z = self.loop.forward(x, x, mask, training_loops=loops)

        logits = self.head(z, mask)
        if return_state:
            return logits, z
        return logits


__all__ = [
    "DenseBlock",
    "RouteCore",
    "StableLoopCppLM",
    "StableLoopCppLMConfig",
    "SwiGLUMLP",
]
