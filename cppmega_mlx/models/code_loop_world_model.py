"""Code-edit world model whose dynamics core IS the FPRM stable looped core.

``CodeLoopWorldModel`` predicts the *next observation* (the post-edit code
region) of a code-edit transition from the *current observation* (the pre-edit
context).  Its latent dynamics are driven by the SAME weight-tied
:class:`~cppmega_mlx.nn.stable_loop.StableFixedPointLoop` that backs the looped
LM — this module does NOT reimplement the loop; it constructs one and asserts
``isinstance(self.dynamics, StableFixedPointLoop)``.

Leaves are reused cppmega blocks:

  * token embedding from :class:`~cppmega_mlx.models.dense_cpp_lm.DenseCppLM`
    (``nn.Embedding`` + learned positions), RMSNorm pre-norm, and the SwiGLU
    route core (``RouteCore`` from ``stable_loop_cpp_lm``) swept ``2L`` times per
    loop iteration.

Heads:

  * ``decode`` — predicts next-observation token logits from a rolled latent
    (tied to the token embedding).
  * ``reward_head`` / ``done_head`` — scalar control heads, trained ONLY where a
    transition carries an explicitly-synthetic label (RULE #1: no fabricated
    labels; real transitions never touch these heads).

Rollout:

  * :meth:`rollout` rolls the latent forward ``horizon`` loop windows and decodes
    tokens ONLY at the terminal step (deferred decode).  :meth:`rollout_decode_all`
    decodes at every step; the terminal-step shape of the deferred rollout EQUALS
    the per-step decode at the same step.

Regularizers:

  * latent-consistency: the rolled latent should match the latent obtained by
    re-encoding the (teacher) next observation.
  * entropy regularization on the next-observation predictive distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.models.stable_loop_cpp_lm import RouteCore, StableLoopCppLMConfig, SwiGLUMLP
from cppmega_mlx.nn.stable_loop import StableFixedPointLoop


@dataclass(frozen=True)
class CodeLoopWorldModelConfig:
    """Config for the code-edit world model (smoke scale by default)."""

    vocab_size: int = 64
    hidden_size: int = 32
    num_heads: int = 4
    ffn_hidden_size: int = 64
    max_seq_length: int = 512
    route_layers: int = 2
    a1_init: float = 0.75
    a2_init: float = 0.25
    tau: float = 0.1
    max_loops: int = 32
    train_loops: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.route_layers < 1:
            raise ValueError("route_layers must be positive")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be >= 2")
        if self.train_loops < 1:
            raise ValueError("train_loops must be positive")

    @property
    def n_sublayers(self) -> int:
        return 2 * self.route_layers

    def stable_loop_config(self) -> StableLoopCppLMConfig:
        """Build the StableLoopCppLMConfig that parameterises the shared core."""
        return StableLoopCppLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            ffn_hidden_size=self.ffn_hidden_size,
            max_seq_length=self.max_seq_length,
            route_layers=self.route_layers,
            prelude_blocks=0,
            a1_init=self.a1_init,
            a2_init=self.a2_init,
            tau=self.tau,
            max_loops=self.max_loops,
            train_loops=self.train_loops,
        )


class CodeLoopWorldModel(nn.Module):
    """World model with an FPRM stable looped dynamics core."""

    def __init__(self, config: CodeLoopWorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CodeLoopWorldModelConfig()
        cfg = self.config
        loop_cfg = cfg.stable_loop_config()

        # Reused leaves: token + position embedding (tied decode head).
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)

        # Encoder over an observation: pre-norm SwiGLU mixer then mean-pool to a
        # single (B, D) latent that the dynamics core operates on.
        self.encoder_norm = nn.RMSNorm(cfg.hidden_size)
        self.encoder_mlp = SwiGLUMLP(cfg.hidden_size, cfg.ffn_hidden_size)

        # Dynamics core: THE StableFixedPointLoop over a shared RouteCore.
        self.route_core = RouteCore(loop_cfg)
        self.dynamics = StableFixedPointLoop(
            self.route_core,
            d_model=cfg.hidden_size,
            n_sublayers=cfg.n_sublayers,
            a1_init=cfg.a1_init,
            a2_init=cfg.a2_init,
            tau=cfg.tau,
            max_loops=cfg.max_loops,
        )
        # Hard requirement: the dynamics core IS the stable fixed-point loop.
        assert isinstance(self.dynamics, StableFixedPointLoop), (
            "CodeLoopWorldModel.dynamics must be a StableFixedPointLoop instance"
        )

        # Decode head: latent (B, D) -> per-position next-obs logits. The decoder
        # broadcasts the rolled latent over the target positions, adds the target
        # position embedding, mixes, and projects with the tied embedding weight.
        self.decode_norm = nn.RMSNorm(cfg.hidden_size)
        self.decode_mlp = SwiGLUMLP(cfg.hidden_size, cfg.ffn_hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied

        # Control heads (synthetic-only): scalar reward + done logit from latent.
        self.reward_head = nn.Linear(cfg.hidden_size, 1, bias=True)
        self.done_head = nn.Linear(cfg.hidden_size, 1, bias=True)

    # ------------------------------------------------------------------ #
    # Encoding / dynamics
    # ------------------------------------------------------------------ #
    def _embed_tokens(self, tokens: mx.array) -> mx.array:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (B, S), got {tuple(tokens.shape)}")
        seq = int(tokens.shape[1])
        if seq > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_length {self.config.max_seq_length}"
            )
        positions = mx.arange(seq)[None, :]
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def encode(self, tokens: mx.array) -> mx.array:
        """Encode an observation ``(B, S)`` to a latent ``(B, D)`` (mean pool)."""
        hidden = self._embed_tokens(tokens)
        hidden = hidden + self.encoder_mlp(self.encoder_norm(hidden))
        return mx.mean(hidden, axis=1)

    def step_latent(
        self,
        latent: mx.array,
        *,
        training_loops: int | None = None,
    ) -> mx.array:
        """Roll the latent forward ONE world-model step via the looped core.

        The latent ``(B, D)`` is treated as a length-1 sequence so the shared
        route-core attention/FFN sublayers apply unchanged; the injected input to
        the iteration mix is the same latent (autonomous dynamics).
        """

        z = latent[:, None, :]  # (B, 1, D)
        loops = training_loops if training_loops is not None else self.config.train_loops
        if loops < 0:
            rolled = self.dynamics.forward(z, z, None, training_loops=None)
        else:
            rolled = self.dynamics.forward(z, z, None, training_loops=loops)
        return rolled[:, 0, :]  # (B, D)

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(self, latent: mx.array, length: int) -> mx.array:
        """Decode ``length`` next-observation token logits from a latent.

        Returns logits ``(B, length, vocab)``.  The latent is broadcast over the
        ``length`` target positions, summed with the target position embeddings,
        mixed by a SwiGLU block, and projected through the tied head.
        """

        if length < 1:
            raise ValueError(f"decode length must be >= 1, got {length}")
        if length > self.config.max_seq_length:
            raise ValueError(
                f"decode length {length} exceeds max_seq_length {self.config.max_seq_length}"
            )
        batch = int(latent.shape[0])
        positions = mx.arange(length)[None, :]
        pos_emb = self.position_embedding(positions)  # (1, L, D)
        broadcast = mx.broadcast_to(latent[:, None, :], (batch, length, latent.shape[-1]))
        hidden = broadcast + pos_emb
        hidden = hidden + self.decode_mlp(self.decode_norm(hidden))
        return self.lm_head(hidden)

    def predict_next(
        self,
        obs: mx.array,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Predict next-observation logits ``(B, L, V)`` and return the latent.

        One world-model step: encode ``obs`` -> roll latent -> decode ``length``.
        """

        latent = self.encode(obs)
        rolled = self.step_latent(latent, training_loops=training_loops)
        logits = self.decode(rolled, length)
        return logits, rolled

    # ------------------------------------------------------------------ #
    # Control heads (synthetic labels only)
    # ------------------------------------------------------------------ #
    def reward(self, latent: mx.array) -> mx.array:
        """Scalar reward prediction ``(B,)`` from a rolled latent."""
        return self.reward_head(latent)[:, 0]

    def done_logit(self, latent: mx.array) -> mx.array:
        """Done logit ``(B,)`` from a rolled latent (pre-sigmoid)."""
        return self.done_head(latent)[:, 0]

    # ------------------------------------------------------------------ #
    # Rollout (deferred vs per-step decode)
    # ------------------------------------------------------------------ #
    def rollout(
        self,
        obs: mx.array,
        horizon: int,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Deferred rollout: roll the latent ``horizon`` steps, decode ONCE.

        Returns ``(terminal_logits (B, L, V), terminal_latent (B, D))``.  Decode
        happens ONLY at the terminal step — intermediate latents are never
        decoded.
        """

        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        latent = self.encode(obs)
        for _ in range(horizon):
            latent = self.step_latent(latent, training_loops=training_loops)
        logits = self.decode(latent, length)
        return logits, latent

    def rollout_decode_all(
        self,
        obs: mx.array,
        horizon: int,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> list[mx.array]:
        """Per-step rollout: decode ``length`` logits at EVERY step.

        Returns a list of ``horizon`` logit tensors, each ``(B, L, V)``.  The
        terminal entry equals (shape AND value) the deferred :meth:`rollout`
        output because both decode the same terminal latent.
        """

        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        latent = self.encode(obs)
        per_step: list[mx.array] = []
        for _ in range(horizon):
            latent = self.step_latent(latent, training_loops=training_loops)
            per_step.append(self.decode(latent, length))
        return per_step

    def __call__(
        self,
        obs: mx.array,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        return self.predict_next(obs, length, training_loops=training_loops)


__all__ = [
    "CodeLoopWorldModel",
    "CodeLoopWorldModelConfig",
]
