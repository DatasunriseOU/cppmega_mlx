"""Code-edit world model whose dynamics core IS the FPRM stable looped core.

``CodeLoopWorldModel`` predicts the *next observation* (the post-edit code
region) of a code-edit transition from the *current observation* and the action
applied to it.  Its latent dynamics are driven by the SAME weight-tied
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

from collections.abc import Sequence
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

        # Actions use the same token vocabulary but a distinct encoder. The
        # transition projection makes the stable loop's injected input depend on
        # both the previous latent state and the action latent.
        self.action_encoder_norm = nn.RMSNorm(cfg.hidden_size)
        self.action_encoder_mlp = SwiGLUMLP(cfg.hidden_size, cfg.ffn_hidden_size)
        self.transition_norm = nn.RMSNorm(2 * cfg.hidden_size)
        self.transition_projection = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size)

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
    def _embed_tokens(self, tokens: mx.array, *, where: str) -> mx.array:
        if not isinstance(tokens, mx.array):
            raise TypeError(f"{where} tokens must be an mx.array, got {type(tokens).__name__}")
        if tokens.ndim != 2:
            raise ValueError(f"{where} tokens must be (B, S), got {tuple(tokens.shape)}")
        if int(tokens.shape[0]) < 1:
            raise ValueError(f"{where} token batch must be non-empty")
        seq = int(tokens.shape[1])
        if seq < 1:
            raise ValueError(f"{where} token sequence must be non-empty")
        if seq > self.config.max_seq_length:
            raise ValueError(
                f"{where} sequence length {seq} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )
        if not mx.issubdtype(tokens.dtype, mx.integer):
            raise TypeError(f"{where} tokens must contain integer ids, got {tokens.dtype}")
        min_token = int(mx.min(tokens).item())
        max_token = int(mx.max(tokens).item())
        if min_token < 0 or max_token >= self.config.vocab_size:
            raise ValueError(
                f"{where} token ids must be in [0, {self.config.vocab_size}), got "
                f"range [{min_token}, {max_token}]"
            )
        positions = mx.arange(seq)[None, :]
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def encode(self, tokens: mx.array) -> mx.array:
        """Encode an observation ``(B, S)`` to a latent ``(B, D)`` (mean pool)."""
        hidden = self._embed_tokens(tokens, where="observation")
        hidden = hidden + self.encoder_mlp(self.encoder_norm(hidden))
        return mx.mean(hidden, axis=1)

    def encode_action(self, tokens: mx.array) -> mx.array:
        """Encode action tokens ``(B, A)`` to an action latent ``(B, D)``."""

        hidden = self._embed_tokens(tokens, where="action")
        hidden = hidden + self.action_encoder_mlp(self.action_encoder_norm(hidden))
        return mx.mean(hidden, axis=1)

    def step_latent(
        self,
        previous_latent: mx.array,
        action: mx.array,
        *,
        training_loops: int | None = None,
    ) -> mx.array:
        """Roll ``previous_latent`` forward under one typed action.

        The previous latent remains the stable loop's initial state ``z0``. The
        injected input ``x`` is a learned fusion of the previous state and encoded
        action, so action conditioning survives fixed-point iteration instead of
        existing only in the initial state.
        """

        if not isinstance(previous_latent, mx.array):
            raise TypeError(
                "previous_latent must be an mx.array, got "
                f"{type(previous_latent).__name__}"
            )
        if (
            previous_latent.ndim != 2
            or int(previous_latent.shape[1]) != self.config.hidden_size
        ):
            raise ValueError(
                "previous_latent must be (B, D) with "
                f"D={self.config.hidden_size}, got {tuple(previous_latent.shape)}"
            )
        if int(previous_latent.shape[0]) < 1:
            raise ValueError("previous_latent batch must be non-empty")
        if not mx.issubdtype(previous_latent.dtype, mx.floating):
            raise TypeError(
                "previous_latent must contain floating-point values, got "
                f"{previous_latent.dtype}"
            )

        action_latent = self.encode_action(action)
        if int(action_latent.shape[0]) != int(previous_latent.shape[0]):
            raise ValueError(
                f"action batch {int(action_latent.shape[0])} != previous_latent batch "
                f"{int(previous_latent.shape[0])}"
            )

        transition_features = mx.concatenate((previous_latent, action_latent), axis=-1)
        injected = self.transition_projection(self.transition_norm(transition_features))
        z = previous_latent[:, None, :]  # (B, 1, D)
        x = injected[:, None, :]  # (B, 1, D)
        loops = training_loops if training_loops is not None else self.config.train_loops
        if loops < 0:
            rolled = self.dynamics.forward(z, x, None, training_loops=None)
        else:
            rolled = self.dynamics.forward(z, x, None, training_loops=loops)
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
        action: mx.array,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Predict next-observation logits ``(B, L, V)`` and return the latent.

        One world-model step: encode ``obs`` -> apply ``action`` -> decode.
        """

        latent = self.encode(obs)
        rolled = self.step_latent(latent, action, training_loops=training_loops)
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
        actions: Sequence[mx.array],
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

        action_steps = self._validate_rollout_actions(actions, horizon)
        latent = self.encode(obs)
        for action in action_steps:
            latent = self.step_latent(latent, action, training_loops=training_loops)
        logits = self.decode(latent, length)
        return logits, latent

    def rollout_decode_all(
        self,
        obs: mx.array,
        actions: Sequence[mx.array],
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

        action_steps = self._validate_rollout_actions(actions, horizon)
        latent = self.encode(obs)
        per_step: list[mx.array] = []
        for action in action_steps:
            latent = self.step_latent(latent, action, training_loops=training_loops)
            per_step.append(self.decode(latent, length))
        return per_step

    @staticmethod
    def _validate_rollout_actions(
        actions: Sequence[mx.array], horizon: int
    ) -> tuple[mx.array, ...]:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if isinstance(actions, mx.array) or not isinstance(actions, Sequence):
            raise TypeError("actions must be a sequence containing one (B, A) array per step")
        action_steps = tuple(actions)
        if len(action_steps) != horizon:
            raise ValueError(
                f"actions length {len(action_steps)} != rollout horizon {horizon}"
            )
        for index, action in enumerate(action_steps):
            if not isinstance(action, mx.array):
                raise TypeError(
                    f"actions[{index}] must be an mx.array, got {type(action).__name__}"
                )
        return action_steps

    def __call__(
        self,
        obs: mx.array,
        action: mx.array,
        length: int,
        *,
        training_loops: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        return self.predict_next(obs, action, length, training_loops=training_loops)


__all__ = [
    "CodeLoopWorldModel",
    "CodeLoopWorldModelConfig",
]
