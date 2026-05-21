"""A tiny MLX decoder-only LM used to validate local training plumbing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.platform_context import MAX_PLATFORM_IDS, PLATFORM_VOCAB_SIZE
from cppmega_mlx.nn.platform_embedding import CppMegaPlatformEmbedding


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int = 64
    hidden_size: int = 32
    num_layers: int = 1
    num_heads: int = 4
    ffn_hidden_size: int = 64
    max_seq_length: int = 64
    structure_vocab_size: int = 32
    platform_vocab_size: int = PLATFORM_VOCAB_SIZE
    platform_max_ids: int = MAX_PLATFORM_IDS

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.platform_vocab_size <= 1:
            raise ValueError("platform_vocab_size must be greater than one")
        if self.platform_max_ids <= 0:
            raise ValueError("platform_max_ids must be positive")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class TinyDecoderBlock(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.attn = nn.MultiHeadAttention(config.hidden_size, config.num_heads, bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size)
        self.ffn_gate = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.ffn_down = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array, mask: mx.array) -> mx.array:
        attn_in = self.attn_norm(hidden_states)
        hidden_states = hidden_states + self.attn(attn_in, attn_in, attn_in, mask)
        ffn_in = self.ffn_norm(hidden_states)
        return hidden_states + self.ffn_down(nn.gelu(self.ffn_gate(ffn_in)))


class TinyLM(nn.Module):
    """Minimal causal LM with optional structure embeddings for smoke coverage."""

    def __init__(self, config: TinyLMConfig | None = None):
        super().__init__()
        self.config = config or TinyLMConfig()
        cfg = self.config
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)
        self.structure_embedding = nn.Embedding(
            cfg.structure_vocab_size, cfg.hidden_size
        )
        self.platform_embedding = CppMegaPlatformEmbedding(
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.platform_vocab_size,
            max_ids=cfg.platform_max_ids,
        )
        self.layers = [TinyDecoderBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _structure_core(
        self,
        structure_ids: mx.array | None,
        dep_levels: mx.array | None,
        seq_length: int,
    ) -> mx.array | None:
        if structure_ids is None and dep_levels is None:
            return None
        if structure_ids is None:
            assert dep_levels is not None
            core = dep_levels[:, :seq_length]
        elif dep_levels is None:
            core = structure_ids[:, :seq_length]
        else:
            core = structure_ids[:, :seq_length] + dep_levels[:, :seq_length]
        return core % self.config.structure_vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        platform_ids: mx.array | None = None,
    ) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be shaped (B, S), got {input_ids.shape}")

        seq_length = input_ids.shape[1]
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq_length} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )

        positions = mx.arange(seq_length)[None, :]
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        structure_core = self._structure_core(structure_ids, dep_levels, seq_length)
        if structure_core is not None:
            hidden_states = hidden_states + self.structure_embedding(structure_core)

        if platform_ids is not None:
            if platform_ids.ndim not in (2, 3):
                raise ValueError(
                    "platform_ids must be shaped (B, K) or (B, S, K), "
                    f"got {platform_ids.shape}"
                )
            if platform_ids.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    "platform_ids batch dimension must match input batch "
                    f"{input_ids.shape[0]}, got {platform_ids.shape[0]}"
                )
            if platform_ids.ndim == 3 and platform_ids.shape[1] != seq_length:
                raise ValueError(
                    "token-local platform_ids sequence dimension must match input "
                    f"sequence {seq_length}, got {platform_ids.shape[1]}"
                )
            hidden_states = hidden_states + self.platform_embedding(
                platform_ids,
                target_dtype=hidden_states.dtype,
            )

        for optional_ids in (ast_depth_ids, sibling_index_ids, node_type_ids):
            if optional_ids is not None:
                channel_ids = optional_ids[:, :seq_length]
                hidden_states = hidden_states + self.structure_embedding(
                    channel_ids % self.config.structure_vocab_size
                )

        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            seq_length, dtype=hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return self.lm_head(self.norm(hidden_states))


__all__ = ["TinyDecoderBlock", "TinyLM", "TinyLMConfig"]
