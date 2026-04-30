"""Tiny MLX token batches used by local trainer smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class LMTokenBatch:
    """A dense next-token LM batch plus optional cppmega structure side-channels."""

    tokens: mx.array
    attention_mask: mx.array | None = None
    structure_ids: mx.array | None = None
    dep_levels: mx.array | None = None
    ast_depth_ids: mx.array | None = None
    sibling_index_ids: mx.array | None = None
    node_type_ids: mx.array | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must be shaped (B, S), got {self.tokens.shape}")
        if self.tokens.shape[1] < 2:
            raise ValueError("tokens sequence length must be at least 2")

        for name, value in self.structure_fields().items():
            if value is not None and value.shape != self.tokens.shape:
                raise ValueError(
                    f"{name} must match tokens shape {self.tokens.shape}, got {value.shape}"
                )

        if self.attention_mask is not None and self.attention_mask.shape != self.tokens.shape:
            raise ValueError(
                "attention_mask must match tokens shape "
                f"{self.tokens.shape}, got {self.attention_mask.shape}"
            )

    @property
    def inputs(self) -> mx.array:
        return self.tokens[:, :-1]

    @property
    def targets(self) -> mx.array:
        return self.tokens[:, 1:]

    @property
    def target_mask(self) -> mx.array:
        if self.attention_mask is None:
            return mx.ones(self.targets.shape, dtype=mx.float32)
        return self.attention_mask[:, 1:].astype(mx.float32)

    def structure_fields(self) -> dict[str, mx.array | None]:
        return {
            "structure_ids": self.structure_ids,
            "dep_levels": self.dep_levels,
            "ast_depth_ids": self.ast_depth_ids,
            "sibling_index_ids": self.sibling_index_ids,
            "node_type_ids": self.node_type_ids,
        }

    def model_kwargs(self) -> dict[str, mx.array]:
        return {
            name: value[:, :-1]
            for name, value in self.structure_fields().items()
            if value is not None
        }

    def as_dict(self) -> dict[str, mx.array]:
        data: dict[str, mx.array] = {"tokens": self.tokens}
        if self.attention_mask is not None:
            data["attention_mask"] = self.attention_mask
        data.update({k: v for k, v in self.structure_fields().items() if v is not None})
        return data


def ensure_lm_batch(batch: LMTokenBatch | Mapping[str, Any] | mx.array) -> LMTokenBatch:
    """Normalize supported tiny-trainer batch inputs into ``LMTokenBatch``."""

    if isinstance(batch, LMTokenBatch):
        return batch
    if isinstance(batch, mx.array):
        return LMTokenBatch(tokens=batch)
    if isinstance(batch, Mapping):
        if "tokens" not in batch:
            raise ValueError("batch mapping must contain a 'tokens' array")
        return LMTokenBatch(
            tokens=batch["tokens"],
            attention_mask=batch.get("attention_mask"),
            structure_ids=batch.get("structure_ids"),
            dep_levels=batch.get("dep_levels"),
            ast_depth_ids=batch.get("ast_depth_ids"),
            sibling_index_ids=batch.get("sibling_index_ids"),
            node_type_ids=batch.get("node_type_ids"),
        )
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def synthetic_token_batch(
    *,
    batch_size: int = 2,
    seq_length: int = 8,
    vocab_size: int = 64,
    seed: int = 0,
    include_structure: bool = False,
) -> LMTokenBatch:
    """Create a deterministic synthetic batch for GPU smoke tests."""

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if seq_length < 2:
        raise ValueError("seq_length must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    tokens = mx.array(
        rng.integers(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)
    )
    attention_mask = mx.ones((batch_size, seq_length), dtype=mx.float32)

    if not include_structure:
        return LMTokenBatch(tokens=tokens, attention_mask=attention_mask)

    structure_vocab = max(2, min(vocab_size, 32))
    structure_ids = mx.array(
        rng.integers(
            0, structure_vocab, size=(batch_size, seq_length), dtype=np.int32
        )
    )
    dep_levels = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    ast_depth_ids = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    sibling_index_ids = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    node_type_ids = mx.array(
        rng.integers(0, structure_vocab, size=(batch_size, seq_length), dtype=np.int32)
    )
    return LMTokenBatch(
        tokens=tokens,
        attention_mask=attention_mask,
        structure_ids=structure_ids,
        dep_levels=dep_levels,
        ast_depth_ids=ast_depth_ids,
        sibling_index_ids=sibling_index_ids,
        node_type_ids=node_type_ids,
    )


__all__ = ["LMTokenBatch", "ensure_lm_batch", "synthetic_token_batch"]
