"""Zero-initialized platform context embedding for cppmega models."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cppmega_mlx.data.batch import batch_values_are_prevalidated
from cppmega_mlx.data.integer_validation import validated_integer_array
from cppmega_mlx.data.platform_context import MAX_PLATFORM_IDS, PLATFORM_VOCAB_SIZE


class PlatformEmbedding(nn.Module):
    """Per-document platform embedding broadcast across the token sequence."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int = PLATFORM_VOCAB_SIZE,
        max_ids: int = MAX_PLATFORM_IDS,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if max_ids <= 0:
            raise ValueError("max_ids must be positive")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.max_ids = int(max_ids)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding.weight = mx.zeros_like(self.embedding.weight)

    def __call__(
        self,
        platform_ids: mx.array | None,
        *,
        target_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        dtype = target_dtype or self.embedding.weight.dtype
        if platform_ids is None:
            return mx.array(0.0, dtype=dtype)
        self._validate_shape(platform_ids)

        if not batch_values_are_prevalidated():
            self._validate_range(platform_ids)
        ids = platform_ids.astype(mx.int64)
        embeddings = self.embedding(ids)
        mask = (ids != 0)[..., None].astype(embeddings.dtype)
        if ids.ndim == 2:
            out = mx.sum(embeddings * mask, axis=1)[:, None, :]
        else:
            out = mx.sum(embeddings * mask, axis=2)
        if out.dtype != dtype:
            out = out.astype(dtype)
        return out

    def validate_input_ids(self, platform_ids: mx.array | None) -> None:
        """Validate host-visible IDs before entering an MLX transform."""

        if platform_ids is None:
            return
        self._validate_shape(platform_ids)
        self._validate_range(platform_ids)

    def _validate_shape(self, platform_ids: mx.array) -> None:
        if platform_ids.ndim not in (2, 3):
            raise ValueError(
                "platform_ids must be shaped (B, K) or (B, S, K), "
                f"got {platform_ids.shape}"
            )
        if platform_ids.shape[-1] > self.max_ids:
            raise ValueError(
                f"platform_ids width {platform_ids.shape[-1]} exceeds max_ids={self.max_ids}"
            )

    def _validate_range(self, ids: mx.array) -> None:
        values = validated_integer_array(ids, where="platform_ids")
        if np.any(values < 0):
            raise ValueError("platform_ids must be non-negative")
        if np.any(values >= self.vocab_size):
            raise ValueError(
                f"platform_ids must be less than vocab_size={self.vocab_size}"
            )


CppMegaPlatformEmbedding = PlatformEmbedding

__all__ = ["CppMegaPlatformEmbedding", "PlatformEmbedding"]
