"""Loss helpers for local MLX language-model training."""

from __future__ import annotations

from typing import Mapping

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch


def next_token_cross_entropy(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
) -> tuple[mx.array, mx.array]:
    """Return masked next-token CE loss and the number of contributing tokens."""

    lm_batch = ensure_lm_batch(batch)
    logits = model(lm_batch.inputs, **lm_batch.model_kwargs())
    targets = lm_batch.targets

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits prefix shape {logits.shape[:2]} must match targets {targets.shape}"
        )

    token_losses = nn.losses.cross_entropy(
        logits.astype(mx.float32), targets, reduction="none"
    )
    mask = lm_batch.target_mask
    ntokens = mask.sum()
    denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
    loss = (token_losses * mask).astype(mx.float32).sum() / denom
    return loss, ntokens


__all__ = ["next_token_cross_entropy"]
