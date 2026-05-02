"""Minimal MLX Multi-Token Prediction loss helper.

The contract mirrors the local cppmega FastMTP lane: K static roll-and-mask
depths, normalized beta-decayed depth weights, and total loss composition as
``next_token + lambda * mtp``. This module intentionally stays training-side
and does not alter model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn


DEFAULT_MTP_DEPTH = 2
DEFAULT_MTP_DECAY = 0.6
DEFAULT_MTP_LAMBDA = 0.3
MTP_IGNORE_INDEX = -1
MTP_INVALID_DOCUMENT_ID = -1


@dataclass(frozen=True)
class MTPLossConfig:
    """Static K-depth MTP loss settings."""

    depth: int = DEFAULT_MTP_DEPTH
    decay: float = DEFAULT_MTP_DECAY
    loss_weight: float = DEFAULT_MTP_LAMBDA
    ignore_index: int = MTP_IGNORE_INDEX

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("MTP depth must be non-negative")
        if self.decay <= 0:
            raise ValueError("MTP decay must be positive")
        if self.loss_weight < 0:
            raise ValueError("MTP loss weight must be non-negative")


@dataclass(frozen=True)
class MTPLossMetrics:
    """Per-depth and composed loss values for logging."""

    next_token_loss: mx.array
    mtp_loss: mx.array
    total_loss: mx.array
    per_depth_losses: tuple[mx.array, ...]
    depth_weights: mx.array
    loss_weight: float = DEFAULT_MTP_LAMBDA


class MTPInferenceHead(Protocol):
    """Small model surface required by MTP loss helpers."""

    token_embedding: nn.Embedding
    lm_head: nn.Linear


class MinimalMTPSharedBlock(nn.Module):
    """Small shared block recurred across MTP depths.

    The production CUDA path uses a transformer layer. For this local MLX helper
    contract, a residual MLP block is enough to prove the important ownership
    rule: one module instance is reused for every depth instead of allocating
    per-depth blocks.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.up = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.down = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states + self.down(nn.gelu(self.up(self.norm(hidden_states))))


def compute_mtp_step_weights(
    depth: int = DEFAULT_MTP_DEPTH,
    decay: float = DEFAULT_MTP_DECAY,
) -> mx.array:
    """Return normalized ``beta**k`` weights for ``k in [0, depth)``."""

    if depth < 0:
        raise ValueError("MTP depth must be non-negative")
    if decay <= 0:
        raise ValueError("MTP decay must be positive")
    if depth == 0:
        return mx.zeros((0,), dtype=mx.float32)

    raw = mx.power(mx.array(decay, dtype=mx.float32), mx.arange(depth))
    return raw / raw.sum()


def roll_and_mask_mtp_labels(
    targets: mx.array,
    *,
    depth: int = DEFAULT_MTP_DEPTH,
    ignore_index: int = MTP_IGNORE_INDEX,
    document_ids: mx.array | None = None,
) -> tuple[mx.array, ...]:
    """Build static-shape future labels for all MTP depths.

    Each returned label tensor keeps the input ``(B, T)`` shape. Depth 1 predicts
    one step beyond the next-token target, so the last token is masked. Depth 2
    masks the last two positions, and so on.
    """

    if targets.ndim != 2:
        raise ValueError(f"targets must be shaped (B, T), got {targets.shape}")
    if depth < 0:
        raise ValueError("MTP depth must be non-negative")

    base_doc_ids = _validate_mtp_document_ids(document_ids, tokens=targets)
    rolled_doc_ids = base_doc_ids
    rolled = targets
    labels: list[mx.array] = []
    for _ in range(depth):
        rolled = _roll_left_with_fill(rolled, ignore_index)
        if base_doc_ids is not None and rolled_doc_ids is not None:
            rolled_doc_ids = _roll_left_with_fill(
                rolled_doc_ids,
                MTP_INVALID_DOCUMENT_ID,
            )
            rolled = _mask_cross_document_roll(
                rolled,
                base_doc_ids,
                rolled_doc_ids,
                fill_value=ignore_index,
            )
        labels.append(rolled)
    return tuple(labels)


def roll_and_mask_mtp_ids(
    token_ids: mx.array,
    *,
    depth: int = DEFAULT_MTP_DEPTH,
    document_ids: mx.array | None = None,
) -> tuple[mx.array, ...]:
    """Build static-shape teacher-forcing token IDs for all MTP depths."""

    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be shaped (B, T), got {token_ids.shape}")
    if depth < 0:
        raise ValueError("MTP depth must be non-negative")

    base_doc_ids = _validate_mtp_document_ids(document_ids, tokens=token_ids)
    rolled_doc_ids = base_doc_ids
    rolled = token_ids
    ids: list[mx.array] = []
    for _ in range(depth):
        rolled = _roll_left_with_fill(rolled, 0)
        if base_doc_ids is not None and rolled_doc_ids is not None:
            rolled_doc_ids = _roll_left_with_fill(
                rolled_doc_ids,
                MTP_INVALID_DOCUMENT_ID,
            )
            rolled = _mask_cross_document_roll(
                rolled,
                base_doc_ids,
                rolled_doc_ids,
                fill_value=0,
            )
        ids.append(rolled)
    return tuple(ids)


def mtp_cross_entropy_from_logits(
    logits: mx.array,
    labels: mx.array,
    *,
    ignore_index: int = MTP_IGNORE_INDEX,
) -> mx.array:
    """Mean CE over non-ignored MTP labels without changing tensor shape."""

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits prefix shape {logits.shape[:2]} must match labels {labels.shape}"
        )
    safe_labels = mx.where(labels == ignore_index, mx.zeros_like(labels), labels)
    token_losses = nn.losses.cross_entropy(
        logits.astype(mx.float32),
        safe_labels,
        reduction="none",
    )
    valid_labels = cast(mx.array, labels != ignore_index)
    valid_mask = valid_labels.astype(mx.float32)
    denom = mx.maximum(valid_mask.sum(), mx.array(1.0, dtype=mx.float32))
    return (token_losses * valid_mask).sum() / denom


def compute_weighted_mtp_loss(
    per_depth_losses: tuple[mx.array, ...],
    *,
    decay: float = DEFAULT_MTP_DECAY,
) -> tuple[mx.array, mx.array]:
    """Return weighted MTP CE and the normalized depth weights used."""

    weights = compute_mtp_step_weights(len(per_depth_losses), decay)
    if not per_depth_losses:
        return mx.array(0.0, dtype=mx.float32), weights

    weighted = mx.array(0.0, dtype=mx.float32)
    for index, loss in enumerate(per_depth_losses):
        weighted = weighted + weights[index] * loss
    return weighted, weights


class MinimalMTPHead(nn.Module):
    """Shared-block MTP head for static K-depth loss tests.

    This is the smallest useful MLX contract for M0.5: it reuses a model's token
    embedding and lm head, recurs one shared block across all depths, emits one
    logits tensor per future depth, and leaves the main model's inference path
    untouched.
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        *,
        config: MTPLossConfig | None = None,
        shared_block: nn.Module | None = None,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.lm_head = lm_head
        self.config = config or MTPLossConfig()
        hidden_size = int(token_embedding.weight.shape[1])
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.embedding_norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.shared_block = (
            shared_block if shared_block is not None else MinimalMTPSharedBlock(hidden_size)
        )
        self.output_norm = nn.RMSNorm(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        target_tokens: mx.array,
        *,
        document_ids: mx.array | None = None,
    ) -> tuple[mx.array, ...]:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be shaped (B, T, D), got {hidden_states.shape}"
            )
        if hidden_states.shape[:2] != target_tokens.shape:
            raise ValueError(
                f"hidden_states prefix shape {hidden_states.shape[:2]} must match "
                f"target_tokens {target_tokens.shape}"
            )

        teacher_ids = roll_and_mask_mtp_ids(
            target_tokens,
            depth=self.config.depth,
            document_ids=document_ids,
        )
        logits_by_depth: list[mx.array] = []
        # Match CUDA FastMTP: MTP supervises the shared head/block, but does
        # not backpropagate through the already-computed main decoder states.
        h = mx.stop_gradient(hidden_states)
        for ids in teacher_ids:
            teacher_emb = self.token_embedding(ids)
            h_mtp = self.proj(
                mx.concatenate(
                    [self.hidden_norm(h), self.embedding_norm(teacher_emb)],
                    axis=-1,
                )
            )
            h = self.output_norm(self.shared_block(h_mtp))
            logits_by_depth.append(_lm_head_with_stopped_weight(self.lm_head, h))
        return tuple(logits_by_depth)

    def loss(
        self,
        hidden_states: mx.array,
        target_tokens: mx.array,
        *,
        document_ids: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...], mx.array]:
        labels = roll_and_mask_mtp_labels(
            target_tokens,
            depth=self.config.depth,
            ignore_index=self.config.ignore_index,
            document_ids=document_ids,
        )
        logits_by_depth = self(
            hidden_states,
            target_tokens,
            document_ids=document_ids,
        )
        per_depth = tuple(
            mtp_cross_entropy_from_logits(
                logits,
                label,
                ignore_index=self.config.ignore_index,
            )
            for logits, label in zip(logits_by_depth, labels, strict=True)
        )
        mtp_loss, depth_weights = compute_weighted_mtp_loss(
            per_depth,
            decay=self.config.decay,
        )
        return mtp_loss, per_depth, depth_weights


def attach_mtp_head(
    model: nn.Module,
    *,
    config: MTPLossConfig | None = None,
) -> MinimalMTPHead:
    """Attach a persistent MTP head to a model using direct module aliasing."""

    token_embedding = getattr(model, "token_embedding", None)
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(token_embedding, nn.Embedding):
        raise TypeError("MTP loss requires model.token_embedding to be an nn.Embedding")
    if not isinstance(lm_head, nn.Linear):
        raise TypeError("MTP loss requires model.lm_head to be an nn.Linear")

    head = MinimalMTPHead(token_embedding, lm_head, config=config)
    setattr(model, "mtp_head", head)
    return head


def get_or_attach_mtp_head(
    model: nn.Module,
    *,
    config: MTPLossConfig | None = None,
) -> MinimalMTPHead:
    """Return the model-owned MTP head, creating it once if absent."""

    cfg = config or MTPLossConfig()
    existing = getattr(model, "mtp_head", None)
    if existing is None:
        return attach_mtp_head(model, config=cfg)
    if not isinstance(existing, MinimalMTPHead):
        raise TypeError("model.mtp_head must be a MinimalMTPHead")
    if existing.config != cfg:
        raise ValueError(
            "model.mtp_head config does not match requested MTP loss config; "
            "attach a head with the desired config before training"
        )
    return existing


def next_token_and_mtp_loss(
    next_token_loss: mx.array,
    mtp_loss: mx.array,
    *,
    loss_weight: float = DEFAULT_MTP_LAMBDA,
) -> mx.array:
    """Compose total training loss as ``NTP + lambda * MTP``."""

    if loss_weight < 0:
        raise ValueError("MTP loss weight must be non-negative")
    return next_token_loss + loss_weight * mtp_loss


def mtp_loss_for_model(
    model: MTPInferenceHead,
    target_tokens: mx.array,
    *,
    config: MTPLossConfig | None = None,
) -> MTPLossMetrics:
    """Compute MTP metrics with a model's embedding and lm head.

    ``next_token_loss`` is returned as zero here because this helper owns only
    the MTP side contract. Use ``next_token_and_mtp_loss`` when composing with an
    externally computed next-token CE.
    """

    cfg = config or MTPLossConfig()
    head = MinimalMTPHead(model.token_embedding, model.lm_head, config=cfg)
    hidden_states = model.token_embedding(target_tokens)
    mtp_loss, per_depth, depth_weights = head.loss(hidden_states, target_tokens)
    next_token_loss = mx.array(0.0, dtype=mx.float32)
    total_loss = next_token_and_mtp_loss(
        next_token_loss,
        mtp_loss,
        loss_weight=cfg.loss_weight,
    )
    return MTPLossMetrics(
        next_token_loss=next_token_loss,
        mtp_loss=mtp_loss,
        total_loss=total_loss,
        per_depth_losses=per_depth,
        depth_weights=depth_weights,
        loss_weight=cfg.loss_weight,
    )


def _roll_left_with_fill(x: mx.array, fill_value: int) -> mx.array:
    rolled = mx.roll(x, -1, 1)
    positions = mx.arange(x.shape[1])[None, :]
    keep = positions < x.shape[1] - 1
    return mx.where(keep, rolled, mx.array(fill_value, dtype=x.dtype))


def _validate_mtp_document_ids(
    document_ids: mx.array | None,
    *,
    tokens: mx.array,
) -> mx.array | None:
    if document_ids is None:
        return None
    if document_ids.ndim != 2:
        raise ValueError(f"document_ids must be shaped (B, T), got {document_ids.shape}")
    if document_ids.shape != tokens.shape:
        raise ValueError(
            f"document_ids shape {document_ids.shape} must match token shape {tokens.shape}"
        )
    return document_ids.astype(mx.int32)


def _mask_cross_document_roll(
    x: mx.array,
    base_doc_ids: mx.array,
    rolled_doc_ids: mx.array,
    *,
    fill_value: int,
) -> mx.array:
    same_doc = mx.equal(base_doc_ids, rolled_doc_ids)
    same_doc = mx.logical_and(same_doc, base_doc_ids >= 0)
    same_doc = mx.logical_and(same_doc, rolled_doc_ids >= 0)
    return mx.where(same_doc, x, mx.array(fill_value, dtype=x.dtype))


def _lm_head_with_stopped_weight(lm_head: nn.Linear, hidden_states: mx.array) -> mx.array:
    """Apply lm_head while matching CUDA FastMTP output-weight detach semantics."""

    logits = hidden_states @ mx.stop_gradient(lm_head.weight).T
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + mx.stop_gradient(bias)
    return logits


__all__ = [
    "DEFAULT_MTP_DECAY",
    "DEFAULT_MTP_DEPTH",
    "DEFAULT_MTP_LAMBDA",
    "MTP_IGNORE_INDEX",
    "MTPLossConfig",
    "MTPLossMetrics",
    "MTPInferenceHead",
    "MinimalMTPHead",
    "MinimalMTPSharedBlock",
    "attach_mtp_head",
    "compute_mtp_step_weights",
    "compute_weighted_mtp_loss",
    "get_or_attach_mtp_head",
    "mtp_cross_entropy_from_logits",
    "mtp_loss_for_model",
    "next_token_and_mtp_loss",
    "roll_and_mask_mtp_ids",
    "roll_and_mask_mtp_labels",
]
