"""Auxiliary losses for domain-routed code/build/shell/diagnostic training."""

from __future__ import annotations

import mlx.core as mx

_NEG_INF = -1e9


def _check_rank(name: str, value: mx.array, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {tuple(value.shape)}")


def _log_softmax(logits: mx.array, axis: int = -1) -> mx.array:
    logits = logits.astype(mx.float32)
    max_v = mx.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_v
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))


def opener_domain_classification_loss(
    domain_logits: mx.array,
    target_domain_ids: mx.array,
    opener_mask: mx.array,
    *,
    loss_coeff: float = 1.0,
) -> mx.array:
    """Cross entropy for predicting the active domain at opener tokens.

    Args:
        domain_logits: ``(B, S, D)`` logits from hidden states.
        target_domain_ids: ``(B, S)`` target ``DomainKind`` ids.
        opener_mask: ``(B, S)`` 1 at domain opener tokens, 0 elsewhere.
    """

    _check_rank("domain_logits", domain_logits, 3)
    _check_rank("target_domain_ids", target_domain_ids, 2)
    _check_rank("opener_mask", opener_mask, 2)
    B, S, D = (int(x) for x in domain_logits.shape)
    if tuple(target_domain_ids.shape) != (B, S):
        raise ValueError(
            f"target_domain_ids shape {tuple(target_domain_ids.shape)} must be ({B},{S})"
        )
    if tuple(opener_mask.shape) != (B, S):
        raise ValueError(f"opener_mask shape {tuple(opener_mask.shape)} must be ({B},{S})")

    targets = target_domain_ids.astype(mx.int32)
    if bool(mx.any((targets < 0) | (targets >= D)).item()):
        raise ValueError(f"target_domain_ids must be in [0,{D})")
    logp = _log_softmax(domain_logits, axis=-1)
    picked = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    mask = opener_mask.astype(mx.float32)
    denom = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return (-mx.sum(picked * mask) / denom) * float(loss_coeff)


def domain_edge_bce_loss(
    edge_logits: mx.array,
    edge_targets: mx.array,
    *,
    edge_mask: mx.array | None = None,
    pos_weight: float = 1.0,
    loss_coeff: float = 1.0,
) -> mx.array:
    """BCE-with-logits for build/shell/diagnostic/cross-domain edge targets."""

    _check_rank("edge_logits", edge_logits, 3)
    if tuple(edge_targets.shape) != tuple(edge_logits.shape):
        raise ValueError(
            f"edge_targets shape {tuple(edge_targets.shape)} must match edge_logits {tuple(edge_logits.shape)}"
        )
    logits = edge_logits.astype(mx.float32)
    targets = edge_targets.astype(mx.float32)
    valid = logits > (_NEG_INF / 2.0)
    if edge_mask is not None:
        if tuple(edge_mask.shape) != tuple(edge_logits.shape):
            raise ValueError(
                f"edge_mask shape {tuple(edge_mask.shape)} must match edge_logits {tuple(edge_logits.shape)}"
            )
        valid = valid & (edge_mask.astype(mx.bool_))
    valid_f = valid.astype(mx.float32)
    max0 = mx.maximum(logits, mx.zeros_like(logits))
    bce = max0 - logits * targets + mx.log1p(mx.exp(-mx.abs(logits)))
    weight = mx.where(
        targets > 0,
        mx.array(float(pos_weight), dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    denom = mx.maximum(mx.sum(valid_f * weight), mx.array(1.0, dtype=mx.float32))
    return (mx.sum(bce * valid_f * weight) / denom) * float(loss_coeff)


def cross_domain_retrieval_ranking_loss(
    scores: mx.array,
    positive_indices: mx.array,
    negative_indices: mx.array,
    *,
    margin: float = 1.0,
    loss_coeff: float = 1.0,
) -> mx.array:
    """Margin loss for diagnostic/build tokens ranking linked spans higher.

    ``scores`` is ``(B, Q, S)``. ``positive_indices`` and ``negative_indices``
    are ``(B, Q, P/N)`` with ``-1`` sentinel padding.
    """

    _check_rank("scores", scores, 3)
    _check_rank("positive_indices", positive_indices, 3)
    _check_rank("negative_indices", negative_indices, 3)
    B, Q, S = (int(x) for x in scores.shape)
    if tuple(positive_indices.shape[:2]) != (B, Q):
        raise ValueError(
            f"positive_indices leading shape {tuple(positive_indices.shape[:2])} must be ({B},{Q})"
        )
    if tuple(negative_indices.shape[:2]) != (B, Q):
        raise ValueError(
            f"negative_indices leading shape {tuple(negative_indices.shape[:2])} must be ({B},{Q})"
        )

    pos_mask = positive_indices >= 0
    neg_mask = negative_indices >= 0
    if bool(mx.any((positive_indices >= S) | (negative_indices >= S)).item()):
        raise ValueError(f"ranking indices must be < score width {S}")

    pos_idx = mx.maximum(positive_indices.astype(mx.int32), mx.zeros_like(positive_indices.astype(mx.int32)))
    neg_idx = mx.maximum(negative_indices.astype(mx.int32), mx.zeros_like(negative_indices.astype(mx.int32)))
    pos_scores = mx.take_along_axis(scores, pos_idx, axis=-1)
    neg_scores = mx.take_along_axis(scores, neg_idx, axis=-1)
    pair = (
        float(margin)
        + neg_scores[..., None, :]
        - pos_scores[..., :, None]
    )
    mask = (pos_mask[..., :, None] & neg_mask[..., None, :]).astype(mx.float32)
    loss = mx.maximum(pair, mx.zeros_like(pair)) * mask
    denom = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return (mx.sum(loss) / denom) * float(loss_coeff)


__all__ = [
    "cross_domain_retrieval_ranking_loss",
    "domain_edge_bce_loss",
    "opener_domain_classification_loss",
]
