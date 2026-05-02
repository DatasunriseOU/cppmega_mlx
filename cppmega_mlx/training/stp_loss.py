"""Minimal MLX Semantic Tube Prediction loss helper.

STP is an auxiliary geodesic regularizer over decoder hidden-state trajectories:
``1 - cosine(h[r] - h[s], h[t] - h[r])`` for ordered triples ``s < r < t``.
The local MLX helper uses deterministic static-shape triples instead of random
sampling so compiled training paths remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


DEFAULT_STP_SPANS = 1
DEFAULT_STP_LAMBDA = 0.0
STP_COSINE_EPSILON = 1e-8


@dataclass(frozen=True)
class STPLossConfig:
    """Static STP auxiliary-loss settings."""

    n_spans: int = DEFAULT_STP_SPANS
    loss_weight: float = DEFAULT_STP_LAMBDA

    def __post_init__(self) -> None:
        _validate_n_spans(self.n_spans)
        if self.loss_weight < 0:
            raise ValueError("STP loss weight must be non-negative")


@dataclass(frozen=True)
class STPLossMetrics:
    """Composed STP loss values for logging."""

    next_token_loss: mx.array
    stp_loss: mx.array
    total_loss: mx.array
    n_spans: int
    loss_weight: float = DEFAULT_STP_LAMBDA


def compute_stp_loss(
    hidden_states: mx.array | tuple[mx.array, ...] | list[mx.array],
    *,
    n_spans: int = DEFAULT_STP_SPANS,
) -> mx.array:
    """Compute deterministic STP geodesic loss.

    ``hidden_states`` accepts one final layer shaped ``(B, T, D)`` or a tuple of
    layer states. Tuple input averages the scalar loss across layers, matching
    nanochat's multi-layer variant while avoiding runtime random sampling.
    """

    spans = _validate_n_spans(n_spans)
    if isinstance(hidden_states, (list, tuple)):
        if not hidden_states:
            return mx.array(0.0, dtype=mx.float32)
        total = mx.array(0.0, dtype=mx.float32)
        for layer_hidden_states in hidden_states:
            total = total + _stp_loss_single(layer_hidden_states, n_spans=spans)
        return total / len(hidden_states)
    return _stp_loss_single(hidden_states, n_spans=spans)


def next_token_and_stp_loss(
    next_token_loss: mx.array,
    stp_loss: mx.array,
    *,
    loss_weight: float = DEFAULT_STP_LAMBDA,
) -> mx.array:
    """Compose total training loss as ``NTP + lambda * STP``."""

    if loss_weight < 0:
        raise ValueError("STP loss weight must be non-negative")
    return next_token_loss + loss_weight * stp_loss


def _stp_loss_single(hidden_states: mx.array, *, n_spans: int) -> mx.array:
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be shaped (B, T, D), got {hidden_states.shape}"
        )
    if n_spans == 0 or hidden_states.shape[1] < 3:
        return hidden_states.astype(mx.float32).sum() * 0.0

    sequence_length = int(hidden_states.shape[1])
    total_loss = mx.array(0.0, dtype=mx.float32)
    for span_index in range(n_spans):
        start, middle, end = _deterministic_triple(
            span_index,
            n_spans=n_spans,
            sequence_length=sequence_length,
        )
        direction_a = (hidden_states[:, middle, :] - hidden_states[:, start, :]).astype(
            mx.float32
        )
        direction_b = (hidden_states[:, end, :] - hidden_states[:, middle, :]).astype(
            mx.float32
        )
        total_loss = total_loss + (1.0 - _cosine_similarity(direction_a, direction_b)).mean()
    return total_loss / n_spans


def _deterministic_triple(
    span_index: int,
    *,
    n_spans: int,
    sequence_length: int,
) -> tuple[int, int, int]:
    max_start = sequence_length - 3
    if n_spans == 1:
        start = 0
    else:
        start = (span_index * max_start) // (n_spans - 1)
    return start, start + 1, start + 2


def _cosine_similarity(direction_a: mx.array, direction_b: mx.array) -> mx.array:
    dot = (direction_a * direction_b).sum(axis=-1)
    norm_a = (direction_a * direction_a).sum(axis=-1)
    norm_b = (direction_b * direction_b).sum(axis=-1)
    denom = mx.sqrt(
        mx.maximum(
            norm_a * norm_b,
            mx.array(STP_COSINE_EPSILON, dtype=mx.float32),
        )
    )
    return dot / denom


def _validate_n_spans(n_spans: int) -> int:
    if not isinstance(n_spans, int):
        raise TypeError("STP n_spans must be an integer")
    if n_spans < 0:
        raise ValueError("STP n_spans must be non-negative")
    return n_spans


__all__ = [
    "DEFAULT_STP_LAMBDA",
    "DEFAULT_STP_SPANS",
    "STP_COSINE_EPSILON",
    "STPLossConfig",
    "STPLossMetrics",
    "compute_stp_loss",
    "next_token_and_stp_loss",
]
