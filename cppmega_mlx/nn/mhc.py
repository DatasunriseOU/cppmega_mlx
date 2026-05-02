"""Standalone pure-MLX manifold hyperconnection branch mixer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_nonnegative_float(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _require_floating(name: str, x: mx.array) -> None:
    if not mx.issubdtype(x.dtype, mx.floating):
        raise TypeError(f"{name} must use a floating dtype, got {x.dtype}")


def _require_rank(name: str, x: mx.array, rank: int) -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {x.shape}")


@dataclass(frozen=True)
class ManifoldBranchMixerConfig:
    """Config for the standalone MLX mHC branch mixer."""

    hidden_size: int
    sinkhorn_iters: int = 5
    temperature: float = 1.0
    epsilon: float = 1e-6
    blend_alpha: float = 1.0
    max_branches: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("hidden_size", self.hidden_size)
        _require_nonnegative_int("sinkhorn_iters", self.sinkhorn_iters)
        _require_positive_float("temperature", self.temperature)
        _require_positive_float("epsilon", self.epsilon)
        _require_nonnegative_float("blend_alpha", self.blend_alpha)
        _require_nonnegative_int("max_branches", self.max_branches)


def sinkhorn_normalize(
    raw_matrix: mx.array,
    *,
    iters: int = 5,
    epsilon: float = 1e-6,
) -> mx.array:
    """Return a fp32 Sinkhorn-normalized transport matrix.

    The normalization follows nanochat's softmax initialization and alternating
    row/column renormalization, but stays standalone so tests can assert the
    routing invariant directly.
    """

    _require_rank("raw_matrix", raw_matrix, 3)
    _require_floating("raw_matrix", raw_matrix)
    _require_nonnegative_int("iters", int(iters))
    _require_positive_float("epsilon", float(epsilon))
    if raw_matrix.shape[-1] != raw_matrix.shape[-2]:
        raise ValueError(f"raw_matrix must be square on the last two dims, got {raw_matrix.shape}")

    transport = mx.softmax(raw_matrix.astype(mx.float32), axis=-2)
    for _ in range(int(iters)):
        row_sum = mx.sum(transport, axis=-1, keepdims=True)
        transport = transport / mx.maximum(row_sum, float(epsilon))
        col_sum = mx.sum(transport, axis=-2, keepdims=True)
        transport = transport / mx.maximum(col_sum, float(epsilon))
    return transport


class ManifoldBranchMixer(nn.Module):
    """Sinkhorn-style branch mixer for independent (B, T, C) branch tensors."""

    def __init__(self, config: ManifoldBranchMixerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        hidden = max(8, min(256, self.hidden_size // 4))
        self.score_proj = nn.Linear(self.hidden_size, hidden, bias=False)
        self.score_out = nn.Linear(hidden, 1, bias=False)

    def _validate_branches(self, branches: Sequence[mx.array]) -> tuple[int, int, int, int]:
        if len(branches) == 0:
            raise ValueError("ManifoldBranchMixer requires at least one branch")
        if self.config.max_branches > 0 and len(branches) > self.config.max_branches:
            raise ValueError(
                f"too many branches: got {len(branches)}, max_branches={self.config.max_branches}"
            )

        ref_shape = branches[0].shape
        for idx, branch in enumerate(branches):
            _require_rank(f"branches[{idx}]", branch, 3)
            _require_floating(f"branches[{idx}]", branch)
            if branch.shape != ref_shape:
                raise ValueError(f"branch shape mismatch: expected {ref_shape}, got {branch.shape}")
            if branch.dtype != branches[0].dtype:
                raise TypeError(
                    f"branches[{idx}] dtype {branch.dtype} must match branches[0] dtype "
                    f"{branches[0].dtype}"
                )
        if ref_shape[-1] != self.hidden_size:
            raise ValueError(f"branch hidden size must be {self.hidden_size}, got {ref_shape[-1]}")
        batch, seq, hidden = ref_shape
        return batch, seq, len(branches), hidden

    def routing_weights(self, branches: Sequence[mx.array]) -> mx.array:
        """Return per-batch branch weights with shape (B, N)."""

        batch, _, n_branches, _ = self._validate_branches(branches)
        if n_branches == 1:
            return mx.ones((batch, 1), dtype=branches[0].dtype)

        stacked = mx.stack(list(branches), axis=2)
        pooled = mx.mean(stacked, axis=1)
        keys = mx.tanh(self.score_proj(pooled))
        logits = mx.squeeze(self.score_out(keys), axis=-1)
        temperature = max(float(self.config.temperature), float(self.config.epsilon))

        if n_branches == 2:
            weights = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)
        else:
            raw_matrix = (keys.astype(mx.float32) @ mx.swapaxes(keys.astype(mx.float32), -1, -2))
            raw_matrix = raw_matrix / temperature
            eye = mx.eye(n_branches, dtype=mx.float32)
            raw_matrix = raw_matrix + logits.astype(mx.float32)[..., None] * eye
            transport = sinkhorn_normalize(
                raw_matrix,
                iters=self.config.sinkhorn_iters,
                epsilon=self.config.epsilon,
            )
            weights = mx.diagonal(transport, axis1=-2, axis2=-1)

        uniform = mx.ones_like(weights) / n_branches
        weights = mx.nan_to_num(weights, nan=1.0 / n_branches, posinf=1.0 / n_branches, neginf=0.0)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + float(self.config.epsilon))
        valid = mx.all(mx.isfinite(weights), axis=-1, keepdims=True)
        weights = mx.where(valid, weights, uniform)

        alpha = min(max(float(self.config.blend_alpha), 0.0), 1.0)
        if alpha < 1.0:
            weights = uniform + alpha * (weights - uniform)
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + float(self.config.epsilon))
        return weights.astype(branches[0].dtype)

    def __call__(self, branches: Sequence[mx.array]) -> mx.array:
        if len(branches) == 1:
            self._validate_branches(branches)
            return branches[0]
        weights = self.routing_weights(branches)
        stacked = mx.stack(list(branches), axis=2)
        return mx.sum(stacked * weights[:, None, :, None], axis=2)


CppMegaManifoldBranchMixer = ManifoldBranchMixer


__all__ = [
    "CppMegaManifoldBranchMixer",
    "ManifoldBranchMixer",
    "ManifoldBranchMixerConfig",
    "sinkhorn_normalize",
]
