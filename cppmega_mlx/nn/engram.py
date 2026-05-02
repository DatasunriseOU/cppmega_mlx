"""Standalone MLX Engram branch for local per-block experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.mamba3 import causal_depthwise_conv1d


def parse_ngram_orders(ngram_orders: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse and de-duplicate positive n-gram orders, matching nanochat defaults."""

    if isinstance(ngram_orders, str):
        raw_orders = [int(part.strip()) for part in ngram_orders.split(",") if part.strip()]
    else:
        raw_orders = [int(order) for order in ngram_orders]

    orders: list[int] = []
    seen: set[int] = set()
    for order in raw_orders:
        if order <= 0 or order in seen:
            continue
        orders.append(order)
        seen.add(order)
    return tuple(orders or (2, 3, 4))


@dataclass(frozen=True)
class EngramConfig:
    """Small config for the standalone local MLX Engram branch."""

    hidden_size: int
    ngram_orders: str | tuple[int, ...] | list[int] = "2,3,4"
    bottleneck_dim: int = 0
    dropout: float = 0.0
    gated: bool = False
    gate_sqrt_compress: bool = False
    conv_kernel: int = 0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _require_positive_int("hidden_size", self.hidden_size)
        parse_ngram_orders(self.ngram_orders)
        if self.bottleneck_dim < 0:
            raise ValueError(f"bottleneck_dim must be non-negative, got {self.bottleneck_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.conv_kernel < 0:
            raise ValueError(f"conv_kernel must be non-negative, got {self.conv_kernel}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    @property
    def orders(self) -> tuple[int, ...]:
        return parse_ngram_orders(self.ngram_orders)

    @property
    def effective_bottleneck_dim(self) -> int:
        return int(self.bottleneck_dim) if self.bottleneck_dim else max(1, self.hidden_size // 4)


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_rank(name: str, x: mx.array, rank: int) -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {x.shape}")


def _require_floating(name: str, x: mx.array) -> None:
    if not mx.issubdtype(x.dtype, mx.floating):
        raise TypeError(f"{name} must use a floating dtype, got {x.dtype}")


def _rms_norm_last(x: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def _same_doc_shift_mask(doc_ids: mx.array, shift: int, dtype: mx.Dtype) -> mx.array:
    batch, seq = doc_ids.shape
    if shift == 0:
        return mx.ones((batch, seq), dtype=dtype)
    if shift >= seq:
        return mx.zeros((batch, seq), dtype=dtype)
    same = mx.equal(doc_ids[:, shift:], doc_ids[:, :-shift]).astype(dtype)
    return mx.pad(same, [(0, 0), (shift, 0)])


def causal_local_average(x: mx.array, order: int, doc_ids: mx.array | None = None) -> mx.array:
    """Average current and previous order-1 tokens without crossing doc boundaries."""

    _require_rank("x", x, 3)
    _require_floating("x", x)
    _require_positive_int("order", order)
    batch, seq, _ = x.shape

    if doc_ids is not None:
        _validate_doc_ids(x, doc_ids)

    total = mx.zeros_like(x)
    for shift in range(order):
        if shift == 0:
            shifted = x
        elif shift < seq:
            shifted = mx.pad(x[:, :-shift, :], [(0, 0), (shift, 0), (0, 0)])
        else:
            shifted = mx.zeros_like(x)
        if doc_ids is not None:
            shifted = shifted * mx.expand_dims(_same_doc_shift_mask(doc_ids, shift, x.dtype), -1)
        total = total + shifted
    return total / order


def _validate_doc_ids(x: mx.array, doc_ids: mx.array) -> None:
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must have shape (batch, seq), got {doc_ids.shape}")
    if doc_ids.shape != x.shape[:2]:
        raise ValueError(f"doc_ids shape {doc_ids.shape} must match x leading dims {x.shape[:2]}")


def causal_depthwise_silu_conv1d(
    x: mx.array,
    weight: mx.array,
    *,
    eps: float = 1e-6,
    doc_ids: mx.array | None = None,
) -> mx.array:
    """RMSNorm + causal grouped SiLU conv over MLX NLC tensors."""

    _require_rank("x", x, 3)
    _require_floating("x", x)
    _require_rank("weight", weight, 3)
    if weight.shape[0] != x.shape[-1] or weight.shape[-1] != 1:
        raise ValueError(f"weight must be shaped (hidden, kernel, 1) for x={x.shape}, got {weight.shape}")

    normed = _rms_norm_last(x, eps)
    if doc_ids is None:
        return nn.silu(causal_depthwise_conv1d(normed, weight.astype(normed.dtype)))

    _validate_doc_ids(x, doc_ids)
    kernel = weight.shape[1]
    result = mx.zeros_like(normed)
    for shift in range(kernel):
        if shift == 0:
            shifted = normed
        elif shift < x.shape[1]:
            shifted = mx.pad(normed[:, :-shift, :], [(0, 0), (shift, 0), (0, 0)])
        else:
            shifted = mx.zeros_like(normed)
        shifted = shifted * mx.expand_dims(_same_doc_shift_mask(doc_ids, shift, x.dtype), -1)
        result = result + shifted * weight[:, kernel - 1 - shift, :].reshape(1, 1, x.shape[-1])
    return nn.silu(result)


class EngramBranch(nn.Module):
    """Pure-MLX causal n-gram branch, not wired into NAM56R by default."""

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ngram_orders = config.orders
        self.bottleneck_dim = config.effective_bottleneck_dim
        self.gated = bool(config.gated)
        self.gate_sqrt_compress = bool(config.gate_sqrt_compress)
        self.conv_kernel = int(config.conv_kernel)
        self.eps = float(config.eps)

        self.in_proj = nn.Linear(config.hidden_size, self.bottleneck_dim, bias=False)
        self.order_mix = [nn.Linear(self.bottleneck_dim, self.bottleneck_dim, bias=False) for _ in self.ngram_orders]
        self.dropout = nn.Dropout(config.dropout)

        if self.gated:
            self.gate_key_proj = nn.Linear(self.bottleneck_dim, config.hidden_size, bias=False)
            self.value_proj = nn.Linear(self.bottleneck_dim, config.hidden_size, bias=False)
            self.value_proj.weight = mx.zeros_like(self.value_proj.weight)
        else:
            self.out_proj = nn.Linear(self.bottleneck_dim, config.hidden_size, bias=False)
            self.out_proj.weight = mx.zeros_like(self.out_proj.weight)

        if self.conv_kernel > 0:
            self.conv_weight = self._init_conv_weight(config.hidden_size, self.conv_kernel)

    @staticmethod
    def _init_conv_weight(channels: int, kernel_size: int) -> mx.array:
        scale = math.sqrt(1 / (channels * kernel_size))
        return mx.random.uniform(-scale, scale, (channels, kernel_size, 1))

    def ngram_features(self, x: mx.array, doc_ids: mx.array | None = None) -> mx.array:
        _require_rank("x", x, 3)
        _require_floating("x", x)
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"x hidden size must be {self.hidden_size}, got {x.shape[-1]}")
        if doc_ids is not None:
            _validate_doc_ids(x, doc_ids)

        z = self.in_proj(x)
        y = mx.zeros_like(z)
        for order, mix in zip(self.ngram_orders, self.order_mix, strict=True):
            y = y + mix(causal_local_average(z, order, doc_ids=doc_ids))
        return y / len(self.ngram_orders)

    def gate_values(self, x: mx.array, features: mx.array) -> mx.array:
        if not self.gated:
            raise ValueError("gate_values is only available when gated=True")
        k = self.gate_key_proj(features)
        h_norm = _rms_norm_last(x, self.eps)
        k_norm = _rms_norm_last(k, self.eps)
        logits = mx.sum(h_norm * k_norm, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        if self.gate_sqrt_compress:
            logits = mx.sign(logits) * mx.sqrt(mx.maximum(mx.abs(logits), mx.array(1e-6, dtype=logits.dtype)))
        return mx.sigmoid(logits)

    def __call__(self, x: mx.array, doc_ids: mx.array | None = None) -> mx.array:
        features = self.ngram_features(x, doc_ids=doc_ids)
        if self.gated:
            out = self.gate_values(x, features) * self.value_proj(features)
        else:
            out = self.out_proj(features)

        if self.conv_kernel > 0:
            out = causal_depthwise_silu_conv1d(
                out,
                self.conv_weight.astype(out.dtype),
                eps=self.eps,
                doc_ids=doc_ids,
            )
        return self.dropout(out)


CppMegaEngramBranch = EngramBranch

__all__ = [
    "CppMegaEngramBranch",
    "EngramBranch",
    "EngramConfig",
    "causal_depthwise_silu_conv1d",
    "causal_local_average",
    "parse_ngram_orders",
]
