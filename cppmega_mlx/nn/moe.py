"""Correctness-first MLX MoE reference blocks for local smoke training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import mlx.core as mx
import mlx.nn as nn


ActivationName = Literal["gelu", "relu2", "swiglu"]


@dataclass(frozen=True)
class MoEConfig:
    """Small MLX config mirroring NAM56R MoE defaults where practical."""

    d_model: int
    num_experts: int = 16
    top_k: int = 4
    expert_hidden_size: int = 896
    shared_expert_hidden_size: int | None = 1024
    activation: ActivationName = "gelu"
    normalize_top_k: bool = True
    router_dtype: str | None = "fp32"
    bias: bool = False

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("num_experts", self.num_experts)
        _require_positive("top_k", self.top_k)
        _require_positive("expert_hidden_size", self.expert_hidden_size)
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if self.shared_expert_hidden_size is not None:
            _require_positive("shared_expert_hidden_size", self.shared_expert_hidden_size)
        if self.activation not in {"gelu", "relu2", "swiglu"}:
            raise ValueError(f"unsupported MoE activation={self.activation!r}")
        if self.router_dtype not in {None, "fp32"}:
            raise ValueError("only router_dtype=None or 'fp32' is supported")


class RouterOutput(NamedTuple):
    logits: mx.array
    probabilities: mx.array
    top_indices: mx.array
    top_weights: mx.array
    aux_loss: mx.array
    load: mx.array
    importance: mx.array


class MoEOutput(NamedTuple):
    output: mx.array
    router: RouterOutput
    routed_output: mx.array
    shared_output: mx.array | None


class FeedForwardExpert(nn.Module):
    """Tiny routed/shared expert MLP.

    The full CUDA lane uses Megatron/TE grouped GEMM.  This local MLX block is a
    readable reference path: every selected expert is a normal MLX module.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        *,
        activation: ActivationName = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=bias)
        self.up_proj = (
            nn.Linear(d_model, hidden_size, bias=bias)
            if activation == "swiglu"
            else None
        )
        self.down_proj = nn.Linear(hidden_size, d_model, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_proj(x)
        if self.activation == "swiglu":
            assert self.up_proj is not None
            h = nn.silu(h) * self.up_proj(x)
        elif self.activation == "relu2":
            h = mx.square(mx.maximum(h, mx.array(0.0, dtype=h.dtype)))
        else:
            h = nn.gelu_approx(h)
        return self.down_proj(h)


class TopKRouter(nn.Module):
    """Softmax top-k router with lightweight auxiliary metrics."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 16,
        top_k: int = 4,
        *,
        normalize_top_k: bool = True,
        router_dtype: str | None = "fp32",
        bias: bool = False,
    ):
        super().__init__()
        _require_positive("d_model", d_model)
        _require_positive("num_experts", num_experts)
        _require_positive("top_k", top_k)
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        if router_dtype not in {None, "fp32"}:
            raise ValueError("only router_dtype=None or 'fp32' is supported")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_top_k = normalize_top_k
        self.router_dtype = router_dtype
        self.gate = nn.Linear(d_model, num_experts, bias=bias)

    def __call__(self, x: mx.array) -> RouterOutput:
        if x.ndim < 2:
            raise ValueError(f"x must have a hidden dimension, got shape {x.shape}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dim must be {self.d_model}, got {x.shape[-1]}")

        flat_x = x.reshape(-1, x.shape[-1])
        logits = self.gate(flat_x)
        router_logits = logits.astype(mx.float32) if self.router_dtype == "fp32" else logits
        probabilities = mx.softmax(router_logits, axis=-1)

        top_indices = mx.stop_gradient(
            mx.argpartition(-probabilities, self.top_k - 1, axis=-1)[:, : self.top_k]
        )
        top_probs = mx.take_along_axis(probabilities, top_indices, axis=-1)
        top_weights = top_probs
        if self.normalize_top_k:
            denom = mx.maximum(top_probs.sum(axis=-1, keepdims=True), mx.array(1e-9))
            top_weights = top_probs / denom
        top_weights = top_weights.astype(x.dtype)

        selected = mx.equal(top_indices[..., None], mx.arange(self.num_experts))
        selected = selected.astype(mx.float32)
        load = selected.mean(axis=(0, 1))
        importance = probabilities.astype(mx.float32).mean(axis=0)
        aux_loss = self.num_experts * mx.sum(load * importance)

        prefix = x.shape[:-1]
        return RouterOutput(
            logits=logits.reshape(*prefix, self.num_experts),
            probabilities=probabilities.reshape(*prefix, self.num_experts),
            top_indices=top_indices.reshape(*prefix, self.top_k),
            top_weights=top_weights.reshape(*prefix, self.top_k),
            aux_loss=aux_loss,
            load=load,
            importance=importance,
        )


class ReferenceMoE(nn.Module):
    """Dense per-expert MLX MoE reference suitable for smoke tests."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = TopKRouter(
            config.d_model,
            config.num_experts,
            config.top_k,
            normalize_top_k=config.normalize_top_k,
            router_dtype=config.router_dtype,
            bias=config.bias,
        )
        self.experts = [
            FeedForwardExpert(
                config.d_model,
                config.expert_hidden_size,
                activation=config.activation,
                bias=config.bias,
            )
            for _ in range(config.num_experts)
        ]
        self.shared_expert = (
            FeedForwardExpert(
                config.d_model,
                config.shared_expert_hidden_size,
                activation=config.activation,
                bias=config.bias,
            )
            if config.shared_expert_hidden_size is not None
            else None
        )

    def __call__(self, x: mx.array) -> MoEOutput:
        if x.ndim < 2:
            raise ValueError(f"x must have a hidden dimension, got shape {x.shape}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"x last dim must be {self.config.d_model}, got {x.shape[-1]}"
            )

        router = self.router(x)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = router.top_indices.reshape(-1, self.config.top_k)
        flat_weights = router.top_weights.reshape(-1, self.config.top_k)

        routed = mx.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(flat_x)
            mask = mx.equal(flat_indices, mx.array(expert_id, dtype=flat_indices.dtype))
            weight = mx.sum(mx.where(mask, flat_weights, mx.zeros_like(flat_weights)), axis=-1)
            routed = routed + expert_out * weight[:, None]

        routed = routed.reshape(x.shape)
        shared = self.shared_expert(x) if self.shared_expert is not None else None
        output = routed + shared if shared is not None else routed
        return MoEOutput(
            output=output,
            router=router,
            routed_output=routed,
            shared_output=shared,
        )


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "ActivationName",
    "FeedForwardExpert",
    "MoEConfig",
    "MoEOutput",
    "ReferenceMoE",
    "RouterOutput",
    "TopKRouter",
]
