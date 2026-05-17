"""DeepSeek-V4-flavored MoE: aux-loss-free balancing + sqrt(softplus) scoring.

This is a side-by-side plugin module. It does not import ``ReferenceMoE`` and
does not modify any file under ``cppmega_mlx/``. It reuses ``FeedForwardExpert``
from ``cppmega_mlx.nn.moe`` only as a leaf module so the per-expert MLP shape
stays consistent with the existing trainer.

Scoring options (per V3/V4 papers):
    - ``softmax``      — legacy V2/V3 default; backward-compatible.
    - ``sigmoid``      — V3 per-expert affinity (no normalization across experts).
    - ``sqrtsoftplus`` — V4-Pro default: ``sqrt(softplus(logits))``.

Aux-loss-free balancing (V3, paper 2408.15664):
    A learnable per-expert bias ``b[i]`` is added to the routing scores **only
    for top-k selection** — the weights of selected experts use the raw scores
    (without bias) divided by their sum. The bias is updated post-step:
    ``b[i] += rate * sign(load[i] - mean_load)``.

Node-limited routing (V3):
    When ``node_limited_routing=N`` is set, the routing first picks the top-N
    expert *groups*, then top-k within those groups. We expose it as a config
    knob; setting it to ``None`` (default) disables grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.moe import (
    ActivationName,
    FeedForwardExpert,
    MoEConfig,
    MoEOutput,
    RouterOutput,
)

V4Scoring = Literal["softmax", "sigmoid", "sqrtsoftplus"]


@dataclass(frozen=True)
class V4MoEConfig:
    """V4 MoE config. Mirrors ``MoEConfig`` fields plus V4-specific knobs."""

    d_model: int
    num_experts: int = 16
    top_k: int = 4
    expert_hidden_size: int = 256
    shared_expert_hidden_size: int | None = None
    activation: ActivationName = "swiglu"
    normalize_top_k: bool = True
    router_dtype: str | None = "fp32"
    bias: bool = False
    scoring: V4Scoring = "softmax"
    aux_loss_free: bool = False
    bias_update_rate: float = 1e-3
    node_limited_routing: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must satisfy 0 < top_k <= num_experts")
        if self.expert_hidden_size <= 0:
            raise ValueError("expert_hidden_size must be positive")
        if self.shared_expert_hidden_size is not None and self.shared_expert_hidden_size <= 0:
            raise ValueError("shared_expert_hidden_size must be positive when set")
        if self.scoring not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError(f"unsupported scoring {self.scoring!r}")
        if self.router_dtype not in (None, "fp32"):
            raise ValueError("only router_dtype=None or 'fp32' is supported")
        if self.bias_update_rate < 0:
            raise ValueError("bias_update_rate must be non-negative")
        if self.node_limited_routing is not None:
            if self.node_limited_routing <= 0:
                raise ValueError("node_limited_routing must be positive when set")
            if self.num_experts % self.node_limited_routing != 0:
                raise ValueError(
                    "num_experts must be divisible by node_limited_routing groups"
                )
            if self.top_k > self.node_limited_routing * (
                self.num_experts // self.node_limited_routing
            ):
                raise ValueError("top_k exceeds capacity of node_limited_routing groups")

    def as_moe_config(self) -> MoEConfig:
        """Project to the legacy MoEConfig (for parity tests against ReferenceMoE)."""
        return MoEConfig(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_size=self.expert_hidden_size,
            shared_expert_hidden_size=self.shared_expert_hidden_size,
            activation=self.activation,
            normalize_top_k=self.normalize_top_k,
            router_dtype=self.router_dtype,
            bias=self.bias,
        )


def _score_logits(logits: mx.array, scoring: V4Scoring) -> mx.array:
    if scoring == "softmax":
        return mx.softmax(logits, axis=-1)
    if scoring == "sigmoid":
        return mx.sigmoid(logits)
    # sqrtsoftplus: sqrt(softplus(x)) — per V4-Pro config (scoring_func=sqrtsoftplus)
    return mx.sqrt(nn.softplus(logits))


class V4MoE(nn.Module):
    """V4 MoE block: scoring + aux-loss-free bias + optional node-limited routing.

    Forward returns a ``MoEOutput`` whose ``router.aux_loss`` is zero when
    ``aux_loss_free=True`` (no auxiliary loss should be added to the training
    objective in that mode — the expert bias does the balancing).
    """

    def __init__(self, config: V4MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=config.bias)
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
        # ``expert_bias`` is a non-trainable parameter mutated by
        # ``update_bias_after_step``. We freeze it so the optimizer ignores it.
        if config.aux_loss_free:
            self.expert_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
            self.freeze(keys=["expert_bias"])

    def _router_scores(self, flat_x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Return (logits, raw_scores, biased_scores) — gate computed once."""
        logits = self.gate(flat_x)
        cast_logits = logits.astype(mx.float32) if self.config.router_dtype == "fp32" else logits
        raw_scores = _score_logits(cast_logits, self.config.scoring)
        if self.config.aux_loss_free:
            biased_scores = raw_scores + self.expert_bias[None, :]
        else:
            biased_scores = raw_scores
        return logits, raw_scores, biased_scores

    def _select_top_k(self, biased_scores: mx.array) -> mx.array:
        """Top-k indices over the (optionally node-limited) biased scores."""
        cfg = self.config
        if cfg.node_limited_routing is None or cfg.node_limited_routing == cfg.num_experts:
            return mx.stop_gradient(
                mx.argpartition(-biased_scores, cfg.top_k - 1, axis=-1)[:, : cfg.top_k]
            )

        groups = cfg.node_limited_routing
        group_size = cfg.num_experts // groups
        reshaped = biased_scores.reshape(-1, groups, group_size)
        # Use max score within each group to rank groups.
        group_scores = reshaped.max(axis=-1)
        # Pick the top groups whose total capacity is at least ``top_k`` —
        # simplest policy is to take all groups whose score is in the top
        # ``ceil(top_k / group_size)`` ranks, then top-k within the masked set.
        keep_groups = max(1, (cfg.top_k + group_size - 1) // group_size)
        keep_groups = min(keep_groups, groups)
        top_group_idx = mx.stop_gradient(
            mx.argpartition(-group_scores, keep_groups - 1, axis=-1)[:, :keep_groups]
        )
        mask = mx.zeros_like(group_scores)
        ones = mx.ones_like(top_group_idx).astype(mask.dtype)
        mask = mx.put_along_axis(mask, top_group_idx, ones, axis=-1)
        mask = mx.repeat(mask[:, :, None], group_size, axis=-1).reshape(biased_scores.shape)
        masked = mx.where(
            mask.astype(mx.bool_),
            biased_scores,
            mx.full(biased_scores.shape, -mx.inf, dtype=biased_scores.dtype),
        )
        return mx.stop_gradient(
            mx.argpartition(-masked, cfg.top_k - 1, axis=-1)[:, : cfg.top_k]
        )

    def __call__(self, x: mx.array) -> MoEOutput:
        if x.ndim < 2:
            raise ValueError(f"x must have a hidden dimension, got shape {x.shape}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"x last dim must be {self.config.d_model}, got {x.shape[-1]}"
            )
        cfg = self.config
        flat_x = x.reshape(-1, x.shape[-1])
        logits, raw_scores, biased_scores = self._router_scores(flat_x)

        top_indices = self._select_top_k(biased_scores)
        # Weights use the **raw** (pre-bias) scores — the bias only affects
        # which experts are selected, per V3 paper.
        top_scores = mx.take_along_axis(raw_scores, top_indices, axis=-1)
        if cfg.normalize_top_k:
            denom = mx.maximum(top_scores.sum(axis=-1, keepdims=True), mx.array(1e-9))
            top_weights = top_scores / denom
        else:
            top_weights = top_scores
        top_weights = top_weights.astype(x.dtype)

        flat_indices = top_indices.reshape(-1, cfg.top_k)
        flat_weights = top_weights.reshape(-1, cfg.top_k)
        routed = mx.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(flat_x)
            mask = mx.equal(flat_indices, mx.array(expert_id, dtype=flat_indices.dtype))
            weight = mx.sum(mx.where(mask, flat_weights, mx.zeros_like(flat_weights)), axis=-1)
            routed = routed + expert_out * weight[:, None]

        routed = routed.reshape(x.shape)
        shared = self.shared_expert(x) if self.shared_expert is not None else None
        output = routed + shared if shared is not None else routed

        selected = mx.equal(flat_indices[..., None], mx.arange(cfg.num_experts))
        selected = selected.astype(mx.float32)
        load = selected.mean(axis=(0, 1))
        importance = raw_scores.astype(mx.float32).mean(axis=0)
        # Aux loss is zero when aux_loss_free=True (bias-update does balancing).
        if cfg.aux_loss_free:
            aux_loss = mx.array(0.0, dtype=mx.float32)
        else:
            aux_loss = cfg.num_experts * mx.sum(load * importance)

        prefix = x.shape[:-1]
        router = RouterOutput(
            logits=logits.reshape(*prefix, cfg.num_experts),
            probabilities=raw_scores.reshape(*prefix, cfg.num_experts),
            top_indices=top_indices.reshape(*prefix, cfg.top_k),
            top_weights=top_weights.reshape(*prefix, cfg.top_k),
            aux_loss=aux_loss,
            load=load,
            importance=importance,
        )
        return MoEOutput(
            output=output,
            router=router,
            routed_output=routed,
            shared_output=shared,
        )

    def update_bias_after_step(self, router_load: mx.array) -> None:
        """Apply the aux-loss-free bias update from V3 (paper 2408.15664).

        ``router_load[i]`` is the fraction of tokens routed to expert ``i``
        in the last training step. The update is ``b[i] += rate * sign(load[i] - mean)``.

        Call this **once per optimizer step**, after the forward+backward pass,
        before the next step starts. No-op when ``aux_loss_free=False``.
        """
        if not self.config.aux_loss_free:
            return
        if router_load.ndim != 1 or router_load.shape[0] != self.config.num_experts:
            raise ValueError(
                f"router_load must be shape ({self.config.num_experts},), "
                f"got {router_load.shape}"
            )
        load = router_load.astype(mx.float32)
        mean_load = load.mean()
        # Underloaded experts (load < mean) need a HIGHER bias so they get
        # selected more often → ``sign(mean - load) = +1``. Overloaded → -1.
        delta = self.config.bias_update_rate * mx.sign(mean_load - load)
        self.expert_bias = self.expert_bias + delta


__all__ = ["V4MoE", "V4MoEConfig", "V4Scoring"]
