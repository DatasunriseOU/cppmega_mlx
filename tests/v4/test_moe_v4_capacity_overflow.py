"""V7-E01: V4MoE top_k dispatch + capacity-factor overflow accounting.

The capacity-bound math itself lives in moe_capacity.py (V7-E02).
This test composes the V4MoE router output with that helper to
prove the closed-loop claim: forced overflow → dropped_token_ratio
> 0.05 AND per-expert assigned count never exceeds cap.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig
from cppmega_v4.nn.moe_capacity import compute_drop_reroute_stats


def _top_indices_from_router(router) -> list[list[int]]:
    """V4MoE returns top_indices shape (B, S, top_k). Flatten to
    (B*S, top_k) python lists for the capacity helper."""
    arr = router.top_indices
    return [
        [int(x) for x in row]
        for row in arr.reshape(-1, arr.shape[-1]).tolist()
    ]


def test_v7_e01_capacity_half_forces_dropped_ratio_over_5pct():
    cfg = V4MoEConfig(
        d_model=64, num_experts=4, top_k=2,
        expert_hidden_size=64,
    )
    moe = V4MoE(cfg)
    # Large batch so capacity bound bites.
    x = mx.random.normal(shape=(1, 128, cfg.d_model),
                          key=mx.random.key(0))
    out = moe(x)
    top = _top_indices_from_router(out.router)
    stats = compute_drop_reroute_stats(
        top, num_experts=cfg.num_experts, capacity_factor=0.5,
        reroute=False)
    assert stats["dropped_token_ratio"] > 0.05, (
        f"forced overflow did not drop tokens: {stats}"
    )
    # Per-expert assigned count never exceeds capacity.
    used = [0] * cfg.num_experts
    cap = stats["capacity_per_expert"]
    for choices in top:
        for e in choices:
            if used[e] < cap:
                used[e] += 1
    for e in used:
        assert e <= cap


def test_v7_e01_high_capacity_keeps_dropped_ratio_zero():
    cfg = V4MoEConfig(
        d_model=64, num_experts=4, top_k=2,
        expert_hidden_size=64,
    )
    moe = V4MoE(cfg)
    # Small batch — even adversarial routing fits.
    x = mx.random.normal(shape=(1, 4, cfg.d_model),
                          key=mx.random.key(7))
    out = moe(x)
    top = _top_indices_from_router(out.router)
    stats = compute_drop_reroute_stats(
        top, num_experts=cfg.num_experts, capacity_factor=2.0,
        reroute=False)
    assert stats["dropped_token_ratio"] == 0.0
