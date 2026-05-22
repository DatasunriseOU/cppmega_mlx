"""V7-E02: capacity-bounded drop / reroute accounting."""

from __future__ import annotations

import pytest

from cppmega_v4.nn.moe_capacity import compute_drop_reroute_stats


def test_v7_e02_no_overflow_when_choices_balanced():
    # 8 tokens × top_k=2 = 16 slots; cap=ceil(16/4)=4 per expert.
    # Balanced round-robin choices → exactly 4 per expert, no overflow.
    top = [[i % 4, (i + 2) % 4] for i in range(8)]
    r = compute_drop_reroute_stats(top, num_experts=4, capacity_factor=1.0)
    assert r["dropped_token_ratio"] == 0.0
    assert r["rerouted_token_ratio"] == 0.0
    assert r["overflow_ratio"] == 0.0


def test_v7_e02_drop_only_when_reroute_false_and_capacity_half():
    # 8 tokens × top_k=2 = 16 slots, num_experts=4 → cap_per_expert
    # = ceil(0.5 * 16 / 4) = 2. All tokens pick expert 0 + 1 → these
    # two experts can absorb 2+2=4 slots; the remaining 12 are dropped.
    top = [[0, 1] for _ in range(8)]
    r = compute_drop_reroute_stats(top, num_experts=4,
                                    capacity_factor=0.5, reroute=False)
    assert r["capacity_per_expert"] == 2
    assert r["dropped_token_ratio"] == 12 / 16
    assert r["rerouted_token_ratio"] == 0.0


def test_v7_e02_reroute_uses_free_experts():
    # Same shape; with reroute=True experts 2 and 3 still have full
    # capacity each. 16 total slots, 4 absorbed by 0/1, 4 rerouted
    # to 2/3 (their caps), 8 still dropped.
    top = [[0, 1] for _ in range(8)]
    r = compute_drop_reroute_stats(top, num_experts=4,
                                    capacity_factor=0.5, reroute=True)
    assert r["capacity_per_expert"] == 2
    assert r["rerouted_token_ratio"] == 4 / 16
    assert r["dropped_token_ratio"] == 8 / 16
    assert (r["dropped_token_ratio"] + r["rerouted_token_ratio"]
            == pytest.approx(r["overflow_ratio"], abs=1e-9))


def test_v7_e02_ratios_in_unit_interval():
    top = [[i % 4, (i + 1) % 4] for i in range(20)]
    r = compute_drop_reroute_stats(top, num_experts=4,
                                    capacity_factor=0.75)
    for k in ("dropped_token_ratio", "rerouted_token_ratio",
              "overflow_ratio"):
        assert 0.0 <= r[k] <= 1.0


def test_v7_e02_empty_input_no_crash():
    r = compute_drop_reroute_stats([], num_experts=4,
                                    capacity_factor=1.0)
    assert r["dropped_token_ratio"] == 0.0
    assert r["total_slots"] == 0
