"""V7-E06 AC#5: gen.run records routed_expert_ids per step for the
UI inference panel + AC#3 dropped_token_ratio=0 at inference."""

from __future__ import annotations

from cppmega_v4.jsonrpc.gen_run_method import (
    GenRunParams, gen_run,
)


def test_v7_e06_gen_run_records_routed_expert_ids():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0,
        max_new_tokens=8, strategy="greedy",
        moe_num_experts=8, moe_top_k=2,
    ))
    assert out.moe is not None
    assert out.moe["num_experts"] == 8
    assert out.moe["top_k"] == 2
    routed = out.moe["routed_expert_ids"]
    assert len(routed) == len(out.tokens) - 1 or len(routed) == 8
    for ids in routed:
        assert len(ids) == 2
        for e in ids:
            assert 0 <= e < 8
        assert ids == sorted(ids)  # canonicalised


def test_v7_e06_gen_run_drop_ratio_zero_at_inference():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0,
        max_new_tokens=4, moe_num_experts=4, moe_top_k=2,
    ))
    assert out.moe is not None
    assert out.moe["dropped_token_ratio"] == 0.0


def test_v7_e06_gen_run_moe_none_when_num_experts_zero():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0,
        max_new_tokens=4, moe_num_experts=0,
    ))
    assert out.moe is None


def test_v7_e06_gen_run_moe_top_k_clamped_to_num_experts():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0,
        max_new_tokens=2,
        moe_num_experts=3, moe_top_k=8,  # > num_experts
    ))
    assert out.moe is not None
    assert out.moe["top_k"] == 3


def test_v7_e06_gen_run_moe_trace_is_deterministic_for_same_seed():
    a = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0, max_new_tokens=6,
        seed=42, moe_num_experts=8, moe_top_k=2,
    ))
    b = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0, max_new_tokens=6,
        seed=42, moe_num_experts=8, moe_top_k=2,
    ))
    assert a.moe["routed_expert_ids"] == b.moe["routed_expert_ids"]


def test_v7_e06_gen_run_dispatcher_route_returns_moe_block():
    from cppmega_v4.jsonrpc.dispatcher import dispatch
    resp = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "gen.run",
        "params": {
            "prompt_tokens": [1], "eos_token_id": 0,
            "max_new_tokens": 4, "moe_num_experts": 4, "moe_top_k": 2,
        },
    })
    assert resp.error is None, resp.error
    assert resp.result["moe"] is not None
    assert resp.result["moe"]["num_experts"] == 4
    assert len(resp.result["moe"]["routed_expert_ids"]) > 0
