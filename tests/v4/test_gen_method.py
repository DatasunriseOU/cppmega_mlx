"""V7-F01: gen.run RPC composes F02/F03/F04/F06."""

from __future__ import annotations

from cppmega_v4.jsonrpc.gen_method import GenParams, gen_run


def test_v7_f01_greedy_strategy_runs_to_length():
    r = gen_run(GenParams(prompt_tokens=[0], eos_token_id=999,
                            max_new_tokens=4, strategy="greedy"))
    # Greedy counter never hits 999 → finish=length, 4 generated.
    assert r.finish_reason == "length"
    assert len(r.tokens) == 5  # prompt + 4
    assert len(r.events) == 4


def test_v7_f01_top_k_strategy_returns_valid_tokens():
    r = gen_run(GenParams(prompt_tokens=[0], eos_token_id=999,
                            max_new_tokens=3, strategy="top_k",
                            top_k=5, seed=42))
    assert r.finish_reason == "length"
    assert len(r.events) == 3
    for tok in r.tokens[1:]:
        assert 0 <= tok < 32


def test_v7_f01_top_p_strategy_returns_valid_tokens():
    r = gen_run(GenParams(prompt_tokens=[0], eos_token_id=999,
                            max_new_tokens=3, strategy="top_p",
                            top_p=0.9, seed=7))
    assert r.finish_reason == "length"
    assert len(r.events) == 3


def test_v7_f01_events_carry_step_token_finish():
    r = gen_run(GenParams(prompt_tokens=[0], eos_token_id=999,
                            max_new_tokens=2, strategy="greedy"))
    assert r.events[0]["step"] == 0
    assert r.events[1]["step"] == 1
    for ev in r.events:
        assert "token_id" in ev
        assert "finish_reason" in ev


def test_v7_f01_eos_halt_when_prompt_already_at_eos_minus_one():
    """Greedy from 0 with eos=4 → token sequence 1,2,3,4 halts."""
    r = gen_run(GenParams(prompt_tokens=[0], eos_token_id=4,
                            max_new_tokens=10, strategy="greedy"))
    assert r.finish_reason == "eos"
    assert r.tokens == [0, 1, 2, 3, 4]
