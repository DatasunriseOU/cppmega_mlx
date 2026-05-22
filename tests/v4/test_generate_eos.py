"""V7-F04: generate_until_eos halts on EOS or hits max_new_tokens."""

from __future__ import annotations

import pytest

from cppmega_v4.runtime.generate import generate_until_eos


def test_v7_f04_simple_eos_halt():
    counter = {"i": 0}
    sequence = [10, 11, 99]  # third call returns EOS

    def step_fn(last: int) -> int:
        v = sequence[counter["i"]]
        counter["i"] += 1
        return v

    out, reason = generate_until_eos(
        initial_tokens=[1, 2],
        step_fn=step_fn,
        eos_token_id=99,
        max_new_tokens=10,
    )
    assert reason == "eos"
    assert out == [1, 2, 10, 11, 99]


def test_v7_f04_length_limit_when_no_eos():
    """No EOS emitted → finish_reason='length' after max_new_tokens."""
    def step_fn(last: int) -> int:
        return last + 1

    out, reason = generate_until_eos(
        initial_tokens=[0],
        step_fn=step_fn,
        eos_token_id=999,
        max_new_tokens=5,
    )
    assert reason == "length"
    assert len(out) == 6  # prompt + 5 generated


def test_v7_f04_zero_max_new_tokens():
    out, reason = generate_until_eos(
        initial_tokens=[7, 8],
        step_fn=lambda _: 0,
        eos_token_id=99,
        max_new_tokens=0,
    )
    assert reason == "length"
    assert out == [7, 8]


def test_v7_f04_negative_max_tokens_rejected():
    with pytest.raises(ValueError):
        generate_until_eos(
            initial_tokens=[1],
            step_fn=lambda _: 0,
            eos_token_id=99,
            max_new_tokens=-1,
        )
