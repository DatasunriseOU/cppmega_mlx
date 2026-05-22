"""V7-F06: per-token streaming generation event tests."""

from __future__ import annotations

from cppmega_v4.runtime.generate_stream import (
    collect_stream, stream_generate,
)


def test_v7_f06_on_token_fires_per_step():
    seq = [10, 11, 99]
    i = {"k": 0}

    def step_fn(_last):
        v = seq[i["k"]]
        i["k"] += 1
        return v

    events: list[dict] = []
    out, reason = stream_generate(
        initial_tokens=[1, 2], step_fn=step_fn,
        eos_token_id=99, max_new_tokens=10,
        on_token=events.append,
    )
    assert reason == "eos"
    assert out == [1, 2, 10, 11, 99]
    assert len(events) == 3
    assert events[0] == {"step": 0, "token_id": 10,
                          "finish_reason": None}
    assert events[1] == {"step": 1, "token_id": 11,
                          "finish_reason": None}
    assert events[2] == {"step": 2, "token_id": 99,
                          "finish_reason": "eos"}


def test_v7_f06_length_finish_no_eos_event():
    def step_fn(last):
        return last + 1

    out, reason, events = collect_stream(
        initial_tokens=[0], step_fn=step_fn,
        eos_token_id=999, max_new_tokens=4,
    )
    assert reason == "length"
    assert len(out) == 5  # prompt + 4 generated
    assert len(events) == 4
    assert all(e["finish_reason"] is None for e in events)


def test_v7_f06_collect_stream_returns_tokens_reason_events():
    out, reason, events = collect_stream(
        initial_tokens=[5], step_fn=lambda _last: 99,
        eos_token_id=99, max_new_tokens=5,
    )
    assert out == [5, 99]
    assert reason == "eos"
    assert len(events) == 1
    assert events[0]["step"] == 0


def test_v7_f06_on_token_none_safe():
    """No on_token callback → no crash, still works."""
    out, reason = stream_generate(
        initial_tokens=[0], step_fn=lambda _l: 99,
        eos_token_id=99, max_new_tokens=2,
    )
    assert reason == "eos"
    assert out == [0, 99]
