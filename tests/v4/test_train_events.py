"""V7-H05: per-step train event generator from extras."""

from __future__ import annotations

from cppmega_v4.runtime.train_events import train_events_from_extras


def _extras() -> dict:
    return {
        "losses": [5.0, 4.5, 4.1],
        "lr_trajectory": [1e-4, 3e-4, 1e-3],
        "elapsed_ms": 30.0,
        "memory_peak_bytes": 4 * 1024 * 1024,
    }


def test_v7_h05_events_per_step_with_documented_keys():
    events = list(train_events_from_extras(_extras(),
                                             batch=1, seq=16))
    assert len(events) == 3
    keys = {"step", "loss", "lr", "grad_norm", "mem_mb",
            "throughput_tok_s", "ts"}
    for e in events:
        assert keys.issubset(e.keys()), f"missing keys in {e}"
    assert [e["step"] for e in events] == [0, 1, 2]
    assert [e["loss"] for e in events] == [5.0, 4.5, 4.1]
    assert [e["lr"] for e in events] == [1e-4, 3e-4, 1e-3]


def test_v7_h05_throughput_positive_and_mem_present():
    events = list(train_events_from_extras(_extras(),
                                             batch=1, seq=16))
    for e in events:
        assert e["throughput_tok_s"] > 0
        assert e["mem_mb"] == 4.0
        assert e["ts"] > 0


def test_v7_h05_handles_missing_lr_trajectory():
    e = {"losses": [1.0, 2.0]}
    events = list(train_events_from_extras(e, batch=1, seq=8))
    assert len(events) == 2
    for ev in events:
        assert ev["lr"] is None
        assert ev["mem_mb"] is None


def test_v7_h05_empty_losses_yields_no_events():
    events = list(train_events_from_extras({"losses": []}))
    assert events == []
