"""V7-H41: long-form sustained generation does not leak memory.

Pins that running gen.run repeatedly with max_new_tokens=1024 doesn't
balloon Metal allocator peak across invocations. Each gen.run builds
+ tears down its own KVCache, samplers, and event-bus subscribers;
if any of those leak references the peak memory after 5 runs would
be N× larger than after the first.

Acceptance: peak_memory after run 5 < 1.2× peak after run 1.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from cppmega_v4.jsonrpc.gen_run_method import GenRunParams, gen_run


def _has_metal_peak() -> bool:
    return hasattr(mx, "get_peak_memory") or hasattr(mx, "metal")


def _peak_bytes() -> int:
    """Read Metal allocator peak in a backward-compatible way."""
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return int(mx.metal.get_peak_memory())
    return 0


def _reset_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


@pytest.fixture(autouse=True)
def _skip_without_metal():
    if not _has_metal_peak():
        pytest.skip("no Metal allocator peak — cannot bound memory")


def _one_long_gen() -> None:
    gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999,
        max_new_tokens=1024, strategy="greedy", seed=11,
        kv_cache_layers=2, kv_cache_head_dim=16,
    ))


def test_v7_h41_repeated_long_gen_memory_bounded():
    """5 × max_new_tokens=1024: peak after 5 < 1.2× peak after 1."""
    # Warm up so the first measurement isn't dominated by import-time
    # allocator initialisation.
    _one_long_gen()
    _reset_peak()
    _one_long_gen()
    peak_after_1 = _peak_bytes()
    for _ in range(4):
        _one_long_gen()
    peak_after_5 = _peak_bytes()
    if peak_after_1 == 0:
        # Some backends report 0 even with metal present — skip cleanly.
        pytest.skip("Metal peak reported 0 — cannot bound delta")
    ratio = peak_after_5 / peak_after_1
    assert ratio < 1.2, (
        f"sustained gen leak: peak_after_1={peak_after_1} bytes, "
        f"peak_after_5={peak_after_5} bytes, ratio={ratio:.3f}× "
        f"(limit 1.2×)")
