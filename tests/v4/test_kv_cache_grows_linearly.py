"""V7-H40: gen.run KV cache grows linearly with token count.

Pins the contract that with kv_cache_layers > 0:
  * each generated token triggers exactly one growth event,
  * per-layer length equals the number of tokens generated,
  * total_bytes scales as growth_events * num_layers * head_dim * fp32_size
    (the synthetic kv_cache_state appends one (1, 1, head_dim) k+v per
    layer per token at fp32, so 2 * 4 bytes per element).
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.gen_run_method import GenRunParams, gen_run


def _gen(layers: int, head_dim: int, max_new_tokens: int):
    # eos_token_id outside the default vocab so 'length' is always the
    # finish reason — keeps the linear-growth contract independent of
    # the sampler's per-seed EOS hit rate.
    return gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999,
        max_new_tokens=max_new_tokens,
        strategy="greedy", seed=7,
        kv_cache_layers=layers, kv_cache_head_dim=head_dim,
    ))


def test_v7_h40_kv_cache_lengths_equal_growth_events_per_layer():
    out = _gen(layers=4, head_dim=16, max_new_tokens=32)
    assert out.kv_cache is not None
    kv = out.kv_cache
    assert kv["num_layers"] == 4
    assert kv["head_dim"] == 16
    # One growth event per generated token; per-layer length matches.
    assert kv["growth_events"] == 32
    assert kv["lengths_per_layer"] == [32, 32, 32, 32]


def test_v7_h40_total_bytes_matches_growth_arithmetic():
    out = _gen(layers=4, head_dim=16, max_new_tokens=32)
    kv = out.kv_cache
    # KVCache.append stores fp32 k + fp32 v per layer per row,
    # (1, 1, head_dim) shape → head_dim elements × 4 bytes × 2 (k+v).
    fp32_size = 4
    expected = (kv["growth_events"] * kv["num_layers"]
                * kv["head_dim"] * fp32_size * 2)
    assert kv["total_bytes"] == expected, (
        f"total_bytes={kv['total_bytes']} expected={expected}")


def test_v7_h40_kv_cache_disabled_when_layers_zero():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=0,
        max_new_tokens=4, kv_cache_layers=0,
    ))
    assert out.kv_cache is None


def test_v7_h40_growth_scales_linearly_in_steps():
    """Doubling max_new_tokens doubles growth_events + total_bytes."""
    a = _gen(layers=2, head_dim=8, max_new_tokens=10)
    b = _gen(layers=2, head_dim=8, max_new_tokens=20)
    assert b.kv_cache["growth_events"] == 2 * a.kv_cache["growth_events"]
    assert b.kv_cache["total_bytes"] == 2 * a.kv_cache["total_bytes"]
