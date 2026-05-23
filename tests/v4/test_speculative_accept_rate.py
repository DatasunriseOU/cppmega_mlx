"""V7-H39: speculative-decode smoke wiring through gen.run.

When speculative_k > 0, gen.run runs the cppmega_mlx.inference.
speculative_decode.speculative_acceptance helper against synthetic
identical draft+target logits and exposes the accept_rate in the
GenRunResult.speculative block. Identical distributions must accept
> 50% of draft tokens — the standard Leviathan sanity gate.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.gen_run_method import GenRunParams, gen_run


def test_v7_h39_speculative_off_by_default():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=4,
    ))
    assert out.speculative is None


def test_v7_h39_speculative_smoke_accept_rate_above_half():
    """K=16 identical draft+target → accept_rate > 0.5 sanity."""
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=4,
        seed=5, vocab_size=64, speculative_k=16,
    ))
    sp = out.speculative
    assert sp is not None
    assert sp["k"] == 16
    assert len(sp["draft_tokens"]) == 16
    assert 0 <= sp["accepted"] <= 16
    assert sp["accept_rate"] > 0.5, (
        f"speculative accept_rate {sp['accept_rate']} <= 0.5 for "
        f"identical draft+target — wiring is broken")


def test_v7_h39_speculative_deterministic_for_same_seed():
    a = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
        seed=7, vocab_size=64, speculative_k=8,
    ))
    b = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
        seed=7, vocab_size=64, speculative_k=8,
    ))
    assert a.speculative == b.speculative


def test_v7_h39_dispatcher_route_returns_speculative_block():
    resp = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "gen.run",
        "params": {
            "prompt_tokens": [1], "eos_token_id": 99_999,
            "max_new_tokens": 2, "vocab_size": 32,
            "speculative_k": 8,
        },
    })
    assert resp.error is None, resp.error
    assert resp.result["speculative"] is not None
    assert resp.result["speculative"]["k"] == 8
