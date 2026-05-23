"""V7-F01: gen.run RPC round-trip across the four sampler strategies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def _call(client, params: dict) -> dict:
    payload = {"jsonrpc": "2.0", "id": "g", "method": "gen.run",
               "params": params}
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body
    return body["result"]


@pytest.mark.parametrize("strategy", ["greedy", "temperature", "top_k", "top_p"])
def test_gen_run_strategy_terminates_at_max_or_eos(client, strategy):
    res = _call(client, {
        "prompt_tokens": [1, 2],
        "eos_token_id": 31,
        "max_new_tokens": 8,
        "strategy": strategy,
        "temperature": 1.0,
        "top_k": 4,
        "top_p": 0.9,
        "seed": 42,
        "vocab_size": 32,
    })
    assert res["strategy"] == strategy
    assert res["finish_reason"] in ("eos", "length")
    # prompt was 2 tokens; tokens grew up to prompt + max_new.
    assert len(res["tokens"]) >= 2
    assert len(res["tokens"]) <= 2 + 8
    # Each event has step + token.
    for e in res["events"]:
        assert "step" in e and "token" in e


def test_gen_run_greedy_is_deterministic_at_same_seed(client):
    a = _call(client, {"prompt_tokens": [5], "eos_token_id": 99,
                        "max_new_tokens": 4, "strategy": "greedy",
                        "seed": 7})
    b = _call(client, {"prompt_tokens": [5], "eos_token_id": 99,
                        "max_new_tokens": 4, "strategy": "greedy",
                        "seed": 7})
    assert a["tokens"] == b["tokens"]


def test_gen_run_in_registry(client):
    methods = set(client.get("/schema/methods").json()["methods"])
    assert "gen.run" in methods


def test_gen_run_kv_cache_grows_per_step(client):
    """V7-F02 honest closure: when kv_cache_layers>0, gen.run wires a
    KVCache and reports its growth in the result."""
    res = _call(client, {
        "prompt_tokens": [0],
        "eos_token_id": -1,            # force max_new_tokens path
        "max_new_tokens": 5,
        "strategy": "greedy",
        "kv_cache_layers": 4,
        "kv_cache_head_dim": 16,
    })
    kv = res["kv_cache"]
    assert kv is not None
    assert kv["num_layers"] == 4
    assert kv["head_dim"] == 16
    assert kv["growth_events"] == 5
    assert kv["total_bytes"] > 0
    # Each layer absorbed one append per step.
    assert kv["lengths_per_layer"] == [5, 5, 5, 5]


def test_gen_run_kv_cache_disabled_by_default(client):
    res = _call(client, {
        "prompt_tokens": [0],
        "max_new_tokens": 2,
        "strategy": "greedy",
    })
    assert res["kv_cache"] is None


def test_gen_run_rejects_oversized_max_new_tokens(client):
    payload = {"jsonrpc": "2.0", "id": "g2", "method": "gen.run",
               "params": {"prompt_tokens": [1], "max_new_tokens": 1_000_000}}
    r = client.post("/rpc", json=payload)
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32602
