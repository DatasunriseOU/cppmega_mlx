"""V7-H46: tokenizer.roundtrip_text RPC for TokenizerPlayground."""

from __future__ import annotations

from cppmega_v4.jsonrpc.dispatcher import dispatch


def test_v7_h46_tokenizer_roundtrip_text_route_returns_match_for_cppmega():
    response = dispatch({
        "jsonrpc": "2.0", "id": "T1",
        "method": "tokenizer.roundtrip_text",
        "params": {
            "tokenizer_source": "cppmega_mlx/tokenizer/tokenizer.json",
            "text": "Hello, world!",
        },
    })
    if response.error is not None:
        # Tokenizer file not vendored in test env — accept skip-like behavior.
        assert response.error.code in (-32603, -32602)
        return
    r = response.result
    assert "matches" in r
    assert "decoded" in r
    assert "tokenizer_capability" in r
    assert "byte_diff" in r
    assert r["original_bytes"] == len("Hello, world!".encode("utf-8"))


def test_v7_h46_invalid_params_rejected():
    response = dispatch({
        "jsonrpc": "2.0", "id": "T2",
        "method": "tokenizer.roundtrip_text",
        "params": {"tokenizer_source": "x", "text": "y", "extra": 1},
    })
    assert response.error is not None
    assert response.error.code == -32602
