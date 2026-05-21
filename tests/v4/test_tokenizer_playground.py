"""F-G tokenizer playground tests — encode_visualize + list_presets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from cppmega_v4.jsonrpc import LRUCache, dispatch
from cppmega_v4.jsonrpc.tokenizer_methods import (
    EncodeVisualizeParams,
    PRESET_LIBRARY,
    encode_visualize,
    list_presets,
)


_VENDORED = "cppmega_mlx/tokenizer/tokenizer.json"


# ---------------------------------------------------------------------------
# encode_visualize
# ---------------------------------------------------------------------------


def test_encode_visualize_returns_spans():
    p = EncodeVisualizeParams(tokenizer_source=_VENDORED, text="hello world")
    r = encode_visualize(p)
    assert r.token_count > 0
    assert all(s.start >= 0 and s.end >= s.start for s in r.tokens)
    assert r.bytes_total == len("hello world".encode("utf-8"))


def test_encode_visualize_marks_specials_when_present():
    p = EncodeVisualizeParams(tokenizer_source=_VENDORED, text="<BOS>code")
    r = encode_visualize(p, cache=LRUCache())
    # At least one token id should be in the special-id set (BOS = 2).
    assert any(s.is_special for s in r.tokens)


def test_encode_visualize_capabilities_match_vendored_tokenizer():
    p = EncodeVisualizeParams(tokenizer_source=_VENDORED, text="x")
    r = encode_visualize(p)
    caps = r.capabilities
    assert caps["vocab_size"] > 0
    assert caps["has_fim"] is True
    assert caps["has_space_nl"] is True


def test_encode_visualize_caches_repeat_calls():
    cache = LRUCache(capacity=4)
    p = EncodeVisualizeParams(tokenizer_source=_VENDORED, text="abc")
    encode_visualize(p, cache=cache)
    encode_visualize(p, cache=cache)
    assert cache.stats()["hits"] >= 1


def test_encode_visualize_under_50ms_short_text():
    p = EncodeVisualizeParams(tokenizer_source=_VENDORED, text="short input")
    # Warm tokenizer cache
    encode_visualize(p)
    t0 = time.perf_counter()
    encode_visualize(p)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50, f"warm encode took {elapsed_ms:.1f} ms"


def test_encode_visualize_missing_source_raises():
    p = EncodeVisualizeParams(tokenizer_source="/nonexistent/tok.json", text="x")
    with pytest.raises(FileNotFoundError):
        encode_visualize(p)


# ---------------------------------------------------------------------------
# list_presets
# ---------------------------------------------------------------------------


def test_list_presets_returns_known_library():
    r = list_presets()
    assert len(r.presets) == len(PRESET_LIBRARY)
    assert "gpt-4-o200k" in r.presets
    assert "cppmega_v3" in r.presets


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_dispatch_tokenizer_encode_visualize_round_trip():
    resp = dispatch({
        "jsonrpc": "2.0", "id": "t1",
        "method": "tokenizer.encode_visualize",
        "params": {"tokenizer_source": _VENDORED, "text": "abc"},
    })
    assert resp.error is None
    assert resp.result["token_count"] >= 1


def test_dispatch_tokenizer_list_presets_round_trip():
    resp = dispatch({
        "jsonrpc": "2.0", "id": "lp", "method": "tokenizer.list_presets",
    })
    assert resp.error is None
    assert "presets" in resp.result
    assert len(resp.result["presets"]) >= 12


def test_dispatch_tokenizer_missing_source_returns_invalid_params():
    resp = dispatch({
        "jsonrpc": "2.0", "id": "tx",
        "method": "tokenizer.encode_visualize",
        "params": {"tokenizer_source": "/nope.json", "text": "x"},
    })
    # FileNotFoundError surfaces as INTERNAL_ERROR (not InvalidParams)
    # because it's a runtime failure, not a schema failure.
    assert resp.error is not None
    assert resp.error.code in (-32603, -32602)


def test_method_registry_includes_tokenizer_endpoints():
    from cppmega_v4.jsonrpc import METHOD_REGISTRY
    assert "tokenizer.encode_visualize" in METHOD_REGISTRY
    assert "tokenizer.list_presets" in METHOD_REGISTRY
