"""E7-8 tests: architectures.list_presets RPC."""

from __future__ import annotations

from cppmega_v4.architectures import PRESETS, available_presets
from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.jsonrpc.schema import METHOD_REGISTRY


def test_method_registered():
    assert "architectures.list_presets" in METHOD_REGISTRY


def test_dispatch_returns_sorted_presets():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1,
        "method": "architectures.list_presets", "params": {},
    })
    assert resp.error is None
    names = resp.result["presets"]
    assert names == sorted(names)
    assert names == sorted(PRESETS.keys())


def test_dispatch_returns_all_62_presets():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1,
        "method": "architectures.list_presets", "params": {},
    })
    assert resp.error is None
    names = set(resp.result["presets"])
    # Must include the 5 entries previously missing from the UI:
    assert {"qwen3_dense_0_6b", "qwen3_dense_4b", "qwen3_dense_8b",
            "qwen3_dense_32b", "qwen3_coder_flash"} <= names


def test_available_presets_matches_dispatch():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1,
        "method": "architectures.list_presets", "params": {},
    })
    assert resp.result["presets"] == available_presets()
