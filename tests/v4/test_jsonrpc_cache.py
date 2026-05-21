"""VBGui F-A cache tests — canonicalisation + LRU semantics."""

from __future__ import annotations

import json

import pytest

from cppmega_v4.jsonrpc.cache import (
    LRUCache,
    canonical_json,
    canonical_sha256,
    strip_layout,
)


def test_strip_layout_drops_position_fields():
    payload = {
        "nodes": [
            {"id": "a", "kind": "mlp", "x": 100, "y": 200},
            {"id": "b", "kind": "mlp", "position": {"x": 0, "y": 0}},
        ],
    }
    stripped = strip_layout(payload)
    assert "x" not in stripped["nodes"][0]
    assert "y" not in stripped["nodes"][0]
    assert "position" not in stripped["nodes"][1]
    assert stripped["nodes"][0]["id"] == "a"


def test_strip_layout_recurses_into_lists():
    payload = [{"x": 1, "id": "a"}, {"y": 2, "id": "b"}]
    stripped = strip_layout(payload)
    assert stripped == [{"id": "a"}, {"id": "b"}]


def test_canonical_json_is_sorted_and_compact():
    payload = {"b": 2, "a": 1}
    s = canonical_json(payload)
    assert s == '{"a":1,"b":2}'


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"weights": float("nan")})


def test_canonical_json_strips_layout_before_emit():
    payload = {"node": {"id": "a", "x": 100}}
    s = canonical_json(payload)
    assert "100" not in s
    assert json.loads(s) == {"node": {"id": "a"}}


def test_sha256_stable_across_key_order():
    a = {"a": 1, "b": [1, 2], "c": {"k": "v"}}
    b = {"c": {"k": "v"}, "b": [1, 2], "a": 1}
    assert canonical_sha256(a) == canonical_sha256(b)


def test_sha256_invariant_to_layout_drift():
    a = {"nodes": [{"id": "x", "kind": "mlp"}]}
    b = {"nodes": [{"id": "x", "kind": "mlp", "x": 12, "y": 34}]}
    assert canonical_sha256(a) == canonical_sha256(b)


# ---------------------------------------------------------------------------
# LRU semantics.
# ---------------------------------------------------------------------------


def test_lru_rejects_zero_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)


def test_lru_get_miss_returns_none_and_increments_misses():
    c = LRUCache(capacity=2)
    assert c.get("k") is None
    assert c.stats()["misses"] == 1


def test_lru_set_then_get_increments_hits():
    c = LRUCache(capacity=2)
    c.set("k", 1)
    assert c.get("k") == 1
    assert c.stats()["hits"] == 1


def test_lru_evicts_least_recently_used():
    c = LRUCache(capacity=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_lru_get_promotes_to_most_recent():
    c = LRUCache(capacity=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")
    c.set("c", 3)  # evicts "b", not "a"
    assert c.get("a") == 1
    assert c.get("b") is None


def test_lru_set_replaces_existing_key_without_eviction():
    c = LRUCache(capacity=2)
    c.set("a", 1)
    c.set("a", 2)
    assert c.get("a") == 2
    assert len(c) == 1


def test_lru_clear_zeros_stats():
    c = LRUCache(capacity=2)
    c.set("a", 1); c.get("a"); c.get("b")
    c.clear()
    s = c.stats()
    assert s["hits"] == 0 and s["misses"] == 0 and s["size"] == 0


def test_default_capacity_matches_vbplan_spec():
    """VisualBuilderPlan.md §5.1 specifies LRU(50)."""
    c = LRUCache()
    assert c.capacity == 50
