"""V7-I07: LRUCache exposes hits/misses/evictions/hit_rate."""

from __future__ import annotations

from cppmega_v4.jsonrpc.cache import LRUCache


def test_v7_i07_stats_initial_shape():
    c = LRUCache(capacity=4)
    s = c.stats()
    for k in ("size", "capacity", "hits", "misses",
              "evictions", "hit_rate"):
        assert k in s
    assert s["capacity"] == 4
    assert s["size"] == 0
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["evictions"] == 0
    assert s["hit_rate"] == 0.0


def test_v7_i07_hits_increment_on_repeat_get():
    c = LRUCache(capacity=4)
    c.set("k1", {"v": 1})
    # First get → hit.
    assert c.get("k1") is not None
    assert c.stats()["hits"] == 1
    # Second get → hit again.
    assert c.get("k1") is not None
    assert c.stats()["hits"] == 2
    # Missing key → miss.
    assert c.get("k_other") is None
    assert c.stats()["misses"] == 1
    s = c.stats()
    # 2 hits / (2 + 1) = 0.6667
    assert abs(s["hit_rate"] - 2 / 3) < 1e-6


def test_v7_i07_evictions_increment_when_capacity_exceeded():
    c = LRUCache(capacity=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.stats()["evictions"] == 0
    c.set("c", 3)  # bumps 'a' out
    assert c.stats()["evictions"] == 1
    c.set("d", 4)  # bumps 'b' out
    assert c.stats()["evictions"] == 2
    assert c.stats()["size"] == 2


def test_v7_i07_clear_resets_counters():
    c = LRUCache(capacity=2)
    c.set("a", 1)
    c.get("a")
    c.get("nothing")
    c.set("b", 2)
    c.set("c", 3)
    assert c.stats()["evictions"] >= 1
    c.clear()
    s = c.stats()
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["evictions"] == 0
    assert s["size"] == 0


def test_v7_i07_http_cache_stats_endpoint_contains_v7_fields():
    """Sanity that the FastAPI /cache/stats hook returns the V7 keys."""
    from fastapi.testclient import TestClient
    from cppmega_v4.jsonrpc.server import create_app

    app = create_app()
    client = TestClient(app)
    r = client.get("/cache/stats")
    assert r.status_code == 200
    body = r.json()
    for k in ("hits", "misses", "evictions", "hit_rate",
              "size", "capacity"):
        assert k in body, f"missing {k} in /cache/stats response"
