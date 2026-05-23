"""V7-I07: cache.metrics JSON-RPC returns LRUCache stats snapshot."""

from __future__ import annotations

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.cache_metrics_method import (
    CacheMetricsParams, cache_metrics,
)
from cppmega_v4.jsonrpc.dispatcher import dispatch


def test_v7_i07_cache_metrics_returns_zeros_for_empty():
    cache = LRUCache(capacity=8)
    r = cache_metrics(CacheMetricsParams(), cache=cache)
    assert r.size == 0
    assert r.capacity == 8
    assert r.hits == 0
    assert r.misses == 0
    assert r.evictions == 0
    assert r.hit_rate == 0.0


def test_v7_i07_cache_metrics_counts_hits_misses():
    cache = LRUCache(capacity=4)
    cache.set("a", {"v": 1})
    cache.get("a")          # hit
    cache.get("a")          # hit
    cache.get("missing")    # miss
    r = cache_metrics(CacheMetricsParams(), cache=cache)
    assert r.size == 1
    assert r.hits == 2
    assert r.misses == 1
    assert abs(r.hit_rate - (2 / 3)) < 1e-9


def test_v7_i07_cache_metrics_counts_evictions():
    cache = LRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts a
    cache.set("d", 4)  # evicts b
    r = cache_metrics(CacheMetricsParams(), cache=cache)
    assert r.size == 2
    assert r.evictions == 2


def test_v7_i07_cache_metrics_handles_missing_cache():
    r = cache_metrics(CacheMetricsParams(), cache=None)
    assert r.size == 0
    assert r.hits == 0
    assert r.misses == 0
    assert r.hit_rate == 0.0


def test_v7_i07_dispatcher_routes_cache_metrics():
    cache = LRUCache(capacity=4)
    cache.set("a", 1)
    cache.get("a")
    resp = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "cache.metrics",
        "params": {},
    }, cache=cache)
    assert resp.error is None, resp.error
    assert resp.result is not None
    assert resp.result["hits"] == 1
    assert resp.result["size"] == 1
    assert resp.result["capacity"] == 4
