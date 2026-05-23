"""V7-I07: cache.metrics RPC — surfaces LRU cache hit-rate to the UI.

The dispatcher already shares a single LRUCache across handlers. This
RPC reads its stats() snapshot so the BottomStrip can render hit-rate
+ size as a tiny dashboard. Reads only; no mutation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cppmega_v4.jsonrpc.cache import LRUCache


class CacheMetricsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CacheMetricsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int
    capacity: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float


def cache_metrics(
    params: CacheMetricsParams,
    *, cache: LRUCache | None = None,
) -> CacheMetricsResult:
    if cache is None:
        return CacheMetricsResult(
            size=0, capacity=0, hits=0, misses=0,
            evictions=0, hit_rate=0.0,
        )
    s = cache.stats()
    return CacheMetricsResult(
        size=int(s["size"]),
        capacity=int(s["capacity"]),
        hits=int(s["hits"]),
        misses=int(s["misses"]),
        evictions=int(s["evictions"]),
        hit_rate=float(s["hit_rate"]),
    )


__all__ = ["CacheMetricsParams", "CacheMetricsResult", "cache_metrics"]
