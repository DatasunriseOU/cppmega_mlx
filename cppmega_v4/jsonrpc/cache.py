"""LRU cache keyed on canonical sha256 of the spec.

Cache key: SHA-256 of ``canonicalize(spec)`` — sorted keys, no
whitespace, NaN/Inf rejected. Node positions are stripped via
:func:`strip_layout` before canonicalisation so that dragging nodes
around in the canvas doesn't bust the cache.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Any, Mapping


DEFAULT_CAPACITY: int = 50


_LAYOUT_KEYS: frozenset[str] = frozenset({"x", "y", "position", "layout"})


def strip_layout(payload: Any) -> Any:
    """Recursively drop layout-only fields before hashing.

    Visual canvas positions never influence backend semantics; stripping
    them lets the cache survive drag operations without rebuilding.
    """
    if isinstance(payload, Mapping):
        return {
            k: strip_layout(v)
            for k, v in payload.items()
            if k not in _LAYOUT_KEYS
        }
    if isinstance(payload, list):
        return [strip_layout(v) for v in payload]
    if isinstance(payload, tuple):
        return [strip_layout(v) for v in payload]
    return payload


def canonical_json(payload: Any) -> str:
    """Produce a deterministic JSON string for cache keying."""
    return json.dumps(
        strip_layout(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(payload: Any) -> str:
    """Stable cache key — SHA-256 over :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class LRUCache:
    """Bounded LRU cache for backend RPC results.

    Thread-safe via a single Lock — RPC backends often share one cache
    across event-loop tasks. Capacity defaults to 50 per VBPlan §5.1.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        self._capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return
            self._data[key] = value
            if len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "capacity": self._capacity,
                "hits": self._hits,
                "misses": self._misses,
            }
