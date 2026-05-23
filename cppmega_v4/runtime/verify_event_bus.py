"""V7-L45: cross-thread pub/sub for live verify progress events.

Mirrors train_event_bus + gen_event_bus, keyed by a stable spec_hash
(sha256 of the canonical JSON of VerifyParams). Used so a long-running
verify can stream progress to the UI via /ws/verify/{spec_hash}.
"""

from __future__ import annotations

import hashlib
import json as _json
import queue
import threading
from typing import Any


_LOCK = threading.Lock()
_QUEUES: dict[str, list[queue.Queue]] = {}


def spec_hash(spec_payload: Any) -> str:
    """Stable sha256 over the canonical JSON of a VerifyParams-like dict.

    Accepts a dict or a Pydantic-ish object via model_dump(). Used both
    by the UI (to build the WS path) and by the verify handler (to know
    which subscribers to notify)."""
    if hasattr(spec_payload, "model_dump"):
        payload = spec_payload.model_dump(mode="json")
    else:
        payload = spec_payload
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True,
                     default=str).encode("utf-8")
    ).hexdigest()


def publish(key: str | None, event: dict[str, Any] | None) -> None:
    if not key:
        return
    with _LOCK:
        subs = list(_QUEUES.get(key, []))
    for q in subs:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def subscribe(key: str) -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=256)
    with _LOCK:
        _QUEUES.setdefault(key, []).append(q)
    return q


def unsubscribe(key: str, q: queue.Queue) -> None:
    with _LOCK:
        subs = _QUEUES.get(key, [])
        if q in subs:
            subs.remove(q)
        if not subs:
            _QUEUES.pop(key, None)


def reset() -> None:
    with _LOCK:
        _QUEUES.clear()


def subscriber_count(key: str) -> int:
    with _LOCK:
        return len(_QUEUES.get(key, []))


__all__ = ["spec_hash", "publish", "subscribe", "unsubscribe", "reset",
            "subscriber_count"]
