"""V7-F06: cross-thread pub/sub for live token-by-token generation events.

Mirrors train_event_bus but keyed by a generation job_id supplied
to gen.run via params.run_id. stream_generate's on_token callback
publishes each {step, token_id, finish_reason} frame onto the bus
and the WS endpoint /ws/gen/{job_id} forwards them to the UI.
"""

from __future__ import annotations

import queue
import threading
from typing import Any


_LOCK = threading.Lock()
_QUEUES: dict[str, list[queue.Queue]] = {}


def publish(job_id: str | None, event: dict[str, Any] | None) -> None:
    if not job_id:
        return
    with _LOCK:
        subs = list(_QUEUES.get(job_id, []))
    for q in subs:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def subscribe(job_id: str) -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=4096)
    with _LOCK:
        _QUEUES.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: queue.Queue) -> None:
    with _LOCK:
        subs = _QUEUES.get(job_id, [])
        if q in subs:
            subs.remove(q)
        if not subs:
            _QUEUES.pop(job_id, None)


def reset() -> None:
    with _LOCK:
        _QUEUES.clear()


def subscriber_count(job_id: str) -> int:
    with _LOCK:
        return len(_QUEUES.get(job_id, []))


__all__ = ["publish", "subscribe", "unsubscribe", "reset",
            "subscriber_count"]
