"""V7-H05: cross-thread pub/sub bus for live per-step training events.

stage_train runs in a worker thread (asyncio.to_thread). The WS
endpoint runs in the asyncio loop. They communicate through a per-
run_id queue.Queue. publish() is thread-safe; subscribe() returns a
queue the WS handler drains until the sentinel `None` arrives marking
train completion.

  bus.publish('run-42', {'step': 3, 'loss': 1.2, 'lr': 1e-3})
  q = bus.subscribe('run-42')
  while True:
      ev = q.get(timeout=0.5)
      if ev is None: break
      ws.send_json(ev)
"""

from __future__ import annotations

import queue
import threading
from typing import Any


_LOCK = threading.Lock()
_QUEUES: dict[str, list[queue.Queue]] = {}


def publish(run_id: str | None, event: dict[str, Any] | None) -> None:
    """Broadcast `event` to every active subscriber for run_id.

    Pass `event=None` to signal completion — subscribers exit their
    consumer loops on the sentinel."""
    if not run_id:
        return
    with _LOCK:
        subs = list(_QUEUES.get(run_id, []))
    for q in subs:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def subscribe(run_id: str) -> queue.Queue:
    """Return a new queue subscribed to run_id events. Caller must
    call unsubscribe() to release the slot when done."""
    q: queue.Queue = queue.Queue(maxsize=1024)
    with _LOCK:
        _QUEUES.setdefault(run_id, []).append(q)
    return q


def unsubscribe(run_id: str, q: queue.Queue) -> None:
    with _LOCK:
        subs = _QUEUES.get(run_id, [])
        if q in subs:
            subs.remove(q)
        if not subs:
            _QUEUES.pop(run_id, None)


def reset() -> None:
    """Test helper — drop all subscribers."""
    with _LOCK:
        _QUEUES.clear()


def subscriber_count(run_id: str) -> int:
    with _LOCK:
        return len(_QUEUES.get(run_id, []))


__all__ = ["publish", "subscribe", "unsubscribe", "reset",
            "subscriber_count"]
