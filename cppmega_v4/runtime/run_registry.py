"""V7-H06b: in-process registry of active pipeline runs.

Tracks the lifecycle of every run_id passed into stage_train so that
pipeline.status RPC can answer:
  - is this run currently running (still in the train loop)?
  - has it been aborted?
  - is it paused?
  - what's the last published (step, loss)?

The UI uses this to gate state transitions on backend confirmation,
not optimistic local flips:
  - pipeline.pause -> poll status until paused=True
  - pipeline.resume -> poll until paused=False
  - pipeline.abort -> poll until running=False

Thread-safe; lives across requests in the FastAPI worker process.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

_LOCK = Lock()
_RUNS: dict[str, dict[str, Any]] = {}


def register(run_id: str | None) -> None:
    """Mark run_id as running. Idempotent; resets last_step / last_loss."""
    if not run_id:
        return
    with _LOCK:
        _RUNS[str(run_id)] = {
            "running": True,
            "aborted": False,
            "last_step": -1,
            "last_loss": None,
            "started_at": time.time(),
            "ended_at": None,
        }


def mark_step(run_id: str | None, step: int, loss: float) -> None:
    """Record the most recent step + loss from the train loop."""
    if not run_id:
        return
    with _LOCK:
        rec = _RUNS.get(str(run_id))
        if rec is None:
            return
        rec["last_step"] = int(step)
        rec["last_loss"] = float(loss)


def mark_aborted(run_id: str | None) -> None:
    if not run_id:
        return
    with _LOCK:
        rec = _RUNS.get(str(run_id))
        if rec is None:
            return
        rec["aborted"] = True


def unregister(run_id: str | None) -> None:
    """Mark run as no longer running. Record is kept so the UI can
    still query its terminal state after train returns."""
    if not run_id:
        return
    with _LOCK:
        rec = _RUNS.get(str(run_id))
        if rec is None:
            return
        rec["running"] = False
        rec["ended_at"] = time.time()


def snapshot(run_id: str | None) -> dict[str, Any] | None:
    """Return a copy of the registry entry, or None if unknown."""
    if not run_id:
        return None
    with _LOCK:
        rec = _RUNS.get(str(run_id))
        if rec is None:
            return None
        return dict(rec)


def reset() -> None:
    """Test helper — drop all registered runs."""
    with _LOCK:
        _RUNS.clear()


__all__ = [
    "register", "mark_step", "mark_aborted", "unregister",
    "snapshot", "reset",
]
