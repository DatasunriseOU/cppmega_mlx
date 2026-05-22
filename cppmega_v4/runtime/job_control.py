"""V7-H06: in-process job-pause/resume registry.

Mirrors V5-G09 abort_token pattern (cppmega_v4.runner.stages.request_abort)
but with a pause→resume cycle instead of a one-shot abort.

The train loop polls is_paused(token) between steps; while paused it
spins on a 50ms sleep without advancing optimizer state. resume(token)
clears the flag. pause(token) sets it.
"""

from __future__ import annotations

import time
from threading import Lock

_PAUSED_JOBS: set[str] = set()
_LOCK = Lock()


def pause(token: str) -> None:
    """Mark a job as paused. Idempotent."""
    if not token:
        return
    with _LOCK:
        _PAUSED_JOBS.add(str(token))


def resume(token: str) -> None:
    """Clear the pause flag. Idempotent."""
    with _LOCK:
        _PAUSED_JOBS.discard(str(token))


def is_paused(token: str | None) -> bool:
    if not token:
        return False
    with _LOCK:
        return str(token) in _PAUSED_JOBS


def wait_while_paused(token: str | None,
                       *, poll_s: float = 0.05,
                       max_wait_s: float = 600.0) -> None:
    """Block while the job is paused, polling every poll_s. Bounded by
    max_wait_s to avoid deadlock if resume() never lands."""
    if not token:
        return
    deadline = time.time() + max_wait_s
    while is_paused(token) and time.time() < deadline:
        time.sleep(poll_s)


def reset() -> None:
    """Test helper — clear the entire paused-jobs set."""
    with _LOCK:
        _PAUSED_JOBS.clear()


__all__ = ["pause", "resume", "is_paused",
           "wait_while_paused", "reset"]
