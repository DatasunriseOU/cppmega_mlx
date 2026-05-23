"""E-AUDIT-01: parquet upload sink + 24h TTL cleanup.

DataInspector previously only accepted text paths to on-disk shards
inside the repo. Users couldn't bring their own parquet without
copying it into the working tree first. This module backs a single
POST /upload/parquet endpoint that:

  * writes the uploaded body to ``/tmp/vbgui_uploads/<uuid>.parquet``,
  * tracks upload timestamps,
  * cleans entries older than 24 h on every new upload (cheap),
  * caps total upload count at 100 (drops oldest first) so a runaway
    client can't fill /tmp.

The endpoint is intentionally simple — no auth, no chunked uploads.
Local-dev scope only; production must front this with a real
auth/scan layer.
"""

from __future__ import annotations

import pathlib
import time
import uuid
from threading import Lock

UPLOAD_ROOT = pathlib.Path("/tmp/vbgui_uploads")
TTL_SECONDS: int = 24 * 60 * 60
MAX_FILES: int = 100

_LOCK = Lock()


def _ensure_root() -> pathlib.Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return UPLOAD_ROOT


def cleanup_stale(now: float | None = None) -> int:
    """Drop files older than TTL_SECONDS; return number removed."""
    root = _ensure_root()
    now = now if now is not None else time.time()
    removed = 0
    with _LOCK:
        for p in sorted(root.iterdir()):
            try:
                age = now - p.stat().st_mtime
            except OSError:
                continue
            if age > TTL_SECONDS:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def _enforce_cap() -> int:
    """Drop oldest files past MAX_FILES; return number removed."""
    root = _ensure_root()
    removed = 0
    with _LOCK:
        entries = sorted(
            (p for p in root.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(entries) - MAX_FILES
        for p in entries[:max(0, excess)]:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def save_upload(body: bytes, *, suffix: str = ".parquet") -> str:
    """Persist an uploaded payload; return the absolute path string.

    A fresh uuid4 stem prevents collision between concurrent uploads."""
    cleanup_stale()
    _enforce_cap()
    root = _ensure_root()
    name = f"{uuid.uuid4().hex}{suffix}"
    path = root / name
    path.write_bytes(body)
    return str(path)


__all__ = [
    "UPLOAD_ROOT", "TTL_SECONDS", "MAX_FILES",
    "save_upload", "cleanup_stale",
]
