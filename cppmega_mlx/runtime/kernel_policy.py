"""Single source of truth for Path A/B/C kernel dispatch decisions.

Operations such as sparse_mla_attention and Mamba3ReferenceBlock may be
served by either the pure-MLX reference (Path A/REFERENCE), a hand-written
Apple Metal kernel (Path B), or an experimental TileLang DSL lowering
(Path C). This module owns the env-var contract that picks among them, plus a
small ring buffer of dispatch decisions used by the training profile receipts
to record the kernel that actually fired during a measured step.

Public surface:

- :class:KernelPath: enum of policy values.
- :func:selected_path: read the env var and return the active policy.
- :func:record_dispatch: append a dispatch decision to the ring buffer.
- :func:get_dispatch_log: return the ring buffer for receipt emission.
- :func:clear_dispatch_log: reset the ring buffer (called at profile begin).

Environment contract: CPPMEGA_KERNEL_PATH selects the policy. Recognized
values (case-insensitive): auto, ref / reference, path_b /
b, path_c / c. Unknown or empty values default to AUTO. The
parser also accepts a per-op override via CPPMEGA_KERNEL_PATH__<OPNAME>
for future expansion (e.g. CPPMEGA_KERNEL_PATH__SPARSE_MLA=ref); when
absent the global variable is used.
"""

from __future__ import annotations

import os
from collections import deque
from enum import Enum
from threading import Lock
from typing import Deque


__all__ = [
    "KernelPath",
    "clear_dispatch_log",
    "get_dispatch_log",
    "record_dispatch",
    "selected_path",
]


_RING_BUFFER_CAPACITY = 256


class KernelPath(Enum):
    """Policy outcome for an op-level dispatch decision."""

    AUTO = "auto"
    REFERENCE = "ref"
    PATH_B = "path_b"
    PATH_C = "path_c"


_VALUE_ALIASES: dict[str, KernelPath] = {
    "auto": KernelPath.AUTO,
    "": KernelPath.AUTO,
    "ref": KernelPath.REFERENCE,
    "reference": KernelPath.REFERENCE,
    "path_a": KernelPath.REFERENCE,
    "a": KernelPath.REFERENCE,
    "path_b": KernelPath.PATH_B,
    "b": KernelPath.PATH_B,
    "path_c": KernelPath.PATH_C,
    "c": KernelPath.PATH_C,
}


def _parse_path_value(raw: str | None) -> KernelPath:
    if raw is None:
        return KernelPath.AUTO
    return _VALUE_ALIASES.get(raw.strip().lower(), KernelPath.AUTO)


def selected_path(op_name: str) -> KernelPath:
    """Return the active :class:KernelPath for op_name.

    The global env var CPPMEGA_KERNEL_PATH provides the default. A
    per-op override is honored at CPPMEGA_KERNEL_PATH__<UPPER_OP_NAME>.
    Unknown op names are treated like any other op (no special-casing): they
    receive the global default. Unknown env values fall back to
    :attr:KernelPath.AUTO.
    """

    if not isinstance(op_name, str):
        raise TypeError(f"op_name must be str, got {type(op_name).__name__}")
    op_key = op_name.strip()
    if op_key:
        per_op = os.environ.get(f"CPPMEGA_KERNEL_PATH__{op_key.upper()}")
        if per_op is not None:
            return _parse_path_value(per_op)
    return _parse_path_value(os.environ.get("CPPMEGA_KERNEL_PATH"))


_dispatch_log: Deque[dict[str, str]] = deque(maxlen=_RING_BUFFER_CAPACITY)
_dispatch_log_lock = Lock()


def record_dispatch(op_name: str, path: KernelPath, kernel_used: str) -> None:
    """Append a dispatch decision to the ring buffer.

    The ring buffer is process-global and bounded to the last
    _RING_BUFFER_CAPACITY records. Concurrent calls are safe.
    """

    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    if not isinstance(path, KernelPath):
        raise TypeError(
            f"path must be KernelPath, got {type(path).__name__}"
        )
    if not isinstance(kernel_used, str) or not kernel_used:
        raise ValueError("kernel_used must be a non-empty string")
    with _dispatch_log_lock:
        _dispatch_log.append(
            {
                "op_name": op_name,
                "path": path.value,
                "kernel_used": kernel_used,
            }
        )


def get_dispatch_log() -> list[dict[str, str]]:
    """Snapshot the ring buffer for receipt emission."""

    with _dispatch_log_lock:
        return [dict(entry) for entry in _dispatch_log]


def clear_dispatch_log() -> None:
    """Reset the ring buffer (called by the profiler at scope begin)."""

    with _dispatch_log_lock:
        _dispatch_log.clear()
