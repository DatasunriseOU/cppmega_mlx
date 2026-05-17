"""Common path-dispatch scaffolding for V4 linear-attention backends.

Mirrors the ``mamba3_path_c_*_status`` / ``*_auto_mode_for_inputs`` pattern
from ``cppmega_mlx/nn/_tilelang/mamba3_path_c.py`` so v4 paths plug into the
same conceptual machinery without depending on cppmega_mlx internals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

PathName = Literal["path_a", "path_b", "path_c", "path_d", "path_e"]


@dataclass(frozen=True)
class PathStatus:
    """Describes whether a given backend is available on this host."""

    path: PathName
    available: bool
    reason: str

    def __bool__(self) -> bool:  # truthy when available
        return self.available


def env_override(env_var: str) -> str | None:
    """Read an env override (e.g., 'path_a', 'path_b', ...). None = auto."""
    value = os.environ.get(env_var, "").strip().lower()
    if not value or value == "auto":
        return None
    if value not in ("path_a", "path_b", "path_c", "path_d", "path_e"):
        raise ValueError(
            f"unsupported {env_var}={value!r}; "
            "expected one of: path_a, path_b, path_c, path_d, path_e, auto"
        )
    return value


def auto_pick(
    statuses: dict[PathName, PathStatus],
    preference: tuple[PathName, ...] = ("path_c", "path_b", "path_e", "path_d", "path_a"),
) -> PathName:
    """Pick the first available path in the preference order."""
    for path in preference:
        st = statuses.get(path)
        if st is not None and st.available:
            return path
    # Always falls back to path_a (the pure-MLX reference is always available).
    return "path_a"
