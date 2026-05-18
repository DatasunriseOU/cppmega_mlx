"""Optional local dependencies for Triton Path D probes.

Path D is a development bridge: cppmega_v4 owns the dispatch surface, while
the Triton frontend and FLA kernels live in sibling checkouts on dev hosts.
This module keeps those lookup rules explicit and cheap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TRITON_FRONTEND_PATH_ENV = "CPPMEGA_MLX_TRITON_FRONTEND_PATH"
FLA_SOURCE_PATH_ENV = "CPPMEGA_MLX_FLA_SOURCE_PATH"

_TRITON_FRONTEND_ROOTS = (
    Path("/Users/dave/sources/tilelang"),
    Path("/Volumes/external/sources/tilelang"),
    Path("/private/tmp/tl_poc_review"),
)

_FLA_ROOTS = (
    Path("/Volumes/external/sources/rent_kernels/flash-linear-attention"),
    Path("/Users/dave/sources/rent_kernels/flash-linear-attention"),
    Path("/Users/dave/sources/flash-linear-attention"),
)


def _prepend_existing(root: Path) -> bool:
    if not root.exists():
        return False
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return True


def ensure_triton_frontend_root() -> str | None:
    """Make ``poc.triton_frontend`` importable from known local checkouts."""

    raw = os.environ.get(TRITON_FRONTEND_PATH_ENV)
    candidates = [Path(raw)] if raw else []
    candidates.extend(_TRITON_FRONTEND_ROOTS)
    for root in candidates:
        if (root / "poc" / "triton_frontend").exists():
            _prepend_existing(root)
            return str(root)
    return None


def ensure_fla_root() -> str | None:
    """Make ``fla`` importable from known local source checkouts."""

    raw = os.environ.get(FLA_SOURCE_PATH_ENV)
    candidates = [Path(raw)] if raw else []
    candidates.extend(_FLA_ROOTS)
    for root in candidates:
        if (root / "fla").exists():
            _prepend_existing(root)
            return str(root)
    return None


__all__ = [
    "FLA_SOURCE_PATH_ENV",
    "TRITON_FRONTEND_PATH_ENV",
    "ensure_fla_root",
    "ensure_triton_frontend_root",
]
