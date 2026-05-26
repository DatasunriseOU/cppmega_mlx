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
TRITON_FRONTEND_UNSAFE_IMPORT_ENV = "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT"

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


def unsafe_triton_frontend_import_enabled() -> bool:
    """Return True only when the caller explicitly accepts unsafe Triton imports.

    Some local Triton checkouts abort the Python process during import instead
    of raising a Python exception. Path D status probes run in normal test and
    benchmark discovery, so they must fail closed unless the developer opts in.
    """

    return os.environ.get(TRITON_FRONTEND_UNSAFE_IMPORT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def unsafe_triton_frontend_import_disabled_reason(root: str | None) -> str:
    root_text = f" root={root}" if root else " root=<not found>"
    return (
        "triton frontend not importable by default: unsafe import disabled "
        f"({TRITON_FRONTEND_UNSAFE_IMPORT_ENV}=1 required); Path D runtime "
        f"adapter not reached;{root_text}"
    )


def unsafe_fla_import_disabled_reason(root: str | None) -> str:
    root_text = f" root={root}" if root else " root=<not found>"
    return (
        "FLA not importable by default: unsafe Path D import disabled "
        f"({TRITON_FRONTEND_UNSAFE_IMPORT_ENV}=1 required); Path D runtime "
        f"adapter not reached;{root_text}"
    )


__all__ = [
    "FLA_SOURCE_PATH_ENV",
    "TRITON_FRONTEND_PATH_ENV",
    "TRITON_FRONTEND_UNSAFE_IMPORT_ENV",
    "ensure_fla_root",
    "ensure_triton_frontend_root",
    "unsafe_fla_import_disabled_reason",
    "unsafe_triton_frontend_import_disabled_reason",
    "unsafe_triton_frontend_import_enabled",
]
