"""Pytest fixtures shared across the cppmega.mlx test suite.

Why the autouse env-isolation fixture below exists
==================================================
Meta agent E flagged that bench-harness scripts (``scripts/bench_tilelang_*``)
read ``TILELANG_ROOT`` / ``TVM_ROOT`` / ``TVM_TARGET`` / ``TVM_METAL_*`` /
``METAL_DEVICE_WRAPPER_TYPE`` / ``TVM_LIBRARY_PATH`` / ``PYTHONPATH`` and
mutate ``sys.path``. Without isolation, a developer's shell env or the
output of one test bleeds into another, masking real configuration bugs
("works on my machine" but fails on CI). The fixture below scrubs those
vars at the start of every test via ``monkeypatch.delenv`` (which is
auto-restored at teardown), giving each test a hermetic env baseline.
Tests that need a specific value should ``monkeypatch.setenv`` it
explicitly inside the test body.
"""

from __future__ import annotations

import os

import pytest


# fix-round-7 finding-5 (security CRIT): the prior approach set
# ``CPPMEGA_ALLOW_UNSAFE_LIBZ3=1`` here so tests could pick up the in-tree
# /tmp/tl_apache_tvm_swap libz3, but that env var was honoured by production
# code paths too — any process that inherited the env (a stray shell export,
# a parent CI job) would silently dlopen a world-writable /tmp dylib. We now
# inject the candidate path directly into the preload helper's private
# candidate list and leave the env var unset, so production never resolves
# /tmp regardless of caller env.
from pathlib import Path as _Path

import cppmega_mlx.nn._tilelang._msl_transform as _msl  # noqa: E402

_msl._LIBZ3_DEV_CANDIDATES = [
    _Path("/tmp/tl_apache_tvm_swap/build/lib/libz3.dylib"),
]
# The module-level preload at the bottom of _msl_transform.py runs at
# import time -- BEFORE we set the candidate list above. So the first
# preload attempt only saw the default empty list (plus the brew
# fallback). Reset the idempotency flag and re-run the preload now that
# the in-tree dev candidate is registered, so libtilelang.dylib's
# basename libz3 reference resolves to the matching dev-build z3.
try:
    if hasattr(_msl._preload_libz3_for_dev_tilelang, "_done"):
        delattr(_msl._preload_libz3_for_dev_tilelang, "_done")
    if hasattr(_msl._preload_libz3_for_dev_tilelang, "_failed_attempts"):
        delattr(_msl._preload_libz3_for_dev_tilelang, "_failed_attempts")
    _msl._preload_libz3_for_dev_tilelang()
except Exception:  # pragma: no cover - best-effort
    pass


# Environment variables that influence TileLang / TVM import resolution and
# Metal/MPS dispatch. Any test that reads or writes these must be explicit.
_TILELANG_TVM_ENV_VARS = (
    "TILELANG_ROOT",
    "TVM_ROOT",
    "TVM_TARGET",
    "TVM_METAL_STORAGE_MODE",
    "METAL_DEVICE_WRAPPER_TYPE",
    "TVM_LIBRARY_PATH",
    "PYTHONPATH",
    "TILELANG_DEV_BUILD_ROOT",
    "TVM_HOME",
    "TVM_SOURCE_DIR",
    "TVM_LIBRARY_PATH_SELECTED",
    "DYLD_LIBRARY_PATH",
)

# Prefix-matched env vars cleared at test start. ``MLX_*`` (e.g.
# ``MLX_DEFAULT_DEVICE``, ``MLX_MEMORY_LIMIT_GB``, ``MLX_GPU_LIMIT``) tune
# the mlx runtime; ``MTL_*`` (e.g. ``MTL_HUD_ENABLED``,
# ``MTL_CAPTURE_ENABLED``, ``MTL_DEBUG_LAYER``) toggle Metal driver
# diagnostics. Both leak under pytest-xdist parallel runners (gpt-5.5-pro G3
# P1 finding) when a developer's shell or a sibling worker has them set.
# NOTE: ``PATH`` is intentionally NOT cleared - wiping it breaks subprocess
# spawning (including pytest's own helpers); the right defence against PATH
# tampering is a snapshot/assert fixture, which is overkill here.
_VOLATILE_ENV_PREFIXES = ("MLX_", "MTL_")


@pytest.fixture(autouse=True)
def _isolate_tilelang_tvm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip TileLang/TVM env vars at test start; auto-restore at teardown.

    Per Meta-E env-leak finding (2026-05-06 review): bench scripts read
    ``TILELANG_ROOT`` etc. at module level. Tests that *just import* the
    module would otherwise inherit whatever the developer's shell exported,
    making CI behaviour shell-dependent. ``monkeypatch.delenv`` restores the
    original value on fixture teardown, so this is safe for tests that
    legitimately want to set one of these vars themselves.

    Extension (gpt-5.5-pro G3 P1, 2026-05-07): also scrub ``MLX_*`` and
    ``MTL_*`` prefix-matched vars to close the pytest-xdist parallel-runner
    leak path. ``PATH`` is intentionally left intact - see
    ``_VOLATILE_ENV_PREFIXES`` docstring.
    """

    for var in _TILELANG_TVM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    for var in list(os.environ):
        if var.startswith(_VOLATILE_ENV_PREFIXES):
            monkeypatch.delenv(var, raising=False)

    # fix-round-7 finding-5: the env-var opt-in approach was inverted —
    # we now inject the in-tree /tmp candidate via
    # ``_msl._LIBZ3_DEV_CANDIDATES`` at conftest import (see top of file)
    # and keep ``CPPMEGA_ALLOW_UNSAFE_LIBZ3`` strictly OFF so any stray
    # production path that still consults the env stays secure-by-default.
    monkeypatch.delenv("CPPMEGA_ALLOW_UNSAFE_LIBZ3", raising=False)
