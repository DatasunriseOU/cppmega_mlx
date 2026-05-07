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
