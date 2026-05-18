"""Shared pytest config for the v4 test suite.

Default-pins both kernel-path env vars to "path_a" for tests that
exercise functional invariants (shapes, parity, doc-id propagation).
Tests that explicitly want a non-Path-A backend opt-in via
`monkeypatch.setenv(...)` and the autouse fixture detects that case
by checking whether the test function's own body sets the var.

Without this default, the block-level dispatch wired in commit
9854e55 routes through Path B/C/E whenever they're available, which
compiles fresh Metal/TileLang kernels per shape — at hundreds of
tests' worth of distinct shapes, the per-process Metal kernel cache
+ tilelang JIT state can SIGABRT mid-suite. Path A (FLA naive) is
the reference; perf tests live in dedicated files that opt out.

Opt-out files (run their own backend selection via monkeypatch):
  - test_block_dispatch_env_override.py — covers all 5 paths
  - test_path_dispatch.py               — env-override tests
  - test_benchmark_matrix.py            — matrix runner
  - test_linear_attention_path_b.py / _c.py / _e_*.py / _d.py
  - test_kda_paths.py
  - test_path_b_bwd.py / test_kda_path_b_bwd.py — bwd kernels
  - test_path_e_training.py
"""

from __future__ import annotations

import pytest


_OPT_OUT_FILES = {
    "test_block_dispatch_env_override.py",
    "test_path_dispatch.py",
    "test_benchmark_matrix.py",
    "test_linear_attention_path_b.py",
    "test_linear_attention_path_c.py",
    "test_linear_attention_path_d.py",
    "test_linear_attention_path_e.py",
    "test_kda_paths.py",
    "test_path_b_bwd.py",
    "test_kda_path_b_bwd.py",
    "test_path_e_training.py",
    "test_benchmark_receipt.py",
}


@pytest.fixture(autouse=True)
def _pin_path_a_by_default(request, monkeypatch):
    """Pin both v4 path env vars to path_a unless the test file opts out.

    Functional tests (shape, parity, doc-id) want the deterministic
    Path A reference. Backend-specific tests opt out and set their
    own env vars.
    """
    fname = request.node.fspath.basename
    if fname not in _OPT_OUT_FILES:
        monkeypatch.setenv("CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION", "path_a")
        monkeypatch.setenv("CPPMEGA_V4_KERNEL_PATH__KDA", "path_a")
    yield
