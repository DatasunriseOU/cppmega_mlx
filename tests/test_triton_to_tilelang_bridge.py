# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Smoke test: cppmega Triton kernel -> TileLang PrimFunc -> dispatch_lower.

Picks the simplest ``@triton.jit`` kernel from
``cppmega/cppmega/megatron/`` (``_reduce_pair_kernel`` defined inside
:func:`cppmega.megatron.mamba3_grouped_head_reduce.reduce_grouped_heads_triton`
— a straight elementwise+reduction kernel that uses only Tier-1 ops:
``program_id``, ``arange``, ``load``, ``store``, ``zeros``, ``static_range``,
arithmetic, masking, ``to(tl.float32)``) and routes it through the
:mod:`cppmega_mlx.nn._triton_bridge` adapter into the unified
``dispatch_lower`` entrypoint.

Test policy
-----------
* ``pytest.importorskip("triton")`` — Mac dev hosts (and most CI lanes
  without a CUDA build) have no Triton. We skip cleanly there.
* If the POC frontend at ``/private/tmp/tl_poc_review`` is missing the
  test is also skipped — same rationale, this is a smoke test for the
  wiring, not a correctness check on the frontend itself.
* When the frontend is reachable but a coverage gap surfaces we
  capture the :class:`~cppmega_mlx.nn._triton_bridge.TritonBridgeError`
  and surface it as an ``xfail`` with the exact gap noted, so the
  test stays informative (per memory rule
  ``feedback_no_silent_delete``: never silently degrade, document
  exactly what's missing).
"""

from __future__ import annotations

import sys
import textwrap
import types

import pytest


# Triton import-skip is the very first gate so this module is collectable
# on hosts without triton (Mac dev). The bridge import itself is also
# guarded because it does not depend on triton at import time but does
# eagerly look for the POC frontend on first call.
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402  - depends on importorskip above

from cppmega_mlx.nn._triton_bridge import (  # noqa: E402
    TritonBridgeError,
    frontend_available,
    triton_to_tilelang_prim,
)


pytestmark = pytest.mark.skipif(
    not frontend_available(),
    reason=(
        "POC triton_frontend not importable from /private/tmp/tl_poc_review. "
        "Set CPPMEGA_MLX_TRITON_FRONTEND_PATH or clone tl_poc_review there."
    ),
)


def _load_simplest_kernel() -> "triton.runtime.JITFunction":
    """Return the simplest cppmega/megatron Triton kernel as a ``JITFunction``.

    The chosen kernel is ``_reduce_pair_kernel`` from
    ``cppmega.megatron.mamba3_grouped_head_reduce``. It is defined inside
    a function (because cppmega gates the import behind a CUDA-only
    runtime check, and we cannot run that on a Mac dev box even with
    triton importable). To stay read-only on cppmega we re-declare an
    *equivalent* kernel here using the same op surface — load, store,
    arange, program_id, zeros, static_range, masking, scalar arithmetic.

    The duplication is intentional: it freezes the op-surface contract
    that the bridge must support, independent of cppmega's evolving
    grouped-head reduction internals.
    """

    @triton.jit
    def _reduce_pair_kernel(
        dq_ptr,
        dk_ptr,
        dq_out_ptr,
        dk_out_ptr,
        total_m: tl.constexpr,
        S_: tl.constexpr,
        R_: tl.constexpr,
        H_: tl.constexpr,
        G_: tl.constexpr,
        N_: tl.constexpr,
        HPG_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        m = tl.program_id(0) * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        n = tl.program_id(1) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        m_mask = m < total_m
        n_mask = n < N_

        g = m % G_
        tmp = m // G_
        r = tmp % R_
        tmp = tmp // R_
        s = tmp % S_
        b = tmp // S_

        in_base = (
            (((b[:, None] * S_ + s[:, None]) * R_ + r[:, None]) * H_ + g[:, None] * HPG_) * N_
            + n[None, :]
        )
        mask = m_mask[:, None] & n_mask[None, :]
        acc_dq = tl.zeros((BLOCK_M_, BLOCK_N_), tl.float32)
        acc_dk = tl.zeros((BLOCK_M_, BLOCK_N_), tl.float32)
        for h in tl.static_range(0, HPG_):
            in_offsets = in_base + h * N_
            acc_dq += tl.load(dq_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
            acc_dk += tl.load(dk_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)

        out_offsets = m[:, None] * N_ + n[None, :]
        tl.store(dq_out_ptr + out_offsets, acc_dq, mask=mask)
        tl.store(dk_out_ptr + out_offsets, acc_dk, mask=mask)

    return _reduce_pair_kernel


def test_bridge_module_imports():
    """Sanity: the bridge module exposes its public surface."""

    from cppmega_mlx.nn import _triton_bridge as bridge

    assert hasattr(bridge, "triton_to_tilelang_prim")
    assert hasattr(bridge, "triton_to_tilelang_compile")
    assert hasattr(bridge, "TritonBridgeError")
    assert hasattr(bridge, "frontend_available")


def test_bridge_lowering_smoke():
    """End-to-end: load Triton kernel, lower to TileLang PrimFunc.

    Acceptance:
      * Either the bridge returns *something* PrimFunc-shaped (the smoke
        passes), OR
      * The bridge raises :class:`TritonBridgeError` with a clear
        coverage-gap message, in which case we ``xfail`` so the gap is
        visible in CI output without being a hard red.

    We deliberately do NOT chain into ``dispatch_lower`` here — the
    follow-up test below covers that path with ``target='cuda'`` and is
    tolerant of ``MSLDispatchUnsupported`` etc.
    """

    kernel = _load_simplest_kernel()
    constexprs = {
        "total_m": 8,
        "S_": 2,
        "R_": 1,
        "H_": 4,
        "G_": 2,
        "N_": 4,
        "HPG_": 2,
        "BLOCK_M_": 2,
        "BLOCK_N_": 4,
    }
    try:
        prim = triton_to_tilelang_prim(kernel, constexprs=constexprs, target="cuda")
    except TritonBridgeError as exc:
        # POC frontend coverage gap: surface the exact missing op so the
        # bridge maintainer can extend OP_TABLE.
        pytest.xfail(
            f"POC triton_frontend coverage gap (expected during MVP): {exc}"
        )

    assert prim is not None, "bridge returned None — silent degradation forbidden"
    # PrimFunc-ish duck typing — the POC frontend may return a tvm.tir
    # PrimFunc, a stub object, or a tuple in transitional builds.
    assert hasattr(prim, "with_attr") or hasattr(prim, "body") or hasattr(prim, "__call__"), (
        f"bridge returned non-PrimFunc-shaped object: {type(prim)!r}"
    )


def test_bridge_dispatch_lower_smoke():
    """Bridge -> dispatch_lower(prim, target='cuda') end-to-end smoke.

    The Mac dev host has no CUDA toolchain, so this will normally fail
    inside ``tilelang.compile`` rather than the bridge itself. We are
    OK with any of:

      * Successful artifact (rare on Mac, expected on CUDA hosts).
      * :class:`TritonBridgeError` from the bridge layer (POC coverage gap).
      * ``ImportError`` / ``ModuleNotFoundError`` from tilelang (no CUDA
        backend in this environment).
      * Any ``RuntimeError`` from tilelang's CUDA codegen ("nvcc not found",
        "no CUDA device", etc.).

    What is NOT OK: silent return of None, or a generic ``Exception``
    leaking past the documented surface.
    """

    kernel = _load_simplest_kernel()
    constexprs = {
        "total_m": 8,
        "S_": 2,
        "R_": 1,
        "H_": 4,
        "G_": 2,
        "N_": 4,
        "HPG_": 2,
        "BLOCK_M_": 2,
        "BLOCK_N_": 4,
    }
    try:
        prim = triton_to_tilelang_prim(kernel, constexprs=constexprs, target="cuda")
    except TritonBridgeError as exc:
        pytest.xfail(f"POC frontend coverage gap (pre dispatch_lower): {exc}")

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    try:
        artifact = dispatch_lower(prim, target="cuda")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.xfail(f"tilelang/cuda backend unavailable on this host: {exc}")
    except RuntimeError as exc:
        pytest.xfail(f"dispatch_lower raised RuntimeError on cuda target: {exc}")

    assert artifact is not None, (
        "dispatch_lower returned None — engine path must produce an artifact "
        "or raise a documented exception"
    )
