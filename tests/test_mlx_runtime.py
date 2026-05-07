"""Tests for ``cppmega_mlx.nn._tilelang._mlx_runtime``.

These tests validate the TileLang Metal -> MLX runtime adapter without
requiring TileLang to be installed -- they hand-author Metal sources
that mimic TileLang's emitted shape (``kernel void name(device <T>*
A, device <T>* B, device <T>* C, ...)``) and check that the rename is
correct and that the resulting kernel actually launches on Mac GPU.
"""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


_HAS_MLX = importlib.util.find_spec("mlx") is not None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not importable on this host")
def test_wrap_renames_and_runs_vector_add() -> None:
    """Hand-authored vector-add Metal kernel renames + runs end-to-end."""
    import mlx.core as mx
    import numpy as np

    from cppmega_mlx.nn._tilelang._mlx_runtime import wrap_tilelang_metal_kernel

    # NOTE: ``mx.fast.metal_kernel`` injects its own grid/threadgroup
    # bindings (``thread_position_in_grid`` etc.) into the body scope, so
    # the body is free to reference them directly. The kernel signature's
    # builtin params (``uint id [[thread_position_in_grid]]``) are NOT
    # carried into the rebuilt MLX kernel -- MLX builds its own signature
    # from ``input_names`` / ``output_names``. We therefore use the MLX
    # idiom in the body and only declare buffer params in the signature.
    src = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]]
) {
    uint id = thread_position_in_grid.x;
    if (id < 16) {
        C[id] = A[id] + B[id];
    }
}
"""

    adapter = wrap_tilelang_metal_kernel(src, input_count=2, output_count=1)
    # Renaming must have happened in the body: ``A`` -> ``inp0``,
    # ``B`` -> ``inp1``, ``C`` -> ``out0``.
    assert "inp0[id]" in adapter.body
    assert "inp1[id]" in adapter.body
    assert "out0[id]" in adapter.body
    # The original positional names must NOT survive in the body.
    assert " A[" not in adapter.body
    assert " B[" not in adapter.body
    assert " C[" not in adapter.body
    assert adapter.input_names == ("inp0", "inp1")
    assert adapter.output_names == ("out0",)
    assert adapter.buffer_names == ("A", "B", "C")

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(16).astype(np.float32)
    b_np = rng.standard_normal(16).astype(np.float32)
    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)

    outputs = adapter(
        inputs=[a_mx, b_mx],
        output_shapes=[(16,)],
        output_dtypes=[mx.float32],
        grid=(16, 1, 1),
        threadgroup=(16, 1, 1),
    )
    mx.eval(outputs)
    result = np.array(outputs[0], copy=False)
    assert np.allclose(result, a_np + b_np, atol=1e-6, rtol=1e-6), (
        f"vector add mismatch: max abs err = {np.abs(result - (a_np + b_np)).max():.3e}"
    )


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not importable on this host")
def test_wrap_multi_output_kernel() -> None:
    """A kernel with two outputs (sum and diff) renames + runs."""
    import mlx.core as mx
    import numpy as np

    from cppmega_mlx.nn._tilelang._mlx_runtime import wrap_tilelang_metal_kernel

    src = """
#include <metal_stdlib>
using namespace metal;

kernel void sum_and_diff(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device float* D [[buffer(3)]]
) {
    uint id = thread_position_in_grid.x;
    if (id < 8) {
        C[id] = A[id] + B[id];
        D[id] = A[id] - B[id];
    }
}
"""

    adapter = wrap_tilelang_metal_kernel(src, input_count=2, output_count=2)
    assert adapter.input_names == ("inp0", "inp1")
    assert adapter.output_names == ("out0", "out1")
    assert adapter.buffer_names == ("A", "B", "C", "D")
    # Both output buffer names must be rewritten in the body.
    assert "out0[id]" in adapter.body
    assert "out1[id]" in adapter.body
    assert "inp0[id]" in adapter.body
    assert "inp1[id]" in adapter.body

    rng = np.random.default_rng(1)
    a_np = rng.standard_normal(8).astype(np.float32)
    b_np = rng.standard_normal(8).astype(np.float32)

    outputs = adapter(
        inputs=[mx.array(a_np), mx.array(b_np)],
        output_shapes=[(8,), (8,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
    )
    mx.eval(outputs)
    sum_result = np.array(outputs[0], copy=False)
    diff_result = np.array(outputs[1], copy=False)
    assert np.allclose(sum_result, a_np + b_np, atol=1e-6, rtol=1e-6)
    assert np.allclose(diff_result, a_np - b_np, atol=1e-6, rtol=1e-6)


def test_wrap_skips_tilelang_args_struct() -> None:
    """TileLang's auto-emitted scalar-args struct must NOT be counted as
    a user data buffer by the parser.

    When TileLang lowers a PrimFunc that has scalar runtime args (e.g.
    ``n_elements`` for vector_add), it emits a ``constant
    <kernel>_kernel_args_t&`` parameter at the *end* of the kernel
    signature. That parameter holds *scalars*, not a user-supplied data
    tensor; counting it would cause the buffer count check to inflate
    beyond ``input_count + output_count`` and the wrap call would fail
    with ``buffer count mismatch`` when the caller can only pass
    ``mx.array`` data buffers, not the scalars struct.

    Regression for the e2e numeric harness vector_add path
    (poc/triton_frontend/_test_harness/numeric_smoke.py).
    """
    from cppmega_mlx.nn._tilelang._mlx_runtime import wrap_tilelang_metal_kernel

    src = """
#include <metal_stdlib>
using namespace metal;

struct vector_add_kernel_args_t {
  int n_elements[2];
  int gridDim_0[2];
};

kernel void vector_add_kernel(
    device float* arg0 [[ buffer(0) ]],
    device float* arg1 [[ buffer(1) ]],
    device float* arg2 [[ buffer(2) ]],
    constant vector_add_kernel_args_t& arg [[ buffer(3) ]],
    uint blockIdx [[threadgroup_position_in_grid]]
) {
    uint id = blockIdx;
    if (id < arg.n_elements[0]) {
        arg2[id] = arg0[id] + arg1[id];
    }
}
"""

    # 3 user data buffers (a, b, c) + the args struct. The wrapper must
    # see only the 3 user buffers; the ``_args_t&`` is filtered out.
    adapter = wrap_tilelang_metal_kernel(src, input_count=2, output_count=1)
    assert adapter.buffer_names == ("arg0", "arg1", "arg2")
    assert adapter.input_names == ("inp0", "inp1")
    assert adapter.output_names == ("out0",)
    # The body's user-buffer references must be renamed.
    assert "inp0[id]" in adapter.body
    assert "inp1[id]" in adapter.body
    assert "out0[id]" in adapter.body
    # The args struct identifier ``arg`` must NOT have been renamed away
    # (it is referenced by the body via ``arg.n_elements[0]``); the
    # rename map only contains user-buffer names. The prefix-collision
    # check below is conservative -- ``arg0/1/2`` start with ``arg`` but
    # the regex uses ``\b`` so the bare ``arg`` identifier is preserved.
    assert "arg.n_elements[0]" in adapter.body


def test_rewrite_blockidx_to_thread_position_in_grid() -> None:
    """``blockIdx``/``threadIdx`` identifiers are rewritten to MLX builtins.

    TileLang emits CUDA-style scalar names (``uint blockIdx
    [[threadgroup_position_in_grid]]``) and references them as bare
    ``blockIdx`` tokens in the body. ``mx.fast.metal_kernel`` rebuilds
    the signature itself and only injects MLX's own ``uint3``-typed
    builtins, so the rewrite must substitute the body identifier.
    """
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        _rewrite_tilelang_metal_to_mlx,
    )

    src = """
kernel void k(
    device float* arg0 [[ buffer(0) ]],
    uint blockIdx [[threadgroup_position_in_grid]],
    uint threadIdx [[thread_position_in_threadgroup]]
) {
    int x = ((int)blockIdx) * 256;
    int y = ((int)threadIdx);
    arg0[x + y] = 0.0f;
}
"""
    out = _rewrite_tilelang_metal_to_mlx(src)
    # Both CUDA identifiers must be replaced with the MLX-injected forms.
    assert "((int)threadgroup_position_in_grid.x)" in out
    assert "((int)thread_position_in_threadgroup.x)" in out
    # The bare CUDA tokens must NOT survive in the body. (They survive in
    # the signature attribute names ``[[threadgroup_position_in_grid]]``,
    # which is fine because the signature is rebuilt by MLX from
    # input_names/output_names anyway -- the identifier rewrite only
    # affects body references.)
    # We expect zero occurrences of the bare ``blockIdx``/``threadIdx``
    # tokens (whole-word) outside the signature attribute markers.
    import re as _re

    body_section = out.split("{", 1)[1] if "{" in out else out
    assert not _re.search(r"\bblockIdx\b", body_section)
    assert not _re.search(r"\bthreadIdx\b", body_section)


def test_rewrite_inlines_args_struct_fields() -> None:
    """``arg.<field>[0]`` accesses are inlined to integer literals.

    TileLang packs scalar runtime args into a ``<kernel>_args_t`` struct
    (e.g. ``int n_elements[2]``) and references them as ``arg.n_elements[0]``
    in the body. ``mx.fast.metal_kernel`` does NOT carry that struct
    parameter into its rebuilt signature, so the rewrite must substitute
    each field access with a literal value supplied by the caller.
    """
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        _rewrite_tilelang_metal_to_mlx,
    )

    src = """
struct k_args_t {
  int n_elements[2];
  int gridDim_0[2];
};

kernel void k(
    device float* arg0 [[ buffer(0) ]],
    constant k_args_t& arg [[ buffer(1) ]]
) {
    if (some_id < arg.n_elements[0]) {
        arg0[some_id] = (float)arg.gridDim_0[0];
    }
}
"""
    out = _rewrite_tilelang_metal_to_mlx(
        src, args_struct_inline={"n_elements": 256, "gridDim_0": 1}
    )
    # The field accesses are substituted with their integer values.
    assert "256" in out
    assert "(float)1" in out
    # The args struct parameter declaration is dropped from the kernel
    # signature (MLX rebuilds the signature itself).
    assert "k_args_t" not in out.split("{", 2)[1]  # signature region
    # No ``arg.<field>`` accesses survive when an inline value is given.
    assert "arg.n_elements" not in out
    assert "arg.gridDim_0" not in out


def test_wrap_skips_cleanly_when_mlx_unavailable() -> None:
    """When mlx.fast.metal_kernel is missing, build() raises MLXRuntimeError.

    The adapter object itself can still be constructed (renaming is pure
    Python); only ``adapter.build()`` / ``adapter(...)`` triggers the
    import. We force the constructor probe to fail (by stashing a stub
    module that has no ``fast`` attribute) and verify the error path is
    a clean ``MLXRuntimeError`` -- not an opaque ``AttributeError`` deep
    inside MLX.

    Note: this test does NOT skip when mlx is importable; the stub
    overrides the real module for the duration of the test, exercising
    the unavailable path even on Mac GPU hosts.
    """
    import sys
    import types

    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        MLXRuntimeError,
        wrap_tilelang_metal_kernel,
    )

    src = """
kernel void k(
    device const float* A,
    device float* C
) {
    uint id = thread_position_in_grid.x;
    C[id] = A[id];
}
"""
    adapter = wrap_tilelang_metal_kernel(src, input_count=1, output_count=1)

    # Construction must succeed regardless of MLX availability.
    assert "inp0" in adapter.body
    assert "out0" in adapter.body

    # Force the lazy ``import mlx.core as mx`` inside ``_build_kernel``
    # to resolve to a stub module that has no ``fast`` attribute -- this
    # exercises the "MLX present but the constructor is unavailable on
    # this build" diagnostic path without relying on whether mlx itself
    # is installed.
    stub_core = types.ModuleType("mlx.core")
    # No ``fast`` attribute on the stub; getattr returns None below.
    saved_core = sys.modules.get("mlx.core")
    saved_mlx = sys.modules.get("mlx")
    try:
        # Always stub both ``mlx`` and ``mlx.core`` -- the import machinery
        # uses the parent's ``core`` attribute when resolving
        # ``import mlx.core as mx``, so even if the real ``mlx`` package is
        # cached we must replace its ``core`` to point at our stub.
        stub_pkg = types.ModuleType("mlx")
        stub_pkg.core = stub_core  # type: ignore[attr-defined]
        sys.modules["mlx"] = stub_pkg
        sys.modules["mlx.core"] = stub_core
        with pytest.raises(MLXRuntimeError, match="mx.fast.metal_kernel"):
            adapter.build()
    finally:
        if saved_core is None:
            sys.modules.pop("mlx.core", None)
        else:
            sys.modules["mlx.core"] = saved_core
        if saved_mlx is None:
            sys.modules.pop("mlx", None)
        else:
            sys.modules["mlx"] = saved_mlx
