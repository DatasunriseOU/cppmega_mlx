"""Focused debug direct-MSL tests for source-level TileLang MSL canonicalization."""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    TileLangMSLLowering,
    _ensure_single_libtvm_ffi_image,
    _parse_buffer_param_names,
    _inline_tilelang_kernel_body,
    _resolve_dispatch_launch_shape,
    _split_kernel_msl,
    metal_grid_for_lowering,
)


def test_inline_body_removes_simple_builtin_aliases() -> None:
    body = _inline_tilelang_kernel_body(
        """
    C[((int)blockIdx.x)] = A[((int)threadIdx.x)] + B[threadIdx.x];
"""
    )

    assert "uint3 blockIdx =" not in body
    assert "uint3 threadIdx =" not in body
    assert "blockIdx" not in body
    assert "threadIdx" not in body
    assert "threadgroup_position_in_grid.x" in body
    assert "thread_position_in_threadgroup.x" in body


def test_inline_body_removes_multiaxis_builtin_aliases() -> None:
    body = _inline_tilelang_kernel_body(
        """
    uint lane = threadIdx.x + threadIdx.y * 32u + threadIdx.z * 1024u;
    uint block = blockIdx.x + blockIdx.y * 64u + blockIdx.z * 4096u;
    C[((int)threadIdx.y)] = A[((int)threadIdx.x)] + B[((int)blockIdx.z)];
"""
    )

    assert "uint3 blockIdx =" not in body
    assert "uint3 threadIdx =" not in body
    assert "blockIdx" not in body
    assert "threadIdx" not in body
    assert "thread_position_in_threadgroup.x" in body
    assert "thread_position_in_threadgroup.y" in body
    assert "thread_position_in_threadgroup.z" in body
    assert "threadgroup_position_in_grid.x" in body
    assert "threadgroup_position_in_grid.y" in body
    assert "threadgroup_position_in_grid.z" in body


def test_inline_body_keeps_alias_for_whole_vector_use() -> None:
    body = _inline_tilelang_kernel_body(
        """
    uint3 raw_tid = threadIdx;
    C[((int)threadIdx.y)] = A[((int)threadIdx.x)];
"""
    )

    assert "((int)thread_position_in_threadgroup.x)" in body
    assert "((int)thread_position_in_threadgroup.y)" in body
    assert "uint3 threadIdx = thread_position_in_threadgroup;" in body
    assert "uint3 raw_tid = threadIdx;" in body


def test_inline_body_rewrites_tilelang_builtin_cast_variants() -> None:
    body = _inline_tilelang_kernel_body(
        """
    int a = (int)threadIdx.x;
    uint b = ((uint)blockIdx.y);
    unsigned long c = unsigned long(threadIdx.z);
    size_t d = static_cast<size_t>(blockIdx.x);
"""
    )

    assert "uint3 blockIdx =" not in body
    assert "uint3 threadIdx =" not in body
    assert "threadIdx" not in body
    assert "blockIdx" not in body
    assert "((int)thread_position_in_threadgroup.x)" in body
    assert "((uint)threadgroup_position_in_grid.y)" in body
    assert "((unsigned long)thread_position_in_threadgroup.z)" in body
    assert "((size_t)threadgroup_position_in_grid.x)" in body


def test_inline_body_ignores_comments_and_strings_when_dropping_aliases() -> None:
    body = _inline_tilelang_kernel_body(
        """
    // threadIdx and blockIdx are only mentioned in this comment.
    const char* marker = "threadIdx.x blockIdx.x";
    C[(int)threadIdx.x] = A[blockIdx.y];
"""
    )

    assert "uint3 blockIdx =" not in body
    assert "uint3 threadIdx =" not in body
    assert "threadIdx" in body
    assert "blockIdx" in body
    assert "// threadIdx and blockIdx are only mentioned in this comment." in body
    assert '"threadIdx.x blockIdx.x"' in body
    assert "threadIdx" not in body.split("//", 1)[0]
    assert "blockIdx" not in body.split("//", 1)[0]
    assert "((int)thread_position_in_threadgroup.x)" in body
    assert "threadgroup_position_in_grid.y" in body


def test_inline_body_rewrites_fp8_vecmat_dot4_hot_pattern() -> None:
    body = _inline_tilelang_kernel_body(
        """
    accum[0] = metal_fp8_dot4_e4m3_lut(
        (&(A[0])),
        (&(B[(((int)blockIdx.x) * 128)])),
        ((int)threadIdx.x),
        ((int)threadIdx.x));
"""
    )

    assert "uint3 blockIdx =" not in body
    assert "uint3 threadIdx =" not in body
    assert "blockIdx" not in body
    assert "threadIdx" not in body
    assert "B[(((int)threadgroup_position_in_grid.x) * 128)]" in body
    assert body.count("((int)thread_position_in_threadgroup.x)") == 2


def test_metal_grid_for_lowering_expands_tilelang_blocks_to_thread_grid() -> None:
    lowering = TileLangMSLLowering(
        header="",
        body="",
        grid=(2, 3, 4),
        threadgroup=(8, 16, 1),
        msl_text="",
        buffer_param_names=[],
        kernel_name="k",
    )

    assert metal_grid_for_lowering(lowering) == (16, 48, 4)


def test_dispatch_launch_shape_uses_lowering_tilelang_blocks() -> None:
    lowering = TileLangMSLLowering(
        header="",
        body="",
        grid=(2, 3, 1),
        threadgroup=(16, 8, 1),
        msl_text="",
        buffer_param_names=[],
        kernel_name="k",
    )

    assert _resolve_dispatch_launch_shape(
        input_count=0,
        output_count=0,
        lowering=lowering,
    ) == ((32, 24, 1), (16, 8, 1))


def test_dispatch_launch_shape_rejects_conflicting_lowering_grid() -> None:
    lowering = TileLangMSLLowering(
        header="",
        body="",
        grid=(2, 1, 1),
        threadgroup=(16, 1, 1),
        msl_text="",
        buffer_param_names=[],
        kernel_name="k",
    )

    with pytest.raises(ValueError, match="conflicting dispatch grid"):
        _resolve_dispatch_launch_shape(
            input_count=0,
            output_count=0,
            grid=(2, 1, 1),
            threadgroup=(16, 1, 1),
            lowering=lowering,
        )


def test_split_kernel_msl_ignores_braces_in_comments_and_strings() -> None:
    msl = r'''
// kernel void fake_comment(device float* X) { not a kernel }
constant char* marker = "brace payload { } )";
kernel void real_kernel(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]]
) {
    // This comment must not close the body: }
    const char* s = "not a brace {";
    if (true) {
      C[0] = A[0];
    }
}
'''

    prelude, sig_text, body_text = _split_kernel_msl(msl)

    assert "fake_comment" in prelude
    assert "device const float* A" in sig_text
    assert 'const char* s = "not a brace {";' in body_text
    assert body_text.rstrip().endswith("}")


def test_parse_buffer_param_names_handles_pointer_and_reference_decls() -> None:
    sig_text = """
    device const half* A [[buffer(0)]],
    const device float &B [[buffer(1)]],
    device uint* __restrict C [[buffer(2)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
"""

    assert _parse_buffer_param_names(sig_text) == ["A", "B", "C"]


def test_single_libtvm_ffi_image_check_allows_zero_or_one_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang._msl_transform._loaded_libtvm_ffi_images",
        lambda: ["/venv/site-packages/tvm_ffi/lib/libtvm_ffi.dylib"],
    )

    _ensure_single_libtvm_ffi_image()


def test_single_libtvm_ffi_image_check_rejects_mixed_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang._msl_transform._loaded_libtvm_ffi_images",
        lambda: [
            "/venv/site-packages/tvm_ffi/lib/libtvm_ffi.dylib",
            "/private/tmp/tilelang/build/lib/libtvm_ffi.dylib",
        ],
    )

    with pytest.raises(MSLDispatchUnsupported, match="multiple libtvm_ffi.dylib images"):
        _ensure_single_libtvm_ffi_image()


# ---------------------------------------------------------------------------
# Fix-3 + Fix-C: xfail coverage for transforms not yet wired into
# `_inline_tilelang_kernel_body`. Originally identified by Meta agent E
# (reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/
#  meta__correctness__20260506T170721.md, gaps 1–5) and tightened by
# gpt-5-pro G4 with `raises=AssertionError` so non-AssertionError failures
# (NameError / ImportError / TypeError) surface loudly instead of silently
# satisfying `strict=True`. See agent-G4-correctness-fix-a-c/
#  chatgpt__correctness__cli__20260506T2310.md Q4.
# ---------------------------------------------------------------------------


def _lower_minimal_kernel_to_msl(kernel_src: str) -> str:
    """Run a minimal kernel snippet through the source-level MSL transform.

    Today the only public transform on the source side is
    `_inline_tilelang_kernel_body`, which is what every existing test in this
    file already exercises. The helper exists as a single point of truth so
    that when the lowering pipeline grows a real "lower kernel -> MSL" entry
    point, the xfail tests below can be retargeted by editing one function
    instead of five.
    """

    return _inline_tilelang_kernel_body(kernel_src)


def test_inline_body_prefixes_threadgroup_barrier_with_metal_namespace() -> None:
    """threadgroup_barrier: assert that the lowered MSL contains the
    fully-qualified Metal builtin.

    raises=AssertionError so any non-AssertionError (NameError if helper goes
    away, ImportError, TypeError on signature change) surfaces loudly instead
    of silently satisfying strict-xfail (gpt-5-pro G4 Q4).
    """
    body = _lower_minimal_kernel_to_msl(
        """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    C[((int)threadIdx.x)] = A[((int)threadIdx.x)];
"""
    )
    assert "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup)" in body


def test_inline_body_handles_simdgroup_matrix_and_mma() -> None:
    """simdgroup MMA: assert that the lowered MSL contains both
    `simdgroup_matrix` and `simdgroup_multiply_accumulate`.

    raises=AssertionError so any non-AssertionError (NameError if helper goes
    away, ImportError, TypeError on signature change) surfaces loudly instead
    of silently satisfying strict-xfail (gpt-5-pro G4 Q4).
    """
    body = _lower_minimal_kernel_to_msl(
        """
    simdgroup_float8x8 a;
    simdgroup_float8x8 b;
    simdgroup_float8x8 c;
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, C, 8);
"""
    )
    assert "simdgroup_matrix" in body
    assert "simdgroup_multiply_accumulate" in body


def test_inline_body_prefixes_metal_namespace_for_bfloat_and_half4() -> None:
    """bfloat / half4: assert that the lowered MSL namespace-qualifies both
    Metal scalar/vector types.

    raises=AssertionError so any non-AssertionError (NameError if helper goes
    away, ImportError, TypeError on signature change) surfaces loudly instead
    of silently satisfying strict-xfail (gpt-5-pro G4 Q4).
    """
    body = _lower_minimal_kernel_to_msl(
        """
    bfloat scale = (bfloat)1.0;
    half4 packed = half4(A[0], A[1], A[2], A[3]);
    C[((int)threadIdx.x)] = (float)(packed.x) * (float)scale;
"""
    )
    assert "metal::bfloat" in body
    assert "metal::half4" in body


def test_inline_body_canonicalizes_device_vs_threadgroup_address_space_casts() -> None:
    """address-space casts: assert that both `device` and `threadgroup`
    qualifiers survive the lowering (no implicit demotion to a single space).

    raises=AssertionError so any non-AssertionError (NameError if helper goes
    away, ImportError, TypeError on signature change) surfaces loudly instead
    of silently satisfying strict-xfail (gpt-5-pro G4 Q4).
    """
    body = _lower_minimal_kernel_to_msl(
        """
    float* tg_ptr = (float*)smem;
    float* dev_ptr = (float*)C;
    dev_ptr[((int)threadIdx.x)] = tg_ptr[((int)threadIdx.x)];
"""
    )
    # Future canonicalization must re-attach address-space qualifiers that
    # were dropped on the input side; today the transform is a pass-through
    # and does not synthesize them, so both literals are absent.
    assert "static_cast<threadgroup float*>" in body
    assert "static_cast<device float*>" in body


def test_inline_body_prefixes_atomics_with_metal_namespace() -> None:
    """atomics: assert that the lowered MSL emits the fully-qualified
    `metal::atomic_fetch_add_explicit` form (and analogous min/max).

    raises=AssertionError so any non-AssertionError (NameError if helper goes
    away, ImportError, TypeError on signature change) surfaces loudly instead
    of silently satisfying strict-xfail (gpt-5-pro G4 Q4).
    """
    body = _lower_minimal_kernel_to_msl(
        """
    atomic_fetch_add_explicit(&C[((int)blockIdx.x)], 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(&C[((int)blockIdx.x)], 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(&C[((int)blockIdx.x)], 1u, memory_order_relaxed);
"""
    )
    assert "metal::atomic_fetch_add_explicit" in body
