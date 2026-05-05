"""Source-level probe for the retired FP8 scaled-matmul patch-B artifact."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PATCH_PATH = THIS_DIR / "0001-metal-fuse-fp8-scaled-matmul-scheduler.patch"
README_PATH = THIS_DIR / "README.md"
PATH_C_VECMAT_SOURCE = REPO_ROOT / "cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _canonical_body(source: str) -> str:
    start = source.index("def canonical_vecmat_runtime_body")
    end = source.index("def _grid_for_lowering", start)
    return source[start:end]


def test_patch_artifact_is_retired_documentation_not_applyable_diff() -> None:
    artifact = _read(PATCH_PATH)

    assert "intentionally not an applyable upstream patch" in artifact
    assert "Retired local artifact" in artifact
    assert "bogus patch-B story" in artifact
    assert "not scale-after-dot" in artifact
    assert "not a packed uint32/LUT\n  dot4 Metal" in artifact
    assert "not a 4-way K unroll" in artifact
    assert "not an M == 1 simd_sum vecmat" in artifact
    assert "There is no accepted upstream TileLang tileop scheduler" in artifact
    assert "no CUDA/H200 acceptance claim" in artifact
    assert "diff --git " not in artifact
    assert "--- a/" not in artifact
    assert "+++ b/" not in artifact
    assert "a_scaled = a_val * sa" not in artifact
    assert "b_scaled = b_val * sb" not in artifact
    assert "C_local[i, j] = C_local[i, j] + a_scaled * b_scaled" not in artifact


def test_readme_describes_honest_local_mlx_metal_path_c_story() -> None:
    readme = _read(README_PATH)

    assert "Path C patch **B**" in readme
    assert "corrected on 2026-05-04" in readme
    assert "no longer carries an applyable upstream patch" in readme
    assert "false performance story" in readme
    assert "algebraic scaled-operands form is retired" in readme
    assert "cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py" in readme
    assert "scale-after-dot" in readme
    assert "C[row] = sum * sx * sw" in readme
    assert "reinterpret_cast<device const uint*>" in readme
    assert "4-way K" in readme
    assert "packed uint32/LUT dot4" in readme
    assert "sum = simd_sum(sum)" in readme
    assert "not CUDA/H200 acceptance" in readme
    assert "no claim that an upstream\nTileLang tileop scheduler exists" in readme
    assert "documentation-only and intentionally not applyable" in readme
    assert "TILELANG_CHECKOUT` environment variable is deliberately ignored" in readme


def test_docs_and_artifact_do_not_reintroduce_scaled_operand_perf_claims() -> None:
    combined = _read(README_PATH) + "\n" + _read(PATCH_PATH)

    forbidden = [
        "a_scaled = a_val * sa",
        "b_scaled = b_val * sb",
        "C_local[i, j] = C_local[i, j] + a_scaled * b_scaled",
        "algebraically identical to a_val * b_val * sa * sb",
        "Same algebraic scaled-operands form",
        "is not a packed-dot4/simd_sum performance path",
        "apply-clean patch point",
        "git apply --check on\npatch B exits 0",
        "GemmMetalFP8ScaledScheduler",
        "def from_fp8_scaled_matmul",
        'if self.op_name == "fp8_scaled_matmul"',
    ]
    for marker in forbidden:
        assert marker not in combined


def test_current_path_c_vecmat_source_contains_scale_after_dot_and_dispatch_markers() -> None:
    assert REPO_ROOT.name == "cppmega.mlx"
    assert PATH_C_VECMAT_SOURCE.is_file()
    source = _read(PATH_C_VECMAT_SOURCE)
    body = _canonical_body(source)

    assert "torch" not in source.lower()
    assert "mps" not in source.lower()
    assert "import mlx.core as mx" in source
    assert "mx.fast.metal_kernel(" in source
    assert "stream=mx.gpu" in source
    assert "def _fp8_vecmat_kernel_for(" in source
    assert "def fp8_scaled_vecmat_path_c(" in source
    assert "source = canonical_vecmat_runtime_body(" in source
    assert 'input_names == ["A", "A_scale", "B", "B_scale"]' in source
    assert "outputs = cast(_msl_transform.MetalKernel, kernel)(" in source
    assert "_msl_transform.dispatch(" in source

    assert "float sum = 0.0f;" in body
    assert "float sx = float(A_scale[0]);" in body
    assert "float sw = float({scale_w_expr});" in body
    assert "C[row] = sum * sx * sw;" in body
    assert body.index("sum = simd_sum(sum);") < body.index("C[row] = sum * sx * sw;")
    assert "a_scaled" not in body
    assert "b_scaled" not in body


def test_current_path_c_vecmat_source_contains_packed_uint32_lut_dot4_and_4way_k_shape() -> None:
    source = _read(PATH_C_VECMAT_SOURCE)
    body = _canonical_body(source)

    assert "The default Metal lowering uses a TileLang intrinsic for packed uint32 e4m3" in source
    assert "dot4 decode plus ``tvm_thread_allreduce`` across K" in source
    assert "T.metal_fp8_e4m3_dot4(" in source
    assert "accum[0] += T.metal_fp8_e4m3_dot4(" in source
    assert "device const uint* A4 = reinterpret_cast<device const uint*>(A);" in body
    assert "device const uint* B4 = reinterpret_cast<device const uint*>(B + row_offset);" in body
    assert "uint K4 = {k_words}u;" in body
    assert "for (uint i = simd_lane; i < K4; i += 32u)" in body
    assert "uint px = A4[i];" in body
    assert "uint pw = B4[i];" in body
    assert "px & 0xFFu" in body
    assert "(px >> 8) & 0xFFu" in body
    assert "(px >> 16) & 0xFFu" in body
    assert "(px >> 24) & 0xFFu" in body
    assert body.count("fp8_e4m3fn_lut[") >= 8


def test_current_path_c_vecmat_source_contains_simd_sum_vecmat_specialization_markers() -> None:
    source = _read(PATH_C_VECMAT_SOURCE)
    body = _canonical_body(source)

    assert "M == 1" in source
    assert "B`` already transposed as ``(N, K)``" in source
    assert "one SIMD-group per output row" in source
    assert "uint row = gid / 32u;" in body
    assert "if (row >= {N}u) return;" in body
    assert "uint row_offset = row * {K}u;" in body
    assert "sum = simd_sum(sum);" in body
    assert "if (simd_lane == 0u)" in body
    assert "threadgroup = (128, 1, 1)" in source
    assert "grid = (((N * 32 + threadgroup[0] - 1) // threadgroup[0]) * threadgroup[0], 1, 1)" in source


def test_env_gated_apply_check_is_skip_gated_because_artifact_is_not_a_patch() -> None:
    if not os.environ.get("TILELANG_CHECKOUT"):
        pytest.skip("TILELANG_CHECKOUT unset; retired patch artifact has no apply check")
    pytest.skip("retired documentation-only artifact; no git apply --check is meaningful")
