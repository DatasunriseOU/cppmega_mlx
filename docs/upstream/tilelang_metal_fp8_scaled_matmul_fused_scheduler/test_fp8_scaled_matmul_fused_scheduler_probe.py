"""Source-level probe for the FP8 scaled-matmul fused scheduler artifact."""

from __future__ import annotations

import re
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PATCH_PATH = THIS_DIR / "0001-metal-fuse-fp8-scaled-matmul-scheduler.patch"
README_PATH = THIS_DIR / "README.md"
PATH_C_VECMAT_SOURCE = REPO_ROOT / "cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_diff_markers(text: str) -> str:
    return re.sub(r"(?m)^[+ -]", "", text)


def _section(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def _added_line_count_for_new_file(patch: str, path: str) -> int:
    marker = f"diff --git a/{path} b/{path}"
    start_index = patch.index(marker)
    next_diff_index = patch.find("\ndiff --git ", start_index + 1)
    section = patch[start_index:] if next_diff_index == -1 else patch[start_index:next_diff_index]
    return sum(1 for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))


def test_patch_artifact_is_clean_and_self_consistent() -> None:
    patch = _read(PATCH_PATH)

    assert "*** End Patch" not in patch
    assert "ScaleSpec" not in patch
    assert "@@ -0,0 +1,428 @@" not in patch
    assert "@@ -0,0 +1,211 @@" not in patch

    scheduler_count = _added_line_count_for_new_file(
        patch, "tilelang/tileop/gemm/gemm_metal_fp8_scaled.py"
    )
    scheduler_hunk = f"@@ -0,0 +1,{scheduler_count} @@"
    assert scheduler_hunk in patch

    test_count = _added_line_count_for_new_file(
        patch, "testing/python/metal/test_fp8_scaled_matmul_fused_scheduler.py"
    )
    test_hunk = f"@@ -0,0 +1,{test_count} @@"
    assert test_hunk in patch


def test_readme_points_at_source_level_local_probe() -> None:
    readme = _read(README_PATH)

    assert "Path C patch **B**" in readme
    assert "0001-metal-fuse-fp8-scaled-matmul-scheduler.patch" in readme
    assert "test_fp8_scaled_matmul_fused_scheduler_probe.py" in readme
    assert "source-level" in readme
    assert "contracted-K loop" in readme
    assert "packed `uint32` FP8 loads" in readme
    assert "LUT-backed e4m3 dot4 decode" in readme
    assert "`simd_sum`" in readme
    assert "per-tensor and per-row scale loads" in readme


def test_patch_exports_tilelang_metal_fp8_scaled_scheduler_surface() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "class GemmMetalFP8ScaledScheduler" in normalized
    assert 'if self.op_name == "fp8_scaled_matmul" and self.target.kind.name == "metal"' in normalized
    assert "GemmMetalFP8ScaledScheduler(self).lower(" in normalized
    assert "def from_fp8_scaled_matmul(cls, op):" in normalized
    assert "schedule.A_scale = op.args[1]" in normalized
    assert "schedule.B_scale = op.args[3]" in normalized
    assert 'schedule.scale_format = op.attrs.get("scale_format", "per_tensor_or_row")' in normalized
    assert "schedule.scale_block_size = int(op.attrs.get(\"scale_block_size\", 0) or 0)" in normalized
    assert "schedule.attrs.update(" in normalized
    assert "A_scale=schedule.A_scale" in normalized
    assert "B_scale=schedule.B_scale" in normalized


def test_patch_declares_required_msl_markers_for_path_c_shape() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "METAL_FP8_SCALED_MSL_MARKERS" in normalized
    assert '"packed_load": "reinterpret_cast<device const uint*>"' in normalized
    assert '"dot4": "__tvm_fp8_e4m3_dot4_packed"' in normalized
    assert '"simd_reduction": "simd_sum"' in normalized
    assert 'E8M0_BLOCK_K32 = "e8m0_block_k32"' in normalized


def test_patch_keeps_scale_selection_keyed_by_contract_k() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    scale_expr = _section(normalized, "def _scale_expr_for(", "def _make_scale_reader(")

    assert "k: Any," in scale_expr
    assert "block = k // 32" in scale_expr
    assert "return T.e8m0_to_float(scale[row, block])" in scale_expr
    assert "return T.e8m0_to_float(scale[col, block])" in scale_expr
    assert 'if axis == "A":' in scale_expr
    assert 'return T.cast(scale[row], "float32")' in scale_expr
    assert 'return T.cast(scale[col], "float32")' in scale_expr


def test_patch_uses_macro_scale_readers_before_entering_tir_body() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    macro_body = _section(normalized, "def _make_scale_reader(", "def _emit_scaled_inner_product(")
    generic_prebody = _section(normalized, "def _emit_scaled_inner_product(", "@T.prim_func")
    vecmat_prebody = _section(normalized, "def _emit_vecmat_reduce(", "@T.prim_func")

    assert "@T.macro" in macro_body
    assert "def scale_at(scale, row, col, k):" in macro_body
    assert 'axis="A"' not in macro_body
    assert 'axis="B"' not in macro_body
    assert generic_prebody.count("self._make_scale_reader(") == 2
    assert vecmat_prebody.count("self._make_scale_reader(") == 2
    assert 'axis="A"' in generic_prebody
    assert 'axis="B"' in generic_prebody
    assert 'axis="A"' in vecmat_prebody
    assert 'axis="B"' in vecmat_prebody


def test_patch_fuses_scale_application_inside_generic_k_loop() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    body = _section(normalized, "def _emit_scaled_inner_product(", "return fp8_scaled_matmul_fused")
    k_loop = _section(body, "for kw in T.serial(K4):", "C_buf[row, col] = acc[0]")

    assert "product = T.metal_fp8_e4m3_dot4(" in k_loop
    assert "b_dot_offset = col * K4 + kw" in k_loop
    assert 'T.access_ptr(B_buf[col, 0], "r", extent=K4 * 4)' in k_loop
    assert "transpose_B else" not in body
    assert "B_buf[0, col]" not in body
    assert "self._scale_expr_for(" not in body
    assert "sa = a_scale_at(A_scale_buf, row, col, k)" in k_loop
    assert "sb = b_scale_at(B_scale_buf, row, col, k)" in k_loop
    assert "acc[0] += product * sa * sb" in k_loop
    assert "C_buf[row, col] = acc[0] *" not in body
    assert "C *= scale" not in body


def test_patch_fuses_scale_application_inside_vecmat_reduction_k_loop() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    body = _section(normalized, "def _emit_vecmat_reduce(", "return fp8_scaled_vecmat_fused")
    k_loop = _section(body, "for kw in T.serial(T.ceildiv(K4, 32)):", "with T.attr(")

    assert "word = kw * 32 + kr" in k_loop
    assert "k = word * 4" in k_loop
    assert "product = T.metal_fp8_e4m3_dot4(" in k_loop
    assert "self._scale_expr_for(" not in body
    assert "sa = a_scale_at(A_scale_buf, 0, col, k)" in k_loop
    assert "sb = b_scale_at(B_scale_buf, 0, col, k)" in k_loop
    assert "acc[0] += product * sa * sb" in k_loop
    assert "C_buf[0, col] = red[0] *" not in body


def test_current_path_c_vecmat_source_still_has_patch_b_target_markers() -> None:
    assert REPO_ROOT.name == "cppmega.mlx"
    assert PATH_C_VECMAT_SOURCE.is_file()
    source = _read(PATH_C_VECMAT_SOURCE)

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
    assert "def canonical_vecmat_runtime_body" in source
    assert "device const uint* A4 = reinterpret_cast<device const uint*>(A);" in source
    assert "device const uint* B4 = reinterpret_cast<device const uint*>(B + row_offset);" in source
    assert "fp8_e4m3fn_lut[px & 0xFFu]" in source
    assert "fp8_e4m3fn_lut[(px >> 24) & 0xFFu]" in source
    assert "sum = simd_sum(sum);" in source
    assert 'scale_w_expr = "B_scale[row]" if scale_w_per_row else "B_scale[0]"' in source
    assert "float sx = float(A_scale[0]);" in source
    assert "float sw = float({scale_w_expr});" in source
    assert "C[row] = sum * sx * sw;" in source
