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


def test_patch_declares_required_msl_markers_for_path_c_shape() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "METAL_FP8_SCALED_MSL_MARKERS" in normalized
    assert '"packed_load": "reinterpret_cast<device const uint*>"' in normalized
    assert '"dot4": "__tvm_fp8_e4m3_dot4_packed"' in normalized
    assert '"simd_reduction": "simd_sum"' in normalized
    assert 'E8M0_BLOCK_K32 = "e8m0_block_k32"' in normalized


def test_patch_keeps_scale_selection_keyed_by_contract_k() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    scale_expr = _section(normalized, "def _scale_expr_for(", "def _emit_scaled_inner_product(")

    assert "k: Any," in scale_expr
    assert "block = k // 32" in scale_expr
    assert "return T.e8m0_to_float(scale[col, block])" in scale_expr
    assert 'if axis == "A":' in scale_expr
    assert 'return T.cast(scale[row], "float32")' in scale_expr
    assert 'return T.cast(scale[col], "float32")' in scale_expr


def test_patch_fuses_scale_application_inside_generic_k_loop() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    body = _section(normalized, "def _emit_scaled_inner_product(", "return fp8_scaled_matmul_fused")
    k_loop = _section(body, "for kw in T.serial(K4):", "C_buf[row, col] = acc[0]")

    assert "product = T.metal_fp8_e4m3_dot4(" in k_loop
    assert "sa = self._scale_expr_for(" in k_loop
    assert "sb = self._scale_expr_for(" in k_loop
    assert k_loop.count("k=k,") == 2
    assert 'axis="A"' in k_loop
    assert 'axis="B"' in k_loop
    assert "scale_format=scale_format" in k_loop
    assert "scale_block_size=scale_block_size" in k_loop
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
    assert "sa = self._scale_expr_for(" in k_loop
    assert "sb = self._scale_expr_for(" in k_loop
    assert k_loop.count("k=k,") == 2
    assert 'axis="A"' in k_loop
    assert 'axis="B"' in k_loop
    assert "acc[0] += product * sa * sb" in k_loop
    assert "C_buf[0, col] = red[0] *" not in body


def test_current_path_c_vecmat_source_still_has_patch_b_target_markers() -> None:
    source = _read(PATH_C_VECMAT_SOURCE)

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
