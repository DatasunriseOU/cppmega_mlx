"""Source-level probe for the E8M0 block-scale TileLang upstream artifact."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PATCH_PATH = THIS_DIR / "0001-tilelang-add-e8m0-blockscaled-layout-primitive.patch"
README_PATH = THIS_DIR / "README.md"
PATH_C_SOURCE = REPO_ROOT / "cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled_path_c.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_diff_markers(text: str) -> str:
    return re.sub(r"(?m)^[+ -]", "", text)


def _patch_file_section(path: str) -> str:
    patch = _read(PATCH_PATH)
    marker = f"diff --git a/{path} b/{path}"
    start = patch.index(marker)
    next_file = patch.find("\ndiff --git ", start + len(marker))
    if next_file == -1:
        return patch[start:]
    return patch[start:next_file]


def test_patch_exports_e8m0_layout_dsl_surface() -> None:
    patch = _read(PATCH_PATH)
    normalized = _without_diff_markers(patch)

    assert "tilelang/language/blockscaled_layout.py" in patch
    assert "BlockScaledLayout" in normalized
    assert "def e8m0_to_float" in normalized
    assert "T.BlockScaledLayout.e8m0_k32()" in normalized
    assert "layout = T.BlockScaledLayout.e8m0_k32()" in normalized
    assert "block_scale_layout=layout" in normalized
    assert "scale_format=\"e8m0_block_k32\"" in normalized
    assert "scale_block_size=32" in normalized
    assert "scale_format != E8M0_BLOCK_K32" in normalized
    assert "scale_block_size is None" in normalized
    assert "int(scale_block_size) != E8M0_BLOCK_SIZE" in normalized
    assert 'layout.scale_axis == "contracted_k"' in normalized
    assert "logical_unswizzled_k_axis_blocks" in normalized


def test_patch_rejects_partial_or_inconsistent_e8m0_metadata() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "if scale_format is None and scale_block_size is None:" in normalized
    assert "scale_format='e8m0_block_k32'" in normalized
    assert "scale_block_size=32" in normalized
    assert (
        "scale_format == E8M0_BLOCK_K32 or int(scale_block_size or 0) == E8M0_BLOCK_SIZE"
        not in normalized
    )


def test_patch_locks_concrete_e8m0_decode_semantics() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    section = _patch_file_section("tilelang/language/blockscaled_layout.py")
    section_normalized = _without_diff_markers(section)

    assert "def e8m0_to_float(bits):" in section_normalized
    assert "bits_i == T.int32(0)" in normalized
    assert "bits_i == T.int32(255)" in normalized
    assert "bits_i - T.int32(127)" in normalized
    assert "T.exp2" in normalized
    assert "T.if_then_else" in normalized


def test_patch_does_not_create_or_import_tileop_metal_quant() -> None:
    patch = _read(PATCH_PATH)

    assert "tilelang/tileop/metal_quant.py" not in patch
    assert "from tilelang.tileop.metal_quant" not in patch


def test_patch_targets_current_fp8_scaled_matmul_macro_prereq_shape() -> None:
    section = _patch_file_section("tilelang/language/fp8_op.py")
    normalized = _without_diff_markers(section)

    assert "def _fp8_scaled_matmul_macro(" in section
    assert "def _fp8_scaled_matmul_macro_trans_b(" in section
    assert "def _body(" not in section
    assert "C_out, layout)" in normalized
    assert "block_scale_layout=None" in normalized
    assert "block_scale_layout: BlockScaledLayout | None = None" in normalized
    assert "return _fp8_scaled_matmul_macro_trans_b(A_fp8, A_scale, B_fp8, B_scale, C_out, layout)" in normalized
    assert "return _fp8_scaled_matmul_macro(A_fp8, A_scale, B_fp8, B_scale, C_out, layout)" in normalized
    assert "_block_scale_value(A_scale, axis=\"A\", col=j, k=k)" in normalized
    assert "_block_scale_value(B_scale, axis=\"B\", col=j, k=k)" in normalized


def test_patch_does_not_reintroduce_false_fused_scale_perf_claims() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    forbidden = [
        "dequant + fused-scale FMA",
        "fuse the selected scales",
        "closes the audiohacking gap",
        "close the audiohacking gap",
        "3-6x gap",
        "3-6× gap",
    ]
    for marker in forbidden:
        assert marker not in normalized

    assert "dequant + algebraic scale form" in normalized
    assert "expose the algebraic scaled-operands form" in normalized


def test_patch_locks_contracted_k_shape_and_indexing_rules() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "layout.a_scale_shape(64) == (2,)" in normalized
    assert "layout.b_scale_shape(16, 64) == (16, 2)" in normalized
    assert "layout.broadcast_b_scale_shape(64) == (2,)" in normalized
    assert "kb = k // 32" in normalized
    assert "return k // 32" in normalized
    assert "K divisible by 32" in normalized
    assert "A_scale for e8m0_block_k32 must have shape" in normalized
    assert "B_scale for e8m0_block_k32 must have shape" in normalized
    assert "B_scale for e8m0_block_k32 must be broadcast (K / 32,)" in normalized


def test_readme_matches_artifact_and_local_verification_contract() -> None:
    readme = _read(README_PATH)

    assert "Path C patch **C**" in readme
    assert "0001-tilelang-add-e8m0-blockscaled-layout-primitive.patch" in readme
    assert "T.BlockScaledLayout.e8m0_k32()" in readme
    assert "T.e8m0_to_float" in readme
    assert "scale_format = \"e8m0_block_k32\"" in readme
    assert "scale_block_size = 32" in readme
    assert "TILELANG_CHECKOUT=/path/to/tilelang" in readme
    assert "apply --check" in readme
    assert "both fields required" in readme
    assert "A bare `block_size=32` is not enough" in readme
    assert "test_blockscaled_e8m0_probe.py" in readme
    assert "Mac M4 Max" in readme
    assert "MLX/Metal" in readme
    assert "H200" not in readme
    assert "CUDA acceptance" not in readme


def test_artifact_does_not_claim_cuda_or_h200_acceptance() -> None:
    combined = _read(PATCH_PATH) + "\n" + _read(README_PATH)
    lower = combined.lower()

    assert "h200" not in lower
    assert "cuda acceptance" not in lower


def test_patch_git_apply_check_when_tilelang_checkout_is_provided() -> None:
    checkout = os.environ.get("TILELANG_CHECKOUT")
    if not checkout:
        pytest.skip("set TILELANG_CHECKOUT to run the upstream git apply --check probe")

    checkout_path = Path(checkout)
    if not checkout_path.is_dir():
        pytest.fail(f"TILELANG_CHECKOUT does not exist or is not a directory: {checkout}")

    result = subprocess.run(
        ["git", "-C", str(checkout_path), "apply", "--check", str(PATCH_PATH)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_path_c_source_still_exposes_e8m0_layout_contract() -> None:
    source = _read(PATH_C_SOURCE)

    assert 'E8M0_BLOCK_SIZE = 32' in source
    assert 'E8M0_SCALE_FORMAT = "e8m0_block_k32"' in source
    assert 'E8M0_LAYOUT = "logical_unswizzled_k_axis_blocks"' in source
    assert "scale_axis" in source and '"contracted_k"' in source
    assert "scale_block_size" in source and "E8M0_BLOCK_SIZE" in source
    assert "A_scale: T.Tensor((_BSFP8_QKR_SCALE_BLOCKS,), \"uint8\")" in source
    assert "B_scale: T.Tensor((_BSFP8_QKR_N, _BSFP8_QKR_SCALE_BLOCKS), \"uint8\")" in source
    assert "B_scale[col, kb]" in source
    assert "B_scale must contain K/" in source and "broadcast bytes" in source


def test_current_path_c_source_still_indexes_scales_by_k32() -> None:
    source = _read(PATH_C_SOURCE)

    assert "kb = k // E8M0_BLOCK_SIZE" in source
    assert "kb = i // (E8M0_BLOCK_SIZE // 4)" in source
    assert "scale_begin = ko * (_BSFP8_BK // E8M0_BLOCK_SIZE)" in source
    assert "scale_end = scale_begin + (_BSFP8_BK // E8M0_BLOCK_SIZE)" in source
    assert "scale_format=E8M0_SCALE_FORMAT" in source
    assert "scale_block_size=E8M0_BLOCK_SIZE" in source


def test_current_path_c_source_still_requires_e8m0_decode_markers() -> None:
    source = _read(PATH_C_SOURCE)

    assert "from tilelang.tileop.metal_quant import e8m0_to_float" in source
    assert "e8m0_to_float(A_scale[kb])" in source
    assert "e8m0_to_float(B_scale[col, kb])" in source
    assert "e8m0_exp2" in source
    assert "e8m0_bias_subtract_127" in source
    assert "e8m0_sentinel_255" in source
    assert "e8m0_zero_sentinel" in source
