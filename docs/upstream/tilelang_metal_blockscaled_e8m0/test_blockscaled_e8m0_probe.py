"""Source-level probe for the E8M0 block-scale TileLang upstream artifact."""

from __future__ import annotations

import re
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PATCH_PATH = THIS_DIR / "0001-tilelang-add-e8m0-blockscaled-layout-primitive.patch"
README_PATH = THIS_DIR / "README.md"
PATH_C_SOURCE = REPO_ROOT / "cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled_path_c.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_diff_markers(text: str) -> str:
    return re.sub(r"(?m)^[+ -]", "", text)


def test_patch_exports_e8m0_layout_dsl_surface() -> None:
    patch = _read(PATCH_PATH)
    normalized = _without_diff_markers(patch)

    assert "tilelang/language/blockscaled_layout.py" in patch
    assert "BlockScaledLayout" in normalized
    assert "def e8m0_to_float" in normalized
    assert "T.BlockScaledLayout.e8m0_k32()" in normalized
    assert "block_scale_layout=T.BlockScaledLayout.e8m0_k32()" in normalized
    assert "scale_format=\"e8m0_block_k32\"" in normalized
    assert "scale_block_size=32" in normalized
    assert "scale_axis=\"contracted_k\"" in normalized
    assert "logical_unswizzled_k_axis_blocks" in normalized


def test_patch_locks_concrete_e8m0_decode_semantics() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert re.search(r"byte\s*==\s*0", normalized)
    assert re.search(r"byte\s*==\s*0xFF", normalized)
    assert "byte - 127" in normalized
    assert "T.exp2" in normalized
    assert "T.if_then_else" in normalized


def test_patch_locks_contracted_k_shape_and_indexing_rules() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    assert "A_scale shape: (K / 32,)" in normalized
    assert "B_scale shape: (N, K / 32)" in normalized
    assert "broadcast (K / 32,)" in normalized
    assert re.search(r"kb\s*=\s*k\s*//\s*32", normalized)
    assert re.search(r"return\s+k\s*//\s*32", normalized)
    assert "K divisible by 32" in normalized
    assert "(K / 32,)" in normalized
    assert "(N, K / 32)" in normalized


def test_readme_matches_artifact_and_local_verification_contract() -> None:
    readme = _read(README_PATH)

    assert "Path C patch **C**" in readme
    assert "0001-tilelang-add-e8m0-blockscaled-layout-primitive.patch" in readme
    assert "T.BlockScaledLayout.e8m0_k32()" in readme
    assert "T.e8m0_to_float" in readme
    assert "scale_format = \"e8m0_block_k32\"" in readme
    assert "scale_block_size = 32" in readme
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
    assert "cuda" not in lower


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
