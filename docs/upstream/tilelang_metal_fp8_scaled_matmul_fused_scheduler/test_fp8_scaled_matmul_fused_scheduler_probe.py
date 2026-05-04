"""Source-level probe for the FP8 scaled-matmul macro patch artifact."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PATCH_PATH = THIS_DIR / "0001-metal-fuse-fp8-scaled-matmul-scheduler.patch"
README_PATH = THIS_DIR / "README.md"
PATH_C_VECMAT_SOURCE = REPO_ROOT / "cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_diff_markers(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("-"):
            continue
        if line.startswith(("+", " ")):
            lines.append(line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def _patch_section(patch: str, path: str) -> str:
    marker = f"diff --git a/{path} b/{path}"
    start_index = patch.index(marker)
    next_diff_index = patch.find("\ndiff --git ", start_index + 1)
    return patch[start_index:] if next_diff_index == -1 else patch[start_index:next_diff_index]


def _section(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def _patch_line_context(path: Path, line_no: int, *, radius: int = 4) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    return "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))


def test_patch_artifact_targets_real_macro_surface_only() -> None:
    patch = _read(PATCH_PATH)

    assert "*** End Patch" not in patch
    assert "diff --git a/tilelang/language/fp8_op.py b/tilelang/language/fp8_op.py" in patch
    assert patch.count("diff --git ") == 1
    assert "diff --git a/tilelang/tileop/gemm" not in patch
    assert "+++ b/tilelang/tileop/gemm" not in patch
    assert "class GemmMetalFP8ScaledScheduler" not in patch
    assert "+from .gemm_metal_fp8_scaled import GemmMetalFP8ScaledScheduler" not in patch
    assert "def from_fp8_scaled_matmul" not in patch
    assert "testing/python/metal/test_fp8_scaled_matmul_fused_scheduler.py" not in patch

    fp8_section = _patch_section(patch, "tilelang/language/fp8_op.py")
    assert "--- a/tilelang/language/fp8_op.py" in fp8_section
    assert "+++ b/tilelang/language/fp8_op.py" in fp8_section
    assert "_fp8_scaled_matmul_macro" in fp8_section
    assert "_fp8_scaled_matmul_macro_trans_b" in fp8_section
    assert "def _body" not in fp8_section


def test_readme_describes_macro_level_prototype() -> None:
    readme = _read(README_PATH)

    assert "Path C patch **B**" in readme
    assert "0001-metal-fuse-fp8-scaled-matmul-scheduler.patch" in readme
    assert "test_fp8_scaled_matmul_fused_scheduler_probe.py" in readme
    assert "macro-level" in readme
    assert "tilelang/language/fp8_op.py" in readme
    assert "_fp8_scaled_matmul_macro" in readme
    assert "_fp8_scaled_matmul_macro_trans_b" in readme
    assert "TILELANG_CHECKOUT" in readme
    assert "not CUDA/H200 acceptance" in readme


def test_patch_reauthors_old_scheduler_claims_away() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))

    forbidden = [
        "diff --git a/tilelang/tileop/gemm",
        "+++ b/tilelang/tileop/gemm",
        "class GemmMetalFP8ScaledScheduler",
        'if self.op_name == "fp8_scaled_matmul"',
        "from_fp8_scaled_matmul",
        "GemmMetalFP8ScaledScheduler(self).lower(",
        "def _body",
    ]
    for marker in forbidden:
        assert marker not in normalized

    assert "real current TileLang surface" in normalized
    assert "tilelang/language/fp8_op.py" in normalized
    assert "There is no tilelang/tileop/gemm scheduler" in normalized


def test_patch_fuses_scale_application_inside_default_macro_k_loop() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    macro_body = _section(
        normalized,
        "def _fp8_scaled_matmul_macro(A_fp8, A_scale, B_fp8, B_scale, C_local):",
        "def _fp8_scaled_matmul_macro_trans_b",
    )
    k_loop = macro_body[macro_body.index("for k in T.serial(K_dim):") :]

    assert "a_val = T.cast(A_fp8[i, k], \"float32\")" in k_loop
    assert "b_val = T.cast(B_fp8[k, j], \"float32\")" in k_loop
    assert "sa = A_scale[0] if sa_size == 1 else A_scale[i]" in k_loop
    assert "sb = B_scale[0] if sb_size == 1 else B_scale[j]" in k_loop
    assert "a_scaled = a_val * sa" in k_loop
    assert "b_scaled = b_val * sb" in k_loop
    assert "C_local[i, j] = C_local[i, j] + a_scaled * b_scaled" in k_loop
    assert "C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb" not in k_loop
    assert "C_local[i, j] = C_local[i, j] *" not in macro_body
    assert "C_local[i, j] *= " not in macro_body


def test_patch_fuses_scale_application_inside_transpose_b_macro_k_loop() -> None:
    normalized = _without_diff_markers(_read(PATCH_PATH))
    macro_body = _section(
        normalized,
        "def _fp8_scaled_matmul_macro_trans_b(A_fp8, A_scale, B_fp8, B_scale, C_local):",
        "def fp8_scaled_matmul(",
    )
    k_loop = macro_body[macro_body.index("for k in T.serial(K_dim):") :]

    assert "b_val = T.cast(B_fp8[j, k], \"float32\")" in k_loop
    assert "sa = A_scale[0] if sa_size == 1 else A_scale[i]" in k_loop
    assert "sb = B_scale[0] if sb_size == 1 else B_scale[j]" in k_loop
    assert "a_scaled = a_val * sa" in k_loop
    assert "b_scaled = b_val * sb" in k_loop
    assert "C_local[i, j] = C_local[i, j] + a_scaled * b_scaled" in k_loop
    assert "M == 1 Path C vecmat shape" in macro_body
    assert "C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb" not in k_loop
    assert "C_local[i, j] = C_local[i, j] *" not in macro_body
    assert "C_local[i, j] *= " not in macro_body


def test_current_path_c_vecmat_source_remains_metal_runtime_reference() -> None:
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


def test_env_gated_patch_applies_to_tilelang_checkout() -> None:
    checkout = os.environ.get("TILELANG_CHECKOUT")
    if not checkout:
        pytest.skip("set TILELANG_CHECKOUT to run git apply --check against a TileLang checkout")

    checkout_path = Path(checkout).expanduser().resolve()
    if not (checkout_path / ".git").exists():
        pytest.fail(f"TILELANG_CHECKOUT is not a git checkout: {checkout_path}")
    if not (checkout_path / "tilelang/language/fp8_op.py").exists():
        pytest.fail(
            "TILELANG_CHECKOUT is missing tilelang/language/fp8_op.py; apply the "
            "tilelang_metal_fp8_scaled_matmul prereq before checking patch B"
        )

    result = subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)],
        cwd=checkout_path,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr or result.stdout
        corrupt = re.search(r"corrupt patch at line (\d+)", details)
        context = ""
        if corrupt:
            line_no = int(corrupt.group(1))
            context = (
                "\nPatch parser failed before hunk matching; this means the "
                "patch artifact is malformed, not merely missing checkout prereqs.\n"
                f"Patch context around line {line_no}:\n{_patch_line_context(PATCH_PATH, line_no)}"
            )
        pytest.fail(
            "patch B git apply --check failed\n"
            f"checkout: {checkout_path}\n"
            f"command: git apply --check {PATCH_PATH}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
            f"{context}"
        )
