from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "lint_mlx.py"


def run_lint(*paths: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(path) for path in paths)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_lint_fails_on_mx_array_scalar_literal(tmp_path: Path) -> None:
    source = tmp_path / "bad.py"
    source.write_text(
        "\n".join(
            [
                "import mlx.core as mx",
                "",
                "alpha = mx.array(1.0)",
                "beta = mx.array(2)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert result.stderr == ""
    assert f"{source}:3: MLX001" in result.stdout
    assert f"{source}:4: MLX001" in result.stdout


def test_lint_passes_on_clean_non_scalar_inputs(tmp_path: Path) -> None:
    source = tmp_path / "clean.py"
    source.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import mlx.core as mx",
                "",
                "values = [1.0, 2.0]",
                "a = mx.array(values)",
                "b = mx.array([1.0, 2.0])",
                "c = mx.array((1.0, 2.0))",
                "d = mx.array(np.array(1.0))",
                "e = mx.array(1.0, dtype=mx.float32)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_recurses_directories_and_respects_mlx_aliases(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    good = package / "good.py"
    bad = package / "bad.py"
    good.write_text(
        "from mlx import core as mx\nsafe = mx.array([1])\n",
        encoding="utf-8",
    )
    bad.write_text(
        "from mlx import core as mx\nunsafe = mx.array(True)\n",
        encoding="utf-8",
    )

    result = run_lint(package)

    assert result.returncode == 1
    assert f"{bad}:2: MLX001" in result.stdout
    assert str(good) not in result.stdout


def test_lint_blocks_ad_hoc_custom_metal_kernel_construction(tmp_path: Path) -> None:
    source = tmp_path / "bad_kernel.py"
    source.write_text(
        "\n".join(
            [
                "import mlx.core as mx",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='ad_hoc',",
                "    input_names=['x'],",
                "    output_names=['y'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert result.stderr == ""
    assert f"{source}:3: MLX002" in result.stdout
    assert "allowlisted legacy/debug direct-MSL modules" in result.stdout


def test_lint_rejects_raw_metal_kernel_even_in_owned_policy_seam(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "cppmega_mlx" / "kernels" / "metal_ops.py"
    owned.parent.mkdir(parents=True)
    owned.write_text(
        "\n".join(
            [
                "from mlx import core as mx",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='owned',",
                "    input_names=['x'],",
                "    output_names=['y'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(owned)

    assert result.returncode == 1
    assert f"{owned}:3: MLX002" in result.stdout
    assert result.stderr == ""


def test_lint_allows_marked_legacy_debug_direct_msl_module(tmp_path: Path) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "mamba3.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                '"""Legacy/debug Path B direct-MSL module."""',
                "import mlx.core as mx",
                "from cppmega_mlx.nn._tilelang import _msl_transform",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='legacy_debug',",
                "    input_names=['x'],",
                "    output_names=['y'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
                "",
                "def launch(x):",
                "    return _msl_transform.dispatch(",
                "        kernel,",
                "        inputs=[x],",
                "        output_shapes=[x.shape],",
                "        output_dtypes=[x.dtype],",
                "    )",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_blocks_msl_transform_dispatch_outside_legacy_debug_modules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "new_path_c.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from cppmega_mlx.nn._tilelang import _msl_transform",
                "",
                "def launch(kernel, x):",
                "    return _msl_transform.dispatch(",
                "        kernel,",
                "        inputs=[x],",
                "        output_shapes=[x.shape],",
                "        output_dtypes=[x.dtype],",
                "    )",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:4: MLX005" in result.stdout
    assert "native TileLang/TVM-FFI" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_imported_msl_transform_dispatch_alias(tmp_path: Path) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "new_alias.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from cppmega_mlx.nn._tilelang._msl_transform import dispatch as msl_dispatch",
                "",
                "def launch(kernel, x):",
                "    return msl_dispatch(",
                "        kernel,",
                "        inputs=[x],",
                "        output_shapes=[x.shape],",
                "        output_dtypes=[x.dtype],",
                "    )",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:4: MLX005" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_direct_native_tvm_ffi_bridge_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "bad_bridge.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from tilelang.contrib.mlx_tvm_ffi import owner_output_buffer",
                "",
                "def launch(shape, dtype):",
                "    return owner_output_buffer(shape, dtype)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:1: MLX007" in result.stdout
    assert "tilelang -> tvm -> tvm-ffi adapter" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_model_level_backend_intrinsic_strings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "models" / "bad_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def lowered_source() -> str:",
                "    return 'tir.metal.simd_sum(acc)'",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX008", source)

    assert result.returncode == 1
    assert f"{source}:2: MLX008" in result.stdout
    assert "framework-owned TileLang adapters" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_model_level_backend_intrinsic_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "models" / "bad_backend_import.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from tir.metal import simd_sum",
                "",
                "def lower(acc):",
                "    return simd_sum(acc)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX008", source)

    assert result.returncode == 1
    assert f"{source}:1: MLX008" in result.stdout
    assert f"{source}:4: MLX008" in result.stdout
    assert result.stderr == ""


def test_lint_allows_framework_owned_backend_intrinsic_receipts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "path_c.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def features(body: str) -> dict[str, int]:",
                "    return {'simd_sum': body.count('simd_sum')}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX008", source)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_blocks_public_partial_output_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "bad_partial.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "import mlx.core as mx",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='bad_partial',",
                "    input_names=['x'],",
                "    output_names=['dkv_partial'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX009", source)

    assert result.returncode == 1
    assert f"{source}:3: MLX009" in result.stdout
    assert "final owner outputs" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_public_api_returning_partial_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "bad_api.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def sparse_mla_bwd(x):",
                "    dkv_partial = x",
                "    return dkv_partial",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX009", source)

    assert result.returncode == 1
    assert f"{source}:3: MLX009" in result.stdout
    assert result.stderr == ""


def test_lint_allows_internal_partial_intermediates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "training" / "ok_partial.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def loss_grad(grad_logits, e_chunk):",
                "    dc = 0",
                "    dc_partial = grad_logits.T @ e_chunk",
                "    return dc + dc_partial",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint("--select", "MLX009", source)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_blocks_native_tvm_ffi_bridge_parent_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "bad_bridge_parent.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from tilelang.contrib import mlx_tvm_ffi",
                "",
                "def launch(func, inputs):",
                "    return mlx_tvm_ffi.metal_call(func, inputs=inputs, output_shapes=[], output_dtypes=[], result_indices=[], num_params=0)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:1: MLX007" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_dynamic_native_tvm_ffi_bridge_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "bad_bridge_dynamic.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "import importlib",
                "",
                "def load():",
                "    return importlib.import_module('tilelang.jit.adapter._mlx_tvm_ffi')",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:4: MLX007" in result.stdout
    assert result.stderr == ""


def test_lint_blocks_monkeypatch_patterns_in_production_modules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cppmega_mlx" / "nn" / "bad_patch.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "def configure(monkeypatch: pytest.MonkeyPatch) -> None:",
                "    monkeypatch.setattr(target, 'value', 1)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:1: MLX006" in result.stdout
    assert f"{source}:3: MLX006" in result.stdout
    assert f"{source}:4: MLX006" in result.stdout
    assert result.stderr == ""


def test_lint_production_direct_msl_and_monkeypatch_guardrails_are_green() -> None:
    result = run_lint(
        "--select",
        "MLX002,MLX005,MLX006,MLX007,MLX008,MLX009",
        ROOT / "cppmega_mlx",
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_direct_msl_and_monkeypatch_guardrails_are_green_with_tests() -> None:
    result = run_lint(
        "--select",
        "MLX002,MLX005,MLX006,MLX007,MLX008,MLX009",
        ROOT / "cppmega_mlx",
        ROOT / "tests",
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_explains_legacy_direct_msl_reduction_allowlist() -> None:
    result = run_lint("--explain-direct-msl-allowlist")

    assert result.returncode == 0
    assert result.stderr == ""
    entries = json.loads(result.stdout)
    by_path = {entry["path"]: entry for entry in entries}
    mamba3 = by_path["cppmega_mlx/nn/_tilelang/mamba3.py"]
    assert mamba3["kind"] == "legacy_path_b_fallback"
    assert "slower on the checked-in receipt" in mamba3["reason"]
    assert mamba3["replacement"].endswith("mamba3_path_c.py")
    assert mamba3["public_partial_outputs"] == []
    assert mamba3["reduction_surface"] == ["atomic_owner_output_p_axis"]
    assert "final owner-output" in mamba3["reason"]
    assert "cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py" not in by_path
    assert "cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py" not in by_path
    assert "cppmega_mlx/nn/_tilelang/m2rnn.py" not in by_path
    assert "tests/test_tilelang_msl_transform.py" not in by_path
    assert "cppmega_mlx/nn/_tilelang/sparse_mla.py" not in by_path


def test_lint_requires_custom_gradient_when_metal_kernel_enters_autodiff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training_kernel.py"
    source.write_text(
        "\n".join(
            [
                "import mlx.core as mx",
                "import mlx.nn as nn",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='training',",
                "    input_names=['x'],",
                "    output_names=['y'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
                "",
                "loss_and_grad = nn.value_and_grad(model, loss_fn)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:4: MLX002" in result.stdout
    assert f"{source}:4: MLX003" in result.stdout
    assert "@mx.custom_function" in result.stdout


def test_lint_accepts_custom_gradient_marker_for_differentiable_metal(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "cppmega_mlx" / "nn" / "_tilelang" / "mamba3.py"
    owned.parent.mkdir(parents=True)
    owned.write_text(
        "\n".join(
            [
                '"""Legacy Path B direct-MSL module."""',
                "import mlx.core as mx",
                "import mlx.nn as nn",
                "",
                "kernel = mx.fast.metal_kernel(",
                "    name='training',",
                "    input_names=['x'],",
                "    output_names=['y'],",
                "    source='uint elem = thread_position_in_grid.x;'",
                ")",
                "",
                "@mx.custom_function",
                "def fused_loss(x):",
                "    return kernel(inputs=[x], output_shapes=[x.shape], output_dtypes=[x.dtype])[0]",
                "",
                "@fused_loss.vjp",
                "def fused_loss_vjp(primals, cotangents, outputs):",
                "    return primals",
                "",
                "loss_and_grad = nn.value_and_grad(model, loss_fn)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(owned)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_lint_rejects_compile_timing_in_steady_state_throughput(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad_bench.py"
    source.write_text(
        "\n".join(
            [
                "tokens_per_step = 128",
                "first_call_time_s = 0.5",
                "tokens_per_second = tokens_per_step / first_call_time_s",
                "",
                "steady_times = [0.1, 0.2]",
                "compile_time_s = sum(steady_times)",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 1
    assert f"{source}:3: MLX004" in result.stdout
    assert f"{source}:6: MLX004" in result.stdout
    assert "steady measured steps" in result.stdout
    assert "must not be derived from warmup or steady-state" in result.stdout


def test_lint_accepts_separate_compile_and_steady_state_timing(tmp_path: Path) -> None:
    source = tmp_path / "good_bench.py"
    source.write_text(
        "\n".join(
            [
                "import statistics",
                "",
                "first_call_time_s = first_call_profile['seconds']",
                "compile_time_s = first_call_time_s if config.compile else 0.0",
                "steady_times = [0.1, 0.2]",
                "mean_step_s = statistics.fmean(steady_times)",
                "tokens_per_second = tokens_per_step / mean_step_s",
                "timing = {",
                "    'compile_time_s': compile_time_s,",
                "    'first_call_time_s': first_call_time_s,",
                "    'step_times_s': steady_times,",
                "    'tokens_per_second': tokens_per_second,",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lint(source)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
