from __future__ import annotations

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
    assert "fallback/parity/VJP/JVP/profile-evidence" in result.stdout


def test_lint_allows_owned_metal_policy_seam(tmp_path: Path) -> None:
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

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


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
    owned = tmp_path / "cppmega_mlx" / "kernels" / "metal_ops.py"
    owned.parent.mkdir(parents=True)
    owned.write_text(
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
