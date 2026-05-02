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
