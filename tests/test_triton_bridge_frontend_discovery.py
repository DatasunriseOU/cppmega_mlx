from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_triton_bridge_discovers_sibling_tilelang_frontend_without_env() -> None:
    sibling_tilelang = Path(__file__).resolve().parents[2] / "tilelang"
    if not (sibling_tilelang / "poc" / "triton_frontend" / "__init__.py").exists():
        pytest.skip(f"sibling TileLang checkout not present: {sibling_tilelang}")

    env = dict(os.environ)
    env.pop("CPPMEGA_MLX_TRITON_FRONTEND_PATH", None)
    script = textwrap.dedent(
        """
        from cppmega_mlx.nn import _triton_bridge as bridge

        root = bridge._frontend_root()
        assert root.name == "tilelang", root
        assert bridge.frontend_available(), root
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
