from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portable_data_submodule_does_not_import_mlx() -> None:
    code = """
import builtins

real_import = builtins.__import__

def reject_mlx(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "mlx" or name.startswith("mlx."):
        raise AssertionError(f"portable data import reached MLX: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_mlx
from cppmega_mlx.data.nanochat_pipeline.language_info import detect_language_info

info = detect_language_info("int main() { return 0; }", filepath="src/example.cpp")
assert info["primary_language"] == "c++"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
