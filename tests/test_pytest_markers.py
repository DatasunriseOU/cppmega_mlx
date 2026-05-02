from __future__ import annotations

import tomllib
from pathlib import Path


REQUIRED_MARKERS = {
    "parity",
    "kernel",
    "training",
    "bench",
    "distributed",
}


def test_pyproject_declares_required_pytest_markers() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    marker_entries = config["tool"]["pytest"]["ini_options"]["markers"]
    marker_names = {entry.split(":", 1)[0].strip() for entry in marker_entries}

    assert REQUIRED_MARKERS <= marker_names
