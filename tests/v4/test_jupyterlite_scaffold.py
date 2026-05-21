"""F-E JupyterLite scaffold tests — config validity + build pipeline.

Lightweight: verify the static scaffold files exist and parse cleanly.
The full `jupyter lite build` runs in CI (see
.github/workflows/vbgui-pages.yml) — locally we just lint the inputs so
a malformed notebook or config can't slip into main.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_VBGUI = Path("vbgui")
_JLITE = _VBGUI / "jupyterlite"


def test_jupyterlite_config_exists_and_parses():
    cfg = _JLITE / "jupyter_lite_config.json"
    assert cfg.is_file(), "missing jupyter_lite_config.json"
    payload = json.loads(cfg.read_text())
    assert "LiteBuildConfig" in payload
    assert payload["LiteBuildConfig"]["contents"] == ["content"]


def test_jupyterlite_demo_notebook_is_valid():
    nb = _JLITE / "content" / "demo.ipynb"
    assert nb.is_file()
    data = json.loads(nb.read_text())
    assert data["nbformat"] >= 4
    assert "cells" in data and len(data["cells"]) >= 1
    src = "".join("".join(c.get("source", [])) for c in data["cells"])
    # Notebook must self-document the estimator-only mode constraint.
    assert "estimator-only" in src.lower()


def test_jupyterlite_demo_imports_widget_class():
    nb = _JLITE / "content" / "demo.ipynb"
    src = nb.read_text()
    assert "VisualBuilderWidget" in src


def test_pages_workflow_exists_and_has_required_jobs():
    wf = Path(".github/workflows/vbgui-pages.yml")
    assert wf.is_file(), "missing GitHub Pages deploy workflow"
    text = wf.read_text()
    assert "actions/upload-pages-artifact" in text
    assert "actions/deploy-pages" in text
    assert "build:widget" in text
    assert "jupyter lite build" in text


@pytest.mark.parametrize("subdir", ["content", "files"])
def test_jupyterlite_subdirs_present(subdir):
    assert (_JLITE / subdir).is_dir()
