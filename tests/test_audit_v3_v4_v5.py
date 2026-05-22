"""Smoke test for scripts/audit_v3_v4_v5.py (H25)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_audit_script_produces_report(tmp_path):
    out = tmp_path / "audit.md"
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.audit_v3_v4_v5",
         "--out", str(out)],
        cwd=REPO, check=False, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    text = out.read_text(encoding="utf-8")
    assert "V6 honesty audit" in text
    assert "math-effect" in text
    assert "propagation" in text
    assert "decorative" in text


def test_audit_classifier_labels_math_effect_assertions():
    """The classifier should label a function that asserts on a loss
    quantity as math-effect."""
    from scripts.audit_v3_v4_v5 import classify_function_body
    src = """
    def test_demo():
        assert abs(losses[0] - losses[-1]) < 1e-6
    """
    assert classify_function_body(src) == "math-effect"


def test_audit_classifier_labels_propagation_assertions():
    from scripts.audit_v3_v4_v5 import classify_function_body
    src = """
    def test_demo():
        assert tr.status == "ok"
        assert extras["saved_path"] == "/tmp/foo"
    """
    assert classify_function_body(src) == "propagation"
