"""Verify the NAM56R readiness assessment is present and consistent."""

from pathlib import Path


def test_nam56r_readiness_assessment_exists_and_has_verdict():
    doc = Path("docs/nam56r_readiness_assessment.md").read_text(encoding="utf-8")
    assert "## 5. Readiness verdict" in doc
    assert "NAM56R is not fully ready in `cppmega_mlx`" in doc
    assert "## 6. Recommended next steps" in doc
    assert "Close TileLang Phase 4" in doc
