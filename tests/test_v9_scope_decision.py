"""Verify the V9 go/no-go decision is recorded in plan and style docs."""

from pathlib import Path


def test_visual_builder_plan_v9_records_scope_decision():
    plan = Path("VisualBuilderPlan-v9.md").read_text(encoding="utf-8")
    assert "## 6.5 V9 scope decision" in plan
    assert "U01 + U02 ship as the V9 scope" in plan
    assert "U03--U10 are deferred to V10" in plan.replace("–", "--")
    assert "Status**: narrowed & in progress 2026-08-01" in plan


def test_style_md_reflects_v9_scope():
    style = Path("vbgui/STYLE.md").read_text(encoding="utf-8")
    assert "| `DraftTabsStrip` | U01 | V9 in progress |" in style
    assert "| `CanvasToolbar` | U02 | V9 in progress |" in style
    assert "| `EmptyState` | U06 | deferred to V10 |" in style
    assert "| `BrickChip` | U07 | deferred to V10 |" in style
    assert "| `KeyboardShortcutsOverlay` | U10 | deferred to V10 |" in style
