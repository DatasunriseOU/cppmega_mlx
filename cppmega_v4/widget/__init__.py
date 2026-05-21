"""Jupyter anywidget for the cppmega Visual Builder.

See ``VisualBuilderPlan.md`` §3.5 for the design.

Stage F-D surface (this commit):
  - widget: VisualBuilderWidget — anywidget.AnyWidget subclass that
    bundles the React/TypeScript canvas + sidebar + chrome and
    exposes the JSON-RPC dispatch through the kernel directly (no
    HTTP round-trip; the widget runs in-process).
"""

from __future__ import annotations

from cppmega_v4.widget.widget import (
    STATIC_DIR,
    VisualBuilderWidget,
    widget_assets_exist,
)

__all__ = [
    "STATIC_DIR",
    "VisualBuilderWidget",
    "widget_assets_exist",
]
