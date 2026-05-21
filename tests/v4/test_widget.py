"""F-D widget tests — anywidget shim + JSON-RPC bridge over custom_msg."""

from __future__ import annotations

from cppmega_v4.jsonrpc import SCHEMA_VERSION
from cppmega_v4.widget import (
    STATIC_DIR,
    VisualBuilderWidget,
    widget_assets_exist,
)


def test_widget_assets_present_after_build():
    """`npm run build:widget` (vbgui/) populates the static dir."""
    assert STATIC_DIR.is_dir()
    assert widget_assets_exist(), (
        "widget assets missing — run `npm run build:widget` in vbgui/"
    )
    assert (STATIC_DIR / "widget.mjs").stat().st_size > 0
    assert (STATIC_DIR / "widget.css").stat().st_size > 0


def test_widget_instantiates_with_bundle_loaded():
    w = VisualBuilderWidget()
    assert w._esm
    assert w._css
    assert w.schema_version == SCHEMA_VERSION


def test_widget_accepts_initial_spec():
    seed = {"graph": {"nodes": [], "edges": []}}
    w = VisualBuilderWidget(spec=seed)
    assert w.spec == seed


def test_widget_responds_to_backend_status_msg():
    """Custom-msg envelope routes through the dispatcher and `send`s a reply."""
    w = VisualBuilderWidget()
    captured: list[object] = []
    w.send = lambda payload, buffers=None: captured.append(payload)  # type: ignore[method-assign]
    w._on_msg(None, {
        "jsonrpc": "2.0", "id": "ws_1", "method": "backend.status",
    }, [])
    assert captured, "no reply emitted"
    reply = captured[0]
    assert reply["id"] == "ws_1"
    assert reply["result"] == {"status": "ok"}


def test_widget_returns_invalid_params_on_bad_envelope():
    w = VisualBuilderWidget()
    captured: list[object] = []
    w.send = lambda payload, buffers=None: captured.append(payload)  # type: ignore[method-assign]
    w._on_msg(None, {
        "jsonrpc": "2.0", "id": 1, "method": "verify",
        "params": {"graph": {"nodes": []}},  # missing required fields
    }, [])
    assert captured[0]["error"]["code"] == -32602


def test_widget_drops_non_dict_messages_silently():
    w = VisualBuilderWidget()
    captured: list[object] = []
    w.send = lambda payload, buffers=None: captured.append(payload)  # type: ignore[method-assign]
    w._on_msg(None, "not a dict", [])  # type: ignore[arg-type]
    assert captured == []


def test_widget_cache_stats_expose_lru_state():
    w = VisualBuilderWidget()
    captured: list[object] = []
    w.send = lambda payload, buffers=None: captured.append(payload)  # type: ignore[method-assign]
    # Dispatch the same backend.status twice; backend.status is short-
    # circuited before cache, so the cache stays empty — verify shape:
    w._on_msg(None, {"jsonrpc": "2.0", "id": 1, "method": "backend.status"}, [])
    stats = w.cache_stats
    assert "size" in stats and "capacity" in stats
    assert stats["capacity"] == 50


def test_widget_schema_version_is_read_only():
    w = VisualBuilderWidget()
    try:
        w.schema_version = "9.9.9"
    except (traitlets.TraitError if (traitlets := __import__("traitlets"))
            else Exception):
        pass
    # Either it raised or it silently rejected; in both cases, value unchanged.
    assert w.schema_version == SCHEMA_VERSION
