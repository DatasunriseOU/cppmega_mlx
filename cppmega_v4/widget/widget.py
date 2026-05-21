"""anywidget shim for the Visual Builder.

The widget mounts the bundled React app (built by ``npm run build:widget``
in ``vbgui/``) and routes JSON-RPC envelopes through the in-kernel
dispatcher — no HTTP server needed when running inside a notebook.

Traitlet contract (kernel ↔ frontend):
  - ``spec`` (Dict): the canonical model spec — graph, loss, optim,
    sharding, rewriters. Frontend mutates, kernel observes.
  - ``last_result`` (Dict): the most recent verify/probe/pipeline
    response, written by the kernel.
  - ``schema_version`` (Unicode, read-only): bumps when the wire
    format changes; frontend reloads.

Frontend-initiated work travels as ``custom_msg`` envelopes shaped like
JSON-RPC 2.0 requests; the kernel runs the same
:func:`cppmega_v4.jsonrpc.dispatch` and sends back a JsonRpcResponse via
``self.send``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anywidget
import traitlets

from cppmega_v4.jsonrpc import SCHEMA_VERSION, LRUCache, dispatch

_log = logging.getLogger(__name__)


STATIC_DIR: Path = Path(__file__).parent / "static"
_ESM_PATH = STATIC_DIR / "widget.mjs"
_CSS_PATH = STATIC_DIR / "widget.css"


def widget_assets_exist() -> bool:
    """True iff the Vite-built widget bundle is present on disk."""
    return _ESM_PATH.is_file() and _CSS_PATH.is_file()


def _load_text(path: Path, missing_hint: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"widget asset not found: {path}\n"
            f"hint: {missing_hint}",
        )
    return path.read_text(encoding="utf-8")


class VisualBuilderWidget(anywidget.AnyWidget):
    """Jupyter widget wrapping the cppmega Visual Builder UI."""

    _esm = traitlets.Unicode().tag(sync=False)
    _css = traitlets.Unicode().tag(sync=False)

    spec = traitlets.Dict().tag(sync=True)
    last_result = traitlets.Dict().tag(sync=True)
    schema_version = traitlets.Unicode(SCHEMA_VERSION, read_only=True).tag(sync=True)

    def __init__(
        self,
        *,
        spec: dict[str, Any] | None = None,
        cache_capacity: int = 50,
        **kwargs: Any,
    ) -> None:
        # Read bundle once at construction; raises with a useful hint if
        # the Vite build hasn't been run.
        kwargs.setdefault("_esm", _load_text(
            _ESM_PATH,
            "run `npm run build:widget` from vbgui/ to populate "
            "cppmega_v4/widget/static/widget.mjs",
        ))
        kwargs.setdefault("_css", _load_text(
            _CSS_PATH,
            "run `npm run build:widget` from vbgui/ to populate "
            "cppmega_v4/widget/static/widget.css",
        ))
        super().__init__(**kwargs)
        if spec is not None:
            self.spec = dict(spec)
        self._cache = LRUCache(capacity=cache_capacity)
        self.on_msg(self._on_msg)

    def _on_msg(self, _: object, content: Any, _buffers: list[bytes]) -> None:
        """Route a kernel-bound JSON-RPC envelope to the dispatcher.

        Side effect: every successful dispatch also lands in the
        ``last_result`` traitlet so Python notebook code can observe RPC
        results without snooping on the wire.
        """
        if not isinstance(content, dict):
            _log.warning("VisualBuilderWidget: dropped non-dict msg %r", content)
            return
        response = dispatch(content, cache=self._cache)
        payload = response.model_dump(mode="json", exclude_none=True)
        self.send(payload)
        if response.result is not None:
            self.last_result = {
                "method": content.get("method"),
                "id": response.id,
                "result": response.result,
            }

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._cache.stats()
