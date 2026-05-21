"""FastAPI server exposing the Visual Builder JSON-RPC 2.0 contract.

Two transports:
  - ``POST /rpc``     — single request/response, no streaming.
  - ``WS   /ws``      — bidirectional channel for high-frequency edits
                        (graph.mutate / param.edit) + server-push
                        ``backend.status`` heartbeat at 1 Hz.

The server is thin: it owns the cache instance + heartbeat loop, then
delegates every envelope to :func:`cppmega_v4.jsonrpc.dispatcher.dispatch`.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import (
    METHOD_REGISTRY,
    SCHEMA_VERSION,
)


_HEARTBEAT_INTERVAL_S: float = 1.0


def create_app(*, cache_capacity: int = 50) -> FastAPI:
    """Build a fresh FastAPI app — used by tests and the prod launcher."""

    cache = LRUCache(capacity=cache_capacity)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="cppmega Visual Builder",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
    )
    app.state.cache = cache

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION}

    @app.get("/schema/methods")
    def methods() -> dict[str, list[str]]:
        return {"methods": sorted(METHOD_REGISTRY)}

    @app.get("/cache/stats")
    def cache_stats() -> dict[str, int]:
        return cache.stats()

    @app.post("/cache/clear")
    def cache_clear() -> dict[str, str]:
        cache.clear()
        return {"status": "cleared"}

    @app.post("/rpc")
    async def rpc(payload: dict) -> dict:
        response = dispatch(payload, cache=cache)
        return response.model_dump(mode="json", exclude_none=True)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        heartbeat_task = asyncio.create_task(_heartbeat(socket))
        try:
            while True:
                raw = await socket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    await socket.send_json({
                        "jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "Parse error",
                                  "data": {"detail": str(exc)}},
                    })
                    continue
                response = dispatch(payload, cache=cache)
                await socket.send_json(
                    response.model_dump(mode="json", exclude_none=True)
                )
        except WebSocketDisconnect:
            pass
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass

    return app


async def _heartbeat(socket: WebSocket) -> None:
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        try:
            await socket.send_json({
                "jsonrpc": "2.0", "id": None,
                "method": "backend.status",
                "params": {"status": "ok"},
            })
        except Exception:
            return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Dev entry point — run with uvicorn."""
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
