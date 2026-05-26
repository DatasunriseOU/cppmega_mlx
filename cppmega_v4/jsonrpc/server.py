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

from fastapi import (
    FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import (
    METHOD_REGISTRY,
    SCHEMA_VERSION,
)


_HEARTBEAT_INTERVAL_S: float = 1.0

# V7-H05 sentinel — distinct from None which marks train completion.
_SENTINEL_EMPTY = object()


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
    # Visual Builder is single-user dev/jupyter — open CORS is correct here.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.state.cache = cache

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": SCHEMA_VERSION}

    @app.get("/schema/methods")
    def methods() -> dict[str, list[str]]:
        return {"methods": sorted(METHOD_REGISTRY)}

    @app.get("/cache/stats")
    def cache_stats() -> dict[str, int | float]:
        return cache.stats()

    @app.post("/cache/clear")
    def cache_clear() -> dict[str, str]:
        cache.clear()
        return {"status": "cleared"}

    @app.post("/upload/parquet")
    async def upload_parquet(file: UploadFile = File(...)) -> dict[str, str | int]:
        """E-AUDIT-01: persist an uploaded parquet to /tmp/vbgui_uploads
        and return its absolute path so DataInspector can populate the
        path field. 24 h TTL + 100-file cap applied on every upload."""
        from cppmega_v4.jsonrpc.uploads import save_upload
        if not file.filename or not file.filename.endswith(".parquet"):
            raise HTTPException(
                status_code=400, detail="only .parquet uploads accepted")
        body = await file.read()
        if not body:
            raise HTTPException(
                status_code=400, detail="empty upload body")
        path = save_upload(body)
        return {"path": path, "bytes": len(body),
                "filename": file.filename}

    @app.post("/rpc")
    async def rpc(payload: dict) -> dict:
        response = await _dispatch(payload, cache)
        return response.model_dump(mode="json", exclude_none=True)

    @app.websocket("/ws/gen/{job_id}")
    async def ws_gen(socket: WebSocket, job_id: str) -> None:
        """V7-F06: live token stream from gen.run.

        Client opens this WS *before* calling gen.run with the same
        job_id, then receives {job_id, event:{step, token_id,
        finish_reason}} frames until {finish:'ok'} fires."""
        from cppmega_v4.runtime import gen_event_bus
        import queue as _queue
        await socket.accept()
        q = gen_event_bus.subscribe(job_id)

        def _try_get(timeout: float = 0.2):
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                return _SENTINEL_EMPTY

        try:
            while True:
                ev = await asyncio.to_thread(_try_get, 0.2)
                if ev is _SENTINEL_EMPTY:
                    await asyncio.sleep(0)
                    continue
                if ev is None:
                    await socket.send_json({"finish": "ok",
                                             "job_id": job_id})
                    return
                await socket.send_json({"job_id": job_id,
                                         "event": ev})
        except WebSocketDisconnect:
            pass
        finally:
            gen_event_bus.unsubscribe(job_id, q)

    @app.websocket("/ws/train/{run_id}")
    async def ws_train(socket: WebSocket, run_id: str) -> None:
        """V7-H05: live per-step training metric stream.

        The client subscribes by run_id; stage_train publishes
        {step, loss, lr, overflow} after each opt.update. A None
        sentinel marks completion; the server then sends a final
        {finish: 'ok'} frame and closes."""
        from cppmega_v4.runtime import train_event_bus
        import queue as _queue
        await socket.accept()
        q = train_event_bus.subscribe(run_id)

        def _try_get(timeout: float = 0.2):
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                return _SENTINEL_EMPTY

        try:
            while True:
                ev = await asyncio.to_thread(_try_get, 0.2)
                if ev is _SENTINEL_EMPTY:
                    # Yield to event loop so disconnect cancellation
                    # can land between polls.
                    await asyncio.sleep(0)
                    continue
                if ev is None:
                    await socket.send_json({"finish": "ok",
                                             "run_id": run_id})
                    return
                await socket.send_json({"run_id": run_id,
                                         "event": ev})
        except WebSocketDisconnect:
            pass
        finally:
            train_event_bus.unsubscribe(run_id, q)

    @app.websocket("/ws/data/{job_id}")
    async def ws_data(socket: WebSocket, job_id: str) -> None:
        """V8-R09: live data-job progress stream.

        The UI opens this WS *before* invoking data.hf_quickstart or
        data.github_corpus with the same ``job_id``. Producer publishes
        {phase: "start"|"progress"|"done", ...} dicts onto
        ``data_event_bus``; the None sentinel ends the stream with a
        final {finish:"ok"} frame."""
        from cppmega_v4.runtime import data_event_bus
        import queue as _queue
        await socket.accept()
        q = data_event_bus.subscribe(job_id)

        def _try_get(timeout: float = 0.2):
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                return _SENTINEL_EMPTY

        try:
            while True:
                ev = await asyncio.to_thread(_try_get, 0.2)
                if ev is _SENTINEL_EMPTY:
                    await asyncio.sleep(0)
                    continue
                if ev is None:
                    await socket.send_json({"finish": "ok",
                                             "job_id": job_id})
                    return
                await socket.send_json({"job_id": job_id,
                                         "event": ev})
        except WebSocketDisconnect:
            pass
        finally:
            data_event_bus.unsubscribe(job_id, q)

    @app.websocket("/ws/verify/{spec_hash}")
    async def ws_verify(socket: WebSocket, spec_hash: str) -> None:
        """V7-L45: live verify progress stream.

        Client subscribes by spec_hash (sha256 of the canonical
        VerifyParams JSON) *before* calling verify. The handler emits
        {phase} frames as it walks resolve/memory/distributed checks
        and a final {finish:'ok'} when verify returns."""
        from cppmega_v4.runtime import verify_event_bus
        import queue as _queue
        await socket.accept()
        q = verify_event_bus.subscribe(spec_hash)

        def _try_get(timeout: float = 0.2):
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                return _SENTINEL_EMPTY

        try:
            while True:
                ev = await asyncio.to_thread(_try_get, 0.2)
                if ev is _SENTINEL_EMPTY:
                    await asyncio.sleep(0)
                    continue
                if ev is None:
                    await socket.send_json({"finish": "ok",
                                             "spec_hash": spec_hash})
                    return
                await socket.send_json({"spec_hash": spec_hash,
                                         "event": ev})
        except WebSocketDisconnect:
            pass
        finally:
            verify_event_bus.unsubscribe(spec_hash, q)

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
                response = await _dispatch(payload, cache)
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

    # V8-R10: Mount static visual builder build directory if present to serve GUI dynamically on a single port
    import os
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    dist_dir = Path("/Users/dave/sources/cppmega.mlx/vbgui/dist")
    if not dist_dir.exists():
        dist_dir = Path(__file__).parent.parent.parent / "vbgui" / "dist"

    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return app


async def _dispatch(payload: dict, cache: LRUCache):
    try:
        if payload.get("method") == "pipeline.run":
            return await asyncio.to_thread(dispatch, payload, cache=cache)
        return dispatch(payload, cache=cache)
    finally:
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and mx.metal.is_available():
                mx.metal.clear_cache()
        except ImportError:
            pass
        except Exception:
            pass


_BUILD_ID: str | None = None


def _backend_build_id() -> str:
    """V7-H48: cached process-lifetime build id.

    Composed from git HEAD sha (best-effort; 'unknown' when not in a
    git checkout) plus the process boot timestamp so a backend restart
    yields a distinct id even when HEAD hasn't moved. UI uses this to
    invalidate caches (e.g. architectures.list_presets, V7-H47)."""
    global _BUILD_ID
    if _BUILD_ID is not None:
        return _BUILD_ID
    import subprocess
    import time as _t
    sha = "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=1.0,
        )
        if out.returncode == 0:
            sha = out.stdout.strip() or "unknown"
    except Exception:
        pass
    _BUILD_ID = f"{sha}.{int(_t.time())}"
    return _BUILD_ID


async def _heartbeat(socket: WebSocket) -> None:
    build_id = _backend_build_id()
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        try:
            await socket.send_json({
                "jsonrpc": "2.0", "id": None,
                "method": "backend.status",
                # V7-H48: build_id surfaces backend git sha + boot
                # timestamp so UI can render it + detect restarts.
                "params": {"status": "ok", "build_id": build_id},
            })
        except Exception:
            return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Dev entry point — run with uvicorn."""
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
