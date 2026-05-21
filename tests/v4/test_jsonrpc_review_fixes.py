"""Regression tests for findings raised in the F-A..F-H code review.

Each test pins a specific defect that the review surfaced so the fix
cannot silently regress.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.jsonrpc import LRUCache, dispatch
from cppmega_v4.jsonrpc.cache import _isolate
from cppmega_v4.jsonrpc.data_methods import (
    PreviewParquetParams,
    preview_parquet,
)
from cppmega_v4.jsonrpc.schema import VerifyResult, ResolvedGraph
from cppmega_v4.jsonrpc.tokenizer_methods import (
    EncodeVisualizeParams,
    _cache_key,
    _load,
)


# ---------------------------------------------------------------------------
# C1 — Cache must return independent objects on hit so mutations don't leak.
# ---------------------------------------------------------------------------


def _empty_verify_result() -> VerifyResult:
    return VerifyResult(
        resolved=ResolvedGraph(edges=[], diagnostics=[], has_errors=False),
        memory_per_brick={},
        elapsed_ms=1.0,
    )


def test_lru_cache_get_returns_independent_pydantic_models():
    cache = LRUCache(capacity=2)
    seed = _empty_verify_result()
    cache.set("k", seed)
    a = cache.get("k")
    b = cache.get("k")
    assert a is not b
    assert a is not seed
    a.memory_per_brick["mutated"] = a.memory_per_brick.get("mutated", None)
    # b must not see the mutation
    assert "mutated" not in b.memory_per_brick


def test_lru_cache_get_returns_independent_dicts():
    cache = LRUCache(capacity=2)
    seed = {"k": [1, 2, 3], "nested": {"a": 1}}
    cache.set("k", seed)
    a = cache.get("k")
    b = cache.get("k")
    a["k"].append(999)
    a["nested"]["a"] = 42
    assert b["k"] == [1, 2, 3]
    assert b["nested"]["a"] == 1


def test_isolate_passes_through_immutables():
    """Strings/ints/tuples need not be deep-copied for safety."""
    assert _isolate("abc") == "abc"
    assert _isolate(42) == 42


# ---------------------------------------------------------------------------
# H4 — Tokenizer cache key must be deterministic across processes.
# ---------------------------------------------------------------------------


def test_tokenizer_cache_key_uses_sha256_not_pythonhash():
    p = EncodeVisualizeParams(
        tokenizer_source="cppmega_mlx/tokenizer/tokenizer.json",
        text="abc",
    )
    key = _cache_key(p)
    expected = hashlib.sha256(b"abc").hexdigest()
    assert expected in key


def test_tokenizer_cache_key_is_stable_across_processes():
    """PYTHONHASHSEED randomises hash(); SHA-256 must survive."""
    text = "deterministic across subprocesses"
    script = (
        "from cppmega_v4.jsonrpc.tokenizer_methods import _cache_key, "
        "EncodeVisualizeParams; "
        "print(_cache_key(EncodeVisualizeParams("
        "tokenizer_source='cppmega_mlx/tokenizer/tokenizer.json', "
        f"text={text!r})))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    out_a = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.strip()
    out_b = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.strip()
    assert out_a == out_b


# ---------------------------------------------------------------------------
# H1 — _TOKENIZER_CACHE must be lock-protected against concurrent _load.
# ---------------------------------------------------------------------------


def test_tokenizer_load_is_thread_safe():
    source = "cppmega_mlx/tokenizer/tokenizer.json"
    # Clear any prior cache entry to force concurrent first-loads.
    from cppmega_v4.jsonrpc import tokenizer_methods as tm
    tm._TOKENIZER_CACHE.pop(source, None)

    loaded: list[object] = []
    errors: list[BaseException] = []

    def worker():
        try:
            loaded.append(_load(source))
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors
    # All threads observed the same cached instance.
    assert len({id(x) for x in loaded}) == 1


# ---------------------------------------------------------------------------
# C2 — data.preview_parquet must distinguish channels=None vs channels=[].
# ---------------------------------------------------------------------------


def _write_full_parquet(p: Path, n_rows: int = 8):
    pq.write_table(pa.table({
        "input_ids":  [list(range(8)) for _ in range(n_rows)],
        "doc_ids":    [i for i in range(n_rows)],
        "loss_mask":  [[1] * 8 for _ in range(n_rows)],
    }), p)


def test_preview_channels_none_emits_all_channels(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_full_parquet(p)
    r = preview_parquet(PreviewParquetParams(path=str(p), limit=1, channels=None))
    assert set(r.rows[0].channels.keys()) == {"doc_ids", "loss_mask"}


def test_preview_channels_empty_emits_no_channels(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_full_parquet(p)
    r = preview_parquet(PreviewParquetParams(path=str(p), limit=1, channels=[]))
    assert r.rows[0].channels == {}


def test_preview_cache_distinguishes_none_vs_empty(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_full_parquet(p)
    cache = LRUCache()
    none_call = preview_parquet(
        PreviewParquetParams(path=str(p), limit=1, channels=None),
        cache=cache,
    )
    empty_call = preview_parquet(
        PreviewParquetParams(path=str(p), limit=1, channels=[]),
        cache=cache,
    )
    # Distinct cache entries → both populated channels payloads differ.
    assert none_call.rows[0].channels != empty_call.rows[0].channels


# ---------------------------------------------------------------------------
# L3 — Widget last_result traitlet must reflect dispatch outcome.
# ---------------------------------------------------------------------------


def test_widget_last_result_populated_after_dispatch():
    from cppmega_v4.widget import VisualBuilderWidget
    w = VisualBuilderWidget()
    captured: list[object] = []
    w.send = lambda payload, buffers=None: captured.append(payload)  # type: ignore[method-assign]
    w._on_msg(None, {
        "jsonrpc": "2.0", "id": "lr1", "method": "backend.status",
    }, [])
    assert w.last_result["method"] == "backend.status"
    assert w.last_result["id"] == "lr1"
    assert w.last_result["result"] == {"status": "ok"}


def test_widget_last_result_unchanged_on_error_envelope():
    from cppmega_v4.widget import VisualBuilderWidget
    w = VisualBuilderWidget()
    sent: list[object] = []
    w.send = lambda payload, buffers=None: sent.append(payload)  # type: ignore[method-assign]
    # First a successful call to set a baseline:
    w._on_msg(None, {"jsonrpc": "2.0", "id": 1, "method": "backend.status"}, [])
    baseline = dict(w.last_result)
    # Then a failure — last_result must NOT be overwritten.
    w._on_msg(None, {
        "jsonrpc": "2.0", "id": 2, "method": "bogus_method",
    }, [])
    assert w.last_result == baseline


# ---------------------------------------------------------------------------
# H8 — FlowCanvas drop coordinates use currentTarget (covered by vbgui suite).
# ---------------------------------------------------------------------------


def test_h8_marker():
    """Placeholder — the actual fix is verified by vbgui tests/FlowCanvas.test.tsx."""
    assert True


# ---------------------------------------------------------------------------
# Dispatcher integration — full round-trip after the fixes.
# ---------------------------------------------------------------------------


def test_dispatcher_data_preview_with_empty_channel_filter(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_full_parquet(p, n_rows=4)
    resp = dispatch({
        "jsonrpc": "2.0", "id": "d1", "method": "data.preview_parquet",
        "params": {"path": str(p), "limit": 2, "channels": []},
    })
    assert resp.error is None
    assert all(row["channels"] == {} for row in resp.result["rows"])
