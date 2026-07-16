"""V7-I03 (cppmega-mlx-lak7): _RUN_CACHE warm-start is race-safe.

Two threads launch pipeline.run with continue_from=<same run_id>
that's already cached. Neither should observe a partially-evicted
or torn opt.state (i.e. both stage_train calls succeed end-to-end
without exception).

Pre-fix the cache was a bare dict mutated from the train loop +
the API thread without synchronisation; under load the eviction
path could pop an entry mid-read on Python 3.13+ where dict
operations are no longer GIL-atomic for compound operations.
"""

from __future__ import annotations

import threading

import pytest
import mlx.core as mx

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
import cppmega_v4.runner.stages as stages
from cppmega_v4.runner.stages import (
    _RUN_CACHE, _RUN_CACHE_LOCK,
    _run_cache_contains, _run_cache_get, _run_cache_set,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    with _RUN_CACHE_LOCK:
        _RUN_CACHE.clear()
    yield
    with _RUN_CACHE_LOCK:
        _RUN_CACHE.clear()


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(opts: dict) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    return {"status": tr.status, "error": tr.error,
            "extras": tr.extras}


def test_v7_i03_run_cache_helpers_are_thread_safe_under_load():
    """1000 mixed reads + writes from 8 threads — no exception, no
    corrupted entries (every value retrieved equals what was set)."""
    errors: list[BaseException] = []

    def _hammer(thread_id: int) -> None:
        try:
            for i in range(125):
                key = f"k-{thread_id}-{i}"
                payload = {"thread": thread_id, "i": i, "data": [i] * 4}
                _run_cache_set(key, payload, cap=64)
                if _run_cache_contains(key):
                    got = _run_cache_get(key)
                    # cap=64 → many entries evicted by other threads;
                    # if the key is still present its value must be
                    # exactly what we wrote, never a torn merge.
                    if got is not None and got != payload:
                        raise AssertionError(
                            f"torn read: set={payload} got={got}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_hammer, args=(t,))
                for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert not errors, errors


def test_v7_i03_concurrent_train_warm_start_completes_both(tmp_path):
    """First train populates _RUN_CACHE['shared']. Then two threads
    each launch a fresh train with continue_from='shared' — both
    must finish status='ok' (no exception from a torn read)."""
    base = _train({"num_steps": 2, "run_id": "shared"})
    assert base["status"] == "ok"
    assert _run_cache_contains("shared")

    results: list[dict] = []
    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            r = _train({
                "num_steps": 2, "run_id": f"warm-{idx}",
                "continue_from_run_id": "shared",
            })
            results.append(r)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=(0,))
    t2 = threading.Thread(target=_worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=30.0)
    t2.join(timeout=30.0)
    assert not errors, errors
    assert len(results) == 2
    for r in results:
        assert r["status"] == "ok", r["error"]
        # Both runs report opt-state actually carried from the shared
        # warm-start entry.
        assert r["extras"].get("opt_state_carried") is True


def test_v7_i03_cache_eviction_does_not_strand_concurrent_readers():
    """Pre-fix path: reader could call cache[k] right after writer
    popped k via eviction → KeyError. Post-fix: _run_cache_get returns
    None safely. Pin that contract."""
    for i in range(100):
        _run_cache_set(f"k{i}", {"i": i}, cap=4)
    # cap=4 enforced → only ~last 4 keys remain.
    surviving = sum(1 for i in range(100)
                     if _run_cache_contains(f"k{i}"))
    assert surviving <= 4
    # Stale key returns None, not KeyError.
    assert _run_cache_get("k0") is None


def test_v7_i03_gpu_state_is_recreated_on_reader_thread() -> None:
    """Cached MLX arrays must not retain the publisher thread's GPU stream."""
    import mlx.core as mx

    value = {"state": mx.arange(4, dtype=mx.float32)}
    mx.eval(value["state"])
    _run_cache_set("gpu-state", value)

    errors: list[BaseException] = []

    def _read() -> None:
        try:
            restored = _run_cache_get("gpu-state")
            assert restored is not None
            mx.eval(restored["state"])
            assert restored["state"].tolist() == [0, 1, 2, 3]
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_read) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not errors, errors


def test_v7_i03_oversized_state_is_rejected_before_host_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages, "_RUN_CACHE_MAX_BYTES", 8)
    monkeypatch.setattr(
        stages,
        "_run_cache_snapshot",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("oversized state must not be materialized")
        ),
    )

    with pytest.raises(MemoryError, match="run cache state"):
        _run_cache_set("oversized", {"state": mx.arange(4, dtype=mx.float32)})
    assert not _run_cache_contains("oversized")


def test_v7_i03_stage_surfaces_cache_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages, "_RUN_CACHE_MAX_BYTES", 1)

    result = _train({"num_steps": 1, "run_id": "too-large"})

    assert result["status"] == "ok", result["error"]
    warning = result["extras"].get("opt_state_cache_warning")
    assert isinstance(warning, str)
    assert "run cache state" in warning
    assert not _run_cache_contains("too-large")
