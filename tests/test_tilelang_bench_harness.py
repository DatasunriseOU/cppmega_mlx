"""Fail-closed policy tests for TileLang Path C benchmark harnesses."""

from __future__ import annotations

import json
import math
import sys

import pytest
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts import bench_tilelang_fp8_path_c as fp8_bench
from scripts import bench_tilelang_sparse_mla as sparse_bench
from scripts import bench_tilelang_topk as topk_bench


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bench_result(*, ok: bool = True, median_ms: float | None = 1.0) -> dict[str, object]:
    return {
        "ok": ok,
        "median_ms": median_ms,
        "min_ms": median_ms,
        "max_ms": median_ms,
        "iters": 5,
        "warmup": 2,
        "error": None if ok else "synthetic failure",
    }


def _load_receipt(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "bench" / "tilelang_ports" / name
    return json.loads(path.read_text())


def _finite_positive_float(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


class _FakeMxArray:
    def __init__(self, value: list[list[float]]) -> None:
        self._value = value

    def __array__(self) -> list[list[float]]:
        return self._value


def test_tilelang_imports_resolve_to_local_tree_and_pinned_ffi() -> None:
    tilelang_root = fp8_bench.TILELANG_ROOT.resolve()
    local_tilelang_init = tilelang_root / "tilelang" / "__init__.py"
    local_tvm_init = tilelang_root / "3rdparty" / "tvm" / "python" / "tvm" / "__init__.py"
    if not local_tilelang_init.is_file() or not local_tvm_init.is_file():
        pytest.skip(f"local apple-head TileLang checkout is not available at {tilelang_root}")

    tilelang = import_module("tilelang")
    tvm = import_module("tvm")
    tvm_ffi = import_module("tvm_ffi")
    assert isinstance(tilelang.__file__, str)
    assert isinstance(tvm.__file__, str)

    tilelang_path = Path(tilelang.__file__).resolve()
    tvm_path = Path(tvm.__file__).resolve()

    assert tilelang_path.is_relative_to(tilelang_root)
    assert tvm_path.is_relative_to(tilelang_root / "3rdparty" / "tvm" / "python")
    assert getattr(tvm_ffi, "__version__", None) == "0.1.11rc2"
    assert importlib_metadata.version("apache-tvm-ffi") == "0.1.11rc2"


def test_topk_strict_requires_both_paths_and_ratio_no_worse() -> None:
    row = {
        "strategies": {
            "path_b_msl": {"ran": True},
            "path_c_tilelang": {"ran": True},
        },
        "ratios": {"path_c_over_path_b": 1.0},
    }

    assert topk_bench._row_strict_ok(row, max_ratio=1.0)

    row["strategies"]["path_b_msl"]["ran"] = False
    assert not topk_bench._row_strict_ok(row, max_ratio=1.0)
    row["strategies"]["path_b_msl"]["ran"] = True

    row["strategies"]["path_c_tilelang"]["ran"] = False
    assert not topk_bench._row_strict_ok(row, max_ratio=1.0)
    row["strategies"]["path_c_tilelang"]["ran"] = True

    row["ratios"]["path_c_over_path_b"] = math.inf
    assert not topk_bench._row_strict_ok(row, max_ratio=1.0)

    row["ratios"]["path_c_over_path_b"] = 1.0001
    assert not topk_bench._row_strict_ok(row, max_ratio=1.0)


def test_topk_timing_records_timed_exception_as_failed_row(monkeypatch) -> None:
    calls = {"count": 0}

    def boom(_scores, _k):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("timed failure")
        return object()

    monkeypatch.setattr(topk_bench, "_eval_result", lambda _out: None)
    scores = topk_bench.mx.zeros((1, 4), dtype=topk_bench.mx.float32)
    row = topk_bench._time_strategy(boom, scores, 1, warmup=1, iters=2)

    assert row["ran"] is False
    assert row["median_ms"] is None
    assert row["error"] == "timed failure"


def test_topk_main_strict_exit_2_and_writes_policy_receipt(monkeypatch, tmp_path) -> None:
    out = tmp_path / "topk.json"
    payload = {
        "schema_version": topk_bench.BENCH_RECEIPT_SCHEMA_VERSION,
        "path_b_status": {"available": True, "reason": "ok"},
        "path_c_status": {"available": True, "reason": "ok"},
        "warmup": 1,
        "iters": 1,
        "seed": 1,
        "strict_policy": {
            "path_c_over_path_b_max_ratio": 1.0,
            "requires_path_b_and_path_c": True,
        },
        "rows": [
            {
                "batch": 1,
                "seq_len": 64,
                "k": 8,
                "dtype": "float32",
                "strategies": {
                    "path_b_msl": {"ran": True, "median_ms": 1.0},
                    "path_c_tilelang": {"ran": True, "median_ms": 1.1},
                },
                "ratios": {"path_c_over_path_b": 1.1},
            }
        ],
    }
    monkeypatch.setattr(topk_bench, "_build_payload", lambda **_kwargs: payload)

    rc = topk_bench.main(
        [
            "--strict",
            "--json",
            "--output",
            str(out),
            "--warmup",
            "1",
            "--iters",
            "1",
        ]
    )

    assert rc == 2
    written = json.loads(out.read_text())
    assert written["schema_version"] == topk_bench.BENCH_RECEIPT_SCHEMA_VERSION
    assert written["strict_policy"] == payload["strict_policy"]
    assert written["rows"][0]["ratios"]["path_c_over_path_b"] == 1.1


def test_topk_payload_points_to_checked_in_source(monkeypatch) -> None:
    monkeypatch.setattr(topk_bench, "_default_shapes", lambda: [])
    payload = topk_bench._build_payload(shapes=[], warmup=1, iters=1, seed=1)

    assert payload["source"] == "cppmega_mlx/nn/_tilelang/topk_selector.py"


def test_topk_auto_dispatch_prefers_path_c_when_available(monkeypatch) -> None:
    topk_module = import_module("cppmega_mlx.nn._tilelang.topk_selector")

    calls: list[str] = []
    sentinel = object()
    scores = topk_module.mx.zeros((4, 2048), dtype=topk_module.mx.float32)

    def path_c(_scores, _k, *, starts=None, ends=None):
        calls.append("path_c")
        return sentinel

    def path_b(_scores, _k, *, starts=None, ends=None):
        calls.append("path_b")
        return object()

    monkeypatch.setattr(topk_module, "topk_selector_tilelang", path_c)
    monkeypatch.setattr(topk_module, "topk_selector_metal", path_b)

    assert topk_module.topk_selector(scores, 64) is sentinel
    assert calls == ["path_c"]


def test_checked_in_topk_receipt_keeps_path_c_dispatch_gate_green() -> None:
    receipt = _load_receipt("topk_selector.json")
    policy = receipt["strict_policy"]
    rows = receipt["rows"]

    assert receipt["schema_version"] == topk_bench.BENCH_RECEIPT_SCHEMA_VERSION
    assert receipt["source"] == "cppmega_mlx/nn/_tilelang/topk_selector.py"
    assert receipt["path_b_status"]["available"] is True
    assert receipt["path_c_status"]["available"] is True
    assert policy == {
        "path_c_over_path_b_max_ratio": 1.0,
        "requires_path_b_and_path_c": True,
    }
    assert isinstance(rows, list) and rows

    max_ratio = float(policy["path_c_over_path_b_max_ratio"])
    for row in rows:
        strategies = row["strategies"]
        ratio = row["ratios"]["path_c_over_path_b"]
        assert strategies["path_b_msl"]["ran"] is True
        assert strategies["path_c_tilelang"]["ran"] is True
        assert _finite_positive_float(strategies["path_b_msl"]["median_ms"])
        assert _finite_positive_float(strategies["path_c_tilelang"]["median_ms"])
        assert _finite_positive_float(ratio)
        assert float(ratio) <= max_ratio


def test_sparse_strict_requires_available_ok_finite_forward() -> None:
    row = {
        "shape": {"name": "synthetic"},
        "path_b": {"available": True, "reason": "ok"},
        "path_c": {"available": True, "reason": "ok"},
        "fwd_msl_paired_ms": _bench_result(ok=True),
        "fwd_path_c_paired_ms": _bench_result(ok=True),
        "fwd_path_c_over_path_b_paired_ratio": 1.0,
    }

    assert (
        sparse_bench._strict_row_failures(
            row,
            fwd_only=True,
            max_ratio=1.0,
            strict_phase="all",
        )
        == []
    )

    row["path_b"]["available"] = False
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=True,
        max_ratio=1.0,
        strict_phase="all",
    )
    row["path_b"]["available"] = True

    row["path_c"]["available"] = False
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=True,
        max_ratio=1.0,
        strict_phase="all",
    )
    row["path_c"]["available"] = True

    row["fwd_path_c_paired_ms"] = _bench_result(ok=False, median_ms=None)
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=True,
        max_ratio=1.0,
        strict_phase="all",
    )
    row["fwd_path_c_paired_ms"] = _bench_result(ok=True)

    row["fwd_path_c_over_path_b_paired_ratio"] = math.inf
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=True,
        max_ratio=1.0,
        strict_phase="all",
    )

    row["fwd_path_c_over_path_b_paired_ratio"] = 1.0001
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=True,
        max_ratio=1.0,
        strict_phase="all",
    )


def test_sparse_strict_checks_backward_when_requested() -> None:
    row = {
        "shape": {"name": "synthetic"},
        "path_b": {"available": True, "reason": "ok"},
        "path_c": {"available": True, "reason": "ok"},
        "fwd_msl_paired_ms": _bench_result(ok=True),
        "fwd_path_c_paired_ms": _bench_result(ok=True),
        "fwd_path_c_over_path_b_paired_ratio": 1.0,
        "bwd_msl_paired_ms": _bench_result(ok=True),
        "bwd_path_c_paired_ms": _bench_result(ok=True),
        "bwd_path_c_over_path_b_paired_ratio": 1.0,
    }

    assert (
        sparse_bench._strict_row_failures(
            row,
            fwd_only=False,
            max_ratio=1.0,
            strict_phase="all",
        )
        == []
    )

    row["bwd_path_c_paired_ms"] = _bench_result(ok=False, median_ms=None)
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=False,
        max_ratio=1.0,
        strict_phase="all",
    )
    row["bwd_path_c_paired_ms"] = _bench_result(ok=True)

    row["bwd_path_c_over_path_b_paired_ratio"] = 1.0001
    assert sparse_bench._strict_row_failures(
        row,
        fwd_only=False,
        max_ratio=1.0,
        strict_phase="all",
    )


def test_sparse_main_strict_exit_2_writes_failed_policy_receipt(monkeypatch, tmp_path) -> None:
    out = tmp_path / "sparse.json"
    row = {
        "shape": {"name": "synthetic"},
        "path_b": {"available": True, "reason": "ok"},
        "path_c": {"available": True, "reason": "ok"},
        "fwd_msl_paired_ms": _bench_result(ok=True),
        "fwd_path_c_paired_ms": _bench_result(ok=True, median_ms=1.1),
        "fwd_path_c_over_path_b_paired_ratio": 1.1,
    }
    monkeypatch.setattr(sparse_bench, "DEFAULT_SHAPES", [{"name": "synthetic"}])
    monkeypatch.setattr(sparse_bench, "_bench_shape", lambda *_args, **_kwargs: row)
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_metal_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_path_c_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_tilelang_sparse_mla.py",
            "--strict",
            "--fwd-only",
            "--shape",
            "synthetic",
            "--out",
            str(out),
            "--warmup",
            "1",
            "--iters",
            "1",
        ],
    )

    rc = sparse_bench.main()

    assert rc == 2
    written = json.loads(out.read_text())
    assert written["strict"]["passed"] is False
    assert written["strict"]["failures"] == [
        "synthetic: forward strict gate failed paired C/B=1.1 "
        "path_b_ok=True path_c_ok=True"
    ]
    assert written["rows"][0]["fwd_path_c_over_path_b_paired_ratio"] == 1.1


def test_sparse_shape_receipt_uses_strict_max_ratio_for_no_worse_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sparse_bench,
        "_make_inputs",
        lambda _cfg, _rng: {
            "q": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            "kv": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            "indices": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.int32),
            "sm_scale": 1.0,
            "d_v": 1,
        },
    )
    monkeypatch.setattr(
        sparse_bench,
        "_bench_callable",
        lambda label, *_args, **_kwargs: _bench_result(
            ok=True,
            median_ms=1.01 if label == "path_c_tilelang_fwd" else 1.0,
        ),
    )
    monkeypatch.setattr(
        sparse_bench,
        "_bench_pair_interleaved",
        lambda *_args, **_kwargs: (
            _bench_result(ok=True, median_ms=1.0),
            _bench_result(ok=True, median_ms=1.01),
            1.01,
        ),
    )
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_metal_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_path_c_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(sparse_bench.mx, "eval", lambda *_args, **_kwargs: None)

    row = sparse_bench._bench_shape(
        {"name": "synthetic"},
        warmup=1,
        iters=1,
        fwd_only=True,
        max_ratio=1.0,
    )

    assert row["path_c_over_path_b_max_ratio"] == 1.0
    assert row["fwd_path_c_over_path_b_ratio"] == 1.01
    assert row["fwd_path_c_over_path_b_paired_ratio"] == 1.01
    assert row["fwd_path_c_no_worse_than_path_b"] is False
    assert row["fwd_path_c_no_worse_than_path_b_paired"] is False


def test_sparse_shape_receipt_sets_backward_paired_no_worse_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sparse_bench,
        "_make_inputs",
        lambda _cfg, _rng: {
            "q": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            "kv": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            "indices": sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.int32),
            "sm_scale": 1.0,
            "d_v": 1,
        },
    )
    monkeypatch.setattr(
        sparse_bench,
        "_bench_callable",
        lambda *_args, **_kwargs: _bench_result(ok=True, median_ms=1.0),
    )
    monkeypatch.setattr(
        sparse_bench,
        "_bench_pair_interleaved",
        lambda *_args, **_kwargs: (
            _bench_result(ok=True, median_ms=1.0),
            _bench_result(ok=True, median_ms=1.01),
            1.01,
        ),
    )
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_metal_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(
        sparse_bench,
        "sparse_mla_path_c_status",
        lambda *_args, **_kwargs: SimpleNamespace(available=True, reason="ok"),
    )
    monkeypatch.setattr(sparse_bench.mx, "eval", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sparse_bench,
        "_sparse_mla_bwd_path_c_partial",
        lambda *_args, **_kwargs: (
            sparse_bench.mx.zeros((1, 1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.float16),
            sparse_bench.mx.zeros((1, 1, 1, 1), dtype=sparse_bench.mx.int32),
            SimpleNamespace(
                batch=1,
                seq_len=1,
                heads=1,
                kv_group=1,
                d_v=1,
                qk_dim=1,
                seq_len_kv=1,
            ),
        ),
    )
    monkeypatch.setattr(
        sparse_bench,
        "_reduce_dkv_partial",
        lambda *_args, **_kwargs: sparse_bench.mx.zeros(
            (1, 1, 1, 1), dtype=sparse_bench.mx.float16
        ),
    )

    row = sparse_bench._bench_shape(
        {"name": "synthetic", "B": 1, "S": 1, "H": 1, "D": 1},
        warmup=1,
        iters=1,
        fwd_only=False,
        max_ratio=1.0,
    )

    assert row["bwd_path_c_over_path_b_paired_ratio"] == 1.01
    assert row["bwd_path_c_no_worse_than_path_b"] is False
    assert row["bwd_path_c_no_worse_than_path_b_paired"] is False


def test_checked_in_sparse_mla_forward_paired_receipt_is_no_worse() -> None:
    receipt = _load_receipt("sparse_mla.json")
    policy = receipt["strict_policy"]
    rows = receipt["rows"]

    assert receipt["schema"] == 1
    assert receipt["path_b_status"]["available"] is True
    assert receipt["path_c_status"]["available"] is True
    assert policy == {
        "path_c_over_path_b_max_ratio": 1.0,
        "requires_path_b_and_path_c": True,
        "fwd_only": False,
        "phase": "all",
    }
    assert isinstance(rows, list) and rows

    max_ratio = float(policy["path_c_over_path_b_max_ratio"])
    for row in rows:
        assert row["path_b"]["available"] is True
        assert row["path_c"]["available"] is True
        assert row["fwd_msl_paired_ms"]["ok"] is True
        assert row["fwd_path_c_paired_ms"]["ok"] is True
        assert _finite_positive_float(row["fwd_msl_paired_ms"]["median_ms"])
        assert _finite_positive_float(row["fwd_path_c_paired_ms"]["median_ms"])
        ratio = row["fwd_path_c_over_path_b_paired_ratio"]
        assert _finite_positive_float(ratio)
        assert row["fwd_path_c_no_worse_than_path_b_paired"] is (float(ratio) <= max_ratio)


def test_checked_in_sparse_mla_receipt_blocks_auto_path_c_flip() -> None:
    receipt = _load_receipt("sparse_mla.json")
    policy = receipt["strict_policy"]
    max_ratio = float(policy["path_c_over_path_b_max_ratio"])
    rows = receipt["rows"]

    assert policy["phase"] == "all"
    assert policy["fwd_only"] is False
    assert isinstance(rows, list) and rows

    for row in rows:
        assert row["bwd_msl_paired_ms"]["ok"] is True
        assert row["bwd_path_c_paired_ms"]["ok"] is True
        paired_ratio = row["bwd_path_c_over_path_b_paired_ratio"]
        assert _finite_positive_float(paired_ratio)
        assert row["bwd_path_c_no_worse_than_path_b"] is (float(paired_ratio) <= max_ratio)
        assert row["bwd_path_c_no_worse_than_path_b_paired"] is (
            float(paired_ratio) <= max_ratio
        )

        row_failures = sparse_bench._strict_row_failures(
            row,
            fwd_only=False,
            max_ratio=max_ratio,
            strict_phase="all",
        )
        assert row_failures == []


def test_checked_in_sparse_mla_fp8_receipt_records_full_path_c_strict_state() -> None:
    receipt = _load_receipt("sparse_mla_fp8.json")
    strict = receipt["strict"]
    ratios = receipt["ratios"]
    indexed_reduce_ratio = ratios["path_c_indexed_qk_reduce_over_path_b_fwd"]

    assert receipt["schema_version"] == 1
    assert receipt["path_c_tilelang_qk_reduce_status"]["available"] is True
    assert receipt["path_c_tilelang_indexed_qk_reduce_status"]["available"] is True
    qk_strict = receipt["qk_reducer_strict"]
    assert qk_strict["enabled"] is True
    assert qk_strict["scope"] == "qk_reducer_dispatch"
    assert qk_strict["passed"] is (float(indexed_reduce_ratio) <= float(qk_strict["max_ratio"]))
    if qk_strict["passed"] is True:
        assert qk_strict["failures"] == []
    else:
        assert any("path_c_indexed_qk_reduce_over_path_b_fwd" in item for item in qk_strict["failures"])
    _assert_full_path_c_status_matches_strict_gate(
        receipt=receipt,
        strict=strict,
        status_key="path_c_tilelang_qk_status",
    )
    assert strict["max_ratio"] == 1.0
    qk_reduce_ratio = ratios["path_c_qk_reduce_over_path_b_qk_vecmat"]
    assert _finite_positive_float(qk_reduce_ratio)
    assert _finite_positive_float(indexed_reduce_ratio)


def _assert_full_path_c_status_matches_strict_gate(
    *,
    receipt: dict[str, object],
    strict: dict[str, object],
    status_key: str,
) -> None:
    status = receipt[status_key]

    assert isinstance(status, dict)
    assert strict["scope"] == "full_path_c_dispatch"
    assert strict["enabled"] is True
    features = status.get("features", {})
    assert isinstance(features, dict)
    if (
        status["available"] is True
        and features.get("dispatch_surface") == "full_fwd_bwd"
        and features.get("full_fwd_bwd_available") is True
    ):
        assert strict["passed"] is True
        assert strict["failures"] == []
        return

    assert strict["passed"] is False
    failures = strict["failures"]
    assert isinstance(failures, list)
    if status["available"] is False:
        assert any(f"{status_key}.available=false" in item for item in failures)
    else:
        if features.get("dispatch_surface") != "full_fwd_bwd":
            assert any("dispatch_surface" in item and "full_fwd_bwd" in item for item in failures)
        if features.get("full_fwd_bwd_available") is not True:
            assert any("full_fwd_bwd_available is not true" in item for item in failures)


def test_checked_in_sparse_mla_blockscaled_receipt_keeps_full_path_c_blocked() -> None:
    receipt = _load_receipt("sparse_mla_blockscaled.json")
    strict = receipt["strict"]

    assert receipt["schema_version"] == 1
    assert receipt["path_c_tilelang_e8m0_qk_reduce_status"]["available"] is True
    qk_strict = receipt["qk_reducer_strict"]
    assert qk_strict["enabled"] is True
    assert qk_strict["scope"] == "qk_reducer_dispatch"
    assert qk_strict["passed"] is True
    assert qk_strict["failures"] == []
    _assert_full_path_c_status_matches_strict_gate(
        receipt=receipt,
        strict=strict,
        status_key="path_c_tilelang_e8m0_qk_status",
    )
    assert receipt["path_c_tilelang_e8m0_qk_reduce_status"]["reason"].startswith(
        "TileLang Path C E8M0 Sparse-MLA real QK reducer"
    )
    ratio = receipt["ratios"]["path_c_e8m0_qk_reduce_over_path_b_blockscaled_fwd"]
    assert _finite_positive_float(ratio)
    assert float(ratio) <= float(strict["max_ratio"])


def _fp8_shape_row(*, kind: str, ratio: float = 1.0, parity: dict[str, float] | None = None) -> dict[str, object]:
    if kind == "vecmat":
        path_b = "path_b_msl_fp8_scaled_vecmat"
        path_c = "path_c_mlx_tilelang_fp8_scaled_vecmat"
    else:
        path_b = "path_b_msl_fp8_scaled_matmul"
        path_c = "matmul_tl_fp8_scaled_matmul"
    row_c: dict[str, object] = {"label": path_c, "bench": _bench_result(ok=True)}
    if parity is not None:
        row_c["parity_vs_path_b_msl"] = parity
    return {
        "shape": {"kind": kind},
        "parity_required": True,
        "rows": [
            {"label": path_b, "bench": _bench_result(ok=True)},
            row_c,
        ],
        "ratios": {f"{path_c}_over_path_b": ratio},
    }


def test_fp8_strict_requires_paths_ratio_and_parity_for_matmul_and_vecmat() -> None:
    parity = {"max_abs": 0.0, "max_rel": 0.0}
    assert fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="matmul", parity=parity),
        max_ratio=1.0,
    )
    assert fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", parity=parity),
        max_ratio=1.0,
    )

    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="matmul", ratio=1.0001, parity=parity),
        max_ratio=1.0,
    )
    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", ratio=math.inf, parity=parity),
        max_ratio=1.0,
    )
    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", parity=None),
        max_ratio=1.0,
    )
    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", parity={"max_abs": math.nan, "max_rel": 0.0}),
        max_ratio=1.0,
    )
    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", parity={"max_abs": 1.0e-3, "max_rel": 0.0}),
        max_ratio=1.0,
    )
    assert not fp8_bench._shape_row_strict_ok(
        _fp8_shape_row(kind="vecmat", parity={"max_abs": 0.0, "max_rel": 1.0e-3}),
        max_ratio=1.0,
    )


def test_fp8_strict_fails_when_path_b_or_path_c_bench_failed() -> None:
    row = _fp8_shape_row(kind="matmul", parity={"max_abs": 0.0, "max_rel": 0.0})
    rows = row["rows"]
    assert isinstance(rows, list)

    rows[0]["bench"] = _bench_result(ok=False, median_ms=None)
    assert not fp8_bench._shape_row_strict_ok(row, max_ratio=1.0)

    rows[0]["bench"] = _bench_result(ok=True)
    rows[1]["bench"] = _bench_result(ok=False, median_ms=None)
    assert not fp8_bench._shape_row_strict_ok(row, max_ratio=1.0)


def test_fp8_paired_timing_alternates_paths_and_fails_closed() -> None:
    calls: list[str] = []

    def path_b() -> None:
        calls.append("b")

    def path_c() -> None:
        calls.append("c")

    result = fp8_bench._bench_paired_callables(
        (("path_b", path_b), ("path_c", path_c)),
        lambda: None,
        flops=1.0,
        warmup=2,
        iters=2,
    )

    assert calls[:4] == ["b", "c", "c", "b"]
    assert calls[4:] == ["b", "c", "c", "b"]
    stats = result.stats
    assert stats["path_b"].ok
    assert stats["path_b"].paired
    assert stats["path_c"].ok
    assert stats["path_c"].paired
    assert result.paired_ratios["path_c_over_path_b_paired_median"] > 0.0

    def boom() -> None:
        raise RuntimeError("paired failure")

    failed = fp8_bench._bench_paired_callables(
        (("path_b", path_b), ("path_c", boom)),
        lambda: None,
        flops=1.0,
        warmup=1,
        iters=1,
    )
    failed_stats = failed.stats
    assert failed_stats["path_c"].ok is False
    assert failed_stats["path_c"].paired
    assert failed_stats["path_c"].median_ms is None
    assert "paired failure" in str(failed_stats["path_c"].error)


def test_fp8_matmul_shape_uses_paired_timing_not_sequential_helpers(monkeypatch) -> None:
    calls: list[str] = []
    path_b = _FakeMxArray([[1.0]])

    class FakeTorchOut:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [[1.0]]

    def compiled(*_args):
        calls.append("path_c_run")

    def paired(strategies, sync, *, flops, warmup, iters):
        assert [label for label, _fn in strategies] == [
            "path_b_msl_fp8_scaled_matmul",
            "matmul_tl_fp8_scaled_matmul",
        ]
        assert flops == 2.0
        assert warmup == 1
        assert iters == 1
        for label, fn in strategies:
            calls.append(label)
            fn()
            sync()
        return fp8_bench.PairedBenchResult(
            stats={
                "path_b_msl_fp8_scaled_matmul": fp8_bench.BenchStats(
                    label="path_b_msl_fp8_scaled_matmul",
                    ok=True,
                    median_ms=1.0,
                    paired=True,
                ),
                "matmul_tl_fp8_scaled_matmul": fp8_bench.BenchStats(
                    label="matmul_tl_fp8_scaled_matmul",
                    ok=True,
                    median_ms=0.9,
                    paired=True,
                ),
            },
            paired_ratios={
                "matmul_tl_fp8_scaled_matmul_over_path_b_msl_fp8_scaled_matmul_paired_median": 0.9,
                "matmul_tl_fp8_scaled_matmul_over_path_b_msl_fp8_scaled_matmul_paired_p90": 0.9,
            },
        )

    monkeypatch.setattr(
        fp8_bench,
        "_build_inputs",
        lambda *_args, **_kwargs: {
            "a_mx": object(),
            "b_t_mx": object(),
            "scale_a_mx": object(),
            "scale_b_mx": object(),
            "a_fp8_mps": object(),
            "a_scale_mps": object(),
            "b_fp8_mps": object(),
            "b_scale_mps": object(),
            "c_out_mps": FakeTorchOut(),
        },
    )
    monkeypatch.setattr(fp8_bench, "_make_scaled_matmul_kernel", lambda **_kwargs: object())
    monkeypatch.setattr(fp8_bench, "_lower_source", lambda _prim: "kernel void path_c() {}")
    monkeypatch.setattr(fp8_bench, "_source_metrics", lambda _src: {"source_len": 1, "markers": {}})
    monkeypatch.setattr(fp8_bench, "_compile_tilelang", lambda _prim: compiled)
    monkeypatch.setattr(fp8_bench, "_sync_all", lambda: None)
    monkeypatch.setattr(fp8_bench, "_xcrun_compile", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(fp8_bench, "_max_error", lambda _a, _b: {"max_abs": 0.0, "max_rel": 0.0})
    monkeypatch.setattr(
        fp8_bench,
        "_parity_for_matmul",
        lambda _inputs, _actual: {"max_abs": 0.0, "max_rel": 0.0},
    )
    monkeypatch.setattr(fp8_bench, "mx", SimpleNamespace(eval=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(fp8_bench, "np", SimpleNamespace(asarray=lambda value: value))
    monkeypatch.setattr(
        fp8_bench,
        "fp8_scaled_matmul_raw",
        lambda *_args, **_kwargs: path_b,
    )
    monkeypatch.setattr(
        fp8_bench,
        "_bench_path_b_matmul",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sequential Path B matmul bench must not run")
        ),
    )
    monkeypatch.setattr(
        fp8_bench,
        "_bench_path_c_scaled_matmul",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sequential Path C matmul bench must not run")
        ),
    )
    monkeypatch.setattr(fp8_bench, "_bench_paired_callables", paired)

    row = fp8_bench._bench_shape(
        "unit_matmul",
        {
            "kind": "matmul",
            "M": 1,
            "N": 1,
            "K": 1,
            "BM": 1,
            "BN": 1,
            "BK": 1,
            "num_stages": 0,
            "parity": True,
        },
        warmup=1,
        iters=1,
        seed=1,
        input_scale=1.0,
        scale_a=1.0,
        scale_b=1.0,
        skip_xcrun=True,
        dump_dir=None,
        include_vecmat_diagnostics=False,
    )

    labels = {item["label"]: item for item in row["rows"]}
    assert labels["path_b_msl_fp8_scaled_matmul"]["bench"]["paired"] is True
    assert labels["matmul_tl_fp8_scaled_matmul"]["bench"]["paired"] is True
    assert labels["matmul_tl_fp8_scaled_matmul"]["parity_vs_path_b_msl"] == {
        "max_abs": 0.0,
        "max_rel": 0.0,
    }
    assert row["ratios"]["matmul_tl_fp8_scaled_matmul_over_path_b"] == 0.9
    assert calls == [
        "path_b_msl_fp8_scaled_matmul",
        "matmul_tl_fp8_scaled_matmul",
        "path_c_run",
    ]


def test_fp8_main_strict_exit_2_does_not_write_receipt(monkeypatch, tmp_path) -> None:
    out = tmp_path / "fp8.json"
    row = _fp8_shape_row(
        kind="matmul",
        ratio=1.1,
        parity={"max_abs": 0.0, "max_rel": 0.0},
    )
    row["shape_name"] = "matmul_128"
    monkeypatch.setattr(fp8_bench, "_require_runtime", lambda: None)
    monkeypatch.setattr(fp8_bench, "_bench_shape", lambda *_args, **_kwargs: row)
    monkeypatch.setattr(fp8_bench, "_print_summary", lambda _payload: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_tilelang_fp8_path_c.py",
            "--strict",
            "--skip-sparse",
            "--shapes",
            "matmul_128",
            "--out",
            str(out),
            "--warmup",
            "1",
            "--iters",
            "1",
        ],
    )

    rc = fp8_bench.main()

    assert rc == 2
    assert not out.exists()


def test_fp8_path_c_bench_sparse_status_uses_checked_in_reducers() -> None:
    source = (REPO_ROOT / "scripts" / "bench_tilelang_fp8_path_c.py").read_text()

    assert "No checked-in TileLang DSL Sparse-MLA Path C reference found in this lane." not in source
    assert "path_c_tilelang_qk_reduce_status" in source
    assert "path_c_tilelang_indexed_qk_reduce_status" in source
    assert "path_c_tilelang_e8m0_qk_reduce_status" in source


def test_fp8_sparse_full_dispatch_gate_rejects_reducer_only_status() -> None:
    payload = {
        "path_c_tilelang_qk_status": {
            "available": True,
            "reason": "TileLang Path C FP8 Sparse-MLA QK reducer is runnable",
            "features": {
                "dispatch_surface": "qk_reduce",
                "runnable_qk_reduce_available": True,
            },
        }
    }

    failures = fp8_bench._full_dispatch_strict_failures(
        payload,
        status_key="path_c_tilelang_qk_status",
        label="FP8",
    )

    assert any("dispatch_surface" in item and "full_fwd_bwd" in item for item in failures)
    assert any("full_fwd_bwd_available is not true" in item for item in failures)


def test_fp8_sparse_status_payload_records_reducers_but_keeps_full_gate_closed() -> None:
    reducer_status = SimpleNamespace(
        available=True,
        reason="TileLang Path C reducer runnable",
        target="metal",
        n=16,
        k=64,
        outputs_per_block=16,
        reduce_threads=32,
        vec=4,
        features={"dispatch_surface": "qk_reduce"},
    )
    indexed_status = SimpleNamespace(
        available=True,
        reason="TileLang Path C indexed reducer runnable",
        target="metal",
        batch=1,
        seq_len=1,
        heads=1,
        seq_len_kv=16,
        kv_group=1,
        head_kv=1,
        topk=16,
        k=64,
        outputs_per_block=16,
        reduce_threads=32,
        vec=4,
        features={"dispatch_surface": "indexed_qk_reduce"},
    )
    full_status = SimpleNamespace(
        available=True,
        reason="QK reducer available, not full Sparse-MLA fwd/bwd",
        target="metal",
        m=1,
        n=16,
        k=64,
        transpose_B=True,
        features={
            "dispatch_surface": "qk_reduce",
            "runnable_qk_reduce_available": True,
        },
    )
    e8m0_full_status = SimpleNamespace(
        available=True,
        reason="E8M0 reducer available, not full Sparse-MLA fwd/bwd",
        target="metal",
        m=1,
        n=16,
        k=64,
        transpose_B=True,
        scale_block_size=32,
        scale_layout="logical_unswizzled_k_axis_blocks",
        features={
            "dispatch_surface": "qk_reduce",
            "runnable_qk_reduce_available": True,
        },
    )
    e8m0_reducer_status = SimpleNamespace(
        available=True,
        reason="TileLang Path C E8M0 Sparse-MLA real QK reducer runnable",
        target="metal",
        n=16,
        k=64,
        outputs_per_block=4,
        reduce_threads=32,
        vec=4,
        scale_block_size=32,
        scale_layout="logical_unswizzled_k_axis_blocks",
        features={"dispatch_surface": "qk_reduce"},
    )

    payload = fp8_bench._sparse_path_c_status_payload(
        fp8_qk_status=full_status,
        fp8_qk_reduce_status=reducer_status,
        fp8_indexed_qk_reduce_status=indexed_status,
        e8m0_qk_status=e8m0_full_status,
        e8m0_qk_reduce_status=e8m0_reducer_status,
    )

    assert payload["path_c_tilelang_qk_reduce_status"]["available"] is True
    assert payload["path_c_tilelang_indexed_qk_reduce_status"]["available"] is True
    assert payload["path_c_tilelang_e8m0_qk_reduce_status"]["available"] is True
    assert payload["path_c_tilelang_e8m0_qk_reduce_status"]["scale_block_size"] == 32
    assert payload["path_c_status"] == payload["path_c_tilelang_qk_status"]
    assert payload["strict"]["scope"] == "full_path_c_dispatch"
    assert payload["strict"]["passed"] is False
    assert any("full_fwd_bwd" in item for item in payload["strict"]["failures"])
