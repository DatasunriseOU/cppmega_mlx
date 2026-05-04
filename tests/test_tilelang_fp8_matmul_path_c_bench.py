"""Regression checks for the FP8 scaled-matmul Path C bench config."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.bench_tilelang_fp8_path_c import SHAPES, _compare_ratios, _shape_row_strict_ok

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = [
    REPO_ROOT / "bench" / "tilelang_ports" / "fp8_path_c_vs_path_b.json",
    REPO_ROOT / "bench" / "tilelang_ports" / "fp8_path_c.json",
]


def test_matmul_128_path_c_uses_tuned_m4_tile() -> None:
    """Keep the 128^3 Path C tile off the noise-prone 32x32x32 config."""

    assert SHAPES["matmul_128"] == {
        "kind": "matmul",
        "M": 128,
        "N": 128,
        "K": 128,
        "BM": 16,
        "BN": 16,
        "BK": 16,
        "num_stages": 0,
        "parity": True,
    }


def test_tiny_128_path_c_matches_matmul_128_tile() -> None:
    """The alias shape should exercise the same tuned 128^3 kernel."""

    assert SHAPES["tiny_128"] == SHAPES["matmul_128"]


def test_strict_gate_requires_matmul_ratio_and_path_b_parity() -> None:
    row = {
        "shape": {"kind": "matmul", "parity": True},
        "parity_required": True,
        "rows": [
            {"label": "path_b_msl_fp8_scaled_matmul", "bench": {"ok": True}},
            {
                "label": "matmul_tl_fp8_scaled_matmul",
                "bench": {"ok": True},
                "parity_vs_path_b_msl": {"max_abs": 0.0, "max_rel": 0.0},
            },
        ],
        "ratios": {"matmul_tl_fp8_scaled_matmul_over_path_b": 0.9},
    }

    assert _shape_row_strict_ok(row)


def test_strict_gate_rejects_matmul_without_path_b_parity() -> None:
    row = {
        "shape": {"kind": "matmul", "parity": True},
        "parity_required": True,
        "rows": [
            {"label": "path_b_msl_fp8_scaled_matmul", "bench": {"ok": True}},
            {"label": "matmul_tl_fp8_scaled_matmul", "bench": {"ok": True}},
        ],
        "ratios": {"matmul_tl_fp8_scaled_matmul_over_path_b": 0.9},
    }

    assert not _shape_row_strict_ok(row)


def test_compare_ratios_prefers_paired_vecmat_median() -> None:
    rows = [
        {
            "label": "path_b_msl_fp8_scaled_vecmat",
            "bench": {"ok": True, "median_ms": 0.22},
        },
        {
            "label": "path_c_mlx_tilelang_fp8_scaled_vecmat",
            "bench": {"ok": True, "median_ms": 0.24},
            "paired_ratios": {
                "path_c_mlx_tilelang_fp8_scaled_vecmat_over_path_b_msl_fp8_scaled_vecmat_paired_median": 0.98
            },
        },
    ]

    assert _compare_ratios(rows) == {"path_c_mlx_tilelang_fp8_scaled_vecmat_over_path_b": 0.98}


def _receipt_shape(payload: dict, name: str) -> dict:
    for row in payload["results"]:
        if row["shape_name"] == name:
            return row
    raise AssertionError(f"receipt missing shape {name}")


def _path_c_receipt_row(shape_row: dict) -> dict:
    kind = shape_row["shape"]["kind"]
    label = "path_c_mlx_tilelang_fp8_scaled_vecmat" if kind == "vecmat" else "matmul_tl_fp8_scaled_matmul"
    for row in shape_row["rows"]:
        if row["label"] == label:
            return row
    raise AssertionError(f"receipt missing Path C row {label}")


def test_checked_in_matmul_receipts_keep_compact_simdgroup_msl() -> None:
    expected = {
        "simdgroup_multiply_accumulate": 1,
        "simdgroup_load": 2,
        "simdgroup_store": 1,
        "threadgroup_half": 2,
        "threadgroup_uchar": 4,
        "threadgroup_barrier": 2,
        "A_scale_loads": 1,
        "B_scale_loads": 1,
        "scalar_float_a_val": 0,
        "scalar_float_b_val": 0,
        "tvm_thread_allreduce": 0,
        "simd_sum": 0,
        "kernel_void": 1,
        "packed_uint_loads": 0,
        "fp8_e4m3_lut": 0,
    }

    for receipt in RECEIPTS:
        payload = json.loads(receipt.read_text())
        row = _receipt_shape(payload, "matmul_128")
        path_c = _path_c_receipt_row(row)
        markers = path_c["source_metrics"]["markers"]

        for key, value in expected.items():
            assert markers[key] == value, f"{receipt} matmul_128 source metric {key}"


def test_checked_in_receipts_satisfy_strict_path_b_gate() -> None:
    for receipt in RECEIPTS:
        payload = json.loads(receipt.read_text())
        policy = payload["strict_policy"]
        assert policy["path_c_over_path_b_max_ratio"] == 1.0
        assert policy["requires_path_b_and_path_c"] is True

        for shape_name in ("matmul_128", "vecmat_4096"):
            row = _receipt_shape(payload, shape_name)
            assert _shape_row_strict_ok(
                row,
                max_ratio=policy["path_c_over_path_b_max_ratio"],
                parity_max_abs=policy["path_c_vs_path_b_parity_max_abs"],
                parity_max_rel=policy["path_c_vs_path_b_parity_max_rel"],
            ), f"{receipt} {shape_name} violates strict Path C <= Path B gate"


def test_checked_in_receipts_reject_reintroduced_slow_ratios() -> None:
    for receipt in RECEIPTS:
        payload = json.loads(receipt.read_text())
        policy = payload["strict_policy"]

        for shape_name in ("matmul_128", "vecmat_4096"):
            row = copy.deepcopy(_receipt_shape(payload, shape_name))
            path_c = _path_c_receipt_row(row)

            for key in list(row["ratios"]):
                if key.endswith("_over_path_b"):
                    row["ratios"][key] = 1.01
            for key in list(path_c.get("paired_ratios", {})):
                if key.endswith("_paired_median"):
                    path_c["paired_ratios"][key] = 1.01

            assert not _shape_row_strict_ok(
                row,
                max_ratio=policy["path_c_over_path_b_max_ratio"],
                parity_max_abs=policy["path_c_vs_path_b_parity_max_abs"],
                parity_max_rel=policy["path_c_vs_path_b_parity_max_rel"],
            ), f"{receipt} {shape_name} accepted stale slow Path C ratio"
