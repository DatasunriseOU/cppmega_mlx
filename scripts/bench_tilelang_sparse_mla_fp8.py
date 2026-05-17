# pyright: reportMissingImports=false
"""Benchmark FP8 / MXFP8 sparse-MLA paths on Apple Silicon.

Path B is the full direct-MSL FP8 sparse-MLA forward/backward path. The
blockscaled direct-MSL Path B surface is retired, so the blockscaled report keeps
the MXFP8 reference timing plus prepared-buffer Path C reducer status. Path C now
has a runnable TileLang DSL reducer for the real FP8 QK tile
``A_fp8(1, K) @ B_fp8(N, K).T`` while the older ``T.fp8_scaled_matmul`` probe
remains fail-closed for the same M=1/top-k shape. This script records both the
full Path B forward timing and the Path C QK tile timing/status, alongside:

1. BF16 reference at ``cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference``
2. FP8 reference (per-tensor scale via ``mx.to_fp8`` / ``mx.from_fp8``)
3. MXFP8 reference (per-32-block scale via ``mx.quantize(mode='mxfp8')``)
4. Hand-built quantized_matmul side-path (regular FP32 matmul on dequantized
   tensors)

Outputs:
    bench/tilelang_ports/sparse_mla_fp8.json
    bench/tilelang_ports/sparse_mla_blockscaled.json only with --include-blockscaled

Each JSON describes the median/min/max ms across ``warmup`` + ``iters`` runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cppmega_mlx.nn._tilelang.fp8_msl_kernels import fp8_scaled_vecmat  # noqa: E402
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled import (  # noqa: E402
    _quantize_mxfp8,
    _unpack_mxfp8_to_uint8,
    sparse_mla_blockscaled_fwd_metal,
    sparse_mla_blockscaled_metal_status,
    sparse_mla_blockscaled_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (  # noqa: E402
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_path_c,
    blockscaled_sparse_mla_qk_reduce_path_c_status,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (  # noqa: E402
    _to_fp8_with_per_tensor_scale,
    sparse_mla_fp8_fwd_metal,
    sparse_mla_fp8_metal_status,
    sparse_mla_fp8_reference,
    sparse_mla_quantized_matmul_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (  # noqa: E402
    fp8_sparse_mla_indexed_qk_reduce_path_c,
    fp8_sparse_mla_indexed_qk_reduce_path_c_status,
    fp8_sparse_mla_qk_path_c_status,
    fp8_sparse_mla_qk_reduce_path_c,
    fp8_sparse_mla_qk_reduce_path_c_status,
    fp8_sparse_mla_qk_scaled_matmul_probe_status,
)
from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference  # noqa: E402


def _bench(
    label: str,
    fn: Callable[[], mx.array | tuple[mx.array, ...]],
    *,
    warmup: int = 5,
    iters: int = 25,
) -> dict[str, float | str]:
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "label": label,
        "median_ms": float(times[len(times) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "iters": iters,
    }


def _make_inputs(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    kv_group: int,
    qk_dim: int,
    topk: int,
    scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    q_np = (rng.standard_normal((batch, seq_len, heads, qk_dim)) * scale).astype(np.float32)
    kv_np = (rng.standard_normal((batch, seq_len, kv_group, qk_dim)) * scale).astype(np.float32)
    ind_np = np.tile(
        np.arange(topk, dtype=np.int32).reshape(1, 1, 1, topk),
        (batch, seq_len, kv_group, 1),
    )
    ind_np[:, :, :, topk // 2:] = -1
    ind_np[ind_np >= seq_len] = -1
    return mx.array(q_np), mx.array(kv_np), mx.array(ind_np)


def _shape_metadata(q, kv, indices, d_v):
    return {
        "q_shape": list(q.shape),
        "kv_shape": list(kv.shape),
        "indices_shape": list(indices.shape),
        "d_v": d_v,
        "q_dtype": str(q.dtype),
        "kv_dtype": str(kv.dtype),
    }


def _max_abs_err(actual: mx.array, ref: mx.array) -> dict[str, float]:
    actual_np = np.asarray(actual.astype(mx.float32))
    ref_np = np.asarray(ref.astype(mx.float32))
    err = float(np.abs(actual_np - ref_np).max())
    rel = err / (float(np.abs(ref_np).max()) + 1e-9)
    return {"max_abs_err": err, "max_rel_err": rel}


def _indexed_qk_score_oracle_np(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
) -> np.ndarray:
    q = mx.from_fp8(q_fp8, dtype=mx.float32) * q_scale.astype(mx.float32)[..., None]
    kv = mx.from_fp8(kv_fp8, dtype=mx.float32) * kv_scale.astype(mx.float32)[..., None]
    mx.eval(q, kv, indices)

    q_np = np.asarray(q).astype(np.float32)
    kv_np = np.asarray(kv).astype(np.float32)
    indices_np = np.asarray(indices).astype(np.int32)
    batch, seq_len, heads, k_dim = q_np.shape
    kv_group = kv_np.shape[2]
    topk = indices_np.shape[-1]
    head_kv = heads // kv_group
    scores = np.full((batch, seq_len, heads, topk), -np.inf, dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            for h in range(heads):
                group = h // head_kv
                for col in range(topk):
                    kv_pos = int(indices_np[b, s, group, col])
                    if kv_pos >= 0:
                        scores[b, s, h, col] = float(
                            np.dot(q_np[b, s, h, :k_dim], kv_np[b, kv_pos, group, :k_dim])
                            * sm_scale
                        )
    return scores


def _indexed_qk_err(actual: mx.array, ref: np.ndarray, indices: mx.array) -> dict[str, float | int]:
    actual_np = np.asarray(actual.astype(mx.float32))
    indices_np = np.asarray(indices).astype(np.int32)
    head_kv = actual_np.shape[2] // indices_np.shape[2]
    invalid = np.repeat(indices_np == -1, repeats=head_kv, axis=2)
    finite = ~invalid
    invalid_mismatch = int(np.count_nonzero(actual_np[invalid] > -3.0e38))
    if np.any(finite):
        err = float(np.abs(actual_np[finite] - ref[finite]).max())
        rel = err / (float(np.abs(ref[finite]).max()) + 1e-9)
    else:
        err = 0.0
        rel = 0.0
    return {
        "max_abs_err": err,
        "max_rel_err": rel,
        "invalid_mismatch_count": invalid_mismatch,
    }


def _e8m0_decode_np(x: np.ndarray) -> np.ndarray:
    x_i = x.astype(np.int32)
    return np.where((x_i == 0) | (x_i == 255), 0.0, np.exp2(x_i - 127)).astype(np.float32)


def _blockscaled_qk_oracle_np(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
) -> np.ndarray:
    A_dec = np.asarray(mx.from_fp8(A_fp8, dtype=mx.float32)).astype(np.float32)
    B_dec = np.asarray(mx.from_fp8(B_fp8, dtype=mx.float32)).astype(np.float32)
    A_scale_dec = _e8m0_decode_np(np.asarray(A_scale).astype(np.uint8).reshape((-1,)))
    B_scale_np = np.asarray(B_scale).astype(np.uint8)
    if B_scale_np.ndim == 1:
        B_scale_np = np.broadcast_to(B_scale_np.reshape((1, -1)), (B_dec.shape[0], A_scale_dec.shape[0]))
    B_scale_dec = _e8m0_decode_np(B_scale_np)

    out = np.zeros((1, B_dec.shape[0]), dtype=np.float32)
    for row in range(B_dec.shape[0]):
        for kb in range(A_scale_dec.shape[0]):
            start = kb * 32
            stop = start + 32
            partial = np.float32(np.dot(A_dec[0, start:stop], B_dec[row, start:stop]))
            out[0, row] += partial * A_scale_dec[kb] * B_scale_dec[row, kb]
    return out


def _array_only(value: mx.array | tuple[mx.array, mx.array]) -> mx.array:
    return value[0] if isinstance(value, tuple) else value


def _require_pair(value: tuple[mx.array, mx.array] | None, label: str) -> tuple[mx.array, mx.array]:
    if value is None:
        raise RuntimeError(f"{label} unexpectedly unavailable during bench")
    return value


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    metric = float(value)
    return metric if np.isfinite(metric) else None


def _strict_failures(
    payload: dict[str, Any],
    *,
    max_abs_err: float,
    max_ratio: float,
) -> list[str]:
    failures: list[str] = []

    if not payload["metal_status"]["available"]:
        failures.append(f"Path B FP8 Metal unavailable: {payload['metal_status']['dispatch_reason']}")

    qk_status = payload["path_c_tilelang_qk_reduce_status"]
    if not qk_status["available"]:
        failures.append(f"Path C QK reducer unavailable: {qk_status['reason']}")

    indexed_status = payload["path_c_tilelang_indexed_qk_reduce_status"]
    if not indexed_status["available"]:
        failures.append(f"Path C indexed QK reducer unavailable: {indexed_status['reason']}")

    parity = payload["parity"]
    for key in (
        "path_c_qk_reduce_vs_oracle",
        "path_c_qk_reduce_vs_path_b_qk_vecmat",
        "path_c_indexed_qk_reduce_vs_oracle",
    ):
        entry = parity[key]
        err = _finite_float(entry.get("max_abs_err"))
        if err is None:
            failures.append(f"{key} missing finite max_abs_err: {entry}")
        elif err > max_abs_err:
            failures.append(f"{key} max_abs_err={err:.6g} exceeds {max_abs_err:.6g}")

    indexed_parity = parity["path_c_indexed_qk_reduce_vs_oracle"]
    invalid_mismatch = indexed_parity.get("invalid_mismatch_count")
    if invalid_mismatch != 0:
        failures.append(f"path_c_indexed_qk_reduce invalid_mismatch_count={invalid_mismatch}")

    ratios = payload["ratios"]
    for key in (
        "path_c_qk_reduce_over_path_b_qk_vecmat",
        "path_c_indexed_qk_reduce_over_path_b_fwd",
    ):
        ratio = _finite_float(ratios.get(key))
        if ratio is None:
            failures.append(f"{key} missing finite ratio: {ratios.get(key)!r}")
        elif ratio > max_ratio:
            failures.append(f"{key}={ratio:.6g} exceeds {max_ratio:.6g}")

    return failures


def _full_dispatch_strict_failures(
    payload: dict[str, Any],
    *,
    status_key: str,
    label: str,
) -> list[str]:
    status = payload[status_key]
    failures: list[str] = []
    if not status["available"]:
        failures.append(f"{status_key}.available=false blocks full Path C {label} dispatch")
        return failures
    features = status.get("features", {})
    dispatch_surface = features.get("dispatch_surface")
    if dispatch_surface != "full_fwd_bwd":
        failures.append(
            f"{status_key}.features.dispatch_surface={dispatch_surface!r} "
            f"is not full_fwd_bwd Path C {label} dispatch"
        )
    if features.get("full_fwd_bwd_available") is not True:
        failures.append(f"{status_key}.features.full_fwd_bwd_available is not true")
    return failures


def _blockscaled_strict_failures(
    payload: dict[str, Any],
    *,
    max_abs_err: float,
    max_ratio: float,
) -> list[str]:
    failures: list[str] = []

    qk_status = payload["path_c_tilelang_e8m0_qk_reduce_status"]
    if not qk_status["available"]:
        failures.append(f"Path C E8M0 QK reducer unavailable: {qk_status['reason']}")

    parity = payload["parity"]["path_c_e8m0_qk_reduce_vs_oracle"]
    err = _finite_float(parity.get("max_abs_err"))
    if err is None:
        failures.append(f"path_c_e8m0_qk_reduce_vs_oracle missing finite max_abs_err: {parity}")
    elif err > max_abs_err:
        failures.append(f"path_c_e8m0_qk_reduce_vs_oracle max_abs_err={err:.6g} exceeds {max_abs_err:.6g}")

    ratio = _finite_float(payload["ratios"].get("path_c_e8m0_qk_reduce_over_blockscaled_reference"))
    if ratio is None:
        failures.append(
            "path_c_e8m0_qk_reduce_over_blockscaled_reference missing finite ratio: "
            f"{payload['ratios'].get('path_c_e8m0_qk_reduce_over_blockscaled_reference')!r}"
        )
    elif ratio > max_ratio:
        failures.append(
            f"path_c_e8m0_qk_reduce_over_blockscaled_reference={ratio:.6g} exceeds {max_ratio:.6g}"
        )

    return failures


def _strict_exit_failures(
    *,
    fp8_reducer_failures: list[str],
    fp8_full_dispatch_failures: list[str],
    include_blockscaled: bool,
    blockscaled_reducer_failures: list[str],
    blockscaled_full_dispatch_failures: list[str],
) -> list[str]:
    del fp8_full_dispatch_failures, blockscaled_full_dispatch_failures
    failures = fp8_reducer_failures
    if include_blockscaled:
        failures += blockscaled_reducer_failures
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-group", type=int, default=1)
    parser.add_argument("--qk-dim", type=int, default=64)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero unless FP8 Path C QK reducers are available, parity-clean, and not slower than Path B.",
    )
    parser.add_argument(
        "--include-blockscaled",
        action="store_true",
        help="Also run/write the unrelated blockscaled E8M0 report. Strict mode includes it only when this flag is set.",
    )
    parser.add_argument(
        "--strict-max-abs-err",
        type=float,
        default=1e-5,
        help="Maximum allowed Path C FP8 QK parity error under --strict.",
    )
    parser.add_argument(
        "--strict-max-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed Path C / Path B median ratio under --strict.",
    )
    parser.add_argument(
        "--strict-max-blockscaled-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed blockscaled Path C E8M0 QK reducer / blockscaled reference median ratio with --include-blockscaled.",
    )
    parser.add_argument(
        "--strict-max-blockscaled-abs-err",
        type=float,
        default=1e-5,
        help="Maximum allowed blockscaled Path C E8M0 QK reducer oracle error with --include-blockscaled --strict.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "bench" / "tilelang_ports",
        help="Output directory for JSON results.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    q, kv, indices = _make_inputs(
        batch=args.batch,
        seq_len=args.seq,
        heads=args.heads,
        kv_group=args.kv_group,
        qk_dim=args.qk_dim,
        topk=args.topk,
        scale=args.scale,
        seed=args.seed,
    )
    mx.eval(q, kv, indices)
    d_v = args.d_v

    bench_kwargs = {"warmup": args.warmup, "iters": args.iters}

    # Reference forward (BF16 oracle)
    bf16_ref_out = _array_only(sparse_mla_attention_reference(q, kv, indices, d_v=d_v))
    fp8_ref_out = _array_only(sparse_mla_fp8_reference(q, kv, indices, d_v=d_v))
    bs_ref_out = _array_only(sparse_mla_blockscaled_reference(q, kv, indices, d_v=d_v))
    qm_out = _array_only(sparse_mla_quantized_matmul_reference(q, kv, indices, d_v=d_v))
    mx.eval(bf16_ref_out, fp8_ref_out, bs_ref_out, qm_out)

    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale)
    q_packed_bs, q_scale_bs = _quantize_mxfp8(q)
    kv_packed_bs, kv_scale_bs = _quantize_mxfp8(kv)
    q_unpacked_bs = _unpack_mxfp8_to_uint8(q_packed_bs, args.qk_dim)
    kv_unpacked_bs = _unpack_mxfp8_to_uint8(kv_packed_bs, args.qk_dim)
    mx.eval(q_packed_bs, q_scale_bs, kv_packed_bs, kv_scale_bs, q_unpacked_bs, kv_unpacked_bs)

    sm_scale = args.qk_dim**-0.5
    qk_A_fp8 = q_fp8[0, 0, 0, :].reshape((1, args.qk_dim))
    qk_A_scale = q_scale[0, 0, 0].reshape((1,))
    qk_indices_np = np.asarray(indices[0, 0, 0, :]).astype(np.int32)
    kv_fp8_np = np.asarray(kv_fp8[0, :, 0, :]).astype(np.uint8)
    kv_scale_np = np.asarray(kv_scale[0, :, 0]).astype(np.float32)
    qk_B_fp8_np = np.zeros((args.topk, args.qk_dim), dtype=np.uint8)
    qk_B_scale_np = np.zeros((args.topk,), dtype=np.float32)
    for row, kv_pos in enumerate(qk_indices_np):
        if 0 <= kv_pos < kv_fp8_np.shape[0]:
            qk_B_fp8_np[row, :] = kv_fp8_np[kv_pos, :]
            qk_B_scale_np[row] = kv_scale_np[kv_pos]
    qk_B_fp8 = mx.array(qk_B_fp8_np)
    qk_B_scale = mx.array(qk_B_scale_np)
    mx.eval(qk_A_fp8, qk_A_scale, qk_B_fp8, qk_B_scale)

    scale_blocks = args.qk_dim // 32
    bs_A_fp8 = q_unpacked_bs[0, 0, 0, :].reshape((1, args.qk_dim))
    bs_A_scale = q_scale_bs[0, 0, 0, :].reshape((scale_blocks,))
    kv_unpacked_bs_np = np.asarray(kv_unpacked_bs[0, :, 0, :]).astype(np.uint8)
    kv_scale_bs_np = np.asarray(kv_scale_bs[0, :, 0, :]).astype(np.uint8)
    bs_B_fp8_np = np.zeros((args.topk, args.qk_dim), dtype=np.uint8)
    bs_B_scale_np = np.zeros((args.topk, scale_blocks), dtype=np.uint8)
    for row, kv_pos in enumerate(qk_indices_np):
        if 0 <= kv_pos < kv_unpacked_bs_np.shape[0]:
            bs_B_fp8_np[row, :] = kv_unpacked_bs_np[kv_pos, :]
            bs_B_scale_np[row, :] = kv_scale_bs_np[kv_pos, :]
    bs_B_fp8 = mx.array(bs_B_fp8_np)
    bs_B_scale = mx.array(bs_B_scale_np)
    bs_qk_reduce_oracle = mx.array(
        _blockscaled_qk_oracle_np(bs_A_fp8, bs_A_scale, bs_B_fp8, bs_B_scale)
    )
    mx.eval(bs_A_fp8, bs_A_scale, bs_B_fp8, bs_B_scale, bs_qk_reduce_oracle)

    path_b_qk_vecmat_out = fp8_scaled_vecmat(
        qk_A_fp8.reshape((args.qk_dim,)),
        qk_B_fp8,
        scale_x=qk_A_scale,
        scale_w=qk_B_scale,
    ).reshape((1, args.topk))
    mx.eval(path_b_qk_vecmat_out)

    # Bench each path
    bf16_bench = _bench(
        "bf16_reference",
        lambda: sparse_mla_attention_reference(q, kv, indices, d_v=d_v),
        **bench_kwargs,
    )
    fp8_bench = _bench(
        "fp8_reference",
        lambda: sparse_mla_fp8_reference(q, kv, indices, d_v=d_v),
        **bench_kwargs,
    )
    bs_bench = _bench(
        "blockscaled_reference",
        lambda: sparse_mla_blockscaled_reference(q, kv, indices, d_v=d_v),
        **bench_kwargs,
    )
    qm_bench = _bench(
        "quantized_matmul_reference",
        lambda: sparse_mla_quantized_matmul_reference(q, kv, indices, d_v=d_v),
        **bench_kwargs,
    )

    bs_status_with_arrays = sparse_mla_blockscaled_metal_status(q, kv, indices)
    bs_status_codegen = sparse_mla_blockscaled_metal_status()

    # Direct-MSL Path B kernels.
    def _msl_fp8():
        result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
        return result[0] if result is not None else mx.zeros((1,))

    fp8_msl_bench = _bench("path_b_msl_fp8_fwd", _msl_fp8, **bench_kwargs)
    bs_msl_pair = sparse_mla_blockscaled_fwd_metal(q, kv, indices, d_v=d_v)
    bs_msl_bench: dict[str, float | str | bool]
    if bs_msl_pair is None:
        bs_msl_bench = {
            "label": "path_b_msl_blockscaled_fwd",
            "available": False,
            "reason": bs_status_with_arrays.reason,
        }
    else:
        def _msl_bs():
            return bs_msl_pair[0]

        bs_msl_bench = _bench("path_b_msl_blockscaled_fwd", _msl_bs, **bench_kwargs)

    fp8_qk_outputs_per_block = 2
    fp8_qk_reduce_threads = 8
    fp8_qk_vec = 4
    qk_reduce_status = fp8_sparse_mla_qk_reduce_path_c_status(
        N=args.topk,
        K=args.qk_dim,
        outputs_per_block=fp8_qk_outputs_per_block,
        reduce_threads=fp8_qk_reduce_threads,
        vec=fp8_qk_vec,
    )
    qk_reduce_out = fp8_sparse_mla_qk_reduce_path_c(
        qk_A_fp8,
        qk_A_scale,
        qk_B_fp8,
        qk_B_scale,
        outputs_per_block=fp8_qk_outputs_per_block,
        reduce_threads=fp8_qk_reduce_threads,
        vec=fp8_qk_vec,
    )
    qk_reduce_oracle = (
        mx.matmul(
            mx.from_fp8(qk_A_fp8, dtype=mx.float32),
            mx.swapaxes(mx.from_fp8(qk_B_fp8, dtype=mx.float32), 0, 1),
        )
        * qk_A_scale.reshape((1, 1)).astype(mx.float32)
        * qk_B_scale.reshape((1, args.topk)).astype(mx.float32)
    )
    if qk_reduce_out is not None:
        mx.eval(qk_reduce_out, qk_reduce_oracle)

    indexed_qk_status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=args.batch,
        seq_len=args.seq,
        heads=args.heads,
        seq_len_kv=args.seq,
        kv_group=args.kv_group,
        topk=args.topk,
        K=args.qk_dim,
        outputs_per_block=fp8_qk_outputs_per_block,
        reduce_threads=fp8_qk_reduce_threads,
        vec=fp8_qk_vec,
    )
    indexed_qk_out = fp8_sparse_mla_indexed_qk_reduce_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        outputs_per_block=fp8_qk_outputs_per_block,
        reduce_threads=fp8_qk_reduce_threads,
        vec=fp8_qk_vec,
    )
    indexed_qk_oracle_np = _indexed_qk_score_oracle_np(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
    )
    if indexed_qk_out is not None:
        mx.eval(indexed_qk_out)

    def _path_b_fp8_qk_vecmat():
        return fp8_scaled_vecmat(
            qk_A_fp8.reshape((args.qk_dim,)),
            qk_B_fp8,
            scale_x=qk_A_scale,
            scale_w=qk_B_scale,
        ).reshape((1, args.topk))

    path_b_qk_vecmat_bench = _bench("path_b_msl_fp8_qk_vecmat", _path_b_fp8_qk_vecmat, **bench_kwargs)

    def _tilelang_fp8_qk_reduce():
        result = fp8_sparse_mla_qk_reduce_path_c(
            qk_A_fp8,
            qk_A_scale,
            qk_B_fp8,
            qk_B_scale,
            outputs_per_block=fp8_qk_outputs_per_block,
            reduce_threads=fp8_qk_reduce_threads,
            vec=fp8_qk_vec,
        )
        return result if result is not None else mx.zeros((1,), dtype=mx.float32)

    qk_reduce_bench = (
        _bench("path_c_tilelang_fp8_qk_reduce", _tilelang_fp8_qk_reduce, **bench_kwargs)
        if qk_reduce_out is not None
        else {
            "label": "path_c_tilelang_fp8_qk_reduce",
            "available": False,
            "reason": qk_reduce_status.reason,
        }
    )

    def _tilelang_fp8_indexed_qk_reduce():
        result = fp8_sparse_mla_indexed_qk_reduce_path_c(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=sm_scale,
            outputs_per_block=fp8_qk_outputs_per_block,
            reduce_threads=fp8_qk_reduce_threads,
            vec=fp8_qk_vec,
        )
        return result if result is not None else mx.zeros((1,), dtype=mx.float32)

    indexed_qk_bench = (
        _bench("path_c_tilelang_fp8_indexed_qk_reduce", _tilelang_fp8_indexed_qk_reduce, **bench_kwargs)
        if indexed_qk_out is not None
        else {
            "label": "path_c_tilelang_fp8_indexed_qk_reduce",
            "available": False,
            "reason": indexed_qk_status.reason,
        }
    )

    bs_qk_reduce_status = blockscaled_sparse_mla_qk_reduce_path_c_status(
        N=args.topk,
        K=args.qk_dim,
        outputs_per_block=4,
        reduce_threads=32,
        vec=4,
    )
    bs_qk_reduce_out = blockscaled_sparse_mla_qk_reduce_path_c(
        bs_A_fp8,
        bs_A_scale,
        bs_B_fp8,
        bs_B_scale,
        outputs_per_block=4,
        reduce_threads=32,
        vec=4,
    )
    if bs_qk_reduce_out is not None:
        mx.eval(bs_qk_reduce_out)

    def _tilelang_blockscaled_e8m0_qk_reduce():
        result = blockscaled_sparse_mla_qk_reduce_path_c(
            bs_A_fp8,
            bs_A_scale,
            bs_B_fp8,
            bs_B_scale,
            outputs_per_block=4,
            reduce_threads=32,
            vec=4,
        )
        return result if result is not None else mx.zeros((1,), dtype=mx.float32)

    bs_qk_reduce_bench = (
        _bench("path_c_tilelang_e8m0_qk_reduce", _tilelang_blockscaled_e8m0_qk_reduce, **bench_kwargs)
        if bs_qk_reduce_out is not None
        else {
            "label": "path_c_tilelang_e8m0_qk_reduce",
            "available": False,
            "reason": bs_qk_reduce_status.reason,
        }
    )

    # Capture parity vs reference for the MSL paths.
    msl_fp8_out = _require_pair(sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v), "Path B FP8 forward")[0]
    msl_bs_out = bs_msl_pair[0] if bs_msl_pair is not None else None
    if msl_bs_out is None:
        mx.eval(msl_fp8_out)
    else:
        mx.eval(msl_fp8_out, msl_bs_out)

    # Capture both dispatch status (which may report dispatcher-level rejections
    # such as int32 indices) and codegen blocker status (no arrays passed) so
    # downstream tooling can distinguish the two layers.
    fp8_status_with_arrays = sparse_mla_fp8_metal_status(q, kv, indices)
    fp8_status_codegen = sparse_mla_fp8_metal_status()
    fp8_path_c_qk_status = fp8_sparse_mla_qk_path_c_status(
        M=1,
        N=args.topk,
        K=args.qk_dim,
        BM=1,
        BN=args.topk,
        BK=args.qk_dim,
        a_scale_size=1,
        b_scale_size=args.topk,
        transpose_B=True,
    )
    fp8_path_c_qk_scaled_matmul_probe_status = fp8_sparse_mla_qk_scaled_matmul_probe_status(
        M=1,
        N=args.topk,
        K=args.qk_dim,
        BM=1,
        BN=args.topk,
        BK=args.qk_dim,
        a_scale_size=1,
        b_scale_size=args.topk,
        transpose_B=True,
    )
    bs_path_c_qk_status = blockscaled_sparse_mla_qk_path_c_status(
        M=1,
        N=args.topk,
        K=args.qk_dim,
        BM=1,
        BN=args.topk,
        BK=args.qk_dim,
        a_scale_size=args.qk_dim // 32,
        b_scale_size=args.qk_dim // 32,
        transpose_B=True,
    )
    shape_meta = _shape_metadata(q, kv, indices, d_v)

    fp8_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tilelang_port_bench",
        "port": "sparse_mla_fp8",
        "shape": shape_meta,
        "metal_status": {
            "available": bool(fp8_status_with_arrays.available),
            "dispatch_reason": fp8_status_with_arrays.reason,
            "codegen_blocker_reason": fp8_status_codegen.reason,
            "fp8_dtype": fp8_status_with_arrays.fp8_dtype,
        },
        "path_c_tilelang_qk_status": {
            "available": bool(fp8_path_c_qk_status.available),
            "reason": fp8_path_c_qk_status.reason,
            "target": fp8_path_c_qk_status.target,
            "m": fp8_path_c_qk_status.m,
            "n": fp8_path_c_qk_status.n,
            "k": fp8_path_c_qk_status.k,
            "transpose_B": fp8_path_c_qk_status.transpose_B,
            "features": fp8_path_c_qk_status.features,
        },
        "path_c_tilelang_qk_scaled_matmul_probe_status": {
            "available": bool(fp8_path_c_qk_scaled_matmul_probe_status.available),
            "reason": fp8_path_c_qk_scaled_matmul_probe_status.reason,
            "target": fp8_path_c_qk_scaled_matmul_probe_status.target,
            "m": fp8_path_c_qk_scaled_matmul_probe_status.m,
            "n": fp8_path_c_qk_scaled_matmul_probe_status.n,
            "k": fp8_path_c_qk_scaled_matmul_probe_status.k,
            "transpose_B": fp8_path_c_qk_scaled_matmul_probe_status.transpose_B,
            "features": fp8_path_c_qk_scaled_matmul_probe_status.features,
        },
        "path_c_tilelang_qk_reduce_status": {
            "available": bool(qk_reduce_status.available),
            "reason": qk_reduce_status.reason,
            "target": qk_reduce_status.target,
            "n": qk_reduce_status.n,
            "k": qk_reduce_status.k,
            "outputs_per_block": qk_reduce_status.outputs_per_block,
            "reduce_threads": qk_reduce_status.reduce_threads,
            "vec": qk_reduce_status.vec,
            "features": qk_reduce_status.features,
        },
        "path_c_tilelang_indexed_qk_reduce_status": {
            "available": bool(indexed_qk_status.available),
            "reason": indexed_qk_status.reason,
            "target": indexed_qk_status.target,
            "batch": indexed_qk_status.batch,
            "seq_len": indexed_qk_status.seq_len,
            "heads": indexed_qk_status.heads,
            "seq_len_kv": indexed_qk_status.seq_len_kv,
            "kv_group": indexed_qk_status.kv_group,
            "head_kv": indexed_qk_status.head_kv,
            "topk": indexed_qk_status.topk,
            "k": indexed_qk_status.k,
            "outputs_per_block": indexed_qk_status.outputs_per_block,
            "reduce_threads": indexed_qk_status.reduce_threads,
            "vec": indexed_qk_status.vec,
            "features": indexed_qk_status.features,
        },
        "parity": {
            "fp8_vs_bf16": _max_abs_err(fp8_ref_out, bf16_ref_out),
            "quantized_matmul_vs_bf16": _max_abs_err(qm_out, bf16_ref_out),
            "msl_fp8_vs_bf16": _max_abs_err(msl_fp8_out, bf16_ref_out),
            "msl_fp8_vs_fp8_ref": _max_abs_err(msl_fp8_out, fp8_ref_out),
            "path_b_qk_vecmat_vs_oracle": _max_abs_err(path_b_qk_vecmat_out, qk_reduce_oracle),
            "path_c_qk_reduce_vs_oracle": (
                _max_abs_err(qk_reduce_out, qk_reduce_oracle)
                if qk_reduce_out is not None
                else {"available": False, "reason": qk_reduce_status.reason}
            ),
            "path_c_qk_reduce_vs_path_b_qk_vecmat": (
                _max_abs_err(qk_reduce_out, path_b_qk_vecmat_out)
                if qk_reduce_out is not None
                else {"available": False, "reason": qk_reduce_status.reason}
            ),
            "path_c_indexed_qk_reduce_vs_oracle": (
                _indexed_qk_err(indexed_qk_out, indexed_qk_oracle_np, indices)
                if indexed_qk_out is not None
                else {"available": False, "reason": indexed_qk_status.reason}
            ),
        },
        "bench": {
            "bf16_reference": bf16_bench,
            "fp8_reference": fp8_bench,
            "quantized_matmul_reference": qm_bench,
            "path_b_msl_fp8_fwd": fp8_msl_bench,
            "path_b_msl_fp8_qk_vecmat": path_b_qk_vecmat_bench,
            "path_c_tilelang_fp8_qk_reduce": qk_reduce_bench,
            "path_c_tilelang_fp8_indexed_qk_reduce": indexed_qk_bench,
        },
        "ratios": {
            "path_c_qk_reduce_over_path_b_qk_vecmat": (
                cast(float, qk_reduce_bench["median_ms"])
                / cast(float, path_b_qk_vecmat_bench["median_ms"])
                if qk_reduce_out is not None
                else None
            ),
            "path_c_indexed_qk_reduce_over_path_b_fwd": (
                cast(float, indexed_qk_bench["median_ms"])
                / cast(float, fp8_msl_bench["median_ms"])
                if indexed_qk_out is not None
                else None
            ),
        },
    }
    strict_failures = _strict_failures(
        fp8_payload,
        max_abs_err=args.strict_max_abs_err,
        max_ratio=args.strict_max_ratio,
    )
    fp8_payload["qk_reducer_strict"] = {
        "enabled": bool(args.strict),
        "scope": "qk_reducer_dispatch",
        "passed": not strict_failures,
        "max_abs_err": args.strict_max_abs_err,
        "max_ratio": args.strict_max_ratio,
        "failures": strict_failures,
    }
    full_dispatch_failures = _full_dispatch_strict_failures(
        fp8_payload,
        status_key="path_c_tilelang_qk_status",
        label="FP8",
    )
    fp8_payload["strict"] = {
        "enabled": bool(args.strict),
        "scope": "full_path_c_dispatch",
        "passed": not full_dispatch_failures,
        "max_abs_err": args.strict_max_abs_err,
        "max_ratio": args.strict_max_ratio,
        "failures": full_dispatch_failures,
    }

    bs_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tilelang_port_bench",
        "port": "sparse_mla_blockscaled",
        "shape": shape_meta,
        "metal_status": {
            "available": bool(bs_status_with_arrays.available),
            "dispatch_reason": bs_status_with_arrays.reason,
            "codegen_blocker_reason": bs_status_codegen.reason,
            "block_size": bs_status_with_arrays.block_size,
        },
        "path_c_tilelang_e8m0_qk_status": {
            "available": bool(bs_path_c_qk_status.available),
            "reason": bs_path_c_qk_status.reason,
            "target": bs_path_c_qk_status.target,
            "m": bs_path_c_qk_status.m,
            "n": bs_path_c_qk_status.n,
            "k": bs_path_c_qk_status.k,
            "transpose_B": bs_path_c_qk_status.transpose_B,
            "scale_block_size": bs_path_c_qk_status.scale_block_size,
            "scale_layout": bs_path_c_qk_status.scale_layout,
            "features": bs_path_c_qk_status.features,
        },
        "path_c_tilelang_e8m0_qk_reduce_status": {
            "available": bool(bs_qk_reduce_status.available),
            "reason": bs_qk_reduce_status.reason,
            "target": bs_qk_reduce_status.target,
            "n": bs_qk_reduce_status.n,
            "k": bs_qk_reduce_status.k,
            "outputs_per_block": bs_qk_reduce_status.outputs_per_block,
            "reduce_threads": bs_qk_reduce_status.reduce_threads,
            "vec": bs_qk_reduce_status.vec,
            "scale_block_size": bs_qk_reduce_status.scale_block_size,
            "scale_layout": bs_qk_reduce_status.scale_layout,
            "features": bs_qk_reduce_status.features,
        },
        "parity": {
            "blockscaled_vs_bf16": _max_abs_err(bs_ref_out, bf16_ref_out),
            "quantized_matmul_vs_bf16": _max_abs_err(qm_out, bf16_ref_out),
            "msl_blockscaled_vs_bf16": (
                _max_abs_err(msl_bs_out, bf16_ref_out)
                if msl_bs_out is not None
                else {"available": False, "reason": bs_status_with_arrays.reason}
            ),
            "msl_blockscaled_vs_bs_ref": (
                _max_abs_err(msl_bs_out, bs_ref_out)
                if msl_bs_out is not None
                else {"available": False, "reason": bs_status_with_arrays.reason}
            ),
            "path_c_e8m0_qk_reduce_vs_oracle": (
                _max_abs_err(bs_qk_reduce_out, bs_qk_reduce_oracle)
                if bs_qk_reduce_out is not None
                else {"available": False, "reason": bs_qk_reduce_status.reason}
            ),
        },
        "bench": {
            "bf16_reference": bf16_bench,
            "blockscaled_reference": bs_bench,
            "quantized_matmul_reference": qm_bench,
            "path_b_msl_blockscaled_fwd": bs_msl_bench,
            "path_c_tilelang_e8m0_qk_reduce": bs_qk_reduce_bench,
        },
        "ratios": {
            "path_c_e8m0_qk_reduce_over_blockscaled_reference": (
                cast(float, bs_qk_reduce_bench["median_ms"])
                / cast(float, bs_bench["median_ms"])
                if bs_qk_reduce_out is not None
                else None
            ),
        },
    }
    bs_strict_failures = _blockscaled_strict_failures(
        bs_payload,
        max_abs_err=args.strict_max_blockscaled_abs_err,
        max_ratio=args.strict_max_blockscaled_ratio,
    )
    bs_payload["qk_reducer_strict"] = {
        "enabled": bool(args.strict),
        "scope": "qk_reducer_dispatch",
        "passed": not bs_strict_failures,
        "max_abs_err": args.strict_max_blockscaled_abs_err,
        "max_ratio": args.strict_max_blockscaled_ratio,
        "failures": bs_strict_failures,
    }
    bs_full_dispatch_failures = _full_dispatch_strict_failures(
        bs_payload,
        status_key="path_c_tilelang_e8m0_qk_status",
        label="blockscaled",
    )
    bs_payload["strict"] = {
        "enabled": bool(args.strict),
        "scope": "full_path_c_dispatch",
        "passed": not bs_full_dispatch_failures,
        "max_abs_err": args.strict_max_blockscaled_abs_err,
        "max_ratio": args.strict_max_blockscaled_ratio,
        "failures": bs_full_dispatch_failures,
    }

    fp8_path = args.out_dir / "sparse_mla_fp8.json"
    bs_path = args.out_dir / "sparse_mla_blockscaled.json"
    fp8_path.write_text(json.dumps(fp8_payload, indent=2))
    if args.include_blockscaled:
        bs_path.write_text(json.dumps(bs_payload, indent=2))

    if args.strict:
        all_strict_failures = _strict_exit_failures(
            fp8_reducer_failures=strict_failures,
            fp8_full_dispatch_failures=full_dispatch_failures,
            include_blockscaled=bool(args.include_blockscaled),
            blockscaled_reducer_failures=bs_strict_failures,
            blockscaled_full_dispatch_failures=bs_full_dispatch_failures,
        )
        if all_strict_failures:
            print("[strict] FAIL")
            for failure in all_strict_failures:
                print(f"  - {failure}")
            raise SystemExit(2)

    print(f"[bench] {fp8_path}")
    print(f"  bf16_reference        median={bf16_bench['median_ms']:.4f} ms")
    print(f"  fp8_reference         median={fp8_bench['median_ms']:.4f} ms")
    print(f"  quantized_matmul_ref  median={qm_bench['median_ms']:.4f} ms")
    print(f"  path_b_msl_fp8_fwd    median={fp8_msl_bench['median_ms']:.4f} ms (Path B direct-MSL)")
    print(
        "  path_b_msl_fp8_qk_vecmat "
        f"median={path_b_qk_vecmat_bench['median_ms']:.4f} ms "
        f"(fair Path B QK tile, N={args.topk}, K={args.qk_dim})"
    )
    print(
        "  path_c_tilelang_qk    "
        f"available={fp8_path_c_qk_status.available} "
        f"surface={fp8_path_c_qk_status.features.get('dispatch_surface')} "
        f"({fp8_path_c_qk_status.reason})"
    )
    print(
        "  path_c_tilelang_qk_scaled_matmul_probe "
        f"available={fp8_path_c_qk_scaled_matmul_probe_status.available} "
        f"({fp8_path_c_qk_scaled_matmul_probe_status.reason})"
    )
    if qk_reduce_out is not None:
        print(
            "  path_c_tilelang_fp8_qk_reduce "
            f"median={cast(float, qk_reduce_bench['median_ms']):.4f} ms "
            f"(TileLang real QK tile, N={args.topk}, K={args.qk_dim})"
        )
        qk_ratio = fp8_payload["ratios"]["path_c_qk_reduce_over_path_b_qk_vecmat"]
        print(f"  path_c/path_b qk ratio={cast(float, qk_ratio):.3f}x")
    else:
        print(f"  path_c_tilelang_fp8_qk_reduce unavailable ({qk_reduce_status.reason})")
    if indexed_qk_out is not None:
        print(
            "  path_c_tilelang_fp8_indexed_qk_reduce "
            f"median={cast(float, indexed_qk_bench['median_ms']):.4f} ms "
            f"(TileLang full-shape indexed QK, B={args.batch}, S={args.seq}, H={args.heads}, "
            f"TOPK={args.topk}, K={args.qk_dim})"
        )
    else:
        print(f"  path_c_tilelang_fp8_indexed_qk_reduce unavailable ({indexed_qk_status.reason})")
    print(f"  fp8 vs bf16 max_abs_err={fp8_payload['parity']['fp8_vs_bf16']['max_abs_err']:.4e}")
    print(f"  msl_fp8 vs bf16 max_abs_err={fp8_payload['parity']['msl_fp8_vs_bf16']['max_abs_err']:.4e}")
    if qk_reduce_out is not None:
        qk_parity = fp8_payload["parity"]["path_c_qk_reduce_vs_oracle"]
        print(f"  path_c_qk_reduce vs oracle max_abs_err={qk_parity['max_abs_err']:.4e}")
        qk_b_parity = fp8_payload["parity"]["path_c_qk_reduce_vs_path_b_qk_vecmat"]
        print(f"  path_c_qk_reduce vs path_b_qk_vecmat max_abs_err={qk_b_parity['max_abs_err']:.4e}")
    if indexed_qk_out is not None:
        indexed_qk_parity = fp8_payload["parity"]["path_c_indexed_qk_reduce_vs_oracle"]
        print(
            "  path_c_indexed_qk_reduce vs oracle "
            f"max_abs_err={indexed_qk_parity['max_abs_err']:.4e} "
            f"invalid_mismatch={indexed_qk_parity['invalid_mismatch_count']}"
        )

    if args.include_blockscaled:
        print(f"[bench] {bs_path}")
        print(f"  blockscaled_reference median={bs_bench['median_ms']:.4f} ms")
        if bs_msl_bench.get("available") is False:
            print(f"  path_b_msl_blockscaled_fwd unavailable ({bs_msl_bench['reason']})")
        else:
            print(f"  path_b_msl_blockscaled_fwd median={bs_msl_bench['median_ms']:.4f} ms (Path B direct-MSL)")
        print(
            "  path_c_tilelang_e8m0_qk "
            f"available={bs_path_c_qk_status.available} ({bs_path_c_qk_status.reason})"
        )
        if bs_qk_reduce_out is not None:
            print(
                "  path_c_tilelang_e8m0_qk_reduce "
                f"median={cast(float, bs_qk_reduce_bench['median_ms']):.4f} ms "
                f"(TileLang E8M0 QK tile, N={args.topk}, K={args.qk_dim})"
            )
            bs_qk_ratio = bs_payload["ratios"]["path_c_e8m0_qk_reduce_over_blockscaled_reference"]
            print(f"  path_c/blockscaled_reference qk-vs-fwd ratio={cast(float, bs_qk_ratio):.3f}x")
            bs_qk_parity = bs_payload["parity"]["path_c_e8m0_qk_reduce_vs_oracle"]
            print(f"  path_c_e8m0_qk_reduce vs oracle max_abs_err={bs_qk_parity['max_abs_err']:.4e}")
        else:
            print(f"  path_c_tilelang_e8m0_qk_reduce unavailable ({bs_qk_reduce_status.reason})")
        print(f"  blockscaled vs bf16 max_abs_err={bs_payload['parity']['blockscaled_vs_bf16']['max_abs_err']:.4e}")
        msl_bs_parity = bs_payload["parity"]["msl_blockscaled_vs_bf16"]
        if "max_abs_err" in msl_bs_parity:
            print(f"  msl_bs vs bf16 max_abs_err={msl_bs_parity['max_abs_err']:.4e}")
        else:
            print(f"  msl_bs unavailable ({msl_bs_parity['reason']})")
    if args.strict:
        print("[strict] PASS")


if __name__ == "__main__":
    main()
