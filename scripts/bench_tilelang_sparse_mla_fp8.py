"""Benchmark FP8 / MXFP8 sparse-MLA paths on Apple Silicon.

The TileLang Path B port of FP8 sparse-MLA is currently blocked by tilelang
0.1.9 metal codegen (see cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py and
sparse_mla_blockscaled.py module docstrings). Until both blockers lift this
script benchmarks the *available* alternatives:

1. BF16 reference at ``cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference``
2. FP8 reference (per-tensor scale via ``mx.to_fp8`` / ``mx.from_fp8``)
3. MXFP8 reference (per-32-block scale via ``mx.quantize(mode='mxfp8')``)
4. Hand-built quantized_matmul side-path (regular FP32 matmul on dequantized
   tensors — bench upper bound for what mxfp8 quantized_matmul could reach
   if tilelang were unblocked).

Outputs:
    bench/tilelang_ports/sparse_mla_fp8.json
    bench/tilelang_ports/sparse_mla_blockscaled.json

Each JSON describes the median/min/max ms across ``warmup`` + ``iters`` runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled import (  # noqa: E402
    sparse_mla_blockscaled_fwd_metal,
    sparse_mla_blockscaled_metal_status,
    sparse_mla_blockscaled_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (  # noqa: E402
    blockscaled_sparse_mla_qk_path_c_status,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (  # noqa: E402
    _to_fp8_with_per_tensor_scale,
    sparse_mla_fp8_fwd_metal,
    sparse_mla_fp8_metal_status,
    sparse_mla_fp8_reference,
    sparse_mla_quantized_matmul_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (  # noqa: E402
    fp8_sparse_mla_qk_path_c_status,
    fp8_sparse_mla_qk_reduce_path_c,
    fp8_sparse_mla_qk_reduce_path_c_status,
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
    bf16_ref_out = sparse_mla_attention_reference(q, kv, indices, d_v=d_v)
    fp8_ref_out = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v)
    bs_ref_out = sparse_mla_blockscaled_reference(q, kv, indices, d_v=d_v)
    qm_out = sparse_mla_quantized_matmul_reference(q, kv, indices, d_v=d_v)
    mx.eval(bf16_ref_out, fp8_ref_out, bs_ref_out, qm_out)

    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale)
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

    # Direct-MSL Path B kernels.
    def _msl_fp8():
        result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
        return result[0] if result is not None else mx.zeros((1,))

    def _msl_bs():
        result = sparse_mla_blockscaled_fwd_metal(q, kv, indices, d_v=d_v)
        return result[0] if result is not None else mx.zeros((1,))

    fp8_msl_bench = _bench("path_b_msl_fp8_fwd", _msl_fp8, **bench_kwargs)
    bs_msl_bench = _bench("path_b_msl_blockscaled_fwd", _msl_bs, **bench_kwargs)

    qk_reduce_status = fp8_sparse_mla_qk_reduce_path_c_status(
        N=args.topk,
        K=args.qk_dim,
        outputs_per_block=4,
        reduce_threads=32,
        vec=4,
    )
    qk_reduce_out = fp8_sparse_mla_qk_reduce_path_c(
        qk_A_fp8,
        qk_A_scale,
        qk_B_fp8,
        qk_B_scale,
        outputs_per_block=4,
        reduce_threads=32,
        vec=4,
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

    def _tilelang_fp8_qk_reduce():
        result = fp8_sparse_mla_qk_reduce_path_c(
            qk_A_fp8,
            qk_A_scale,
            qk_B_fp8,
            qk_B_scale,
            outputs_per_block=4,
            reduce_threads=32,
            vec=4,
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

    # Capture parity vs reference for the MSL paths.
    msl_fp8_out = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)[0]
    msl_bs_out = sparse_mla_blockscaled_fwd_metal(q, kv, indices, d_v=d_v)[0]
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
    bs_status_with_arrays = sparse_mla_blockscaled_metal_status(q, kv, indices)
    bs_status_codegen = sparse_mla_blockscaled_metal_status()
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
        "parity": {
            "fp8_vs_bf16": _max_abs_err(fp8_ref_out, bf16_ref_out),
            "quantized_matmul_vs_bf16": _max_abs_err(qm_out, bf16_ref_out),
            "msl_fp8_vs_bf16": _max_abs_err(msl_fp8_out, bf16_ref_out),
            "msl_fp8_vs_fp8_ref": _max_abs_err(msl_fp8_out, fp8_ref_out),
            "path_c_qk_reduce_vs_oracle": (
                _max_abs_err(qk_reduce_out, qk_reduce_oracle)
                if qk_reduce_out is not None
                else {"available": False, "reason": qk_reduce_status.reason}
            ),
        },
        "bench": {
            "bf16_reference": bf16_bench,
            "fp8_reference": fp8_bench,
            "quantized_matmul_reference": qm_bench,
            "path_b_msl_fp8_fwd": fp8_msl_bench,
            "path_c_tilelang_fp8_qk_reduce": qk_reduce_bench,
        },
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
        "parity": {
            "blockscaled_vs_bf16": _max_abs_err(bs_ref_out, bf16_ref_out),
            "quantized_matmul_vs_bf16": _max_abs_err(qm_out, bf16_ref_out),
            "msl_blockscaled_vs_bf16": _max_abs_err(msl_bs_out, bf16_ref_out),
            "msl_blockscaled_vs_bs_ref": _max_abs_err(msl_bs_out, bs_ref_out),
        },
        "bench": {
            "bf16_reference": bf16_bench,
            "blockscaled_reference": bs_bench,
            "quantized_matmul_reference": qm_bench,
            "path_b_msl_blockscaled_fwd": bs_msl_bench,
        },
    }

    fp8_path = args.out_dir / "sparse_mla_fp8.json"
    bs_path = args.out_dir / "sparse_mla_blockscaled.json"
    fp8_path.write_text(json.dumps(fp8_payload, indent=2))
    bs_path.write_text(json.dumps(bs_payload, indent=2))

    print(f"[bench] {fp8_path}")
    print(f"  bf16_reference        median={bf16_bench['median_ms']:.4f} ms")
    print(f"  fp8_reference         median={fp8_bench['median_ms']:.4f} ms")
    print(f"  quantized_matmul_ref  median={qm_bench['median_ms']:.4f} ms")
    print(f"  path_b_msl_fp8_fwd    median={fp8_msl_bench['median_ms']:.4f} ms (Path B direct-MSL)")
    print(f"  path_c_tilelang_qk    available={fp8_path_c_qk_status.available} ({fp8_path_c_qk_status.reason})")
    if qk_reduce_out is not None:
        print(
            "  path_c_tilelang_fp8_qk_reduce "
            f"median={qk_reduce_bench['median_ms']:.4f} ms "
            f"(TileLang real QK tile, N={args.topk}, K={args.qk_dim})"
        )
    else:
        print(f"  path_c_tilelang_fp8_qk_reduce unavailable ({qk_reduce_status.reason})")
    print(f"  fp8 vs bf16 max_abs_err={fp8_payload['parity']['fp8_vs_bf16']['max_abs_err']:.4e}")
    print(f"  msl_fp8 vs bf16 max_abs_err={fp8_payload['parity']['msl_fp8_vs_bf16']['max_abs_err']:.4e}")
    if qk_reduce_out is not None:
        qk_parity = fp8_payload["parity"]["path_c_qk_reduce_vs_oracle"]
        print(f"  path_c_qk_reduce vs oracle max_abs_err={qk_parity['max_abs_err']:.4e}")

    print(f"[bench] {bs_path}")
    print(f"  blockscaled_reference median={bs_bench['median_ms']:.4f} ms")
    print(f"  path_b_msl_blockscaled_fwd median={bs_msl_bench['median_ms']:.4f} ms (Path B direct-MSL)")
    print(
        "  path_c_tilelang_e8m0_qk "
        f"available={bs_path_c_qk_status.available} ({bs_path_c_qk_status.reason})"
    )
    print(f"  blockscaled vs bf16 max_abs_err={bs_payload['parity']['blockscaled_vs_bf16']['max_abs_err']:.4e}")
    print(f"  msl_bs vs bf16 max_abs_err={bs_payload['parity']['msl_blockscaled_vs_bf16']['max_abs_err']:.4e}")


if __name__ == "__main__":
    main()
