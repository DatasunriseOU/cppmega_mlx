"""Standalone gb10 CUTLASS native MXFP8 (block-scaled) GEMM bench + parity (LEVER 3).

lever="cutlass-mxfp8". MEASURES, on the production bs4 transformer-block GEMM
shapes, the native SM120/SM121 block-scaled MXFP8 x MXFP8 tensor-core route from
``cppmega_mlx.nn._tilelang.cutlass_mxfp8_sm120`` (the CUTLASS >= 4.5.1 C++
collective-builder kernel built by ``_cutlass_mxfp8_sm120_build.sh``):

  * native MXFP8 TFLOPs vs the bf16 reference (target: beat bf16 33-34 TFLOPs;
    the NVIDIA Dev Forum 359960 FP8 ceiling on a real DGX Spark is ~188 TFLOPS),
  * the fp8-vs-bf16 numeric rel_err (target ~0.0376 like §21 R2; a >0.10 gate
    catches a secret garbage run),
  * the operand-byte halving (e4m3 vs bf16).

This wires NOTHING into the pipeline — it only MEASURES the standalone route and
reports it side-by-side with the bf16 reference. RULE #1: a compile/dispatch/
launch failure is RECORDED with where+what and RAISES; there is NO silent fall to
the fp16-decode MMA, to cuBLAS, or to bf16. NaN/Inf RAISES (no degraded path).

Single-owner serial discipline (the GB10 phase ONLY): before running, poll gb10
IDLE (no other python/fp8/probe procs) + free > 105GB; after, fuser -v
/dev/nvidia-uvm + drop_caches; leave gb10 > 115GB free. SIGTERM (not -9) above
113GB.

BUILD (HOST/CPU-only, do FIRST, does not touch the GPU):
  bash /home/dave/source/cppmega_mlx/cppmega_mlx/nn/_tilelang/_cutlass_mxfp8_sm120_build.sh

RUN on gb10 (single SAFE config first; --prod for the 4 prod shapes):
  ssh gb10; cd /home/dave/source/cppmega_mlx
  LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
  TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
  /home/dave/cppmega-venv/bin/python scratch/cutlass_mxfp8_bench.py --smoke
  # then, once smoke passes parity:
  ... scratch/cutlass_mxfp8_bench.py --prod
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass, field

# -- prod local_gb10_quarter config (mirrors scratch/fp8_gemm_microbench.py) ----
HIDDEN = 3584
FFN = 18944
SEQ = 4096
BS = 4
M_TOK = BS * SEQ  # 16384

# The native MXFP8 rel_err gate: a run above this is treated as garbage/secret-
# bf16 and RAISES (RULE #1). §21 R2 parity was ~0.0376; block-scaled MXFP8 RtN
# should land in a similar band. We gate generously at 0.12 to allow the
# block-scale quant noise but still catch a broken (column-shuffled / wrong
# layout / all-zero) result.
REL_ERR_GATE = 0.12


@dataclass
class GemmShape:
    name: str
    M: int
    N: int
    K: int

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K


def prod_shapes() -> list[GemmShape]:
    """The four transformer-block GEMMs that dominate bs4 FLOPs."""
    return [
        GemmShape("mlp_up_gate", M_TOK, 2 * FFN, HIDDEN),     # 16384 x 37888 x 3584
        GemmShape("mlp_down", M_TOK, HIDDEN, FFN),            # 16384 x 3584 x 18944
        GemmShape("attn_qkv", M_TOK, 3 * HIDDEN, HIDDEN),     # 16384 x 10752 x 3584
        GemmShape("attn_out", M_TOK, HIDDEN, HIDDEN),         # 16384 x 3584 x 3584
    ]


def smoke_shapes() -> list[GemmShape]:
    """SAFE small shapes measured FIRST (parity-gate the route before prod)."""
    return [
        GemmShape("smoke_256", 256, 256, 256),
        GemmShape("smoke_512", 512, 512, 512),
        GemmShape("smoke_1024", 1024, 1024, 1024),
    ]


@dataclass
class ShapeResult:
    shape: GemmShape
    bf16_tflops: float | None = None
    bf16_median_ms: float | None = None
    mxfp8_tflops: float | None = None
    mxfp8_median_ms: float | None = None
    rel_err: float | None = None
    finite: bool | None = None
    speedup: float | None = None
    bf16_operand_bytes: int | None = None
    mxfp8_operand_bytes: int | None = None
    ran: bool = False
    reason: str = ""


def _require_exclusive_gpu() -> None:
    """Fail loud unless this process is the sole GPU owner with >105GB free.

    Mirrors the gb10 single-owner serial discipline. RAISES (RULE #1) rather than
    racing another probe; the GB10 phase polls IDLE before invoking this.
    """

    import os
    import subprocess

    try:
        out = subprocess.run(
            ["bash", "-lc",
             "ps aux | grep -iE 'python|fp8|probe|cutlass' | grep -v grep | grep -v "
             + str(os.getpid())],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cutlass_mxfp8_bench: cannot poll for GPU contenders: {exc}") from exc
    # Filter out this very python invocation's own line.
    contenders = [
        ln for ln in out.splitlines()
        if "cutlass_mxfp8_bench" not in ln and ln.strip()
    ]
    if contenders:
        raise RuntimeError(
            "cutlass_mxfp8_bench: refusing to run — other GPU contenders present "
            "(single-owner serial discipline). Lines:\n  " + "\n  ".join(contenders[:10])
        )


def _bench_ms(fn, *, warmup: int = 3, iters: int = 10) -> float:
    """Median wall-clock ms of ``fn`` with CUDA sync around each timed iter."""

    import time
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def run_one(shape: GemmShape, *, so_path: str | None) -> ShapeResult:
    import torch

    from cppmega_mlx.nn._tilelang import cutlass_mxfp8_sm120 as cm

    res = ShapeResult(shape=shape)
    M, N, K = shape.M, shape.N, shape.K

    # Random TN operands. A=(M,K), B=(N,K). bf16 reference uses A @ B^T.
    g = torch.Generator(device="cuda").manual_seed(1234)
    a = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5)
    b = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5)

    # --- bf16 reference ---
    def _bf16():
        return torch.matmul(a, b.t())

    ref = _bf16()
    if not torch.isfinite(ref).all():
        raise FloatingPointError(
            f"cutlass_mxfp8_bench[{shape.name}]: bf16 reference is non-finite (RULE #1)."
        )
    bf16_ms = _bench_ms(_bf16)
    res.bf16_median_ms = bf16_ms
    res.bf16_tflops = shape.flops / (bf16_ms * 1e-3) / 1e12
    res.bf16_operand_bytes = a.numel() * 2 + b.numel() * 2

    # --- native MXFP8 route (the ONE path) ---
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def _mxfp8():
        return cm.mxfp8_gemm_from_hp(a, b, out=out, so_path=so_path)

    got = _mxfp8()
    if not torch.isfinite(got).all():
        raise FloatingPointError(
            f"cutlass_mxfp8_bench[{shape.name}]: MXFP8 result is non-finite (RULE #1)."
        )
    mxfp8_ms = _bench_ms(_mxfp8)
    res.mxfp8_median_ms = mxfp8_ms
    res.mxfp8_tflops = shape.flops / (mxfp8_ms * 1e-3) / 1e12
    # e4m3 operands are 1 byte each; scales are negligible (1 byte / 32 elems).
    res.mxfp8_operand_bytes = a.numel() + b.numel() + (a.numel() + b.numel()) // 32

    # --- parity ---
    rel = (got.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)
    res.rel_err = float(rel)
    res.finite = True
    res.speedup = (res.mxfp8_tflops or 0.0) / (res.bf16_tflops or 1.0)
    res.ran = True

    if res.rel_err > REL_ERR_GATE:
        raise FloatingPointError(
            f"cutlass_mxfp8_bench[{shape.name}]: native MXFP8 rel_err {res.rel_err:.4e} "
            f"> gate {REL_ERR_GATE} — the result is garbage / wrong layout / secret "
            f"non-MXFP8. RULE #1: refuse to report a passing route. Check the E8M0 "
            f"SFA/SFB atom layout vs Sm1xxBlkScaledConfig."
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prod", action="store_true", help="run the 4 prod bs4 GEMM shapes")
    ap.add_argument("--smoke", action="store_true", help="run the SAFE small parity shapes (default)")
    ap.add_argument("--so", default=None, help="path to _cutlass_mxfp8_sm120.so (else default/env)")
    ap.add_argument("--no-gpu-guard", action="store_true",
                    help="skip the exclusive-owner check (the GB10 phase polls separately)")
    ap.add_argument("--json", default=None, help="write the result records to this JSON path")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] torch import failed: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[FATAL] torch.cuda.is_available() is False — this bench is gb10-only.",
              file=sys.stderr)
        return 2

    if not args.no_gpu_guard:
        _require_exclusive_gpu()

    from cppmega_mlx.nn._tilelang.cutlass_mxfp8_sm120 import cutlass_mxfp8_status

    st = cutlass_mxfp8_status(args.so)
    print(f"# CUTLASS MXFP8 status: available={st.available} so={st.so_path}")
    print(f"#   reason: {st.reason}")
    if not st.available:
        # RULE #1: the .so must be built first; do NOT silently substitute a route.
        print("[FATAL] native MXFP8 .so unavailable; build it first "
              "(_cutlass_mxfp8_sm120_build.sh). No fallback (RULE #1).", file=sys.stderr)
        return 3

    shapes = prod_shapes() if args.prod else smoke_shapes()
    results: list[ShapeResult] = []
    for shape in shapes:
        print(f"\n## {shape.name}: M={shape.M} N={shape.N} K={shape.K} "
              f"(FLOPs={shape.flops/1e9:.1f}G)", flush=True)
        res = run_one(shape, so_path=args.so)
        results.append(res)
        print(f"   bf16 : {res.bf16_tflops:8.1f} TFLOPs ({res.bf16_median_ms:.3f} ms)", flush=True)
        print(f"   mxfp8: {res.mxfp8_tflops:8.1f} TFLOPs ({res.mxfp8_median_ms:.3f} ms)  "
              f"speedup {res.speedup:.2f}x  rel_err {res.rel_err:.3e}", flush=True)
        byte_ratio = (res.bf16_operand_bytes or 1) / (res.mxfp8_operand_bytes or 1)
        print(f"   operand bytes: bf16={res.bf16_operand_bytes} mxfp8={res.mxfp8_operand_bytes} "
              f"({byte_ratio:.2f}x smaller)", flush=True)

    print("\n# ============ SUMMARY ============")
    print("# shape                  bf16_TFLOPs  mxfp8_TFLOPs  speedup  rel_err", flush=True)
    for res in results:
        print(f"# {res.shape.name:22s} {res.bf16_tflops:10.1f}  "
              f"{res.mxfp8_tflops:11.1f}  {res.speedup:6.2f}x  {res.rel_err:.2e}", flush=True)

    if args.json:
        payload = [
            {**{k: v for k, v in asdict(r).items() if k != "shape"},
             "shape": asdict(r.shape)}
            for r in results
        ]
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n# wrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
