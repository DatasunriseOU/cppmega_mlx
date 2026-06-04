"""Standalone gb10 FP8(e4m3) vs bf16 GEMM microbench — DE-RISK for FP8 activations e2e.

Track 2 of the Megatron-gap workflow (docs/FP8-ACTIVATIONS-PATHC.md). This probe
MEASURES, at the production bs4 transformer-block GEMM shapes, the realized
fp8-vs-bf16 tensor-core TFLOPs + the operand-byte halving + the fp8-vs-bf16
numeric rel_err — BEFORE any e2e rewrite. It wires NOTHING into the pipeline.

It reports TWO fp8 routes side by side (RULE #1 — report both honestly; the
e2e plan picks the measured winner; a route that cannot run on this host is
RECORDED as the measured gap, never silently skipped):

  R1 = TransformerEngine te.Linear under MXFP8BlockScaling / Float8 per-tensor,
       wrapped in te.fp8_autocast — the proven gb10 cuBLASLt FP8 CUDA path
       (scripts/_nvfp4_route.py already drives te.Linear/te.fp8_autocast here).
  R2 = our cppmega_mlx fp8 cooperative T.gemm (fp8->fp16 dequant +
       T.gemm). On gb10 this now drives the CUDA fragment-C twin
       (fp8_scaled_matmul_path_c_cuda_prim, compiled target=cuda with the
       sm_121 TMA + warp-spec escape-hatch pass_configs) — a REAL e4m3
       tensor-core measurement, with the honest e4m3-vs-bf16 rel_err over all
       elements (the >0.10 gate catches a secret bf16/garbage run). RULE #1:
       compile/dispatch failure is RECORDED with where+what, never a silent
       fall to the Metal LUT route, to TE, or to bf16.

Single-run discipline: ONLY the Profile agent runs this, sequentially, after
polling gb10 IDLE + free>105GB. It RAISES on NaN/Inf (no degraded path).

Run on gb10:
  ssh gb10; cd /home/dave/source/cppmega_mlx
  PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
  TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
  /home/dave/cppmega-venv/bin/python scratch/fp8_gemm_microbench.py --prod

Flags:
  --prod        prod local_gb10_quarter shapes (hidden=3584 ffn=18944 bs=4 seq=4096)
  --iters N     timed calls per shape (default 20, median reported)
  --warmup N    untimed warmup calls (default 5)
  --no-te       skip the TransformerEngine R1 route (record as skipped-by-flag)
  --no-tl       skip the TileLang R2 route
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass, field

# -- prod local_gb10_quarter config (MEGATRON-VS-MLX-PATHS.md §config) ----------
HIDDEN = 3584
FFN = 18944
SEQ = 4096
BS = 4
M_TOK = BS * SEQ  # 16384 — the bs4 token count that makes these GEMMs dominate

FP8_E4M3_MAX = 448.0


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
    """The four transformer-block GEMMs that dominate bs4 FLOPs + one SSD tile."""
    return [
        GemmShape("mlp_up_gate", M_TOK, 2 * FFN, HIDDEN),     # SwiGLU gate+up
        GemmShape("mlp_down", M_TOK, HIDDEN, FFN),            # longest K
        GemmShape("attn_qkv", M_TOK, 3 * HIDDEN, HIDDEN),     # QKV proj
        GemmShape("attn_out", M_TOK, HIDDEN, HIDDEN),         # out proj
        GemmShape("ssd_f2_tile", 64, 64, 64),                # F2 chunk x headdim x dstate
    ]


@dataclass
class RouteResult:
    route: str
    ran: bool
    reason: str = ""
    tflops: float | None = None
    median_ms: float | None = None
    rel_err: float | None = None
    finite: bool | None = None


@dataclass
class ShapeResult:
    shape: GemmShape
    bf16_tflops: float | None = None
    bf16_median_ms: float | None = None
    bytes_bf16: int = 0
    bytes_fp8: int = 0
    scale_bytes_pertensor: int = 0
    scale_bytes_mxfp8: int = 0
    routes: list[RouteResult] = field(default_factory=list)


# -----------------------------------------------------------------------------
# timing helper (torch.cuda events; RAISE on NaN/Inf in the output)
# -----------------------------------------------------------------------------
def _time_cuda(fn, *, iters: int, warmup: int) -> tuple[float, "object"]:
    """Return (median_ms, last_output). RAISES on non-finite output."""
    import torch

    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    last = None
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    if last is not None and not bool(torch.isfinite(last.float()).all().item()):
        raise FloatingPointError(
            "fp8_gemm_microbench: GEMM produced non-finite output — refuse to "
            "report a TFLOPs number for a NaN/Inf result (RULE #1 fail-loud)."
        )
    return statistics.median(times), last


def _rel_err(approx, ref) -> float:
    import torch

    a = approx.float()
    r = ref.float()
    denom = float(torch.linalg.vector_norm(r).item())
    if denom == 0.0:
        raise FloatingPointError(
            "fp8_gemm_microbench: bf16 reference has zero norm — cannot form "
            "rel_err; check the input generation."
        )
    return float(torch.linalg.vector_norm(a - r).item()) / denom


# -----------------------------------------------------------------------------
# bf16 baseline (cuBLAS)
# -----------------------------------------------------------------------------
def run_bf16(shape: GemmShape, *, iters: int, warmup: int):
    import torch

    dev = "cuda"
    a = torch.randn(shape.M, shape.K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(shape.N, shape.K, device=dev, dtype=torch.bfloat16)  # (N,K), B.T form
    median_ms, out = _time_cuda(lambda: a @ w.t(), iters=iters, warmup=warmup)
    tflops = shape.flops / (median_ms * 1e-3) / 1e12
    return tflops, median_ms, out, a, w


# -----------------------------------------------------------------------------
# R1 — TransformerEngine fp8 (the proven gb10 cuBLASLt CUDA route)
# -----------------------------------------------------------------------------
def run_te_fp8(shape: GemmShape, ref_out, a_bf16, w_bf16, *, iters: int, warmup: int,
               recipe_kind: str) -> RouteResult:
    """te.Linear under MXFP8BlockScaling (recipe_kind='mxfp8') or Float8 per-tensor
    delayed (recipe_kind='tensorwise'), in te.fp8_autocast. RAISES with the precise
    reason if TE is absent/blocked (never silently skipped)."""
    name = f"R1_te_{recipe_kind}"
    try:
        import torch
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe
    except Exception as exc:
        return RouteResult(name, ran=False, reason=f"TE import failed: {exc!r}")

    try:
        if recipe_kind == "mxfp8":
            fp8_recipe = te_recipe.MXFP8BlockScaling()
        elif recipe_kind == "tensorwise":
            fp8_recipe = te_recipe.DelayedScaling(
                fp8_format=te_recipe.Format.E4M3,
            )
        else:
            return RouteResult(name, ran=False, reason=f"unknown recipe {recipe_kind!r}")

        # te.Linear computes y = x @ W.T ; W is (out=N, in=K).
        lin = te.Linear(shape.K, shape.N, bias=False, params_dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            lin.weight.copy_(w_bf16)
        x = a_bf16

        def _call():
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                return lin(x)

        median_ms, out = _time_cuda(_call, iters=iters, warmup=warmup)
        tflops = shape.flops / (median_ms * 1e-3) / 1e12
        rel = _rel_err(out, ref_out)
        return RouteResult(
            name, ran=True, tflops=tflops, median_ms=median_ms,
            rel_err=rel, finite=True,
        )
    except Exception as exc:
        # RULE #1: a TE failure is RECORDED with where+what, not hidden.
        return RouteResult(
            name, ran=False,
            reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
        )


# -----------------------------------------------------------------------------
# R2 — our cppmega_mlx fp8 cooperative T.gemm (CUDA fragment-C twin on gb10)
# -----------------------------------------------------------------------------
def _quantize_pertensor_fp8(t):
    """Per-tensor amax -> e4m3 (uint8 storage) + fp32 dequant scale, via fp8_amax.

    Reuses fp8_pack_tilelang (the CUDA-ready per-tensor amax + RNE e4m3 cast);
    returns (fp8_u8, scale_f32_scalar) where ``scale = amax/448`` is the
    post-gemm dequant multiplier. RAISES on non-finite amax (fp8_pack_tilelang).
    """
    import torch

    from cppmega_mlx.nn._tilelang.fp8_amax import fp8_pack_tilelang

    fp8_e4m3, scale, _orig_dtype = fp8_pack_tilelang(t)
    # The cooperative kernel consumes the e4m3 *bytes* as uint8 storage.
    fp8_u8 = fp8_e4m3.view(torch.uint8).contiguous()
    scale_buf = scale.reshape(1).to(dtype=torch.float32, device=t.device)
    return fp8_u8, scale_buf


def run_tilelang_fp8(shape: GemmShape, ref_out, a_bf16, w_bf16, *,
                     iters: int, warmup: int) -> RouteResult:
    """In-house fp8 e4m3 dequant->half->T.gemm, CUDA fragment-C twin (sm_121).

    Quantizes the SAME a_bf16/w_bf16 the bf16 reference used (per-tensor amax via
    fp8_amax) to e4m3, compiles the CUDA prim
    (``fp8_scaled_matmul_path_c_cuda_prim`` -> tilelang.compile(target=cuda,
    pass_configs disable TMA + warp-spec)), and times a REAL e4m3 tensor-core
    GEMM. The realized value is (A/scale_a · W/scale_b)·scale_a·scale_b ≈ A·W —
    the same logical matmul as the bf16 reference, differing ONLY in 3-bit-mantissa
    operand precision; rel_err is reported over ALL elements (honest, no subset).

    RULE #1: a compile/dispatch failure RAISES (recorded ran=False with the
    precise where+what). It NEVER falls back to the Metal LUT route, to TE, or to
    bf16; the >0.10 rel_err parity gate would catch a secret bf16/garbage run.
    """
    name = "R2_tilelang_coop"
    try:
        import tilelang
        import torch

        from cppmega_mlx.nn._tilelang._msl_transform import _as_cuda_target
        from cppmega_mlx.nn._tilelang.fp8_matmul_path_c import (
            fp8_matmul_path_c_status,
            fp8_scaled_matmul_path_c_cuda_prim,
        )
    except Exception as exc:
        return RouteResult(name, ran=False, reason=f"import failed: {exc!r}")

    status = fp8_matmul_path_c_status(target="cuda")
    if not status.available:
        return RouteResult(
            name, ran=False,
            reason=(
                f"fp8_matmul_path_c (cuda target) unavailable on this host: "
                f"{status.reason}."
            ),
        )

    M, N, K = shape.M, shape.N, shape.K
    try:
        # Quantize the IDENTICAL reference operands to e4m3 (per-tensor amax).
        a_fp8, scale_a = _quantize_pertensor_fp8(a_bf16)          # (M,K) u8, (1,) f32
        w_fp8, scale_b = _quantize_pertensor_fp8(w_bf16)          # (N,K) u8, (1,) f32

        # Build + compile the CUDA fragment-C dequant->T.gemm prim. out_idx=[4]
        # -> the kernel owns/returns the (M,N) fp32 output C (the 5th tensor).
        prim = fp8_scaled_matmul_path_c_cuda_prim(M=M, N=N, K=K, c_dtype="float32")
        if prim is None:
            return RouteResult(
                name, ran=False,
                reason=(
                    f"no legal cooperative tile divides M={M} N={N} K={K}; the "
                    "CUDA dequant->T.gemm kernel cannot tile this shape (RULE #1: "
                    "no silent partial-tile / no Metal-dot4 sibling on CUDA)."
                ),
            )
        kernel = tilelang.compile(
            prim,
            target=_as_cuda_target("cuda"),
            out_idx=[4],
            pass_configs={
                "tl.disable_tma_lower": True,
                "tl.disable_warp_specialized": True,
            },
        )
    except Exception as exc:
        # RULE #1: a compile failure is RECORDED with where+what, never hidden.
        return RouteResult(
            name, ran=False,
            reason=(
                f"cuda coop fp8 T.gemm compile failed: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=4)}"
            ),
        )

    try:
        def _call():
            return kernel(a_fp8, scale_a, w_fp8, scale_b)

        median_ms, out = _time_cuda(_call, iters=iters, warmup=warmup)
    except Exception as exc:
        # RULE #1: a dispatch failure is RECORDED with where+what, never hidden.
        return RouteResult(
            name, ran=False,
            reason=(
                f"cuda coop fp8 T.gemm dispatch failed: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=4)}"
            ),
        )

    tflops = shape.flops / (median_ms * 1e-3) / 1e12
    rel = _rel_err(out, ref_out)
    return RouteResult(
        name, ran=True, tflops=tflops, median_ms=median_ms,
        rel_err=rel, finite=True,
        reason="cuda fragment-C dequant->T.gemm (fp8_matmul_path_c_cuda_prim, sm_121)",
    )


# -----------------------------------------------------------------------------
# fp8 activation cast/round helper (reference; the e2e producer reuses this)
# -----------------------------------------------------------------------------
def fp8_quantize_activation_reference(x, *, recipe: str = "tensorwise"):
    """Reference fp8 activation quantizer. per-tensor reuses fp8_amax.py (CUDA-ready);
    mxfp8 is a block-32 reference (the e2e producer ports the swizzled codec from
    sparse_mla_blockscaled). RAISES on non-finite amax (reuse fail-loud)."""
    import torch

    if recipe == "tensorwise":
        from cppmega_mlx.nn._tilelang.fp8_amax import (
            fp8_amax_tilelang,
            fp8_quantize_tilelang,
        )
        amax = fp8_amax_tilelang(x).item()
        if not (amax == amax) or amax in (float("inf"), float("-inf")):
            raise FloatingPointError("fp8_quantize_activation: non-finite amax")
        inv_scale = FP8_E4M3_MAX / amax if amax > 0 else 1.0
        q = fp8_quantize_tilelang(x, inv_scale)
        scale = torch.tensor([amax / FP8_E4M3_MAX], dtype=torch.float32, device=x.device)
        return q, scale
    if recipe == "mxfp8":
        # block-32 E8M0 reference (power-of-2 scale, E4M3 cast). The e2e path
        # ports the swizzled hardware-layout codec from sparse_mla_blockscaled.
        xf = x.reshape(-1, 32).float()
        amax = xf.abs().amax(dim=1, keepdim=True)
        if not bool(torch.isfinite(amax).all().item()):
            raise FloatingPointError("fp8_quantize_activation(mxfp8): non-finite block amax")
        exp = torch.clamp(torch.floor(torch.log2(amax.clamp_min(1e-30))), -127, 127)
        scale = torch.pow(2.0, exp)  # E8M0 power-of-2
        q = torch.clamp(xf / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return q.reshape(x.shape), scale.reshape(x.shape[:-1] + (x.shape[-1] // 32,))
    raise ValueError(f"unknown recipe {recipe!r}")


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
def run(shapes: list[GemmShape], *, iters: int, warmup: int,
        do_te: bool, do_tl: bool) -> list[ShapeResult]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fp8_gemm_microbench requires CUDA (gb10 sm_121). RULE #1: refuse to "
            "report a CPU number as a tensor-core result."
        )
    cc = torch.cuda.get_device_capability(0)
    print(f"# device={torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]} "
          f"prod M={M_TOK} (bs={BS} seq={SEQ}) hidden={HIDDEN} ffn={FFN}", flush=True)

    results: list[ShapeResult] = []
    for shape in shapes:
        print(f"\n## {shape.name}  {shape.M}x{shape.N}x{shape.K}  "
              f"({shape.flops/1e9:.1f} GFLOP)", flush=True)
        bf16_tflops, bf16_ms, ref_out, a_bf16, w_bf16 = run_bf16(
            shape, iters=iters, warmup=warmup
        )
        res = ShapeResult(shape=shape, bf16_tflops=bf16_tflops, bf16_median_ms=bf16_ms)
        # operand bytes (A + W); fp8 = 1 byte, bf16 = 2 bytes
        a_numel = shape.M * shape.K
        w_numel = shape.N * shape.K
        res.bytes_bf16 = 2 * (a_numel + w_numel)
        res.bytes_fp8 = 1 * (a_numel + w_numel)
        res.scale_bytes_pertensor = 4 * 2  # two fp32 scalars
        res.scale_bytes_mxfp8 = (a_numel + w_numel) // 32  # one uint8 E8M0 per 32
        print(f"   bf16: {bf16_tflops:8.1f} TFLOPs  ({bf16_ms:.3f} ms)  "
              f"operand_bytes bf16={res.bytes_bf16/1e6:.1f}MB "
              f"fp8={res.bytes_fp8/1e6:.1f}MB (+{res.scale_bytes_mxfp8/1e6:.3f}MB mxfp8 scales)",
              flush=True)

        if do_te:
            for rk in ("mxfp8", "tensorwise"):
                r = run_te_fp8(shape, ref_out, a_bf16, w_bf16,
                               iters=iters, warmup=warmup, recipe_kind=rk)
                res.routes.append(r)
                if r.ran:
                    speedup = (r.tflops or 0) / bf16_tflops
                    print(f"   {r.route}: {r.tflops:8.1f} TFLOPs  ({r.median_ms:.3f} ms)  "
                          f"{speedup:.2f}x vs bf16  rel_err={r.rel_err:.4e}", flush=True)
                else:
                    print(f"   {r.route}: SKIP/GAP — {r.reason.splitlines()[0]}", flush=True)
        else:
            res.routes.append(RouteResult("R1_te", ran=False, reason="--no-te"))

        if do_tl:
            r = run_tilelang_fp8(shape, ref_out, a_bf16, w_bf16,
                                 iters=iters, warmup=warmup)
            res.routes.append(r)
            if r.ran:
                speedup = (r.tflops or 0) / bf16_tflops
                print(f"   {r.route}: {r.tflops:8.1f} TFLOPs  ({r.median_ms:.3f} ms)  "
                      f"{speedup:.2f}x vs bf16  rel_err={r.rel_err:.4e}", flush=True)
            else:
                print(f"   {r.route}: GAP — "
                      f"{r.reason.splitlines()[0] if r.reason else 'ok'}", flush=True)
        else:
            res.routes.append(RouteResult("R2_tilelang", ran=False, reason="--no-tl"))

        results.append(res)
    return results


def _best_fp8_route(res: ShapeResult) -> "RouteResult | None":
    """The fastest fp8 route that actually RAN for this shape (None if none ran)."""
    best: RouteResult | None = None
    for r in res.routes:
        if r.ran and r.tflops is not None:
            if best is None or r.tflops > (best.tflops or 0.0):
                best = r
    return best


def _summary(results: list[ShapeResult]) -> None:
    print("\n# SUMMARY (MEASURED) — fp8 vs bf16 at prod bs4 shapes", flush=True)
    print("# shape                  bf16_TFLOPs  best_fp8_TFLOPs  speedup  rel_err  route", flush=True)
    for res in results:
        best = _best_fp8_route(res)
        if best is not None:
            sp = (best.tflops or 0) / (res.bf16_tflops or 1)
            print(f"# {res.shape.name:22s} {res.bf16_tflops:10.1f}  "
                  f"{best.tflops:13.1f}  {sp:6.2f}x  {best.rel_err:.2e}  {best.route}",
                  flush=True)
        else:
            print(f"# {res.shape.name:22s} {res.bf16_tflops:10.1f}  "
                  f"{'(no fp8 route ran)':>13s}     —       —     —", flush=True)


def _emit_json(results: list[ShapeResult]) -> None:
    """Machine-parseable RESULT block. The Profile agent / orchestrator greps the
    single ``RESULT_JSON:`` line; a pretty block follows for humans. Per-shape:
    bf16 + best-fp8 TFLOPs, speedup, fp8-vs-bf16 operand byte sizes, every route
    (incl. the ones that RECORDED a gap — RULE #1: gaps are in the data, not hidden)."""
    payload: dict[str, object] = {
        "schema": "fp8_gemm_microbench/v1",
        "config": {
            "hidden": HIDDEN, "ffn": FFN, "seq": SEQ, "bs": BS, "m_tok": M_TOK,
            "fp8_e4m3_max": FP8_E4M3_MAX,
        },
        "shapes": [],
    }
    for res in results:
        best = _best_fp8_route(res)
        shape_obj: dict[str, object] = {
            "name": res.shape.name,
            "M": res.shape.M, "N": res.shape.N, "K": res.shape.K,
            "gflop": res.shape.flops / 1e9,
            "bf16_tflops": res.bf16_tflops,
            "bf16_median_ms": res.bf16_median_ms,
            "best_fp8_tflops": (best.tflops if best else None),
            "best_fp8_route": (best.route if best else None),
            "best_fp8_rel_err": (best.rel_err if best else None),
            "fp8_vs_bf16_speedup": (
                (best.tflops or 0.0) / res.bf16_tflops
                if (best and res.bf16_tflops) else None
            ),
            "bytes": {
                "operand_bf16": res.bytes_bf16,
                "operand_fp8": res.bytes_fp8,
                "operand_halving_ratio": (
                    res.bytes_bf16 / res.bytes_fp8 if res.bytes_fp8 else None
                ),
                "scale_sidecar_pertensor": res.scale_bytes_pertensor,
                "scale_sidecar_mxfp8": res.scale_bytes_mxfp8,
            },
            "routes": [asdict(r) for r in res.routes],
        }
        # asdict captures multi-line tracebacks; keep them but they JSON-escape fine.
        payload["shapes"].append(shape_obj)  # type: ignore[attr-defined]

    line = json.dumps(payload, separators=(",", ":"), default=str)
    print("\nRESULT_JSON: " + line, flush=True)
    print("\n# RESULT (pretty)\n" + json.dumps(payload, indent=2, default=str), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true",
                    help="prod local_gb10_quarter shapes (the only supported mode)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-te", action="store_true")
    ap.add_argument("--no-tl", action="store_true")
    args = ap.parse_args()

    if not args.prod:
        # RULE #1: do not silently invent a fake small shape; require --prod so the
        # measured numbers are always the production GEMM shapes.
        print("ERROR: pass --prod (the microbench only measures prod bs4 shapes).",
              file=sys.stderr)
        return 2

    shapes = prod_shapes()
    results = run(
        shapes, iters=args.iters, warmup=args.warmup,
        do_te=not args.no_te, do_tl=not args.no_tl,
    )
    _summary(results)
    _emit_json(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
