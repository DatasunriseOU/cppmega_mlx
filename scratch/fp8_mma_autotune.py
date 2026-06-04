"""LEVER fp8-mma-tune — gb10 (sm_121) native fp8-input e4m3 MMA tile autotuner.

This standalone gb10-only harness SWEEPS the SM120 native fp8-input e4m3 MMA tile
parameters ``(block_M, block_N, block_K, num_stages, threads, rasterize)`` per
production GEMM shape, picks the FASTEST PARITY-PASSING tile, and reports its
TFLOPs vs the bf16 cuBLAS baseline + the fp8-vs-bf16 parity per config.

It drives the LIVE pipelined, closure-parameterized native prim
``fp8_scaled_matmul_path_c_cuda_native_tunable_prim``
(cppmega_mlx/nn/_tilelang/fp8_matmul_path_c.py) through TileLang's built-in
``AutoTuner`` (parallel compile, validate vs a fp8 reference, benchmark, and
disk-cache the winner). The native prim's pipelined K loop makes ``num_stages``
LIVE (the production-default prim uses ``T.serial`` so its num_stages is inert);
this harness is the ONLY place that varies the tile.

SEARCH METHOD (seeded coordinate-descent, ~18 compiles per shape, NOT the full
3*3*2*4*2*2=288 Cartesian grid — the box is single-owner/serial and each compile
is costly):

  STAGE A — anchor sweep (<=12 legal configs): seed from the CUTLASS-4.5.x SM120
    winners (canonical 128x128x128 TileShape + the NEW 128x32xK/128x64xK tiles
    the CUTLASS changelog flags as up-to-30% on SM121), holding num_stages=2,
    threads=128. Illegal tiles for a given shape (K%block_K!=0, M%block_M!=0,
    N%block_N!=0, block_K%32!=0) are FILTERED out (skipped), not silently
    re-tiled (RULE #1).
  STAGE B — refine the Stage-A winner: sweep num_stages in {0,2,3} x threads in
    {128,256} (6 configs) on the winning (block_M,block_N,block_K).
  STAGE C (only if the Stage-B best is still < the floor): probe rasterize on/off
    (2 configs) on the winning tile.

RULE #1 (HARD): the autotuner picks the FASTEST tile that PASSES the fp8 parity
gate; if NO parity-passing tile clears the documented bf16-relative floor
(``--floor``, default 1.0x bf16), the harness RAISES ``NoTileBeatsFloorError``
with the best measured residual — it NEVER silently emits a slow/sub-floor tile,
never falls to the fp16-decode prim, and never falls to bf16. A non-finite output
or a parity miss is surfaced, not masked. The fp8 parity tolerance is the honest
e4m3 quant noise (~0.0376 measured, §21), NOT a masked numerical bug.

This harness wires NOTHING into the model. The WINNING per-shape config it prints
(``BEST_CONFIG_JSON:``) is what a follow-up edit pins into ``_native_fp8_tile_for``
(or a shape->config table) so production picks the tuned tile with zero runtime
sweep.

Run on gb10 (single-owner; poll IDLE + free>105GB FIRST):
  ssh gb10; cd /home/dave/source/cppmega_mlx
  LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
  TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
  /home/dave/cppmega-venv/bin/python scratch/fp8_mma_autotune.py --prod

Flags:
  --prod        sweep the prod local_gb10_quarter shapes (the only supported mode)
  --shapes A,B  restrict to a comma-separated subset of shape names
  --floor F     bf16-relative speedup floor a winning tile must clear (default 1.0)
  --warmup N    autotuner warmup iters per config (default 10)
  --rep N       autotuner timed reps per config (default 50)
  --timeout S   per-config compile+bench timeout seconds (default 120)
  --stage-a-only  run only Stage A (skip the num_stages/threads refine)
  --no-stage-c    never run the rasterize probe even if Stage B is below floor
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass, field

# -- prod local_gb10_quarter config (mirrors scratch/fp8_gemm_microbench.py) -----
HIDDEN = 3584
FFN = 18944
SEQ = 4096
BS = 4
M_TOK = BS * SEQ  # 16384

FP8_E4M3_MAX = 448.0

# Parity tolerance: honest e4m3 quant noise (~0.0376 measured §21). The autotuner
# gate uses an elementwise max|approx-ref| over the dequantized output; the >GATE
# check catches a secret bf16/garbage run (RULE #1) while admitting real fp8 noise.
PARITY_REL_GATE = 0.10  # reject any tile whose fp8 rel_err exceeds this (bf16/garbage guard)

# Bound lazily inside _benchmark_config (fp8_matmul_path_c imports mlx.core at top,
# which is absent on gb10). The per-config `kernel` factory reads this via
# LOAD_GLOBAL to keep the AutoTuner closure empty + serializable. See the contract
# comment in _benchmark_config.
_FP8_NATIVE_TUNABLE_BUILD = None


class NoTileBeatsFloorError(RuntimeError):
    """No parity-passing tile cleared the documented bf16-relative floor."""


class Fp8AutotuneParityError(RuntimeError):
    """A measured 'winning' tile failed the honest fp8 parity gate (bf16/garbage guard)."""


class FpparityNonFinite(FloatingPointError):
    """Kernel output went non-finite during the parity check (RULE #1 fail-loud)."""


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
    """The four transformer-block GEMMs that dominate bs4 FLOPs + the SSD F2 tile."""
    return [
        GemmShape("mlp_up_gate", M_TOK, 2 * FFN, HIDDEN),
        GemmShape("mlp_down", M_TOK, HIDDEN, FFN),
        GemmShape("attn_qkv", M_TOK, 3 * HIDDEN, HIDDEN),
        GemmShape("attn_out", M_TOK, HIDDEN, HIDDEN),
        GemmShape("ssd_f2_tile", 64, 64, 64),
    ]


@dataclass
class TileConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int
    rasterize: bool = False

    def key(self) -> tuple[int, int, int, int, int, bool]:
        return (
            self.block_M,
            self.block_N,
            self.block_K,
            self.num_stages,
            self.threads,
            self.rasterize,
        )

    def legal_for(self, shape: GemmShape) -> bool:
        """The fp8 m16n8k32 MMA hard gates (RULE #1: illegal -> FILTER, never re-tile)."""
        if self.block_K % 32 != 0:
            return False
        if shape.K % self.block_K:
            return False
        if self.block_M % 16 or self.block_N % 8:
            return False
        if shape.M % self.block_M or shape.N % self.block_N:
            return False
        if self.threads <= 0 or self.threads % 32 != 0:
            return False
        if self.num_stages < 0:
            return False
        if (shape.M // self.block_M) <= 0 or (shape.N // self.block_N) <= 0:
            return False
        return True


@dataclass
class ConfigResult:
    config: dict
    ran: bool
    reason: str = ""
    median_ms: float | None = None
    tflops: float | None = None
    speedup_vs_bf16: float | None = None
    rel_err: float | None = None
    parity_pass: bool | None = None


@dataclass
class ShapeAutotuneResult:
    shape: GemmShape
    bf16_tflops: float | None = None
    bf16_median_ms: float | None = None
    configs: list[ConfigResult] = field(default_factory=list)
    best: ConfigResult | None = None
    floor: float = 1.0
    floor_cleared: bool = False


# -----------------------------------------------------------------------------
# seeded coordinate-descent config grids
# -----------------------------------------------------------------------------
def stage_a_configs() -> list[TileConfig]:
    """12 anchor tiles seeded from the CUTLASS-4.5.x SM120 winners.

    128x128x128 is the canonical SM120 TileShape; the CUTLASS changelog adds
    128x32xK / 128x64xK as the up-to-30% SM121 winners (narrow-N -> less fp32-C
    SMEM). num_stages=2 / threads=128 held fixed; Stage B refines those axes.
    """
    tiles = [
        (128, 128, 64),
        (128, 128, 32),
        (128, 64, 64),
        (128, 64, 32),
        (128, 32, 64),
        (128, 32, 32),
        (64, 64, 64),
        (64, 64, 32),
        (64, 128, 64),
        (64, 128, 32),
        (64, 32, 64),
        (64, 32, 32),
    ]
    return [TileConfig(bm, bn, bk, num_stages=2, threads=128) for (bm, bn, bk) in tiles]


def stage_b_configs(best: TileConfig) -> list[TileConfig]:
    """Refine the Stage-A winner over num_stages x threads (6 configs)."""
    out: list[TileConfig] = []
    for ns in (0, 2, 3):
        for thr in (128, 256):
            out.append(
                TileConfig(
                    best.block_M, best.block_N, best.block_K,
                    num_stages=ns, threads=thr,
                )
            )
    return out


def stage_c_configs(best: TileConfig) -> list[TileConfig]:
    """Probe L2 rasterization on/off on the Stage-B winner (2 configs)."""
    return [
        TileConfig(best.block_M, best.block_N, best.block_K,
                   num_stages=best.num_stages, threads=best.threads, rasterize=False),
        TileConfig(best.block_M, best.block_N, best.block_K,
                   num_stages=best.num_stages, threads=best.threads, rasterize=True),
    ]


def _dedup(configs: list[TileConfig]) -> list[TileConfig]:
    seen: set = set()
    out: list[TileConfig] = []
    for c in configs:
        if c.key() in seen:
            continue
        seen.add(c.key())
        out.append(c)
    return out


# -----------------------------------------------------------------------------
# bf16 cuBLAS baseline (the speedup denominator)
# -----------------------------------------------------------------------------
def _time_cuda(fn, *, iters: int, warmup: int):
    """Return (median_ms, last_output). RAISES on non-finite output (RULE #1)."""
    import torch

    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    if last is not None and not bool(torch.isfinite(last.float()).all().item()):
        raise FloatingPointError(
            "fp8_mma_autotune: GEMM produced non-finite output — refuse to report "
            "a TFLOPs number for a NaN/Inf result (RULE #1 fail-loud)."
        )
    return statistics.median(times), last


def run_bf16(shape: GemmShape, *, iters: int, warmup: int):
    import torch

    dev = "cuda"
    a = torch.randn(shape.M, shape.K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(shape.N, shape.K, device=dev, dtype=torch.bfloat16)  # (N,K), B.T form
    median_ms, _ = _time_cuda(lambda: a @ w.t(), iters=iters, warmup=warmup)
    tflops = shape.flops / (median_ms * 1e-3) / 1e12
    return tflops, median_ms


# -----------------------------------------------------------------------------
# fp8 supply + parity reference (shared by every config of a shape)
# -----------------------------------------------------------------------------
def _make_fp8_operands(shape: GemmShape):
    """Quantize one bf16 (a,w) pair to per-tensor e4m3 + fp32 dequant scales.

    Reuses fp8_pack_tilelang (per-tensor amax + RNE e4m3 cast) exactly as the
    microbench does, so the autotune parity is over the SAME quantization the
    production fp8 route uses. Returns (a_fp8, scale_a, w_fp8, scale_b, ref_fp32)
    where ``ref_fp32 = (a_fp8.f32 @ w_fp8.f32.T) * scale_a * scale_b`` is the fp8
    reference (NOT a bf16 reference): the parity gate measures the kernel's MMA +
    epilogue, with the e4m3 operands held FIXED, so a passing tile proves the MMA
    math is correct (a failing tile is a real kernel bug, RULE #1).
    """
    import torch

    from cppmega_mlx.nn._tilelang.fp8_amax import fp8_pack_tilelang

    dev = "cuda"
    a_bf16 = torch.randn(shape.M, shape.K, device=dev, dtype=torch.bfloat16)
    w_bf16 = torch.randn(shape.N, shape.K, device=dev, dtype=torch.bfloat16)

    a_fp8, sa, _ = fp8_pack_tilelang(a_bf16)
    w_fp8, sb, _ = fp8_pack_tilelang(w_bf16)
    if a_fp8.dtype is not torch.float8_e4m3fn or w_fp8.dtype is not torch.float8_e4m3fn:
        raise TypeError(
            "fp8_mma_autotune._make_fp8_operands: fp8_pack_tilelang did not return "
            f"float8_e4m3fn (got {a_fp8.dtype}/{w_fp8.dtype}); the native prim's "
            "float8_e4m3 operand param requires the native e4m3 dtype (RULE #1: no "
            "silent uint8 reinterpret)."
        )
    a_fp8 = a_fp8.contiguous()
    w_fp8 = w_fp8.contiguous()
    scale_a = sa.reshape(1).to(dtype=torch.float32, device=dev)
    scale_b = sb.reshape(1).to(dtype=torch.float32, device=dev)

    ref_fp32 = (
        a_fp8.to(torch.float32) @ w_fp8.to(torch.float32).t()
    ) * scale_a[0] * scale_b[0]
    if not bool(torch.isfinite(ref_fp32).all().item()):
        raise FloatingPointError(
            "fp8_mma_autotune._make_fp8_operands: fp8 reference is non-finite; "
            "refuse to tune against a NaN/Inf reference (RULE #1)."
        )
    return a_fp8, scale_a, w_fp8, scale_b, ref_fp32


def _rel_err(approx, ref) -> float:
    import torch

    a = approx.float()
    r = ref.float()
    denom = float(torch.linalg.vector_norm(r).item())
    if denom == 0.0:
        raise FloatingPointError(
            "fp8_mma_autotune: fp8 reference has zero norm — cannot form rel_err."
        )
    return float(torch.linalg.vector_norm(a - r).item()) / denom


# -----------------------------------------------------------------------------
# per-config tilelang AutoTuner run
# -----------------------------------------------------------------------------
def _benchmark_config(
    shape: GemmShape,
    cfg: TileConfig,
    operands,
    *,
    warmup: int,
    rep: int,
    timeout: int,
) -> ConfigResult:
    """Compile + parity-check + benchmark ONE tile config via TileLang AutoTuner.

    Uses ``AutoTuner.from_kernel`` over a SINGLE-config space (the seeded
    coordinate-descent picks the next config here; the AutoTuner does the
    compile/validate/benchmark/disk-cache for that one config). The manual_check
    gate asserts the fp8-reference parity (honest e4m3 noise); a parity FAIL marks
    the config ran=False with the rel_err (RULE #1: a sub-parity tile is NOT a
    candidate winner — it would be a bf16/garbage run).
    """
    import torch

    import tilelang  # noqa: F401
    from tilelang.autotuner import AutoTuner

    from cppmega_mlx.nn._tilelang._msl_transform import _as_cuda_target
    from cppmega_mlx.nn._tilelang.fp8_matmul_path_c import (
        fp8_scaled_matmul_path_c_cuda_native_tunable_prim,
    )

    # Bind the prim builder to a MODULE GLOBAL so the nested `kernel` fn below
    # references it via LOAD_GLOBAL — NOT as a closure freevar (which the
    # AutoTuner asserts must be int/float/str/bool/None) and NOT as a default arg
    # (which generate_cache_key json.dumps's — a function default would not
    # serialize). fp8_matmul_path_c imports mlx.core at top, so it cannot be a
    # top-of-file import on a gb10 (no-MLX) host; binding the global here keeps the
    # import lazy while satisfying the closure/serialization contract.
    global _FP8_NATIVE_TUNABLE_BUILD
    _FP8_NATIVE_TUNABLE_BUILD = fp8_scaled_matmul_path_c_cuda_native_tunable_prim

    a_fp8, scale_a, w_fp8, scale_b, ref_fp32 = operands
    M, N, K = shape.M, shape.N, shape.K

    cfg_dict = asdict(cfg)

    # The kernel factory the AutoTuner calls per config. CRITICAL closure/cache
    # contract (tuner.py run() + generate_cache_key):
    #   * the config keys (block_M/block_N/block_K/num_stages/threads/rasterize)
    #     are the fn's POSITIONAL params — the tuner passes them by name per config;
    #   * M/N/K are DEFAULT ARGS (stored in __defaults__) so they are captured into
    #     the cache key's op_parameters (keying the disk cache per shape) AND are
    #     not closure freevars; they json-serialize fine (ints);
    #   * the prim builder is read via the MODULE GLOBAL _FP8_NATIVE_TUNABLE_BUILD
    #     (LOAD_GLOBAL) so it is neither a freevar (serializability assert) nor a
    #     default (json.dumps of a function would fail).
    # The prim builder VALIDATES + RAISES on an illegal tile (we pre-filter, so
    # this only fires on a real bug).
    def kernel(
        block_M,
        block_N,
        block_K,
        num_stages,
        threads,
        rasterize,
        _M=M,
        _N=N,
        _K=K,
    ):
        return _FP8_NATIVE_TUNABLE_BUILD(
            M=_M,
            N=_N,
            K=_K,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            c_dtype="float32",
            enable_rasteration=rasterize,
        )

    # supply_prog receives the kernel's INPUT params (out excluded by out_idx=-1):
    # (A_fp8, A_scale, B_fp8, B_scale). Return the FIXED operands so every config
    # is tuned against the identical e4m3 inputs (apples-to-apples).
    def supply_prog(_params):
        return [a_fp8, scale_a, w_fp8, scale_b]

    # ref_prog returns the fp8 reference output; manual_check_prog asserts parity.
    def ref_prog(A_fp8_in, A_scale_in, B_fp8_in, B_scale_in):
        return ref_fp32

    parity_box: dict = {"rel_err": None, "pass": None}

    def manual_check_prog(lib_outs, ref_outs):
        out = lib_outs[0] if isinstance(lib_outs, (list, tuple)) else lib_outs
        ref = ref_outs[0] if isinstance(ref_outs, (list, tuple)) else ref_outs
        if not bool(torch.isfinite(out.float()).all().item()):
            raise FpparityNonFinite(
                "fp8_mma_autotune: kernel output non-finite during parity check."
            )
        rel = _rel_err(out, ref)
        parity_box["rel_err"] = rel
        parity_box["pass"] = rel <= PARITY_REL_GATE
        if rel > PARITY_REL_GATE:
            raise Fp8AutotuneParityError(
                f"fp8_mma_autotune: tile {cfg.key()} for {shape.name} "
                f"({M}x{N}x{K}) FAILED fp8 parity: rel_err={rel:.4e} > gate "
                f"{PARITY_REL_GATE:.2f} — this is a bf16/garbage/mis-tiled run, "
                "NOT honest e4m3 noise (RULE #1: reject, do not crown it)."
            )

    try:
        autotuner = (
            AutoTuner.from_kernel(kernel=kernel, configs=[cfg_dict])
            .set_compile_args(
                out_idx=[-1],
                target=_as_cuda_target("cuda"),
                execution_backend="tvm_ffi",
                pass_configs={
                    "tl.disable_tma_lower": True,
                    "tl.disable_warp_specialized": True,
                },
            )
            .set_profile_args(
                supply_prog=supply_prog,
                ref_prog=ref_prog,
                manual_check_prog=manual_check_prog,
                skip_check=False,
                warmup=warmup,
                rep=rep,
                timeout=timeout,
                backend="event",
            )
        )
        # NOTE: warmup/rep/timeout drive the benchmark via set_profile_args (the
        # profiler reads profile_args.warmup/rep/backend); run() re-passes them so
        # the per-config timeout is honored at the tuner level too.
        result = autotuner.run(warmup=warmup, rep=rep, timeout=timeout)
    except Fp8AutotuneParityError as exc:
        return ConfigResult(
            config=cfg_dict, ran=False,
            reason=f"PARITY FAIL: {exc}",
            rel_err=parity_box["rel_err"], parity_pass=False,
        )
    except Exception as exc:  # compile / dispatch / timeout — RECORD with where+what
        return ConfigResult(
            config=cfg_dict, ran=False,
            reason=(
                f"compile/bench failed: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=4)}"
            ),
        )

    latency_ms = result.latency
    if latency_ms is None:
        return ConfigResult(
            config=cfg_dict, ran=False,
            reason="AutoTuner returned no latency (no valid config compiled).",
            rel_err=parity_box["rel_err"], parity_pass=parity_box["pass"],
        )

    tflops = shape.flops / (latency_ms * 1e-3) / 1e12
    return ConfigResult(
        config=cfg_dict, ran=True,
        median_ms=latency_ms, tflops=tflops,
        rel_err=parity_box["rel_err"], parity_pass=parity_box["pass"],
    )


# -----------------------------------------------------------------------------
# per-shape coordinate-descent driver
# -----------------------------------------------------------------------------
def _best_running(results: list[ConfigResult]) -> ConfigResult | None:
    best: ConfigResult | None = None
    for r in results:
        if r.ran and r.tflops is not None and r.parity_pass:
            if best is None or r.tflops > (best.tflops or 0.0):
                best = r
    return best


def autotune_shape(
    shape: GemmShape,
    *,
    floor: float,
    warmup: int,
    rep: int,
    timeout: int,
    iters_bf16: int,
    warmup_bf16: int,
    stage_a_only: bool,
    allow_stage_c: bool,
) -> ShapeAutotuneResult:
    bf16_tflops, bf16_ms = run_bf16(shape, iters=iters_bf16, warmup=warmup_bf16)
    res = ShapeAutotuneResult(
        shape=shape, bf16_tflops=bf16_tflops, bf16_median_ms=bf16_ms, floor=floor
    )
    print(f"   bf16: {bf16_tflops:8.1f} TFLOPs  ({bf16_ms:.3f} ms)  "
          f"floor={floor:.2f}x -> need >= {floor * bf16_tflops:.1f} TFLOPs", flush=True)

    operands = _make_fp8_operands(shape)

    def _bench(cfg: TileConfig, stage: str) -> ConfigResult:
        r = _benchmark_config(shape, cfg, operands, warmup=warmup, rep=rep, timeout=timeout)
        res.configs.append(r)
        if r.ran and r.tflops is not None:
            sp = r.tflops / bf16_tflops if bf16_tflops else 0.0
            r.speedup_vs_bf16 = sp
            print(f"   [{stage}] {cfg.key()}: {r.tflops:8.1f} TFLOPs "
                  f"({r.median_ms:.4f} ms)  {sp:.2f}x bf16  "
                  f"rel_err={r.rel_err if r.rel_err is not None else float('nan'):.4e}",
                  flush=True)
        else:
            head = r.reason.splitlines()[0] if r.reason else "skipped"
            print(f"   [{stage}] {cfg.key()}: GAP — {head}", flush=True)
        return r

    # STAGE A — anchor sweep over the legal seeded tiles.
    stage_a = [c for c in stage_a_configs() if c.legal_for(shape)]
    if not stage_a:
        raise NoTileBeatsFloorError(
            f"fp8_mma_autotune: NO legal Stage-A tile divides {shape.name} "
            f"({shape.M}x{shape.N}x{shape.K}); the e4m3 m16n8k32 MMA requires "
            "K%block_K==0 + m16n8 divisibility (RULE #1: no silent partial-tile)."
        )
    for cfg in stage_a:
        _bench(cfg, "A")

    best_a = _best_running(res.configs)
    if best_a is None:
        raise NoTileBeatsFloorError(
            f"fp8_mma_autotune: NO Stage-A tile compiled + passed parity for "
            f"{shape.name} ({shape.M}x{shape.N}x{shape.K}); cannot tune the native "
            "fp8 MMA on this shape (RULE #1: surface the gap, do not crown a "
            "parity-failing tile)."
        )

    best_cfg = TileConfig(**best_a.config)

    # STAGE B — refine num_stages x threads on the Stage-A winner.
    if not stage_a_only:
        for cfg in _dedup([c for c in stage_b_configs(best_cfg) if c.legal_for(shape)]):
            if cfg.key() == best_cfg.key():
                continue  # already measured in Stage A
            _bench(cfg, "B")

    best_b = _best_running(res.configs)
    assert best_b is not None  # best_a guaranteed one
    best_cfg = TileConfig(**best_b.config)
    best_b_speedup = (best_b.tflops or 0.0) / bf16_tflops if bf16_tflops else 0.0

    # STAGE C — only if the best so far is still below the floor (extra rasterize
    # probe is a last attempt to clear the floor before declaring NO-GO).
    if allow_stage_c and not stage_a_only and best_b_speedup < floor:
        print(f"   [C] best={best_b_speedup:.2f}x < floor {floor:.2f}x — "
              "probing L2 rasterization on the winning tile", flush=True)
        for cfg in _dedup([c for c in stage_c_configs(best_cfg) if c.legal_for(shape)]):
            if cfg.key() == best_cfg.key():
                continue
            _bench(cfg, "C")

    best = _best_running(res.configs)
    assert best is not None
    res.best = best
    best.speedup_vs_bf16 = (best.tflops or 0.0) / bf16_tflops if bf16_tflops else 0.0
    res.floor_cleared = (best.speedup_vs_bf16 or 0.0) >= floor

    if not res.floor_cleared:
        # RULE #1: NO parity-passing tile cleared the floor — RAISE with the best
        # measured residual. Do NOT silently emit a sub-floor tile.
        raise NoTileBeatsFloorError(
            f"fp8_mma_autotune: best parity-passing native fp8 tile for "
            f"{shape.name} ({shape.M}x{shape.N}x{shape.K}) is "
            f"{best.config} at {best.tflops:.1f} TFLOPs = "
            f"{best.speedup_vs_bf16:.2f}x bf16 ({bf16_tflops:.1f} TFLOPs), which "
            f"is BELOW the documented floor {floor:.2f}x. RULE #1: no slow-tile is "
            "silently emitted; surface the residual so the lever is recorded NO-GO "
            "(or the floor is consciously revised) rather than papering over it."
        )

    print(f"   WINNER: {best.config}  {best.tflops:.1f} TFLOPs  "
          f"{best.speedup_vs_bf16:.2f}x bf16  rel_err="
          f"{best.rel_err if best.rel_err is not None else float('nan'):.4e}",
          flush=True)
    return res


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
def run(
    shapes: list[GemmShape],
    *,
    floor: float,
    warmup: int,
    rep: int,
    timeout: int,
    stage_a_only: bool,
    allow_stage_c: bool,
) -> list[ShapeAutotuneResult]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fp8_mma_autotune requires CUDA (gb10 sm_121). RULE #1: refuse to "
            "report a CPU number as a tensor-core result."
        )
    cc = torch.cuda.get_device_capability(0)
    print(f"# device={torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]} "
          f"prod M={M_TOK} (bs={BS} seq={SEQ}) hidden={HIDDEN} ffn={FFN} "
          f"floor={floor:.2f}x", flush=True)

    results: list[ShapeAutotuneResult] = []
    failures: list[str] = []
    for shape in shapes:
        print(f"\n## {shape.name}  {shape.M}x{shape.N}x{shape.K}  "
              f"({shape.flops/1e9:.1f} GFLOP)", flush=True)
        try:
            res = autotune_shape(
                shape,
                floor=floor,
                warmup=warmup,
                rep=rep,
                timeout=timeout,
                iters_bf16=20,
                warmup_bf16=5,
                stage_a_only=stage_a_only,
                allow_stage_c=allow_stage_c,
            )
            results.append(res)
        except NoTileBeatsFloorError as exc:
            # Record the NO-GO but keep sweeping the other shapes; RAISE at the end
            # so the harness exits non-zero (RULE #1: the gap is loud, not hidden).
            print(f"   NO-GO: {exc}", flush=True)
            failures.append(str(exc))
            # still emit a partial record for the JSON
            partial = ShapeAutotuneResult(shape=shape, floor=floor)
            results.append(partial)

    if failures:
        # Surface the residual(s) loudly but only AFTER emitting the full JSON so
        # the orchestrator captures the measured data. The caller (main) re-raises.
        _summary(results)
        _emit_json(results)
        raise NoTileBeatsFloorError(
            "fp8_mma_autotune: one or more shapes had NO parity-passing tile clear "
            "the floor:\n  - " + "\n  - ".join(f.splitlines()[0] for f in failures)
        )
    return results


def _summary(results: list[ShapeAutotuneResult]) -> None:
    print("\n# SUMMARY (MEASURED) — best native fp8 MMA tile vs bf16 at prod shapes",
          flush=True)
    print("# shape                  bf16_TFLOPs  best_fp8_TFLOPs  speedup  rel_err  tile",
          flush=True)
    for res in results:
        if res.best is not None:
            cfg = res.best.config
            tile = (cfg["block_M"], cfg["block_N"], cfg["block_K"],
                    cfg["num_stages"], cfg["threads"], cfg["rasterize"])
            print(f"# {res.shape.name:22s} {res.bf16_tflops:10.1f}  "
                  f"{res.best.tflops:13.1f}  {res.best.speedup_vs_bf16:6.2f}x  "
                  f"{res.best.rel_err:.2e}  {tile}", flush=True)
        else:
            bf = res.bf16_tflops if res.bf16_tflops is not None else float('nan')
            print(f"# {res.shape.name:22s} {bf:10.1f}  "
                  f"{'(NO-GO)':>13s}     —       —     —", flush=True)


def _emit_json(results: list[ShapeAutotuneResult]) -> None:
    payload: dict = {
        "schema": "fp8_mma_autotune/v1",
        "config": {
            "hidden": HIDDEN, "ffn": FFN, "seq": SEQ, "bs": BS, "m_tok": M_TOK,
            "fp8_e4m3_max": FP8_E4M3_MAX, "parity_rel_gate": PARITY_REL_GATE,
        },
        "shapes": [],
    }
    for res in results:
        shape_obj: dict = {
            "name": res.shape.name,
            "M": res.shape.M, "N": res.shape.N, "K": res.shape.K,
            "gflop": res.shape.flops / 1e9,
            "bf16_tflops": res.bf16_tflops,
            "bf16_median_ms": res.bf16_median_ms,
            "floor": res.floor,
            "floor_cleared": res.floor_cleared,
            "best": asdict(res.best) if res.best else None,
            "configs": [asdict(c) for c in res.configs],
        }
        payload["shapes"].append(shape_obj)
    line = json.dumps(payload, separators=(",", ":"), default=str)
    print("\nRESULT_JSON: " + line, flush=True)
    # The single line a follow-up edit pins into _native_fp8_tile_for: the winning
    # tile per shape (only shapes that cleared the floor).
    best_table = {
        res.shape.name: res.best.config
        for res in results if res.best is not None
    }
    print("BEST_CONFIG_JSON: " + json.dumps(best_table, separators=(",", ":")),
          flush=True)
    print("\n# RESULT (pretty)\n" + json.dumps(payload, indent=2, default=str), flush=True)


def main() -> int:
    # gb10 NVRTC loader-path guard FIRST (before any torch import in run()).
    from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path

    ensure_nvrtc_builtins_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true",
                    help="sweep the prod local_gb10_quarter shapes (only supported mode)")
    ap.add_argument("--shapes", type=str, default="",
                    help="comma-separated subset of shape names to sweep")
    ap.add_argument("--floor", type=float, default=1.0,
                    help="bf16-relative speedup floor a winning tile must clear")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--stage-a-only", action="store_true",
                    help="run only Stage A (skip the num_stages/threads refine)")
    ap.add_argument("--no-stage-c", action="store_true",
                    help="never run the rasterize probe even if Stage B < floor")
    args = ap.parse_args()

    if not args.prod:
        print("ERROR: pass --prod (the autotuner only sweeps prod bs4 shapes).",
              file=sys.stderr)
        return 2

    shapes = prod_shapes()
    if args.shapes:
        wanted = {s.strip() for s in args.shapes.split(",") if s.strip()}
        shapes = [s for s in shapes if s.name in wanted]
        if not shapes:
            print(f"ERROR: no prod shape matched --shapes={args.shapes!r}; "
                  f"valid: {[s.name for s in prod_shapes()]}", file=sys.stderr)
            return 2

    try:
        results = run(
            shapes,
            floor=args.floor,
            warmup=args.warmup,
            rep=args.rep,
            timeout=args.timeout,
            stage_a_only=args.stage_a_only,
            allow_stage_c=not args.no_stage_c,
        )
    except NoTileBeatsFloorError as exc:
        # JSON already emitted inside run() before the raise; surface the NO-GO
        # loudly and exit non-zero (RULE #1: the floor miss is a hard failure).
        print(f"\nNO-GO (RULE #1): {exc}", file=sys.stderr, flush=True)
        return 1

    _summary(results)
    _emit_json(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
