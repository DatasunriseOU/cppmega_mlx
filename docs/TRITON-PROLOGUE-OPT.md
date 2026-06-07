# Triton Prologue Optimizer — routed Tri-Dao chunk kernels

Gated prologue optimizer for the routed-triton path in the TileLang triton
frontend (`poc/triton_frontend`). Three transforms were scoped; this records
which landed, with MEASURED numbers reproducible from committed HEAD on gb10
(sm_121, aarch64-linux).

Source HEAD (tilelang): `29888fee` — "feat(triton-frontend): prologue-opt
transforms 1+3 (drop int32-overflow guards, fold masks) + gated
thread-distribute machinery". The gb10 source tree
(`/home/dave/source/tilelang`) working copy of the 4 touched files is
byte-identical (shasum) to this commit; the measurements below regenerate the
PrimFunc + CUDA from HEAD and re-run.

## The gap (recap)

Routed `_chunk_scan_bwd_dstates` is CORRECT through our stack at §P1 but
~1290x slower than native (≈1473 ms vs 1.15 ms/kernel). strip-and-time:
~100% of the ms is the PROLOGUE; the cooperative GEMM half (T.copy/T.gemm,
tensor-core, threadIdx-swizzled) is ~0 ms. The prologue is serial scalar
index/mask arrays that every thread of every block rebuilds element-by-element.

## What landed

| Transform | Status | Evidence |
|---|---|---|
| (1) Fold address arithmetic into T.copy region | PARTIAL — mask/guard-chain folding only | `_emit_andi` folds `constant-true & x -> x` (gated). Deeper region-index folding NOT done. |
| (2) Thread-distribute the prologue | IMPLEMENTED, gated OFF | machinery works (184 threadIdx when forced ON) but ptxas: too much shared data. |
| (3) Drop int32-overflow guards | LANDED | `arith.cmpi addr<=INT32_MAX / INT32_MIN<=addr -> constant-true tile`. |

## MEASURED — IR / CUDA shape (regen from HEAD, gb10)

Generated CUDA for `_chunk_scan_bwd_dstates` (de-monomorphized §P1 PrimFunc,
symbolic flat-arg extents):

| build | json bytes | .cu bytes | for-loops | int32 guards | threadIdx | mma |
|---|---|---|---|---|---|---|
| `prologue_opt=False` (baseline) | 497670 | 37174 | 158 | 28 | 25 | 2 |
| `prologue_opt=True`  (HEAD)     | 332061 | 26685 | 107 |  0 | 25 | 2 |

Transforms 1+3 remove **51 for-loops and all 28 int32-overflow-guard
references** (the `2147483647` / `-2147483648` literals + their bool[] AND
loops). threadIdx count is unchanged (25, all inside the GEMM) — confirming
transform 2 did NOT alter thread distribution in the shipped build.

## MEASURED — §P1 parity (gb10, CUDA_LAUNCH_BLOCKING)

`parity_prod_dstates.py`, real production config
(b1 nh112 hd64 ds64 nc64 cs64, numel 29,360,128, grid (1,64,112)),
2 K-trips/chunk (cs64 / BK32):

```
MAXDIFF = 4.882812e-04   ALLCLOSE_1e-3 = True   PASS
```

Bit-identical to the pre-change baseline (the gate is a numeric no-op).

Small real-strided multi-K-trip (b1 nh8 s128 nc2, cs64/BK32 → 2 trips,
vs native mamba_ssm kernel):

```
MAXDIFF = 4.577637e-05   ALLCLOSE_1e-3 = True   PASS
```

## MEASURED — §P1 EXEC ms (routed, interleaved OFF/OPT/OFF/OPT)

```
OFF (prologue_opt=False): 1468.21, 1477.25 ms   mean ~1472.7
OPT (prologue_opt=True):  1469.99, 1476.92 ms   mean ~1473.5
```

**Per-transform delta (1+3 together): +0.8 ms (~0.05%) — within run-to-run
noise. No measurable §P1 speedup.** Ratio vs native 1.15 ms: ~1281x (was
~1290x). The 51 dropped serial loops are cheap relative to the dominant cost;
the remaining serial prologue (materialized [64]/[2048]/[4096] index+mask
arrays and per-lane redundant address arithmetic) is untouched by 1+3. Only
transform 2 would move the needle, and it is blocked (below).

## Transform 2 blocker (MEASURED, RULE #1 fail-closed)

Forcing `thread_distribute=True` (prologue_opt=True): the walker DOES
thread-distribute — threadIdx jumps 25 → 184, the [64]/[4096] tiles become
cooperative shared-fill loops with `__syncthreads`. But it promotes 63 prologue
tiles to `__shared__` (multiple [4096] int64/int/bool arrays), and NVRTC/ptxas
fails:

```
ptxas error : Entry function '_chunk_scan_bwd_dstates_kernel' uses too much
              shared data (0x1b460 bytes, 0x18c00 max)
```

111712 bytes/block vs the 101376-byte (99 KiB) sm_121 limit. Distribution
requires shared scope (cooperative fill + barrier; a thread-local distributed
write would leave 127/128 slots per lane uninitialized — forbidden). The
correct fix is full transform (1): fold the addressing into the T.copy region
indices/predicate so these tiles are never materialized in ANY scope; the small
residue can then be thread-distributed in shared within budget. Until
address-folding lands, distribution is gated OFF — it RAISES (ptxas error)
rather than emit a racy local write or silently overflow. This is the honest
blocker, not a silent fallback.

## MEASURED — other-paths no-regression

- **fla** (`fla_chunk_delta_h_real_ttir.mlir`): generated CUDA is
  BYTE-IDENTICAL with prologue_opt on/off (sha `581db03dbb91`, len 29515). The
  gate is a no-op — fla has no int32-overflow guards / constant-true ANDs.
- **matmul** (M=N=K=64): the .cu DIFFERS (opt_ON 14727 B vs opt_OFF 22126 B —
  opt also folds matmul's addressing guards), but the OUTPUT is **bit-identical
  on vs off (maxdelta 0.000e+00)** and **EXEC ms is identical: 1.343 ms (opt)
  vs 1.343/1.346 ms (off)**. The .cu size delta is dead-code prologue arrays
  ptxas eliminates; the GEMM compute is unchanged. No regression.

Note: the IMPLEMENT-note claim "non-routed matmul/fla paths default OFF … and
are untouched" is imprecise — any kernel routed through `from_ttir` gets
`prologue_opt=True` by default, so the gate DOES fire on matmul's guards. But
it is a proven numeric + perf no-op there (bit-identical output, identical ms).

## Gate

`from_ttir(prologue_opt=bool)` (default True). `prologue_opt=False` reproduces
the 497670-byte / 37174-byte / 158-loop / 28-guard baseline exactly.
`routed_triton_thread_distribute` (transform 2) is a distinct sub-gate, OFF by
default and only set via `from_ttir(..., thread_distribute=True)` for dev.

## GO / NO-GO

**Partial GO.** Transforms (3) + (1-partial) LANDED, correct, committed, and
§P1 bit-correct (MAXDIFF 4.88e-04) + small multi-K-trip (4.58e-05). They strip
51 serial loops + 28 guards from the IR with ZERO §P1 perf regression and ZERO
other-path regression. They do NOT yet move the §P1 ms (still ~1473 ms) because
the dominant serial prologue survives. Transform (2) is implemented but blocked
on shared-mem budget; landing the perf win requires full transform (1)
region-index folding first. Honest: no §P1 speedup yet; the cleanup is real and
safe; the speedup path is identified and gated correctly.

## Reproduce from HEAD (gb10)

```
cd /home/dave/source/tilelang   # working files == tilelang 29888fee (shasum-verified)
source /home/dave/cppmega-venv/bin/activate
# regen §P1 PrimFunc (prologue_opt=True default) + parity + timing:
python poc/triton_frontend/_test_harness/tridao_parity/emit_pf_json.py _chunk_scan_bwd_dstates
python poc/triton_frontend/_test_harness/tridao_parity/parity_prod_dstates.py
```
