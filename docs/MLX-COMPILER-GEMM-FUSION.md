# MLX SSD Chunk-Region Fusion — honest scope (auto vs hand-call)

Status: **MEASURED on Apple Metal** (this Mac, MLX fork 0.32.0.dev, python3.13,
metal=True). CUDA build+measure on gb10 is a follow-up (no CUDA on this Mac).

## TL;DR — what is and is NOT "mx.compile auto-fuses"

- **NO-GO on the literal ask**: `mx.compile` (MLX's native C++ fuser) does **NOT**
  auto-recognize the SSD chunk region and does **NOT** emit one kernel for it.
  MLX's `compile.cpp` `is_fusable()` is an elementwise typeid allowlist
  (Add/Mul/Exp/Tanh/Select/Broadcast); a Matmul/Reduce/Scan node is a hard fusion
  boundary, and `Compiled::eval_gpu` only emits per-element loops. We did **not**
  modify `compile.cpp` — there is no general tensor-core GEMM/scan codegen inside
  `Compiled::eval_gpu`, and writing one is a separate large subsystem.
  - Direct evidence: `mx.compile(reference_chain)` traces **2564 nodes** and the
    fuser produces only elementwise `CompiledMultiplyExp` /
    `CompiledBroadcastMultiply...` Compiled nodes — **zero `CustomKernel`**. MLX
    by itself never routes to the mono SSD kernel.

- **The reachable win (GO)**: a fusable-**REGION recognizer + proof-gated route**
  at the Python level — `mlx.ssd_region_fuse.compile_chunk_region()`. It
  RECOGNIZES the F0/F1/F2 chunk-scan composite (presented as a `ChunkScanRegion`
  marker), runs the z3 algebraic-equivalence proof + TLA/TLC race escalation, and
  on PROVEN **auto-substitutes** the whole region with ONE custom-primitive op
  (`ssd_chunk_scan_fused`, one `mx.fast.{cuda,metal}_kernel`). The collapse to ONE
  kernel is real and measured. The honest caveat: the **call site invokes
  `compile_chunk_region`**, not bare `mx.compile`; and the mono BODY is the
  hand-written wave8/`ssd_fused` kernel — the recognizer ROUTES, it does not
  synthesize tensor-core GEMMs from the graph.

So: "compiler-style recognizer + proof-gated route to a prebuilt mono kernel" is
TRUE and demonstrated; "mx.compile now synthesizes the fused tensor-core kernel
by itself" would be FALSE.

## Evidence (Apple Metal, bounded shapes batch=1, seqlen=128, nheads=2,
## headdim=16, dstate=8, chunk=64; LOCAL 80GB guard respected)

### Kernel count (graph export, unevaluated)
- MONO (`ssd_chunk_scan_fused`): graph = exactly **ONE `CustomKernel`** node
  (7 inputs A–G, 2 outputs). This is the mono-fusion proof.
- REFERENCE (unfused per-timestep recurrence): **4483** primitive nodes
  (ExpandDims 1024, Slice 640, Squeeze 770, Multiply 768, Exp 128, Broadcast 768,
  Add 256, Sum 128, Concatenate 1). The "6-kernel" SSD chain in production lowers
  to this many ops in the eager graph; the mono kernel collapses all of it to one
  launch with chunk state resident in threadgroup memory.

### Parity (fp32, over all elements)
- `ssd_chunk_scan_fused` vs reference recurrence: **max|dY| = 9.5e-7**,
  **max|d final_state| = 1.8e-7**.
- Via the recognizer route `compile_chunk_region` (proven → mono): same
  **max|dY| = 9.5e-7**.

### Timing (MEASURED, Metal, 50-iter mean after warmup)
- mono = **0.92 ms**, reference (unfused) = **5.0 ms** → **5.47x**.
- LABEL: MEASURED on Apple Metal. This is mono-custom-kernel vs the eager unfused
  recurrence, NOT vs a separately-tuned production 6-kernel CUDA path. The
  "vs 6-kernel" CUDA number is to be MEASURED on gb10 (EXTRAPOLATION until then).

### Proof gate (z3 + TLA/TLC) — NON-VACUOUS
- Correct F1 `cb = C @ B^T` rewrite → z3 **unsat** → `proved_by=z3` → route to mono.
- Injected transposed-A bug (`_inject_transpose_bug=True`) → z3 **sat** →
  `proved_by=refuted` → substitution **BLOCKED + RAISE**. The gate refutes a wrong
  fusion (non-vacuous). Counter-witness reported: `operand_maps: COUNTER-WITNESS
  [k = 0, ...]`.
- z3-unknown → TLA+/TLC bounded single-writer race obligation (tlc-bounded), via
  `verify_escalation.verify_with_escalation`. Prover genuinely uses z3
  (`Solver(`, `unsat`, `sat` present in `_gemm_rewrite_proof`).

### Control paths (RULE #1, all verified)
- proven → route to mono.
- z3-refuted → RAISE (substitution blocked).
- unproven + forced (`MLX_SSD_REGION_FUSE_FORCE`) → RAISE.
- unproven + auto → unfused reference chain (correct math, more kernels — NOT a
  degraded/zeroed fallback). Default OFF behind `MLX_SSD_REGION_FUSE`.

## Build note
Both `python/mlx/ssd_fused.py` and `python/mlx/ssd_region_fuse.py` are **pure
Python** modules in the MLX package; our two commits touch **no C++**
(`compile.cpp` untouched). The editable fork install (mlx 0.32.0.dev,
cpython-313 `core.so`) loads them directly — **no MLX C++ rebuild required**, and
the existing working build was not clobbered.

## Files
- `/Volumes/external/sources/mlx/python/mlx/ssd_fused.py` — mono custom-primitive
  kernel (Metal MSL + CUDA bodies), ONE `mx.fast.{metal,cuda}_kernel` op.
- `/Volumes/external/sources/mlx/python/mlx/ssd_region_fuse.py` — region
  recognizer + proof-gated route (`ssd_chunk_region`, `compile_chunk_region`,
  `prove_region`, `_escalate_tlc`).
- Prover: `cppmega_mlx/nn/_tilelang/_gemm_rewrite_proof.py` +
  `verify_escalation.py` (z3 + TLA/TLC escalation).
