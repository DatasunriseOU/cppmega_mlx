# Mamba3 (+M2RNN) Backward → Chunked Parallel Scan: Ready-Code Sources & Port Guide

Date: 2026-05-31
Status: HOW + READY-CODE consolidation. The premise is correct — the chunked
parallel-scan forward AND backward is a solved, documented problem. We REUSE/PORT
the proven implementations rather than reinvent. This doc ranks the ready code,
gives the concrete algorithm for fwd+bwd, maps it onto **this** codebase's mamba3
(selective SSM + ngroups=1 B/C RMSNorm + causal conv), specifies the Metal port
path, settles licensing, and gives an incremental port plan with parity gates.

---

## 0. Where this codebase is today (the gap)

The current mamba3/m2rnn Path C is a **serial-over-time, parallel-over-channels**
scan, NOT a chunked parallel scan. Concretely, in
`/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/mamba3_path_c.py`
the forward `@T.prim_func fwd` (lines ~948-998) is:

- `with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS)` — one lane per
  `(b, h, p)` = `(BATCH * HEADS * HEADDIM)` lanes.
- `h_state = T.alloc_local((STATE,))` — per-lane register state of size `STATE`.
- `for t in T.serial(SEQ):` — **the whole sequence is walked serially per lane**,
  inner `for n in T.serial(STATE)` does `h = decay*h + x*B; y += h*C`.

The backward (lines ~1100-1500+) is a **serial reverse scan with snapshot/replay**
(`_fallback_recurrence_scan_plan`, `_BWD_SNAPSHOT_BLOCK`, "reverse pass consumes
explicit state snapshots", reconstruct `h_{t-1}` from `h_t`). This is precisely the
"serial reverse scan + checkpoint-replay" the research flags as the slow path.

The B/C RMSNorm (ngroups=1) is done in **host MLX** before the scan, in
`/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/mamba3.py`:
`Mamba3ReferenceBlock.transform_bc` → `_rms_norm_last(B/C)` over the last (state)
axis, then `* B_norm_weight + B_bias`. It is already a **separate reduction**, not
fused into the scan — which (Section 4) is exactly what the production references do.

**Goal:** replace the serial-over-time scan (fwd) and serial-reverse+snapshot scan
(bwd) with a two-level chunked decomposition: intra-chunk dense matmuls + a short
inter-chunk (reverse, for bwd) state-boundary scan. The serial dependency shrinks
from `O(SEQ)` to `O(SEQ/chunk)`, and the per-token/per-chunk gradient work becomes
embarrassingly parallel across `(batch, nchunks, heads)`.

Local port baseline (forward already mirrored in TileLang, no backward exists):
- `/Volumes/external/sources/tilelang/examples/linear_attention/example_mamba_chunk_scan.py`
- `/Volumes/external/sources/tilelang/examples/linear_attention/example_mamba_chunk_state.py`
- Existing TileLang chunked **backward** to generalize:
  `/Volumes/external/sources/tilelang/examples/linear_attention/example_linear_attn_bwd.py`
  and the GDN/KDA bwd kernels under `/Volumes/external/sources/tilelang/examples/gdn/`
  and `/Volumes/external/sources/tilelang/examples/kda/`.

---

## 1. RANKED ready-to-port sources

Ranking criterion: closeness to mamba3's recurrence (scalar/diagonal real decay +
per-step data-dependent B/C, the trapezoidal/RoPE/MIMO pieces layered on top),
presence of a real **backward**, and portability of the *algorithm* (not the
backend) into the TileLang emitter / MSL.

### TIER 1 — primary algorithm spec (fwd + bwd), closest math

**1. state-spaces/mamba — SSD chunked scan (THE primary source).** Apache-2.0.
Mamba3's per-step recurrence `h[t] = exp(A[t]*dt[t])*h[t-1] + x[t]*B[t]`,
`y[t] = sum(h[t]*C[t]) + D*x[t]` is the Mamba2/SSD scalar-decay recurrence with
extra (trapezoidal/RoPE/MIMO) parameterization on top. SSD is the canonical chunked
fwd+bwd. Port these 6 files (the bwd is split across them):
  - `mamba_ssm/ops/triton/ssd_combined.py` — `MambaChunkScanCombinedFn` (autograd),
    `_mamba_chunk_scan_combined_fwd` / `_bwd` (5-stage orchestration), and the fused
    `_chunk_scan_chunk_state_bwd_dx_kernel` (dx/ddt/dD in one pass).
    REUSE: the **end-to-end fwd+bwd orchestration order** (the blueprint).
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py
  - `mamba_ssm/ops/triton/ssd_state_passing.py` — `_state_passing_fwd` (serial
    chunk-boundary scan `states[c]=exp(dA_chunk[c])*states[c-1]+new_states[c]`) and
    **`_state_passing_bwd`** (reverse chunk scan `dstates=exp(dA_cs)*dstates+dout` →
    `dinitstates, ddA_cs`). REUSE: this **one-line reverse chunk-boundary recurrence**
    is the exact replacement for the current snapshot/replay bwd. Lowest port cost,
    highest leverage.
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py
  - `mamba_ssm/ops/triton/ssd_chunk_state.py` — `_chunk_cumsum_fwd/_bwd` (dt*A cumsum
    + softplus/dt_limit chain rule — maps to our `dt = softplus(dd_dt + dt_bias)` and
    `A = min(-softplus(dd_A), -A_floor)`), `_chunk_state_fwd`, `_chunk_state_bwd_dx/db`.
    REUSE: per-chunk state fwd/bwd + the dt/A cumsum gradient (our `compute_dacs_segsum`,
    `bwd_dadt_fused` helpers already mirror this — see Section 5).
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py
  - `mamba_ssm/ops/triton/ssd_chunk_scan.py` — `_chunk_scan_fwd` (already mirrored by
    our local `example_mamba_chunk_scan.py`), `_chunk_scan_bwd_dz/_dstates/_dC/_dcb/
    _ddAcs_stable`. REUSE: the output-side (Y path) gradients incl. the `silu(z)` gate
    split (`_chunk_scan_bwd_dz` ⇒ our `dz`).
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py
  - `mamba_ssm/ops/triton/ssd_bmm.py` — `_bmm_chunk_fwd/_bwd` (per-chunk `CB=C@B^T`,
    causal). REUSE: trivial chunked GEMM, easiest piece; bwd routes dCB→dB,dC.
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_bmm.py
  - `mamba_ssm/modules/ssd_minimal.py` — `ssd_minimal_discrete` + `segsum` (~30-line
    pure-PyTorch Listing-1). REUSE: the **numerical oracle** for the chunked forward
    (autograd gives the bwd oracle for free).
    URL: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py

**Why Tier-1:** same scalar/diagonal real-decay recurrence as mamba3's core, has a
full hand-written backward, and the forward is already partially ported in this repo.

### TIER 2 — alternative chunked fwd+bwd, cleaner small-kernel decomposition

**2. fla-org/flash-linear-attention — chunked linear-attention/delta kernels.** MIT.
FLA realizes Mamba2/SSD as the scalar-decay case of `simple_gla`. Cleaner per-kernel
split and a license (MIT) friendlier than Apache for verbatim mirroring. Port targets:
  - `fla/ops/simple_gla/chunk.py` + `fla/ops/common/chunk_h.py` — `chunk_fwd_h` /
    `chunk_bwd_dh` = the **FLA realization of Mamba2 SSD chunked scan** (scalar
    data-dependent decay `h_t=exp(g_t)h_{t-1}+k_t^T v_t`). Lowest-difficulty chunked
    recurrence, best *starting point*. REUSE: the inter-chunk fwd_h + reverse bwd_dh.
    URL: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/simple_gla/chunk.py
  - `fla/ops/common/chunk_o.py` — `chunk_fwd_o` (intra-chunk output =
    `q@h + tril(q@k^T * decay)@v`) and `chunk_bwd_dqkwg`. REUSE: the intra-chunk
    attention-like block + its gradients. Maps to our intra-chunk Y term.
    URL: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
  - `fla/ops/common/chunk_delta_h.py` — `chunk_gated_delta_rule_bwd_dhu`
    (`for i_t in range(NT-1,-1,-1)` reverse chunk scan, decay-weighted dh accumulation).
    REUSE: the **Triton-tile reverse chunk scan** that maps most directly to TileLang
    `T.gemm`/`T.Parallel` (this repo already has analogs under `examples/gdn`/`examples/kda`).
    URL: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
  - `fla/ops/gated_delta_rule/fused_recurrent.py` — token-serial reference encoding the
    exact Mamba2 dt gate (`g=-exp(A_log)*softplus(g+dt_bias)`). REUSE: a **serial
    correctness oracle** that already matches our `A=min(-softplus(dd_A),-A_floor)`,
    `dt=softplus(dd_dt+dt_bias)` parameterization.
    URL: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py

**Why Tier-2:** structurally identical two-level decomposition, MIT-licensed, and the
repo already has TileLang ports of the GDN/KDA siblings to copy index math from.

### TIER 3 — the actual MSL PORT REFERENCE (parallel scan on Apple GPU, fwd + bwd)

**3. D-CSIL/mlx-recurrence — `mlx_recurrence/ssm_scan.py`.** MIT (verified:
`Copyright (c) 2026 Paul O. Derrington, Jr.`). The ONLY Apple-Metal SSM scan with a
**real GPU backward kernel** (`mx.fast.metal_kernel` + `@mx.custom_function` VJP).
  - `_ssm_forward_kernel` / `_ssm_backward_metal` / `_selective_scan_metal_impl + .vjp`.
  - REUSE: (a) the **MSL kernel bodies** (~30 lines of plain arithmetic each) as the
    template for the per-lane recurrence and reverse-adjoint recurrence; (b) the VJP
    wiring that **persists `h_all` as a forward OUTPUT** so bwd does not re-run fwd
    (avoids "Metal buffer aliasing at model scale producing corrupt gradients") — this
    is the same `@mx.custom_function` pattern our `mamba3.py::_mamba3_mimo_apply_with_state`
    already uses; (c) the half-fused pattern (per-thread grads on GPU, contracted-dim
    reductions in MLX) which matches our existing "host sums over P" decomposition.
  URL: https://github.com/D-CSIL/mlx-recurrence/blob/main/mlx_recurrence/ssm_scan.py
  Caveat: small single-author repo (v0.2.0). Treat the bwd math as a reference to
  re-derive/validate against state-spaces/mamba `selective_scan_bwd_kernel.cuh`
  (the authoritative adjoint), not as authoritative itself.

### TIER 4 — Metal idiom references (forward/inference only, NO backward)

**4. ggml-org/llama.cpp — `ggml/src/ggml-metal/ggml-metal.metal`.** MIT.
  - `kernel_ssm_scan_f32` (Mamba-2): simdgroup-lane = d_state index; `s=s0*dA+B[i0]*x_dt`;
    `y=simd_sum(s*C[i0])` reduces over state across 32 lanes in ONE instruction; outer
    loop batches `sgptg` tokens via threadgroup shared mem (precompute softplus(dt)).
  - REUSE: the **`simd_sum` state-reduction + simdgroups-per-threadgroup token-batching**
    idiom — directly relevant to ngroups=1 small-state reductions and to keeping the
    intra-chunk reduction on-GPU. Inference-only (no bwd).
  URL: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal
  (decode single-step `selective_state_update` reference: ml-explore/mlx-lm
  `mlx_lm/models/mamba_selective_state_update.py`, MIT — `fast::exp`+fma update.)

### TIER 5 — true parallel-over-L scan (only if channel parallelism is insufficient)

**5. proger/accelerated-scan (`ref.py`/`warp.py`, MIT)** and **state-spaces/mamba
`csrc/selective_scan/reverse_scan.cuh` (Apache/BSD).** Blelloch up/down-sweep where the
SAME parametric tree does fwd OR reverse by toggling `reverse`; `warp.cuh
warpscan_backward` is the parallel reverse scan. Build this ONLY for long-L/tiny-batch
where `BATCH*HEADS*HEADDIM` lanes underfill the GPU. Caution (alxndrTL/mamba.py author):
naive array-op Blelloch is SLOWER than serial-over-time on Apple GPU — re-derive as an
in-kernel `simd_shuffle_up` scan if needed (template: WebGPU subgroup-scan +
GPU-Gems-3 ch.39). Default to parallel-over-channels (Tier 3/4) first.

---

## 2. The concrete HOW — chunked parallel scan fwd + bwd

Notation matched to this repo: `x (B,S,H,P)`, `B,C (B,S,H,N)` after group→head
broadcast (`_broadcast_groups_to_heads`), `A,dt (B,S,H)`, `D (H,)`, `h0 (B,H,P,N)`,
`y (B,S,H,P)`. Chunk size `L` (start with `L=64`; the local example uses larger).
Per-step decay `a[t] = exp(A[t]*dt[t])`. Work in **log space**: cumulative
`acc[t] = sum_{k<=t within chunk} A[k]*dt[k]`; intra-chunk decay between `m>=k` is
`exp(acc[m]-acc[k])`, clamped `exp(min(acc[m]-acc[k], 0))` so the causal-masked term
never exponentiates a positive argument (numerical-stability rule from SSD).

### 2a. FORWARD — 5 stages (mirror `_mamba_chunk_scan_combined_fwd`)

Reshape `S = nchunks * L`. Per chunk `c`, per head `h`:

1. **chunk cumsum** (`_chunk_cumsum_fwd`): `dA_cumsum[c] = cumsum(A*dt)` within chunk;
   also realize our `dt=softplus(dd_dt+dt_bias)` and `A=min(-softplus(dd_A),-A_floor)`.
2. **chunk state** (`_chunk_state_fwd`): `states[c] = sum_{k in c} B[k] *
   exp(dA_last - dA_k) * x[k]` → shape `(B,nchunks,H,P,N)`.
3. **state passing** (`_state_passing_fwd`, the ONLY serial part, length `nchunks`):
   `states[c] = exp(dA_chunk[c]) * states[c-1] + states_new[c]`, seeded by `h0`.
4. **bmm** (`_bmm_chunk_fwd`): `CB[c] = C[c] @ B[c]^T` (chunk_size×chunk_size, causal).
5. **chunk scan** (`_chunk_scan_fwd`, already in `example_mamba_chunk_scan.py`):
   `Y = tril(CB * exp(segsum decay) * dt) @ x        (intra-chunk, attention-style)`
   `  + C @ states_prev * exp(dA_cumsum)             (inter-chunk, low-rank)`
   `  + D * x                                          (skip)`,
   then gate `y = silu(z) * Y` (our `z_val*sigmoid(z_val)`).

All of stages 1,2,4,5 are **parallel across `(B, nchunks, H)`**; only stage 3 is
serial (over `nchunks`, tiny — a single short TileLang loop or a Blelloch micro-scan).

### 2b. BACKWARD — recompute-chunk-state + REVERSE chunk-boundary scan

Replace the current snapshot/replay reverse scan with this (mirror
`_mamba_chunk_scan_combined_bwd` + `_state_passing_bwd`). The key fact (Martin & Cundy
§2.2; Chiu): **the backward of a first-order linear recurrence is itself a reverse
linear recurrence = another associative scan**, so it is fully chunk-parallel except
for the short inter-chunk carry.

- **P1. Recompute (activation checkpointing).** Recompute per-chunk `dA_cumsum`, `CB`,
  and per-chunk boundary `states` via the forward stages 1,2,3. Memory is `O(nchunks)`,
  not `O(SEQ)`. (Our current code already saves boundary state via the snapshot plan —
  reuse that buffer, drop the per-token replay.)
- **P2. Gate split** (`_chunk_scan_bwd_dz`): split `dz`, `dD`, and `dout_x` (grad after
  the `silu(z)` gate). `dz` and `dx`-direct need no reduction (our existing "direct
  outputs" decomposition).
- **P3. dstates + reverse chunk scan.** `_chunk_scan_bwd_dstates` → grad into chunk
  states; then **`_state_passing_bwd`**: loop chunks `c = nchunks-1 .. 0`:
  `dstates = exp(dA_cs[c]) * dstates + dout[c]`, producing per-chunk `dstates`,
  `dinitstates` (= our `dh0`), and `ddA_cs`. **This single line replaces the entire
  serial-token reverse scan + snapshot/replay.**
- **P4. Parallel per-chunk grads.** With chunk-boundary `dstates` known, run
  independently across chunks: fused `_chunk_scan_chunk_state_bwd_dx` → `dx, ddt, dD`;
  `_chunk_state_bwd_db` → `dB (+ddA_next)`; `_chunk_scan_bwd_dC` → `dC (+ddA_prev)`;
  `_chunk_scan_bwd_dcb` → `dCB`.
- **P5. Route dCB** via `_bmm_chunk_bwd` twice → into `dB` and `dC`.
- **P6. Assemble ddA** = `ddA_next + ddA_prev + ddA_stable` (`_chunk_scan_bwd_ddAcs_stable`).
- **P7. cumsum bwd** (`_chunk_cumsum_bwd`): final `ddt, dA, ddt_bias` with the
  softplus + dt_limit chain rule. For mamba3 also chain through the trapezoidal scale
  and RoPE-angle cumsum (our existing `bwd_dadt_fused`, `bwd_dtrap_ddt`,
  `compute_dacs_segsum` helpers — Section 5).

Accumulation/determinism note (from SSD): `ddt`/`dD`/`ddA` buffers accumulate across
chunks into the same locations — **zero-init the gradient buffers** (Triton uses
`pre_hook init_to_zero`) and prefer a tiled-workspace + finalize step over naive
atomics for determinism.

### 2c. Existing TileLang backward to GENERALIZE (don't start from scratch)

`/Volumes/external/sources/tilelang/examples/linear_attention/example_linear_attn_bwd.py`
already has BOTH levels: the intra-chunk parallel bwd (triangular-masked `ds` via
`T.if_then_else`) AND a cross-chunk **"Calculate dK, dV reversely"** loop
(`for i in Pipelined(1,NT+1); start=NT-i`) that carries `dh` backward via
`T.gemm(q,do,dh)`. This is the dh reverse-chunk carry already in TileLang. The mamba3
bwd is this file generalized with: (1) the decay/state handling of `_state_passing_bwd`,
and (2) the dt/A/trap/RoPE cumsum gradients. Validated there vs serial ref with
`torch.allclose(atol=1e-2, rtol=1e-2)`.

---

## 3. Group-norm / B/C RMSNorm as a SEPARATE reduction kernel

**Finding (verified across mamba2 + FLA):** every production SSM/linear-attn kernel
runs gated RMSNorm/GroupNorm as a **separate kernel dispatch, NOT fused into the
chunk-scan recurrence**. Even the "fully fused" `mamba_split_conv1d_scan_combined`
runs the scan to completion, THEN dispatches `_layer_norm_fwd` on the output. **Do not
put a per-step cross-all-channel norm reduction inside the scan loop.**

This codebase is already structurally correct here:
- B/C RMSNorm is in host MLX (`mamba3.py::transform_bc` → `_rms_norm_last` over the
  state axis) BEFORE the scan dispatch — a separate reduction, ngroups=1 ⇒ one group
  over the full `d_state`. Keep it out of the scan.

**Reduction structure to mirror** (`_layer_norm_fwd_1pass_kernel`,
state-spaces/mamba `mamba_ssm/ops/triton/layernorm_gated.py`, Apache-2.0; and FLA
`fla/modules/layernorm_gated.py`, MIT — byte-for-byte same algorithm, prefer MIT):
- `grid = (M, ngroups)` where `M = B*S` (one program per `(token-row, group)`).
- Reduce over a `BLOCK_N`-wide channel vector held in registers:
  `var = tl.sum(xbar*xbar, axis=0)/N; rstd = 1/sqrt(var+eps)` — an **intra-program
  tree reduction over channels** (`O(log N)`), NO cross-program reduction, NO atomics
  on forward. `BLOCK_N = min(65536//elem, next_pow2(group_size))`,
  `num_warps = min(max(BLOCK_N//256,1),8)`.
- Backward: split `M` rows across row-block programs, accumulate `dw/db` partials per
  block, sum partials across the few row-blocks; `dx` is per-row (no cross-row reduce).

**ngroups=1 / small-dstate optimization** (FLA `fla/modules/layernorm.py`, MIT):
when `D <= 512`, **batch many token-rows per program** (`BT` rows/block, autotune
`num_warps∈[2,4,8]`) so launch cost is amortized when the per-row reduction is tiny;
when `D > 512`, one row per program. `num_groups` handled by reshaping to
`(-1, D//num_groups)` so groups never cross-reduce.

**TileLang port of the norm kernel** (low-medium difficulty): block per `(token,group)`,
`T.alloc_fragment` for the channel vector, `T.reduce_sum` over channels for var,
broadcast `rstd`. No scan interaction. Only subtlety = the optional `silu(z)` gate +
`norm_before_gate` ordering (mamba3 B/C norm has no z-gate; the OUTPUT gated-RMSNorm
of the broader block, if added, does). The repo's `bwd_dadt_fused` doc already notes
the K-reduction wants a parallel `simd`-level reduce — that same primitive is what
makes this norm kernel fast (see `kernel_coverage_matrix.md`).

---

## 4. Metal PORT path (CUDA/Triton not runnable on Metal)

The CUDA/Triton sources are **algorithm references** — we port the math into the
TileLang emitter, then lower to MSL. Two concrete routes already exist in-tree:

- **Path C (preferred): TileLang `@T.prim_func` → `tilelang.compile(target="metal",
  execution_backend="tvm_ffi", out_idx=...)`** into caller-owned MLX buffers. This is
  exactly how `mamba3_path_c.py` and `_mamba3_helpers_tilelang.py` already lower. Write
  the chunked fwd/bwd as `@T.prim_func` mirroring the local examples (`example_mamba_chunk_scan.py`,
  `example_chunk_delta_h.py`, `example_chunk_delta_bwd.py`) and let the emitter produce MSL.
- **Path B (fallback / when shared.dyn or simdgroup ops misbehave): inline MSL via
  `mx.fast.metal_kernel`** through `lower_tilelang_to_msl_inline` in
  `cppmega_mlx/nn/_tilelang/_msl_transform.py`, wrapped by `mx.custom_function` (already
  how `mamba3.py` and the helpers ship).

**MSL parallel-scan reference = D-CSIL/mlx-recurrence `ssm_scan.py`** (Section 1, Tier 3):
- The per-lane forward recurrence and the **reverse-adjoint recurrence** MSL bodies are
  ~30 lines of plain arithmetic — lift them as the template for the intra-chunk register
  sweep and the reverse chunk-boundary carry.
- The VJP wiring (persist saved activations as forward OUTPUTS, do not re-run fwd in bwd)
  matches our existing `_mamba3_mimo_apply_with_state.vjp` contract.
- Half-fused reductions (per-thread grads on GPU, contracted-dim reductions in MLX)
  match our "host sums over P / over (B,P)" decomposition documented in
  `docs/tilelang_ports/mamba3.md`.

**Metal lowering gotchas already cataloged** (`mamba3_helpers_tilelang.md`): use
`scope="shared"` (not `shared.dyn`); `T.sync_threads()` (not
`T.tvm_storage_sync("threadgroup")`); inline functions cannot allocate threadgroup
memory (inline the body into `source=`); avoid `from __future__ import annotations`
breaking PrimFunc dtype capture; use `T.alloc_local((1,),dtype)` instead of reassigning
locals across if/else. bf16 simdgroup MSL codegen is buggy (cubecl#1202) ⇒ **fp16
carrier + fp32 accumulators** (already this repo's policy).

For the intra-chunk `simd_sum` reduction over `d_state` and token-batching, the
**ggml `kernel_ssm_scan_f32`** idiom (Tier 4) is the MSL reference; for the true
parallel-over-L Blelloch (only if needed) use accelerated-scan + reverse_scan.cuh +
`simd_shuffle_up` (Tier 5).

---

## 5. What this repo ALREADY has (reuse, don't re-port)

- `_chunk_cumsum`-style segment cumsum + dt/A gradients are already ported as the three
  Triton-helper replacements: `compute_dacs_segsum`, `bwd_dadt_fused`, `bwd_dtrap_ddt`
  in `cppmega_mlx/nn/_tilelang/_mamba3_helpers.py` (pure MLX) and
  `_mamba3_helpers_tilelang.py` (TileLang `@T.prim_func`). These are the mamba3 analogs
  of SSD's `_chunk_cumsum_bwd` + the trapezoidal/RoPE chain rule. **Reuse verbatim** in
  bwd stage P7; do not re-derive.
- Forward chunk_scan / chunk_state TileLang kernels:
  `examples/linear_attention/example_mamba_chunk_scan.py`, `example_mamba_chunk_state.py`.
- Chunked backward template with dh reverse carry: `example_linear_attn_bwd.py`, plus
  GDN/KDA bwd kernels (`examples/gdn/example_chunk_delta_bwd.py`,
  `examples/gdn/example_chunk_o_bwd.py`, `examples/kda/chunk_delta_bwd.py`,
  `examples/kda/wy_fast_bwd.py`).
- `mx.custom_function` fwd/bwd VJP wiring: `mamba3.py::_mamba3_mimo_apply_with_state`.
- Serial oracles for parity: `mamba3.py::_reference_scan` /
  `_chunked_mamba3_diagonal_scan` (mamba3), `m2rnn.py::m2rnn_scan` /
  `chunked_m2rnn_scan` (m2rnn). These are the "serial oracle" to validate the chunked
  fwd+bwd against (Section 6).

---

## 6. Incremental port plan

1. **Forward chunked scan (fwd only), parity vs serial oracle.** Write the 5-stage
   chunked forward as TileLang `@T.prim_func`(s) for the mamba3 core recurrence
   (`h=exp(A*dt)h+x*B`, `y=sum(h*C)+D*x`, `silu(z)` gate). Build the intra/inter terms
   from `example_mamba_chunk_scan.py` (stage 5) + `example_mamba_chunk_state.py`
   (stage 2) + a short serial `_state_passing_fwd` loop (stage 3). Oracle:
   `ssd_minimal_discrete` (numerical) and `mamba3.py::_reference_scan` (in-repo).
   Gate: max rel-RMS vs serial oracle ≤ 5e-3 (fp16) / ≤ 1e-6 (fp32), per
   `docs/tilelang_ports/mamba3.md` tolerances.
2. **Wire the trapezoidal/RoPE/MIMO + ngroups=1 B/C RMSNorm host pre-pass.** Keep B/C
   RMSNorm as the existing separate MLX reduction (Section 3); verify the chunked
   forward consumes already-normed B/C and matches `Mamba3ReferenceBlock.__call__`
   end-to-end (the full block, not just the scan).
3. **Backward: reverse chunk-boundary scan first.** Implement `_state_passing_bwd`
   (`dstates=exp(dA_cs)*dstates+dout`) to produce `dh0` + chunk `dstates`, REPLACING
   the snapshot/replay reverse pass. Validate `dh0` + state grads vs autograd-through
   `ssd_minimal_discrete` and vs `mamba3.py` autograd.
4. **Backward: parallel per-chunk grads.** Add P4/P5 (`dx,ddt,dD,dB,dC,dCB` +
   `_bmm_chunk_bwd`), then P6/P7 reusing the in-repo `bwd_dadt_fused`/`bwd_dtrap_ddt`/
   `compute_dacs_segsum` helpers for `ddt/dA/dtrap/dRoPE`. Zero-init grad accumulators.
   Gate: all 8 input grads at rel-RMS ≤ 5e-3 (fp16) / ≤ 1e-6 (fp32) vs the serial
   backward oracle (do NOT expect bitwise — the chunked reduction reorders sums).
5. **Lower to Metal.** Run through Path C (`tilelang.compile target="metal"
   execution_backend="tvm_ffi"`); fall back to inline MSL via
   `lower_tilelang_to_msl_inline` where the emitter hits a Metal gotcha (Section 4).
   Use mlx-recurrence MSL bodies as the per-lane template.
6. **Bench + receipt-gate.** Benchmark chunked vs current serial Path B/C with
   `scripts/bench_tilelang_mamba3_path_c.py` / `scripts/bench_tilelang_mamba3.py`;
   only promote chunked in AUTO when the receipt proves it no-worse for the exact shape
   (the existing receipt-gate discipline). Expected win grows with `SEQ` (serial dep
   drops `O(SEQ)→O(SEQ/L)`).
7. **Repeat for m2rnn.** Same recipe on `m2rnn_path_c.py` (rank-1 `q,k,v,W` update +
   `xf` gate); m2rnn already has `chunked_m2rnn_scan` as the serial oracle.

**Apply RULE #1 throughout:** if the chunked Path C errors/misbehaves at runtime,
RAISE with where+what (as `_dispatch_mamba3_scan` already does for Path C) — never add
a silent fallback to the serial path that would hide a chunked-kernel bug.

---

## 7. Licensing for reuse

- **state-spaces/mamba** (SSD chunked fwd+bwd, layernorm_gated, reverse_scan.cuh):
  **Apache-2.0**. Permissive; reuse/port allowed with attribution + license/NOTICE
  retention. Header is `Copyright (c) 2024, Tri Dao, Albert Gu`. We PORT the algorithm
  into TileLang (not vendor the Triton/CUDA verbatim), so attribute source files in the
  port docstring (this repo's `docs/tilelang_ports/mamba3.md` already does this for the
  Path B port).
- **fla-org/flash-linear-attention** (chunked linear-attn/delta, layernorm_gated/
  layernorm): **MIT**. Most permissive; **preferred for any verbatim mirroring** of the
  norm kernel and chunk decomposition. Attribute in port docstrings.
- **D-CSIL/mlx-recurrence** (MSL fwd+bwd scan): **MIT** (verified
  `Copyright (c) 2026 Paul O. Derrington, Jr.`). MSL bodies portable; attribute.
- **ggml-org/llama.cpp** (`kernel_ssm_scan_f32` idiom): **MIT**. Idiom reference; attribute.
- **proger/accelerated-scan**, **alxndrTL/mamba.py**: **MIT**. Blelloch reference; attribute.
- **TileLang** (this repo's emitter + examples we extend): **MIT** (Tile-AI). In-tree.
- Papers (Mamba2/SSD arXiv:2405.21060; Mamba3 arXiv:2603.15569; Martin & Cundy
  arXiv:1709.04057; Futhark SC22): cite, not code.

Policy alignment: per `docs/metal_kernel_policy.md`, external Apple/Metal kernels are
"source-review and parity-fixture inputs only until pinned, vendored or reimplemented
in-tree, licensed, profiled, and covered by the same fallback and VJP/JVP gates." This
plan reimplements in-tree (TileLang/MSL), so the Apache/MIT sources are compliant
references, not runtime dependencies. Pin the mamba commit
`a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` (v2.3.2.post1) when transcribing exact index math.

---

## 8. Gaps where NO ready code exists (we must derive)

- **Mamba3-specific pieces are NOT in any chunked kernel.** Neither SSD nor FLA cover:
  (a) **trapezoidal/exponential discretization** (mixes `h_{t-1}` with an average of
  inputs at `t-1` and `t` — changes BOTH the intra-chunk decay matrix and the
  inter-chunk boundary term; needs a new derivation, not a flag); (b) **complex/rotational
  (RoPE-on-state) transitions** (FLA/SSD decays are real positive diagonal); (c) **MIMO**
  (non-rank-1 B/C state update — outside the `K^T@V` / WY structure); (d) **BCNorm**
  placement inside a chunk kernel. We have the host-MLX reference (`mamba3.py`) and the
  three helper kernels for the cumsum/trapezoidal gradients, but the chunked-form
  trapezoidal/RoPE/MIMO **intra-chunk decay matrix** must be derived from the mamba3
  recurrence and validated against `Mamba3ReferenceBlock`. This is the main net-new math.
- **No TileLang mamba/SSD backward exists** in tilelang upstream — only forward kernels.
  The bwd is net-new TileLang work (the GDN/KDA bwd kernels + `example_linear_attn_bwd.py`
  are the closest templates; generalize their decay/state handling).
- **No Metal SSM backward exists anywhere except D-CSIL/mlx-recurrence** (a small,
  unbattle-tested repo). Its bwd is **half-fused** (grad_B/grad_C/grad_A reductions run
  in MLX, not the kernel). A fully-fused Metal backward (contracted-dim reductions via
  `simd_sum`/threadgroup) does not exist publicly — derive if needed.
- **No production kernel fuses the norm INTO the scan**, and none normalizes B/C
  per-step inside the scan. Our ngroups=1 B/C RMSNorm as a separate pre-scan reduction
  is the correct, supported placement; there is no kernel to copy for an in-scan B/C
  norm (and we should not build one).
- **Decay underflow** over long ranges: keep decays in log space + fp32 accumulation
  (SSD/FLA do this; our `compute_dacs_segsum` already weights by `exp(rev[t])`). The
  safe range per dtype/seqlen must be validated empirically; the existing
  `scan_plan.py` snapshot policy may become unnecessary once log-space chunk-cumsum is in.
- **Gradient parity is fp-tolerance, not bitwise** — chunked reductions reorder sums.
  Validate at rel-RMS ≈ 5e-3 (fp16/bf16) / atol≈1e-2, matching FLA's `get_err_ratio`
  (0.005) and this repo's `mamba3.md` observed `5.3e-3` for A/dt over 512 steps.
