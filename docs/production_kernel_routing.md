# Production Kernel Routing — Path A / Path B / Path C

Date: 2026-05-03

This document is the authoritative source for **which kernel implementation each operation uses in the cppmega.mlx production training/inference path**, why, and how to override it. It complements docs/metal_kernel_policy.md (the policy gates) and docs/kernel_coverage_matrix.md (the upstream landscape).

## Path definitions

<table>
  <thead>
    <tr>
      <th>Path</th>
      <th>What it is</th>
      <th>Where it lives</th>
      <th>Differentiable?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Path A</strong></td>
      <td>Native MLX ops — <code>mx.matmul</code>, <code>mx.fast.scaled_dot_product_attention</code>,<br>
      <code>mx.fast.rms_norm</code>, <code>mx.fast.rope</code>, <code>mx.gather_mm</code>, <code>mx.softmax</code>.</td>
      <td>First-class MLX,<br>no cppmega code.</td>
      <td>Yes — through<br>MLX autograd.</td>
    </tr>
    <tr>
      <td><strong>Path B</strong></td>
      <td>Hand-written Apple MSL kernels via <code>mx.fast.metal_kernel</code>,<br>
      paired with <code>mx.custom_function</code> and a manual VJP.</td>
      <td><code>cppmega_mlx/nn/_tilelang/</code></td>
      <td>Yes — manual VJP<br>per kernel.</td>
    </tr>
    <tr>
      <td><strong>Path C</strong></td>
      <td>TileLang DSL <code>@T.prim_func</code> lowered to MSL via the<br>
      patched apple-head TileLang on Metal target.</td>
      <td><code>cppmega_mlx/nn/_tilelang/mamba3_path_c.py</code>
      (direct tvm-ffi owner-output apply for DLPack-exportable contiguous FP32 buffers)<br>
      <code>cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py</code> (full apply)<br>
      <code>cppmega_mlx/nn/_tilelang/topk_selector.py</code>
      (receipt-gated AUTO plus direct tvm-ffi owner-output entrypoint for
      float32/float16 <code>topk_selector_tilelang_direct(..., out=...)</code>)<br>
      <code>cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled_path_c.py</code>
      (prepared-buffer full apply + E8M0 QK reducer)<br>
      <code>cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py</code>
      (prepared-buffer full apply + QK/indexed-QK reducers)<br>
      <code>cppmega_mlx/nn/_tilelang/m2rnn_path_c.py</code>
      (owner-output forward/backward; opt-in fail-closed route)<br>
      <code>cppmega_mlx/nn/_tilelang/fp8_matmul_path_c.py</code> and
      <code>cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py</code>
      (standalone owner-output surfaces; correctness is useful, but current
      production-speed receipts are not green)</td>
      <td>Mixed: direct owner-output tvm-ffi surfaces are not callable from
      MLX graph transformations yet; legacy <code>mx.custom_function</code>
      MSL-wrapper surfaces remain differentiable where still used.</td>
    </tr>
  </tbody>
</table>

## Per-operation routing decision

Default behavior on Apple Silicon is in the **"Production"** column. Override via CPPMEGA_KERNEL_PATH=auto|ref|path_b|path_c env var (auto = production default with reference fallback when Metal unavailable).

<table>
  <thead>
    <tr>
      <th>Operation</th>
      <th>Production</th>
      <th>Reference fallback</th>
      <th>Path C alt</th>
      <th>Why this choice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Regular attention</strong><br>(Q@K^T, softmax, @V)</td>
      <td><strong>A</strong> — <code>mx.fast.scaled_dot_product_attention</code></td>
      <td>—</td>
      <td>—</td>
      <td>Native MLX wins;<br>no in-tree port needed.</td>
    </tr>
    <tr>
      <td><strong>RMSNorm</strong></td>
      <td><strong>A</strong> — <code>mx.fast.rms_norm</code></td>
      <td>—</td>
      <td>—</td>
      <td>Native MLX has fused fwd+bwd.</td>
    </tr>
    <tr>
      <td><strong>RoPE</strong></td>
      <td><strong>A</strong> — <code>mx.fast.rope</code></td>
      <td>—</td>
      <td>—</td>
      <td>Native MLX, fused.</td>
    </tr>
    <tr>
      <td><strong>Dense GEMM (fp16/fp32)</strong></td>
      <td><strong>A</strong> — <code>mx.matmul</code> (MPS BNNS)</td>
      <td>—</td>
      <td>T.gemm via apple-head TileLang<br>(slower than A, building block only)</td>
      <td>A faster than every<br>in-tree port we have.</td>
    </tr>
    <tr>
      <td><strong>Quantized GEMM (q4)</strong></td>
      <td><strong>A</strong> — <code>mx.gather_qmm</code> / <code>mx.quantized_matmul</code></td>
      <td>—</td>
      <td>—</td>
      <td>Inference path; affine q4 g=64<br>is the canonical mlx-lm path.</td>
    </tr>
    <tr>
      <td><strong>MoE (gate+up routing)</strong></td>
      <td><strong>A</strong> — <code>mx.gather_mm</code> + SwitchGLU pattern</td>
      <td>per-expert loop</td>
      <td>—</td>
      <td>Native gather_mm has fwd+bwd;<br>per-expert loop kept for 4-expert M0 smoke.</td>
    </tr>
    <tr>
      <td><strong>Mamba3 main<br>(chunked SSD fwd+bwd)</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.mamba3_mimo_apply</code></td>
      <td><code>_chunked_mamba3_diagonal_scan</code><br>(pure MLX in <code>nn/mamba3.py</code>)</td>
      <td><code>mamba3_path_c.py</code><br>(direct tvm-ffi owner-output for
      contiguous FP32 buffers; full block route fails closed on non-contiguous
      broadcast/slice views rather than copying)</td>
      <td>No SSM in MLX; Path B is 25.5× faster than reference;<br>Path C parity-validates the DSL through direct owner-output, but MLX graph-transform VJP remains blocked by DLPack export.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd BF16</strong></td>
      <td><strong>C when row-green, else B</strong> — <code>sparse_mla_attention(...)</code></td>
      <td><code>sparse_mla_attention_reference</code></td>
      <td>Path C available via <code>sparse_mla_path_c_apply</code>;<br>
      AUTO promotes only receipt rows where every forward no-worse flag is true.<br>
      <strong>Kwarg rename:</strong> Path B accepts <code>force_metal=True</code> while
      Path C accepts <code>force_path_c=True</code>. There is intentionally no
      backwards-compatible alias — callers migrating from Path B must rename the
      kwarg explicitly. AUTO callers never see this kwarg directly.</td>
      <td><code>B2_S128_H8_D64</code>, <code>B4_S512_H8_D64</code>,<br>
      and <code>B4_S1024_H8_D64</code> promote to Path C today;<br>
      unreceipted shapes stay Path B fail-closed.</td>
    </tr>
    <tr>
      <td><strong>FP8 vecmat (Path C)</strong><br>(M=1, transpose-B contract)</td>
      <td>Opt-in only — see Path B FP8 vecmat row below.</td>
      <td><code>fp8_scaled_vecmat</code> (Path B)</td>
      <td><code>fp8_scaled_vecmat_path_c</code> (in
      <code>cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py</code>):<br>
      <code>x_fp8 (K,) uint8 e4m3</code>,
      <code>W_fp8 (N, K) uint8 e4m3</code>,
      <code>scale_x</code> scalar, <code>scale_w</code> scalar or <code>(N,)</code>.<br>
      Mirrors Path B's vecmat contract. <strong>Status:</strong> opt-in only.
      Current TL-W owner-output/tvm-ffi production-path timing is still about
      1.7x slower than the shipped Path B/audiohacking-style reducer, so AUTO
      must not promote it. Older packed-dot4 probe receipts that showed parity
      or speed parity are diagnostic history, not the current production gate.</td>
      <td>Listed for accuracy — earlier revisions of this table omitted FP8 vecmat
      Path C even though the entrypoint exists in the tree.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd FP8</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.sparse_mla_fp8_apply</code></td>
      <td><code>sparse_mla_fp8_reference</code></td>
      <td><strong>Prepared-buffer Path C apply.</strong>
      <code>sparse_mla_fp8_path_c_apply</code> consumes existing
      <code>q_fp8/q_scale/kv_fp8/kv_scale</code> GPU buffers and performs QK,
      softmax, and SV in one lowered kernel. It is not a high-level
      float-carrier wrapper; float inputs still route through Path B or a
      graph-level FP8 producer.</td>
      <td>FP8 path is software-emulated via uchar storage<br>+ LUT decode (Apple Silicon has no native FP8 ALU).</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd<br>blockscaled (e8m0)</strong></td>
      <td><strong>B</strong> — <code>sparse_mla_blockscaled_apply</code></td>
      <td><code>sparse_mla_blockscaled_reference</code></td>
      <td><strong>Prepared-buffer Path C apply.</strong>
      <code>sparse_mla_blockscaled_path_c_apply</code> consumes existing FP8
      byte tensors and E8M0 K/32 scale tensors, then performs QK, softmax, and
      SV in one lowered kernel. It is not a high-level float-carrier wrapper;
      Path B still owns MXFP8 quantization/unpacking unless the graph creates
      those buffers directly.</td>
      <td>mxfp8 block-scale is a software emulation;<br>Path B handles the 16-element scale tile bookkeeping.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA bwd</strong><br>(chunked dQ/dK/dV)</td>
      <td><strong>C when row-green, else B</strong><br>(called through sparse-MLA custom VJP)</td>
      <td>autograd through reference fwd</td>
      <td>Backward parity is tested and participates in the strict row gate;<br>
      row-level AUTO requires all no-worse flags to stay green.</td>
      <td><code>B2_S128_H8_D64</code>, <code>B4_S512_H8_D64</code>,<br>
      and <code>B4_S1024_H8_D64</code> promote to Path C today;<br>
      unreceipted shapes stay Path B fail-closed.</td>
    </tr>
    <tr>
      <td><strong>topk_selector</strong><br>(per-row top-k indices<br>for sparse-MLA)</td>
      <td><strong>C</strong> — <code>topk_selector(..., backend="auto")</code></td>
      <td><code>topk_selector_reference</code></td>
      <td>Receipt-gated Path C with Path B fallback via
      <code>backend="metal"</code>.<br>
      <code>topk_selector_tilelang_direct(..., out=...)</code> is the
      tvm-ffi owner-output surface for float32/float16 scores: it mutates the
      caller-provided <code>mx.int32</code> output and returns that owner array.
      Direct bf16 fails closed to avoid hidden casts. Public AUTO should not be
      read as a generic owner-output guarantee because compatibility wrapper
      paths still exist.</td>
      <td>Checked-in topk receipt runs both B and C on every row and<br>
      keeps all C/B ratios <= 1.0.</td>
    </tr>
    <tr>
      <td><strong>M2RNN main scan<br>(R blocks, fwd+bwd)</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.m2rnn_apply_with_state</code></td>
      <td><code>chunked_m2rnn_scan</code><br>(pure MLX in <code>nn/m2rnn.py</code>)</td>
      <td><code>m2rnn_path_c.py</code> owner-output/tvm-ffi forward/backward.<br>
      Opt-in only: it requires pre-aligned head counts and caller-provided
      <code>h0</code>, and fails closed instead of broadcasting, allocating
      state, or falling back to Path B. Result/cache/scratch tensors are
      explicit tvm-ffi <code>out=</code> buffers, not input staging.</td>
      <td>One MSL kernel per fwd / bwd; 1 threadgroup per (B, H) lane,<br><code>K_DIM</code> threads per group sharing <code>W</code> via threadgroup memory.<br>fp16 carrier; bf16 upcast to fp16 to dodge simdgroup MSL bugs.</td>
    </tr>
    <tr>
      <td><strong>Cross-entropy<br>(chunked, V=65536)</strong></td>
      <td><strong>Opt-in</strong> — <code>train_hybrid_tiny.py --loss-backend cce</code><br>
      calls <code>cppmega_mlx.training.loss.next_token_cut_cross_entropy</code></td>
      <td><code>nn.losses.cross_entropy</code></td>
      <td>N/A — pure MLX, no TileLang</td>
      <td>Default training and eval still use materialized CE.<br>
      The CCE path is scoped train integration; c08.2 full backward/memory acceptance remains open.</td>
    </tr>
    <tr>
      <td><strong>FP8 scaled matmul /<br>vecmat</strong> (when used in<br>custom paths)</td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.fp8_scaled_matmul</code><br>(audiohacking-style)</td>
      <td>dequant + <code>mx.matmul</code></td>
      <td><code>T.fp8_scaled_matmul</code> owner-output/tvm-ffi path exists,<br>
      but current TL-W timing is not production-green: matmul is about 14x
      slower and vecmat about 1.7x slower than the shipped Path B kernels.</td>
      <td>Standalone correctness is useful, but AUTO remains on Path B until
      strict current receipts show no-worse speed and full fwd/bwd composition.</td>
    </tr>
  </tbody>
</table>

## Combination in the production model

The hybrid mini config (1.2B params, calibrated quarter from 4.8B) at training time uses the following stack per layer, top-to-bottom:

1. **Embedding lookup** → Path A.
2. **Per-block (alternating M=mamba3 / A=attention / E=moe / R=m2rnn — see HybridLMConfig)**:
   - **A blocks** (HybridBackend="attention"): RMSNorm (A) → Linear projections (A — mx.matmul) → RoPE (A — mx.fast.rope) → **sparse-MLA (per-shape AUTO: green receipt rows use C, otherwise B)** → output projection (A) → residual.
   - **M blocks** (HybridBackend="mamba3"): RMSNorm (A) → in-projections (A) → causal depthwise conv1d (A) → **Mamba3 main scan (B)** → out projection (A) → residual.
   - **E blocks** (HybridBackend="moe"): RMSNorm (A) → router (A) → **gather_mm SwitchGLU (A)** → residual.
   - **R blocks** (HybridBackend="m2rnn"): RMSNorm (A) → in-proj (A) → causal depthwise conv1d (A) → **M2RNN main scan (B)** → out projection (A) → residual. Reference fallback: <code>chunked_m2rnn_scan</code> via <code>CPPMEGA_KERNEL_PATH=ref</code>. Path C is an explicit no-copy/tvm-ffi audit route and fails closed unless tensors are already in the required head layout with caller-owned <code>h0</code>.
3. **Final RMSNorm** → A.
4. **lm_head projection** → A — mx.matmul.
5. **Loss** → default **A/reference** — nn.losses.cross_entropy; opt-in **CCE** — next_token_cut_cross_entropy when the recipe passes --loss-backend cce.

### What this means for a typical training step (mini, B=4 T=2048):
- **Path A** dominates raw FLOPs (every Linear, every RoPE, every RMSNorm, every attention QKV, every gather_mm).
- **Path B** carries the sequence-dimension reductions (Mamba3 scan, sparse-MLA attention). Chunked CE is opt-in recipe behavior, not the default production loss.
- **Path C** is hit by topk_selector(..., backend="auto") when that selector is used and the TileLang path is available. The explicit `topk_selector_tilelang_direct(..., out=...)` surface is now a real tvm-ffi owner-output route for float32/float16 scores and caller-owned `mx.int32` outputs; direct bf16 fails closed to avoid hidden casts. Sparse-MLA BF16 uses a per-shape fail-closed AUTO gate: checked green rows use Path C, unreceipted rows stay Path B. Mamba3 remains a proof/override path. M2RNN Path C is opt-in and fails closed rather than broadcasting heads, allocating initial state, or re-entering Path B.

## Override mechanism

bash
# Default — Path B preferred, reference fallback if Metal kernel unavailable
unset CPPMEGA_KERNEL_PATH

# Force pure-MLX reference (parity-tests / debugging)
export CPPMEGA_KERNEL_PATH=ref

# Force Path B; raise if Metal unavailable
export CPPMEGA_KERNEL_PATH=path_b

# Force Path C for ops wired to the env policy (Mamba3, sparse-MLA, and M2RNN today);
# ops without a Path C implementation still fail closed.
export CPPMEGA_KERNEL_PATH=path_c

# Mamba3 Path C uses direct tvm-ffi owner-output for DLPack-exportable
# contiguous FP32 buffers. It fails closed instead of copying broadcast/slice
# views or running inside MLX graph transformations that block __dlpack__.

# topk_selector is selected by its explicit backend argument, not this env var:
# topk_selector(scores, k) / backend="auto" prefers Path C, then Path B.


The dispatch decision is recorded in a process-wide ring buffer (last 256 records) accessible via cppmega_mlx.runtime.kernel_policy.get_dispatch_log(). The training profile snapshots this log at step boundaries and exposes it under ProfileMetrics.kernel_dispatch so the receipt JSON shows which kernels actually fired.

## Bench summary (M4 Max, FP32 unless noted)

<table>
  <thead>
    <tr>
      <th>Op</th>
      <th>Path A</th>
      <th>Path B</th>
      <th>Path C</th>
      <th>Δ B vs A</th>
      <th>Δ C vs B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mamba3 fwd+bwd (B=2 T=512 H=4 P=32 N=64)</td>
      <td>n/a (no SSM in A)</td>
      <td>7.823 ms</td>
      <td>7.707 ms</td>
      <td>n/a</td>
      <td>-1.5%</td>
    </tr>
    <tr>
      <td>M2RNN fwd (B=2 T=512 H=4 K=64 V=16, fp16)</td>
      <td>—</td>
      <td>2.9 ms vs 46.9 ms ref</td>
      <td>n/a</td>
      <td>16.1×</td>
      <td>—</td>
    </tr>
    <tr>
      <td>M2RNN fwd+bwd (B=2 T=512 H=4 K=64 V=16, fp16)</td>
      <td>—</td>
      <td>12.9 ms vs 170.1 ms ref</td>
      <td>n/a</td>
      <td>13.2×</td>
      <td>—</td>
    </tr>
    <tr>
      <td>M2RNN fwd+bwd (B=2 T=2048 H=8 K=64 V=32, fp16)</td>
      <td>—</td>
      <td>132.5 ms vs 3441 ms ref</td>
      <td>n/a</td>
      <td>26.0×</td>
      <td>—</td>
    </tr>
    <tr>
      <td>topk_selector checked shapes</td>
      <td>n/a</td>
      <td>paired Path B</td>
      <td>paired Path C, C/B 0.510-0.880</td>
      <td>—</td>
      <td>C no worse</td>
    </tr>
    <tr>
      <td>Sparse-MLA BF16 B2_S128_H8_D64</td>
      <td>n/a</td>
      <td>Path B available</td>
      <td>fwd C/B 0.973; paired fwd C/B 0.993; paired bwd C/B 0.913</td>
      <td>—</td>
      <td>AUTO promotes C</td>
    </tr>
    <tr>
      <td>Sparse-MLA BF16 B4_S512_H8_D64</td>
      <td>n/a</td>
      <td>Path B available</td>
      <td>fwd C/B 1.048; paired fwd C/B 0.993; paired bwd C/B 0.975</td>
      <td>—</td>
      <td>AUTO promotes C</td>
    </tr>
    <tr>
      <td>Sparse-MLA BF16 B4_S1024_H8_D64</td>
      <td>n/a</td>
      <td>Path B available</td>
      <td>fwd C/B 1.017; paired fwd C/B 0.994; paired bwd C/B 0.997</td>
      <td>—</td>
      <td>AUTO promotes C</td>
    </tr>
    <tr>
      <td>Sparse-MLA FP8 indexed QK reduce</td>
      <td>n/a</td>
      <td>Path B fwd</td>
      <td>Reducer remains available for direct QK tests; full prepared-buffer
      apply is covered separately by <code>sparse_mla_fp8_path_c_apply</code>.</td>
      <td>—</td>
      <td>partial only</td>
    </tr>
    <tr>
      <td>Sparse-MLA blockscaled e8m0 QK reduce</td>
      <td>n/a</td>
      <td>Path B blockscaled fwd</td>
      <td>Reducer remains available for direct QK tests; full prepared-buffer
      apply is covered separately by
      <code>sparse_mla_blockscaled_path_c_apply</code>.</td>
      <td>—</td>
      <td>partial only</td>
    </tr>
    <tr>
      <td>FP8 matmul 128×128×128 e4m3</td>
      <td>n/a</td>
      <td>shipped Path B/audiohacking-style FP8 MSL route</td>
      <td>owner-output/tvm-ffi Path C route is currently about 14x slower</td>
      <td>n/a</td>
      <td>blocked</td>
    </tr>
    <tr>
      <td>FP8 vecmat M=1 N=K=4096</td>
      <td>n/a</td>
      <td>shipped M=1 simd_sum reducer</td>
      <td>owner-output/tvm-ffi Path C route is currently about 1.7x slower</td>
      <td>n/a</td>
      <td>blocked</td>
    </tr>
    <tr>
      <td>Cross-entropy chunked V=65536 fwd peak</td>
      <td>baseline</td>
      <td>-54.6% peak (bench only)</td>
      <td>n/a</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Cross-entropy chunked V=65536 F+B peak</td>
      <td>baseline</td>
      <td>-26.9% peak (bench only; c08.2 not closed)</td>
      <td>n/a</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Regular SDPA</td>
      <td>shipped (fastest)</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>RMSNorm</td>
      <td>shipped (fastest)</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Dense GEMM</td>
      <td>shipped (MPS BNNS)</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
  </tbody>
</table>

## Honest limitations

- **R (m2rnn) blocks now ship a Path B port** (cppmega_mlx/nn/_tilelang/m2rnn.py). Both forward and backward run as hand-written MSL via mx.fast.metal_kernel; the chunked-scan reference remains as the parity oracle and is reachable via CPPMEGA_KERNEL_PATH=ref. The kernel uses one threadgroup per (batch, head) with K_DIM threads per group; W is loaded into threadgroup memory once per (B, H). The fp16 carrier dodges the bf16 simdgroup MSL codegen bugs that Mamba3 also worked around.
- **Path C is narrow, not global.** It is the default for topk_selector where the checked-in receipt proves C no worse than B, and `topk_selector_tilelang_direct(..., out=...)` now closes the float32/float16 owner-output case. That does not make public AUTO a generic owner-output route: compatibility wrapper paths still exist, and direct bf16 fails closed to avoid hidden casts. Sparse-MLA BF16 uses a per-shape fail-closed AUTO gate backed by bench/tilelang_ports/sparse_mla.json: green checked rows promote, unreceipted rows stay Path B. Mamba3 Path C remains a proof/override path. M2RNN Path C is available through owner-output/tvm-ffi, but only for already-aligned tensors with explicit <code>h0</code>; otherwise it raises before data movement.
- **FP8 paths are software emulation.** Apple Silicon (M1–M4) has no native FP8 ALU. The uchar storage + LUT decode + fp32 fma loop pattern (vendored from audiohacking/fp8-mps-metal Apache 2.0) is what we ship. Native FP8 is M5/M6 territory.
- **TileLang DSL on Metal works for FP16/FP32 GEMM, Mamba3, topk_selector, and BF16 sparse-MLA today.** The stale 32×32 T.Pipelined blocker no longer describes the in-tree sparse BF16 port. Remaining Sparse-MLA gaps are FP8 scheduler composition, e8m0 full-layout coverage, and unreceipted BF16 shapes outside the checked routing table.
- **CPPMEGA_KERNEL_PATH=path_c is not a complete global path.** It redirects env-policy ops that have Path C wiring, currently Mamba3, sparse-MLA, and M2RNN. Mamba3 and M2RNN Path C routes use owner-output/tvm-ffi and fail closed before hidden broadcast/repeat/state allocation or graph-transform DLPack export. topk_selector uses its explicit backend argument. Other ops without Path C support must stay on Path A/B or fail closed.
- **Remaining fast-kernel wrappers are intentional debt.** The current list is `_mlx_runtime.py`, `_msl_transform.py`, `_mamba3_helpers_tilelang.py`, `m2rnn.py`, `fp8_msl_kernels.py`, `fp8_matmul_path_c.py`, `fp8_vecmat_path_c.py`, `topk_selector.py` compatibility paths, `sparse_mla.py`, `sparse_mla_path_c.py`, `sparse_mla_fp8.py`, `sparse_mla_fp8_path_c.py`, `sparse_mla_blockscaled.py`, and `sparse_mla_blockscaled_path_c.py`. Removing them requires a no-hidden-copy tvm-ffi route plus equal-or-better receipts.

## Receipts

The training receipt JSON (emitted by cppmega_mlx.training.profile) gains a kernel_dispatch field documenting which path each op took during the profiled step. Example:

json
{
  "kernel_dispatch": [
    {"op": "mamba3_mimo", "path": "auto", "kernel_used": "metal_kernel_fwd_v1"},
    {"op": "sparse_mla", "path": "auto", "kernel_used": "metal_kernel_fwd_v1"},
    {"op": "cut_cross_entropy", "path": "opt_in_cce", "kernel_used": "next_token_cut_cross_entropy"}
  ]
}



Use these receipts in CI to gate adoption decisions: a PR that flips a kernel from Path B to reference must show evidence in the dispatch log + parity within the documented atol/rtol.
