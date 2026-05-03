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
      <td><code>cppmega_mlx/nn/_tilelang/mamba3_path_c.py</code><br>
      (only Mamba3 today).</td>
      <td>Yes — via <code>mx.custom_function</code><br>
      wrapper around the lowered MSL.</td>
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
      <td><code>mamba3_path_c.py</code><br>(1.5% faster than B but not in hot path<br>— kept as proof)</td>
      <td>No SSM in MLX; Path B is 25.5× faster than reference;<br>Path C parity-validates the DSL but doesn't justify swap.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd BF16</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.sparse_mla_apply</code></td>
      <td><code>sparse_mla_attention_reference</code></td>
      <td>❌ blocked (T.Pipelined num_stages&gt;1<br>only works at 16×16 tiles, sparse-MLA wants 32×32)</td>
      <td>Path B 1.4–3.3× speedup measured;<br>reference kept for parity oracle.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd FP8</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.sparse_mla_fp8_apply</code></td>
      <td><code>sparse_mla_fp8_reference</code></td>
      <td>❌ requires both T.Pipelined 32×32 and scheduler glue<br>to T.fp8_scaled_matmul macro</td>
      <td>FP8 path is software-emulated via uchar storage<br>+ LUT decode (Apple Silicon has no native FP8 ALU).</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA fwd<br>blockscaled (e8m0)</strong></td>
      <td><strong>B</strong> — <code>sparse_mla_blockscaled_apply</code></td>
      <td><code>sparse_mla_blockscaled_reference</code></td>
      <td>❌ no e8m0 layout in DSL</td>
      <td>mxfp8 block-scale is a software emulation;<br>Path B handles the 16-element scale tile bookkeeping.</td>
    </tr>
    <tr>
      <td><strong>Sparse-MLA bwd</strong><br>(chunked dQ/dK/dV)</td>
      <td><strong>B</strong> — <code>sparse_mla_bwd_metal</code><br>(called via <code>sparse_mla_apply</code> VJP)</td>
      <td>autograd through reference fwd</td>
      <td>❌</td>
      <td>Path B keeps memory bounded by chunking dKV;<br>reference autograd OOMs at production shapes.</td>
    </tr>
    <tr>
      <td><strong>topk_selector</strong><br>(per-row top-k indices<br>for sparse-MLA)</td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.topk_selector</code></td>
      <td><code>topk_selector_reference</code></td>
      <td>❌</td>
      <td><code>mx.topk</code> doesn't expose the per-row<br>index format we need.</td>
    </tr>
    <tr>
      <td><strong>Cross-entropy<br>(chunked, V=65536)</strong></td>
      <td><strong>B</strong> — <code>cppmega_mlx.training.cut_cross_entropy.linear_cross_entropy_value_and_grad</code></td>
      <td><code>nn.losses.cross_entropy</code></td>
      <td>N/A — pure MLX, no TileLang</td>
      <td>Saves 26.9% F+B peak memory at V=65536<br>vs the materialized [B*T, V] path.</td>
    </tr>
    <tr>
      <td><strong>FP8 scaled matmul /<br>vecmat</strong> (when used in<br>custom paths)</td>
      <td><strong>B</strong> — <code>cppmega_mlx.nn._tilelang.fp8_scaled_matmul</code><br>(audiohacking-style)</td>
      <td>dequant + <code>mx.matmul</code></td>
      <td>macro <code>T.fp8_scaled_matmul</code> exists but is 3.16×<br>slower than B for 128³ matmul, 6.01× slower for vecmat</td>
      <td>Not in the production hot path today;<br>available for custom kernel composition.</td>
    </tr>
  </tbody>
</table>

## Combination in the production model

The hybrid mini config (1.2B params, calibrated quarter from 4.8B) at training time uses the following stack per layer, top-to-bottom:

1. **Embedding lookup** → Path A.
2. **Per-block (alternating M=mamba3 / A=attention / E=moe / R=m2rnn — see HybridLMConfig)**:
   - **A blocks** (HybridBackend="attention"): RMSNorm (A) → Linear projections (A — mx.matmul) → RoPE (A — mx.fast.rope) → **sparse-MLA (B)** → output projection (A) → residual.
   - **M blocks** (HybridBackend="mamba3"): RMSNorm (A) → in-projections (A) → causal depthwise conv1d (A) → **Mamba3 main scan (B)** → out projection (A) → residual.
   - **E blocks** (HybridBackend="moe"): RMSNorm (A) → router (A) → **gather_mm SwitchGLU (A)** → residual.
   - **R blocks** (HybridBackend="m2rnn"): RMSNorm (A) → in-proj (A) → m2rnn cell (pure MLX reference, **no Path B** yet) → residual.
3. **Final RMSNorm** → A.
4. **lm_head projection** → A — mx.matmul.
5. **Loss** → **B** — linear_cross_entropy_value_and_grad (chunked, fused with the lm_head projection inside the chunked loop).

### What this means for a typical training step (mini, B=4 T=2048):
- **Path A** dominates raw FLOPs (every Linear, every RoPE, every RMSNorm, every attention QKV, every gather_mm).
- **Path B** carries the sequence-dimension reductions (Mamba3 scan, sparse-MLA attention, chunked CE) — the ops that have no native MLX equivalent.
- **Path C** is **never hit** in the default production path. It exists only as a reproducibility receipt that the TileLang DSL lowers correctly through patched apple-head TileLang on Metal (proof-of-DSL artifact for docs/upstream/_pr_filing_pack.md).

## Override mechanism

bash
# Default — Path B preferred, reference fallback if Metal kernel unavailable
unset CPPMEGA_KERNEL_PATH

# Force pure-MLX reference (parity-tests / debugging)
export CPPMEGA_KERNEL_PATH=ref

# Force Path B; raise if Metal unavailable
export CPPMEGA_KERNEL_PATH=path_b

# Force Path C (only Mamba3 supported; raises NotImplementedError for sparse-MLA)
export CPPMEGA_KERNEL_PATH=path_c


The dispatch decision is recorded in a process-wide ring buffer (last 256 records) accessible via cppmega_mlx.runtime.kernel_policy.get_dispatch_log(). The training profile snapshots this log at step boundaries and exposes it under ProfileMetrics.kernel_dispatch so the receipt JSON shows which kernels actually fired.

## Bench summary (M4 Max, FP32 unless noted)

| Op                                       |             Path A |                   Path B |                  Path C | Δ B vs A | Δ C vs B |
| ---------------------------------------- | -----------------: | -----------------------: | ----------------------: | -------: | -------: |
| Mamba3 fwd+bwd (B=2 T=512 H=4 P=32 N=64) |  n/a (no SSM in A) |                 7.823 ms |                7.707 ms |      n/a |    -1.5% |
| Sparse-MLA fwd BF16 (production shape)   |                n/a | 1.4-3.3× faster than ref |                     n/a |        — |        — |
| FP8 matmul 128×128×128 e4m3              |                n/a |  0.172 ms / 0.024 TFLOPS | 0.555 ms / 0.008 TFLOPS |      n/a |   -3.16× |
| FP8 vecmat M=1 N=K=4096                  |                n/a |  0.182 ms / 0.184 TFLOPS | 1.098 ms / 0.031 TFLOPS |      n/a |   -6.01× |
| Cross-entropy chunked V=65536 fwd peak   |           baseline |              -54.6% peak |                     n/a |        — |        — |
| Cross-entropy chunked V=65536 F+B peak   |           baseline |              -26.9% peak |                     n/a |        — |        — |
| Regular SDPA                             |  shipped (fastest) |                        — |                       — |        — |        — |
| RMSNorm                                  |  shipped (fastest) |                        — |                       — |        — |        — |
| Dense GEMM                               | shipped (MPS BNNS) |                        — |                       — |        — |        — |

## Honest limitations

- **R (m2rnn) blocks have no Path B port today.** They run pure-MLX reference. Adding a port is post-M0 work; the reference is correct but slow on long sequences.
- **Path C is not in the hot path.** It exists for two reasons: (a) prove the patched apple-head TileLang DSL lowers correctly on Apple Silicon — necessary because we file the patches upstream; (b) reproducibility receipt for future contributors who want to re-derive the kernel from a high-level DSL rather than read 700 lines of MSL.
- **FP8 paths are software emulation.** Apple Silicon (M1–M4) has no native FP8 ALU. The uchar storage + LUT decode + fp32 fma loop pattern (vendored from audiohacking/fp8-mps-metal Apache 2.0) is what we ship. Native FP8 is M5/M6 territory.
- **TileLang DSL on Metal works for FP16/FP32 GEMM and Mamba3 today.** Sparse-MLA via DSL is blocked on T.Pipelined num_stages>1 working at 32×32 tiles (only 16×16 works after Agent D's 3D-buffer fix). FP8 scaled matmul DSL macro exists but the scheduler doesn't fuse per-load scale into GemmMetalScalar's K-loop yet — that's the documented follow-up in docs/upstream/tilelang_metal_fp8_scaled_matmul/README.md.
- **CPPMEGA_KERNEL_PATH=path_c is not a complete path.** It only redirects Mamba3. Other ops that don't have a Path C port raise NotImplementedError with a redirect to Path B. This is by design — Path C is a proof artifact, not a primary execution mode.

## Receipts

The training receipt JSON (emitted by cppmega_mlx.training.profile) gains a kernel_dispatch field documenting which path each op took during the profiled step. Example:

json
{
  "kernel_dispatch": [
    {"op": "mamba3_mimo", "path": "auto", "kernel_used": "metal_kernel_fwd_v1"},
    {"op": "sparse_mla", "path": "auto", "kernel_used": "metal_kernel_fwd_v1"},
    {"op": "cut_cross_entropy", "path": "path_b", "kernel_used": "linear_cross_entropy_value_and_grad"}
  ]
}


Use these receipts in CI to gate adoption decisions: a PR that flips a kernel from Path B to reference must show evidence in the dispatch log + parity within the documented atol/rtol.
