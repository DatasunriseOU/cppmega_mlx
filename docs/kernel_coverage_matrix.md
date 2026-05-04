# MLX Kernel Coverage Matrix (Apple Silicon)

Date: 2026-05-03

Synthesis of six parallel web-search audits (D matmul, E activations, F loss, G MoE, H Mamba/SSM, I quantization). Goal: track which fused-kernel paths exist on Apple Silicon for every hot op in the cppmega.mlx training/inference stack, what to vendor, and what to defer.

Status legend: 🟢 shipped & usable · 🟡 partial / single-direction (fwd-only / inference-only) · 🔴 must write or wait.

## Master matrix

<table>
  <thead>
    <tr>
      <th>Op family</th>
      <th>Built-in MLX path</th>
      <th>Best vendor</th>
      <th>Cost (LOC / risk)</th>
      <th>Backward?</th>
      <th>When it matters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**GEMM (dense)**</td>
      <td>mx.matmul (MPS BNNS), no fused bias/act on Metal</td>
      <td>DIY mx.fast.metal_kernel (~150 LOC/epilogue + Python VJP)</td>
      <td>high — bf16 carrier + fp32 accum, K%8=0</td>
      <td>yes (you write it)</td>
      <td>when matmul+bias+act becomes profile-bottleneck</td>
    </tr>
    <tr>
      <td>**GEMM (quantized)**</td>
      <td>mx.gather_qmm, mx.quantized_matmul (affine q4 g=64;<br>
      mxfp4/mxfp8/nvfp4) — 🟢</td>
      <td>nothing — built-in is the kernel HF/mlx-lm/vllm-metal all<br>
      use</td>
      <td>0</td>
      <td>inference only (no VJP through dequant)</td>
      <td>shipped today; q4 inference target</td>
    </tr>
    <tr>
      <td>**Attention**</td>
      <td>mx.fast.scaled_dot_product_attention — 🟢 fused fwd+bwd</td>
      <td>nothing</td>
      <td>0</td>
      <td>yes</td>
      <td>shipped today</td>
    </tr>
    <tr>
      <td>**RMSNorm / LayerNorm**</td>
      <td>mx.fast.{rms_norm, layer_norm} — 🟢 fused fwd+bwd</td>
      <td>nothing</td>
      <td>0</td>
      <td>yes</td>
      <td>shipped today</td>
    </tr>
    <tr>
      <td>**RoPE**</td>
      <td>mx.fast.rope — 🟢 fused</td>
      <td>nothing</td>
      <td>0</td>
      <td>yes</td>
      <td>shipped today</td>
    </tr>
    <tr>
      <td>**SwiGLU / GLU activations**</td>
      <td>plain nn.silu(gate) * up op-fused</td>
      <td>**mlx-lm models/activations.py** @mx.compile(shapeless=True)<br>
      (PR #753 merged) — 🟢</td>
      <td>low (~70 LOC)</td>
      <td>yes (autograd)</td>
      <td>every MLP forward; landed today</td>
    </tr>
    <tr>
      <td>**GeGLU / ReLU² / xIELU / GELU-topk**</td>
      <td>nn.gelu / nn.relu² (op fuser)</td>
      <td>mlx-lm models/activations.py already covers most; ZMLX<br>
      explicit-VJP for hot loop (~120 LOC)</td>
      <td>low</td>
      <td>yes</td>
      <td>per-layer activation cost</td>
    </tr>
    <tr>
      <td>**fused gate+up matmul (MoE)**</td>
      <td>mx.gather_mm — 🟢 (fwd+bwd)</td>
      <td>mlx-lm SwitchGLU pattern — 🟢 (Mixtral/Llama4/Gemma4-MoE/GLM-<br>
      MoE all use it)</td>
      <td>~50 LOC replace per-expert loop</td>
      <td>yes</td>
      <td>scales with #experts; current 4-expert reference loop fine<br>
      for M0</td>
    </tr>
    <tr>
      <td>**Cross-entropy / loss**</td>
      <td>nn.losses.cross_entropy materializes [B*T, V]</td>
      <td>**cppmega_mlx.training.loss.next_token_cut_cross_entropy**<br>
      opt-in via train_hybrid_tiny.py --loss-backend cce;<br>
      default/eval stay materialized CE — 🟡</td>
      <td>80 LOC core + train-script wiring/tests</td>
      <td>yes via MLX autograd in train path; manual chunked bwd not<br>
      wired</td>
      <td>V=65536 memory bench is evidence only; c08.2 full acceptance<br>
      open</td>
    </tr>
    <tr>
      <td>**Selective scan (Mamba)**</td>
      <td>None — current cppmega_mlx/nn/mamba3.py is reference scan</td>
      <td>**D-CSIL/mlx-recurrence** ssm_scan.py (~430 LOC, MIT, full<br>
      fwd+bwd VJP) — 🟢 (post-M0)</td>
      <td>medium (430 LOC vendor + MIMO rank>1 extension if needed)</td>
      <td>yes</td>
      <td>when long-T forward is bottleneck; 19× fwd+bwd, ~3× e2e on<br>
      M3 Max</td>
    </tr>
    <tr>
      <td>**Selective scan (inference only)**</td>
      <td>reference scan</td>
      <td>mlx-lm PR #1153 — 🟡 fwd-only, 18.7× prefill</td>
      <td>medium</td>
      <td>no</td>
      <td>inference scout / 48 GB peer</td>
    </tr>
    <tr>
      <td>**Mamba3 / MIMO chunked SSD**</td>
      <td>reference (chunked matmul inter-chunk)</td>
      <td>**TileLang Apple Metal** (PR tile-ai/tilelang#799 merged<br>
      2025-10-07) — emits MSL via TVM, lowers<br>
      mamba_ssm.ops.tilelang.mamba3.*. CuTe DSL path is<br>
      sm_90a-only and dead-end on Metal.</td>
      <td>medium (port: strip PyTorch-MPS launcher → rehost via<br>
      mx.fast.metal_kernel; rewrite 3 Triton helpers; fp16 carrier<br>
      — bf16 simdgroup MSL bugs)</td>
      <td>algorithmically portable; needs custom VJP wrapper</td>
      <td>full Mamba3 perf parity with cppmega CUDA TileLang path</td>
    </tr>
    <tr>
      <td>**M2RNN main scan (R blocks)**</td>
      <td>reference per-token recurrence in cppmega_mlx/nn/m2rnn.py</td>
      <td>**cppmega_mlx.nn._tilelang.m2rnn** (Path B, this repo) —<br>
      fp16 carrier, 1 threadgroup per (B,H) lane, K_DIM threads<br>
      sharing W via threadgroup memory</td>
      <td>medium (~700 LOC fwd+bwd MSL + custom_function VJP)</td>
      <td>yes (manual VJP per kernel)</td>
      <td>M2RNN R blocks — 4-16× fwd, 13-26× fwd+bwd at production<br>
      shapes</td>
    </tr>
    <tr>
      <td>**KV-cache q4 (inference)**</td>
      <td>None</td>
      <td>**TurboQuant** (vllm-metal upstream / arozanov / sharpner) —<br>
      🟢 4.6× compression, ~98% fp16 speed</td>
      <td>low (drop-in mlx-lm KVCache)</td>
      <td>inference only</td>
      <td>inference-scout role; long context</td>
    </tr>
    <tr>
      <td>**NF4 / QLoRA training**</td>
      <td>not in MLX 🔴</td>
      <td>none — would need mx.custom_function with manual gradient</td>
      <td>high</td>
      <td>needs writing</td>
      <td>not on master plan (training stays bf16)</td>
    </tr>
    <tr>
      <td>**W8A8 GPTQ training**</td>
      <td>not in MLX 🔴</td>
      <td>none</td>
      <td>high</td>
      <td>needs writing</td>
      <td>not on master plan</td>
    </tr>
    <tr>
      <td>**all-to-all expert dispatch (DeepEP)**</td>
      <td>not in MLX 🔴</td>
      <td>none — only matters at 64+ experts and EP across nodes</td>
      <td>very high</td>
      <td>n/a</td>
      <td>irrelevant at our 4-expert config</td>
    </tr>
  </tbody>
</table>

## Decision summary (action items)

### Adopt now (low cost, high return)

1. **SwitchGLU MoE pattern** via mx.gather_mm — replace per-expert loop in cppmega_mlx/nn/moe.py (~50 LOC). Keeps backward via existing MLX VJP. Makes scaling beyond 4 experts viable.
2. **mlx-lm models/activations.py SwiGLU** wrapper via @mx.compile(shapeless=True) — already-blessed upstream pattern, ~70 LOC. Future ReLU²/GeGLU adopt the same shape.
3. **TurboQuant KV-cache q4** for the 48 GB inference-scout role. Drop-in mlx-lm KVCache replacement, no model code changes.

### Already shipped (this session)

- cppmega_mlx/training/cut_cross_entropy.py + train_hybrid_tiny.py --loss-backend cce — scoped opt-in chunked CE train integration. Default training/eval remain nn.losses.cross_entropy. Existing V=65536 memory benches (−54.6% forward peak / −26.9% F+B peak) are supporting evidence, not the c08.2 full backward/memory closure.
- cppmega_mlx/training/optimizers.py::MuonAdamWMulti (make_muon) — Muon+AdamW group splitter mirroring cppmega CUDA's _is_nonlinear_or_embedding (commit d5c1986). 14 tests pass. cppmega_cuda_parity=True flag for trace-matching gb10.
- cppmega_mlx/runtime/memory_audit.py — runtime audit with attribute alias dedup (commit 0602832).
- cppmega_mlx/training/plasticity/{fire,dash,redo}.py — FIRE/DASH/ReDo (commit 9173e42).

### Defer (write only when profile demands it)

- **Mamba3 MIMO via TileLang Apple Metal** — TileLang's Metal device backend (PR #799, 2025-10-07) is real and mamba_ssm.ops.tilelang.mamba3 is algorithmically portable. Three concrete dispatch paths verified by reading tilelang/engine/lower.py:216, tilelang/jit/adapter/torch/metal.py (70 LOC), and apache/tvm/src/runtime/metal/metal_{module,common,device_api}.{mm,h} (~778 LOC, zero PyTorch references):
  - **Path A (easiest, ~50 LOC glue, +torch dep)**: pip install tilelang on macOS, use MetalKernelAdapter as-is, bridge mx.array ↔ torch.Tensor via __dlpack__(). Works today.
  - **Path B (clean MLX, ~80 LOC glue, no torch)**: target.build.metal lowering produces an MSL source string (kernel_global_source). Feed that string into mx.fast.metal_kernel(source=..., ...) — MLX's API accepts arbitrary MSL. Wrap in mx.custom_function for VJP. **Bypasses MarkHostMetalContext pass entirely** (that pass is TileLang-specific PyTorch-MPS sync; it's NOT in TVM's base Metal runtime).
  - **Path C (~500 LOC vendor, no torch)**: vendor apache/tvm/src/runtime/metal/ (337+441 LOC Obj-C++) — what mlc-ai/mlc-llm uses. TVM Metal runtime is fully self-contained: MTLStorageModeShared, allocates own MTLBuffer, no PyTorch references in the codebase.
  - Force fp16 carrier — bf16 simdgroup MSL codegen has known bugs (cubecl#1202 open, fix in PR #1207 still missing simd_shuffle gate).
  - The CuTe DSL Mamba3 path (cppmega/megatron/cute_dsl_mimo/) is sm_90a-only and a dead-end on Metal — anchor on TileLang only.
- **Selective-scan Metal kernel (Mamba1)** — vendor D-CSIL ssm_scan.py post-M0 if TileLang path proves too heavy. Reference scan in cppmega_mlx/nn/mamba3.py is fine for M0 smoke + correctness.
- **Fused GEMM+bias+act** — DIY mx.fast.metal_kernel only when profile shows BNNS+op-fuser path is the bottleneck. No upstream MLX merge in 0.32; PR #1123 closed stale.
- **Custom Metal CCE** — the current pure-MLX CCE path is an opt-in train backend and enough to keep M0 work unblocked. Metal/manual-backward CCE remains a **measured optimization / c08.2 acceptance item**, not the default path. Pattern: mirror Apple CCE's flash-memory matmul + lse reduction in MSL.

### Don't pursue (false ROI)

- **ZMLX gather_qmm_swiglu** — pinned MLX SHA, ~800 LOC C++/Metal patch, **inference-only no backward**, gated on N%8 && K%512. Wrong tool for a training package.
- **alxndrTL Blelloch parallel scan** — author himself notes it's slower than sequential MLX loop on M-class GPUs due to reshape overhead.
- **Triton-on-Metal** (RobotFlow-Labs/Triton-mlx, triton PR #9701) — early/experimental, no MLX bridge.
- **Megablocks-MLX / DeepEP** — only worth it at 64+ experts; we have 4.

## Risk register

- **bf16 carrier vs fp32 accum**: any custom Metal kernel must pin accum dtype to fp32 to avoid ~1e-3 drift vs MPS BNNS. Match upstream tolerance anchors.
- **MLX autograd memory anti-pattern**: chunked operations recorded inside mx.value_and_grad may use *more* memory than materialized — autograd records the full trace before evaluation. Workaround: run chunked loops *outside* autograd with manual VJP (this is what cut_cross_entropy.linear_cross_entropy_value_and_grad does).
- **gather_qmm indices VJP**: indices are non-differentiable. Always wrap argpartition / top-k selection with mx.stop_gradient.
- **Mamba3 MIMO**: no public Apple Silicon reference; CuTe DSL parity will need custom kernel work.

## Sources

- **Matmul/GEMM (D)**: ml-explore/mlx PRs #1123 (closed), #2569 (CUDA-only), #2078 (gather_qmm); philipturner/metal-flash-attention; Hmbown/ZMLX
- **Activations (E)**: ml-explore/mlx-lm PR #753 (merged), #774 (open); Hmbown/ZMLX
- **Loss (F)**: apple/ml-cross-entropy (Triton, Apple SCL); linkedin/Liger-Kernel; unslothai/unsloth; this repo's cut_cross_entropy.py
- **MoE (G)**: ml-explore/mlx-lm switch_layers.py (SwitchGLU); mx.gather_mm; arXiv 2506.23635 (DBRX EP on Apple Silicon)
- **Mamba (H)**: ml-explore/mlx-lm PR #1153 (open); D-CSIL/mlx-recurrence; shinyoungkang1/mlx-vision-mamba; alxndrTL/mamba.py
- **Quant (I)**: mx.quantized_matmul; mlx-lm AWQ/GPTQ/DWQ; vllm-metal TurboQuant; arozanov/sharpner/helgklaizar/qjl-mlx forks
