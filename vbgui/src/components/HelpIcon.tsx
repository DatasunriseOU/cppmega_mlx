// General-purpose "?" icon + explanation modal for any in-app value.
// Topic content is centralised in `HELP_TOPICS` so writing the help is
// not gated on touching the component being explained.

import { useState } from "react";
import { createPortal } from "react-dom";
import { TENSOR_DIAGRAMS } from "./diagrams";
import { T } from "@/theme";

export interface HelpTopic {
  title: string;
  what: string;
  why: string;
  example?: string;
  reference?: string;
  // V7-Q09: brick-tile help additions. Inputs / outputs / normalization
  // describe upstream-compatibility, downstream-compatibility, and
  // pre/post-norm requirements for a brick so the operator knows what
  // to connect and where to drop a norm.
  inputs?: string;
  outputs?: string;
  normalization?: string;
}

export const HELP_TOPICS: Record<string, HelpTopic> = {
  // ----- dim_env -----
  dim_env_H: {
    title: "dim_env.H — residual stream width",
    what:
      "The hidden / model dimension that flows through residual " +
      "connections. Every brick that reads/writes the residual stream " +
      "exposes a tensor of shape [B, S, H].",
    why:
      "H controls memory + compute scaling roughly quadratically " +
      "(self-attention) or linearly (MLP) in the brick. Most " +
      "in-tree presets accept H as a constructor knob so the same " +
      "preset can be instantiated mini (H=128) or full (H=4096).",
    example:
      "MINI_HIDDEN=128 is the default for smoke runs; production " +
      "Llama-3 8B uses H=4096; DeepSeek-V3 uses H=7168.",
  },
  dim_env_nh: {
    title: "dim_env.nh — number of query heads",
    what:
      "How many attention heads the SDPA / GQA / MLA blocks split " +
      "queries into. Each head has dimension head_dim.",
    why:
      "More heads = more parallel attention patterns but smaller " +
      "per-head dim. The product nh*head_dim is the *query/output* " +
      "projection dim, which need NOT equal H — many architectures " +
      "use an internal W_Q : R^H → R^{nh*head_dim} projection so the " +
      "Q-space is decoupled from the residual stream.",
    example:
      "Llama-3 70B uses nh=64, head_dim=128, H=8192 (nh*head_dim==H). " +
      "DeepSeek-V3 MLA uses nh=128, head_dim=192, H=7168 (decoupled).",
  },
  dim_env_head_dim: {
    title: "dim_env.head_dim — per-head dimension",
    what:
      "The width of each attention head — both the per-head Q/K " +
      "projection size and the softmax-normalisation dim.",
    why:
      "Common values are 64, 96, 128, 256. Larger head_dim trades " +
      "more compute per head for fewer, richer heads. Combined " +
      "with nh, defines the attention sub-block compute as " +
      "nh * head_dim * H per token (roughly).",
  },
  dim_env_B: {
    title: "dim_env.B — batch size",
    what:
      "How many independent sequences are forwarded in one step.",
    why:
      "Higher B = more parallelism + better GPU utilisation, but " +
      "memory scales linearly with B. For the in-GUI smoke run " +
      "B=1 keeps things fast.",
  },
  dim_env_S: {
    title: "dim_env.S — sequence length",
    what:
      "Number of tokens per batch item.",
    why:
      "Attention memory scales O(B*S^2) for vanilla SDPA. The " +
      "GUI default S=64 keeps mini runs interactive.",
  },
  // ----- train options (K-block) -----
  train_val_every: {
    title: "val_every — validation cadence (V7-A04)",
    what:
      "Run a held-out validation pass every N training steps.",
    why:
      "Catches train/val divergence early. Default off (no validation " +
      "during train) keeps mini smoke runs fast; production runs " +
      "typically val every 50-200 steps.",
  },
  train_grad_clip: {
    title: "grad_clip_max_norm — gradient clipping cap",
    what:
      "Global-norm clip on the gradient vector before the optimizer " +
      "step. If ||grad|| > N, scale grad by N / ||grad||.",
    why:
      "Prevents the rare exploding-gradient blow-up from poisoning " +
      "the optimizer state. 1.0 is a near-universal default; some " +
      "MoE/large-LR setups use 0.5 or 2.0.",
  },
  train_loss_scaler: {
    title: "loss_scaler — fp16 dynamic range adapter",
    what:
      "When training in fp16, the loss is multiplied by init_scale " +
      "so its gradients fit fp16's narrow range. Optimizer unscales " +
      "before the parameter update. growth_interval is how many " +
      "consecutive non-overflow steps must pass before the scale " +
      "doubles.",
    why:
      "Without scaling, fp16 grads quickly underflow to zero and " +
      "training stalls. Dynamic scaling handles transient overflow " +
      "by halving + skipping the step.",
    example:
      "init_scale=65536, growth_interval=2000 is a common pair " +
      "(matches the NVIDIA APEX default).",
  },
  train_fake_ranks: {
    title: "fake_ranks — single-process multi-rank simulation",
    what:
      "Backend simulates an N-rank distributed train by running N " +
      "forward/backward passes in sequence and mean-reducing grads. " +
      "Surfaces gradient_reduce_ms in extras.",
    why:
      "Lets the architect validate that a multi-rank shard plan " +
      "doesn't blow numerical convergence — without spinning up " +
      "real GPUs.",
  },
  // ----- train metrics (M-block) -----
  metric_perplexity: {
    title: "perplexity — exp(loss)",
    what: "exp(mean cross-entropy loss) over the last logged step.",
    why: "Loss values change with vocab size, so perplexity is the " +
         "vocab-invariant readout architects compare across runs.",
  },
  metric_bpb: {
    title: "bits_per_byte",
    what: "Loss converted to bits per UTF-8 byte of input. " +
          "loss / ln(2) for byte-level tokenizers; corrected by " +
          "the avg bytes-per-token for BPE-style.",
    why: "Cross-tokenizer comparable. Production language-model " +
         "benchmarks (Pile, RedPajama) quote BPB instead of " +
         "perplexity.",
  },
  metric_dtype: {
    title: "master / actual dtype",
    what: "master_dtype is what the spec asked for; dtype_actual is " +
          "what the runtime ended up using after fp8/bf16 fallback " +
          "(M3 chips lack fp8, MLX downgrades silently).",
    why: "If the architect asked for fp16 but got fp32, the train " +
         "wall-clock will surprise them. Showing both at the same " +
         "time pins the contract.",
  },
  metric_fp8: {
    title: "fp8_active",
    what: "True when at least one fused region ran in fp8 (NV " +
          "transformer-engine path or MX-fp8 emulator).",
    why: "Distinct from the spec.sharding.fp8_enabled toggle, which " +
         "is the *request*. fp8_active is the *fact*.",
  },
  metric_fim: {
    title: "FIM — fill-in-the-middle",
    what: "fim_active=true means the dataset pre-processor masked " +
          "the middle third of each sample and asked the model to " +
          "predict it. fim_ratio is the share of samples touched.",
    why: "Codegen models (Codestral, DeepSeek-Coder) train on FIM. " +
         "Validating the toggle landed is critical because dropping " +
         "FIM accidentally collapses code-completion quality.",
  },
  metric_optimizer: {
    title: "optimizer_kind",
    what: "Which optimizer the train step actually ran (adamw / " +
          "lion / muon / hybrid…). Reflects the runtime state, not " +
          "the spec request — auto-fallback for unsupported opts.",
    why: "Quick sanity check that the architect's spec landed.",
  },
  metric_gradient_reduce: {
    title: "gradient_reduce_ms",
    what: "Wall-clock spent in the (synthetic, fake_ranks-driven) " +
          "all-reduce of gradients per train step. Single-process " +
          "proxy for a real NCCL all-reduce.",
    why: "Lets the architect estimate distributed-train scaling " +
         "without a real cluster.",
  },
  metric_grad_clip: {
    title: "grad-clip activity",
    what: "max_grad_norm_seen is the largest ‖g‖ observed across " +
          "the run; num_clips counts how many steps tripped the " +
          "grad_clip_max_norm threshold.",
    why: "When clips spike, the spec's lr/grad_clip combo is wrong. " +
         "Surfacing both numbers means the architect can spot a " +
         "diverging run before perplexity confirms it.",
  },
  metric_sharding: {
    title: "sharding applied",
    what: "sharding_applied=true when at least one parallel axis " +
          "engaged. per_rank_param_bytes is the per-device share of " +
          "the weights post-shard.",
    why: "Confirms the FSDP / TP / EP plan actually executed.",
  },
  metric_side_channels: {
    title: "side-channels observed",
    what: "Which side-channel feature families the model actually " +
          "consumed during training (doc_ids, token_ids, …).",
    why: "If the architect wired doc_ids in but training ran with " +
         "the placeholder zero channel, this list catches the leak.",
  },
  metric_per_brick_grad: {
    title: "per-brick grad-norm",
    what: "‖g‖ on each brick's parameters after the last train step. " +
         "Surfaces vanishing/exploding-grad patterns localised to a " +
         "subgraph.",
    why: "A flat 0 next to a 1e3 spike on adjacent bricks signals a " +
         "specific module mis-initialised.",
  },
  metric_moe: {
    title: "MoE routing dashboard",
    what: "9 routing keys: routing_entropy (token diversity), " +
          "load_balance_loss, per_expert_load (bars), " +
          "dropped_token_ratio, rerouted_token_ratio, overflow_ratio, " +
          "capacity_per_expert, capacity_factor, num_experts.",
    why: "MoE collapses silently when one expert wins all routes; " +
         "the bar chart catches it visually within a few steps.",
  },
  metric_inference_steps: {
    title: "verify_build_spec inference_steps flow trace",
    what: "Step-by-step walk of how the dimension auto-inferer " +
          "resolved each brick parameter — which dim_env value got " +
          "picked, which fallback rule fired, which user override " +
          "won.",
    why: "When num_heads comes out as something the architect didn't " +
         "expect, this trace shows exactly which rule landed it.",
  },
  gen_run: {
    title: "gen.run — autoregressive token generation",
    what:
      "Fires the gen.run RPC with prompt_tokens + a sampler " +
      "strategy (greedy / temperature / top_k / top_p), returns the " +
      "generated tokens + finish_reason (eos / length / aborted) + " +
      "wall-clock ms.",
    why:
      "Validates the inference path end-to-end before committing to " +
      "a long generation. 'smoke' mode uses synthetic logits so the " +
      "request doesn't need a real model — useful for sanity-checking " +
      "the sampler config.",
    example:
      "prompt_tokens=[1,2,3], strategy=top_p, top_p=0.9, " +
      "max_new_tokens=16 → reproducible token sequence under fixed " +
      "seed.",
  },
  rpc_error_data: {
    title: "RPC error.data — backend field-level details",
    what:
      "JSON-RPC errors carry an optional .data blob the UI used to " +
      "discard. The ErrorDetailsPanel now walks Pydantic-style " +
      "errors[] arrays, runtime traceback strings, and stage names " +
      "to surface them next to the headline message.",
    why:
      "Without the field detail, an INVALID_PARAMS error reads as " +
      "'1 validation error for VerifyParams' — useless. With the " +
      "expanded view, the architect sees that, e.g., " +
      "graph.nodes.2.kind was 'rmsnorm' (an adapter, not a brick).",
    example:
      "errors: [{loc: ['graph','nodes',2,'kind'], msg: 'unknown " +
      "brick', type: 'value_error'}] now renders as a clickable " +
      "list item.",
  },
  spec_validation_recovery: {
    title: "Spec validation recovery — backend-suggested fixes",
    what:
      "When verify_build_spec rejects a spec with a known-fixable " +
      "pattern (missing edge, dim mismatch, bad dtype combo, etc.), " +
      "the gotcha payload carries a `suggested_fix` hint and a known " +
      "id from a fixable-set. The UI surfaces an Apply button.",
    why:
      "The architect shouldn't need to read the message and guess " +
      "which dropdown to change. One-click fixes turn 'broken spec' " +
      "into 'one keystroke away from valid'.",
  },
  warm_start_history: {
    title: "warm_start_history — which prior run to continue from",
    what:
      "Picks the run_id that warm-start should load opt state + " +
      "weights from. '(latest)' uses lastTrainRunId — the previous " +
      "default.",
    why:
      "When the architect has run several trials and wants to branch " +
      "from a specific one (e.g. a low-loss checkpoint two runs " +
      "back), they need to name it explicitly rather than always " +
      "continuing the most recent run.",
  },
  train_live_controls: {
    title: "train_live_controls — mid-run knobs",
    what:
      "Two controls usable around / during a train run: 'Trigger " +
      "checkpoint' enqueues a one-shot ckpt_save path the next run " +
      "honours; 'Apply lr' submits a pipeline.update_lr RPC against " +
      "the active run id.",
    why:
      "Manually checkpointing at an interesting loss valley, or " +
      "stepping LR down when train plateaus, are workflows that " +
      "production runs need but that smoke runs hardcode away.",
  },
  train_abort_token: {
    title: "abort_token — explicit cancel handle",
    what:
      "A string the train run polls every step; if the host sets " +
      "the token's flag (via pipeline.abort RPC), the train exits " +
      "cleanly at the next step boundary.",
    why:
      "When the run_id derived default isn't enough (e.g. a CLI " +
      "wants to abort an in-flight train without knowing the " +
      "auto-generated run_id), the architect can pin a stable token " +
      "here.",
  },
  // ----- parallel composition -----
  parallel_block: {
    title: "Parallel-block composition (tiny-aya style)",
    what:
      "Constructs a graph where a single input fans out into TWO " +
      "parallel computation branches that are then joined by an " +
      "additive residual + final norm. The two branches see the " +
      "same input tensor but their outputs are summed.",
    why:
      "Parallel attention + MLP is the defining feature of Aya-style " +
      "and GPT-J-style architectures (Wang & Komatsuzaki 2021). " +
      "Mixing-and-matching what runs in each branch is one of the " +
      "fastest small-model ablations. The button-driven shortcut " +
      "exists because manually drawing parallel edges through React " +
      "Flow drag-handles is flaky under Playwright; the resulting " +
      "graph is identical to what manual edge-drawing would yield.",
    example:
      "tiny-aya: aya_input → {aya_attn, aya_mlp} → aya_join " +
      "(residual) → aya_norm. Two paths from input, one join, one " +
      "final norm.",
  },
  // ----- insert into edge -----
  insert_into_edge: {
    title: "Insert brick into an edge",
    what:
      "Splits an existing edge A→B by inserting a new brick X " +
      "between them: A→B becomes A→X and X→B. Node ids and the rest " +
      "of the graph topology stay untouched.",
    why:
      "Lets the architect try interleaving new computation (e.g. an " +
      "mLSTM-style nonlinear-rnn brick between an attention and an " +
      "MLP) without re-laying-out the whole canvas. Verify re-runs " +
      "automatically, surfacing any shape mismatch immediately.",
    example:
      "llama3_8b_attn → llama3_8b_mlp is the default llama edge. " +
      "Insert an mLSTM block to get attn → mlstm → mlp; train 2 " +
      "steps to validate it's loss-finite before scaling up.",
  },
  // ----- brick transplant -----
  brick_transplant: {
    title: "Cross-preset brick transplant",
    what:
      "Picks a brick (kind + params) out of one preset and drops " +
      "it onto the current canvas. The new node lands unconnected; " +
      "the architect wires its edges manually.",
    why:
      "Mixing-and-matching architecture choices — e.g. taking the " +
      "Mixtral-style MoE brick and trying it in a Llama scaffold — " +
      "is the fastest way to validate an ablation hypothesis without " +
      "writing a new preset factory. The transplanted brick keeps " +
      "the source preset's parameter choices, so the architect sees " +
      "exactly what the upstream author shipped.",
    example:
      "Source preset llama4_maverick → moe brick (num_experts=8, " +
      "top_k=2). Drop into a llama3_8b canvas, connect after the " +
      "attention block, train 2 steps to compare against the dense " +
      "MLP variant.",
  },
  // ----- tokenizer matrix -----
  tokenizer_matrix: {
    title: "Tokenizer × preset compatibility matrix",
    what:
      "A grid showing each preset paired with each available " +
      "tokenizer. Cells are 'ok' (encodes cleanly), 'incompat' " +
      "(empty token stream — the tokenizer has no vocabulary for " +
      "the probe text), or 'error' (file missing / corrupt).",
    why:
      "Mixing a preset (which expects a particular vocab_size + " +
      "special-token contract) with the wrong tokenizer is the " +
      "single most common training failure mode in practice. " +
      "Validating the pair *before* train tells the architect " +
      "instantly whether their fixture is even self-consistent. " +
      "Click any cell to inspect the first 10 token ids it " +
      "produced for the canonical probe text.",
    example:
      "T1_cppmega_v3.json + llama3_8b → ok (vocab=32000 fits). " +
      "T4_fim_only.json + mistral_small_3_1 → incompat (no " +
      "non-FIM tokens emitted).",
  },
  // ----- F56b convention -----
  symbolic_dim_mismatch: {
    title: "Why nh*head_dim != H is allowed (but warned)",
    what:
      "When dim_env pins H, nh, and head_dim, the codebase does " +
      "*not* enforce nh*head_dim == H — but it does warn.",
    why:
      "Many recent architectures (DeepSeek-V3 MLA, Gemma 3 MQA, " +
      "Qwen3) intentionally decouple the Q-space from the residual " +
      "stream by inserting an internal W_Q : R^H → R^{nh*head_dim} " +
      "projection. So the contradiction is a real-world pattern, " +
      "not a bug. The warning exists because in *most* user-typed " +
      "specs the mismatch is a typo (e.g. Llama-style where the " +
      "architect meant H = nh*head_dim).",
    example:
      "Llama-3 8B  → H=4096, nh=32, head_dim=128 → 32*128=4096 ✓ " +
      "DeepSeek-V3 → H=7168, nh=128, head_dim=192 → 128*192=24576 " +
      "(decoupled Q via W_Q projection).",
    reference: "https://arxiv.org/abs/2412.19437",
  },

  // V7-Q09: per-brick palette help. Each brick entry covers what it
  // does, why you'd reach for it, expected input/output shapes, and
  // where normalization lives.

  // ----- SDPA / GQA attention family -----
  brick_attention: {
    title: "attention — Vanilla SDPA",
    what:
      "Standard scaled-dot-product attention with N query heads, KV "
      + "projections, and a residual-shaped output. Backed by "
      + "mx.fast.scaled_dot_product_attention on MPS.",
    why:
      "Default attention block. Use when the architecture is plain "
      + "Llama / GPT / Mistral and you don't need linear / sliding / "
      + "sparse variants.",
    inputs:
      "x: (B, S, H). Optional KV cache + attention mask (causal or "
      + "doc-aware). num_heads, head_dim, num_kv_heads from params.",
    outputs:
      "y: (B, S, H). Shape-preserving — drops into a residual stream.",
    normalization:
      "Apply RMSNorm BEFORE the block (pre_norm='rmsnorm'). Post-norm "
      + "is optional — Llama-style architectures use pre-norm only.",
  },
  brick_gated_attention: {
    title: "gated_attention — Qwen3-Next style",
    what:
      "Attention block with an output gate + partial RoPE + Q/K "
      + "RMSNorm + asymmetric GQA. Re-export of "
      + "mlx_lm.models.qwen3_next.Qwen3NextAttention.",
    why:
      "Better at long-context routing — the output gate suppresses "
      + "attention contribution per-token when the residual already "
      + "carries enough info.",
    inputs:
      "x: (B, S, H). num_attention_heads, num_key_value_heads, "
      + "head_dim params. Internally splits GQA: nh queries, nkv kv.",
    outputs: "y: (B, S, H). Shape-preserving.",
    normalization:
      "Q/K RMSNorm is built-in; you still need a residual pre-norm "
      + "outside the brick.",
    reference: "Qwen3 paper §3.4",
  },
  brick_mla: {
    title: "mla — Multi-Latent Attention (DeepSeek V2/V3)",
    what:
      "Attention with LoRA-compressed Q + LoRA-compressed KV + RoPE "
      + "on the pe-only split. Decouples key/value memory from "
      + "head_dim — KV cache shrinks dramatically.",
    why:
      "Long-context inference: a 64k context costs roughly half the "
      + "KV memory of vanilla GQA at the same head_dim because of "
      + "the LoRA-compressed latent.",
    inputs:
      "x: (B, S, H). q_lora_rank + kv_lora_rank + head_dim params. "
      + "Position rotates only the pe-split (rope_dim) channel.",
    outputs: "y: (B, S, H). Same shape as vanilla attention.",
    normalization:
      "Pre-norm required. Built-in Q/K RMSNorm scales the LoRA "
      + "products before the attention dot-product.",
    reference: "https://arxiv.org/abs/2412.19437",
  },
  brick_mla_absorb: {
    title: "mla_absorb — MLA with absorbed fast-path",
    what:
      "Same as mla but prefers the absorbed-LoRA fast-path at decode "
      + "step (merges W_Q*W_K_up and W_O*W_V_up into a single matmul).",
    why:
      "Inference speedup at the cost of bigger weight tensors. Use "
      + "for serving; use plain mla for training.",
    inputs: "Same as mla. Switching is a deploy-time decision.",
    outputs: "Same as mla.",
    normalization: "Same as mla.",
  },
  brick_mistral4_mla: {
    title: "mistral4_mla — Mistral Small 4 absorbed MLA",
    what:
      "Mistral-Small-4 variant of MLA: absorbed Q/KV LoRA + INT4 "
      + "latent cache. Vendored from mlx-lm PR #1037.",
    why:
      "Production Mistral Small 4 architecture; INT4 cache further "
      + "halves long-context memory vs DeepSeek V3.",
    inputs: "x: (B, S, H). Same LoRA-rank knobs as mla.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required; built-in Q/K RMSNorm.",
  },
  brick_dsv4_attention: {
    title: "dsv4_attention — DeepSeek V4 hash-indexed sparse",
    what:
      "Hash-indexed sparse attention from DeepSeek V4 Flash (mlx-lm "
      + "PR #1201). Routes each query to a small set of keys via a "
      + "learned hash + top-k selector.",
    why:
      "Sub-quadratic attention for very long context (32k+). "
      + "Inherits MLA's KV compression on top.",
    inputs:
      "x: (B, S, H). num_hash_buckets, top_k params + per-head hash "
      + "projection weights.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_bailing_mla: {
    title: "bailing_mla — Ling-2.6 multi-latent attention",
    what:
      "Multi-latent attention block from the Ling-2.6 family "
      + "(mlx-lm PR #1227). MLA variant tuned for Ling's sparse MoE.",
    why:
      "Pair with bailing_moe to assemble Ling-style architectures.",
    inputs: "x: (B, S, H). LoRA-rank params similar to mla.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_gqa_sliding: {
    title: "gqa_sliding — sliding-window GQA",
    what:
      "Grouped-query attention with a fixed-size sliding window "
      + "instead of full causal attention. Window_size param caps "
      + "attended history per token.",
    why:
      "5:1 sliding:global ratio is the Gemma 3 + Arcee Trinity "
      + "convention — caps long-range cost while keeping every 6th "
      + "layer global.",
    inputs:
      "x: (B, S, H). num_heads, num_kv_heads, window_size, head_dim "
      + "params.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required (Gemma uses RMSNorm).",
  },
  brick_cca_attention: {
    title: "cca_attention — ZAYA1 Coarse-Causal Attention",
    what:
      "Compressed-context attention: pools key/value tokens into "
      + "coarse blocks before the dot-product. ZAYA1 architecture.",
    why:
      "Memory-bounded long-context with explicit compression — "
      + "trades fine-grained recall for predictable cost.",
    inputs:
      "x: (B, S, H). pool_size + num_heads params. Causal masking "
      + "respects block boundaries.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_gemma4_drafter: {
    title: "gemma4_drafter — Gemma 4 MTP drafter",
    what:
      "Multi-token-prediction drafter layer with cross-attention "
      + "to the main backbone (mlx-lm PR #1276).",
    why:
      "Speculative decoding head — drafts the next 2-3 tokens "
      + "from the same hidden state.",
    inputs:
      "x: (B, S, H) + cross_kv from main stream. drafter_k param "
      + "controls how many lookahead tokens.",
    outputs: "y: (B, S, H) tied to the drafter head loss.",
    normalization: "Pre-norm on both self + cross paths.",
  },
  brick_nemotron_h_mtp: {
    title: "nemotron_h_mtp — Nemotron-H MTP",
    what:
      "Multi-token-prediction block from NVIDIA Nemotron-H "
      + "(mlx-lm PR #1161).",
    why:
      "Drafter variant tuned for Nemotron's MoE+SSM backbone.",
    inputs: "x: (B, S, H). drafter_k param.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },

  // ----- Linear / nonlinear attention -----
  brick_bailing_linear: {
    title: "bailing_linear — Ling-2.6 linear attention",
    what:
      "Linear attention block from Ling-2.6-flash (mlx-lm "
      + "PR #1227). Replaces softmax with a kernel that runs in "
      + "O(S) instead of O(S²).",
    why:
      "Cheap long-context inference; pair with bailing_mla on "
      + "global layers for hybrid sparsity.",
    inputs: "x: (B, S, H). num_heads + head_dim params.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_gdn: {
    title: "gdn — Gated Delta Net",
    what:
      "Linear attention with a delta-rule update + per-token gate. "
      + "Maintains a recurrent state across timesteps.",
    why:
      "Cheap recurrent block for hybrid architectures (e.g. "
      + "Nemotron H, Qwen3-Next). Strong long-range vs vanilla "
      + "linear attention.",
    inputs: "x: (B, S, H). num_heads, head_dim, conv_size params.",
    outputs: "y: (B, S, H) + state carry forward.",
    normalization: "Pre-norm required; built-in gate.",
  },
  brick_kda: {
    title: "kda — Kernel-Delta Attention",
    what:
      "Kimi Linear's delta-style linear attention. Sliding-window "
      + "convolution + kernelised softmax surrogate.",
    why:
      "Used in Kimi Linear architecture for cheap recurrence at "
      + "training time.",
    inputs: "x: (B, S, H). num_heads + conv_size params.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_mamba3: {
    title: "mamba3 — Selective SSM (Mamba-3)",
    what:
      "Selective state-space model with per-token gating + delta "
      + "discretization. Re-export from cppmega_mlx.nn.mamba3.",
    why:
      "Non-attention sequence model — long context at O(S) cost. "
      + "Used in Nemotron 3 Super, Phi-4-mini, hybrid backbones.",
    inputs:
      "x: (B, S, H). num_heads, head_dim, d_state, conv_size, "
      + "expand params.",
    outputs: "y: (B, S, H). Stateful — can carry h_state forward.",
    normalization:
      "Pre-norm required; internal silu gate + RMSNorm on output "
      + "projection.",
  },
  brick_mlstm: {
    title: "mlstm — Matrix-memory xLSTM",
    what:
      "Matrix-memory variant of LSTM (xLSTM 7B). Stores a per-head "
      + "matrix state rather than a vector hidden state.",
    why:
      "Non-attention long-context sequence model. Tradeoff: more "
      + "expressive state than mamba3, more compute.",
    inputs:
      "x: (B, S, H). num_heads, head_dim, rms_norm_eps params.",
    outputs: "y: (B, S, H). Carries matrix state.",
    normalization: "Pre-norm + internal RMSNorm on output.",
  },

  // ----- MoE -----
  brick_moe: {
    title: "moe — Mixture-of-Experts MLP",
    what:
      "MLP replaced by N expert sub-MLPs + top-K router. Each "
      + "token activates only K experts (typically K=2 of 8).",
    why:
      "Scales MLP capacity sub-linearly with compute. Standard "
      + "above ~10B params in modern architectures.",
    inputs:
      "x: (B, S, H). num_experts, top_k, capacity_factor params. "
      + "Aux load-balance loss surfaces in extras.moe.",
    outputs:
      "y: (B, S, H). Plus auxiliary metrics: routing_entropy, "
      + "load_balance_loss, dropped_token_ratio.",
    normalization:
      "Pre-norm required (same as plain MLP). Watch capacity_factor "
      + "< 1 — dropped tokens degrade quality.",
  },
  brick_bailing_moe: {
    title: "bailing_moe — Ling-2.6 sparse MoE",
    what:
      "MoE variant from Ling-2.6 (mlx-lm PR #1227) with a shared "
      + "expert + N routed experts.",
    why:
      "Stronger small-expert specialization than vanilla MoE; "
      + "shared expert covers the always-on path.",
    inputs:
      "x: (B, S, H). num_experts + num_shared_experts + top_k.",
    outputs: "y: (B, S, H) + load-balance metrics in extras.",
    normalization: "Pre-norm required.",
  },

  // ----- Sparse / specialized attention -----
  brick_nsa: {
    title: "nsa — Natively Sparse Attention",
    what:
      "Multi-branch sparse attention: combines local window + "
      + "compressed coarse + top-k selected branches per query.",
    why:
      "Sub-quadratic full-context attention at training time, not "
      + "just inference. DeepSeek V3.2-style.",
    inputs:
      "x: (B, S, H). num_heads, window_size, top_k, compression_ratio.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_lightning_indexer: {
    title: "lightning_indexer — sparse-MLA indexer",
    what:
      "Lightning Indexer from sparse MLA: picks top-k key positions "
      + "to attend per query. Pair with CSA/HCA.",
    why:
      "Selector head that feeds sparse-MLA. Reduces the per-query "
      + "key set to a small top-k.",
    inputs: "x: (B, S, H). top_k + index_dim params.",
    outputs:
      "sparse_idx: (B, S, top_k) integer indices into the key sequence.",
    normalization: "Pre-norm before; internal projection is bias-free.",
  },

  // ----- Cross-attention / engram bridges -----
  brick_csa_hca: {
    title: "csa_hca — Compressed Self+History Cross Attention",
    what:
      "Hybrid: a compressed-self-attention path + a history-cross "
      + "attention path inside one block. Output is the sparse-attn "
      + "result combined with the lightning_indexer top-k.",
    why:
      "Building block for sparse-MLA gallery models. Reuses the "
      + "indexer to bound key set then attends.",
    inputs:
      "x: (B, S, H) + sparse_idx from lightning_indexer + qr "
      + "(rotation token).",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm required.",
  },
  brick_engram: {
    title: "engram — Long-term memory cross-attention",
    what:
      "Cross-attention against a learned long-term memory bank. "
      + "Tokens read from N persistent memory slots.",
    why:
      "Memory-augmented backbone — slots persist across batches "
      + "for stronger long-context retention.",
    inputs:
      "x: (B, S, H). memory_size + num_heads params; memory bank "
      + "is module-internal state.",
    outputs: "y: (B, S, H).",
    normalization: "Pre-norm on the input; memory bank is unnorm'd.",
  },

  // ----- Norm / projection / embed -----
  brick_mlp: {
    title: "mlp — Gated MLP (SwiGLU / GeGLU / etc.)",
    what:
      "Two-projection MLP with a configurable activation. Default "
      + "is gated (swiglu/sigmoid×up), can switch to gelu/relu/silu "
      + "via params.activation.",
    why:
      "Standard FFN. Most architectures alternate attention block "
      + "+ mlp block in each layer.",
    inputs:
      "x: (B, S, H). intermediate_size (defaults 4*H), activation "
      + "(glu/swiglu/geglu/reglu/gelu/silu/relu/relu2/sqrelu/xielu/"
      + "mish), pre_norm, post_norm params.",
    outputs: "y: (B, S, H). Shape-preserving.",
    normalization:
      "Pre-norm strongly recommended (RMSNorm). post_norm rare — "
      + "only some non-Llama architectures (e.g. OLMo).",
  },
  brick_abs_pos_embed: {
    title: "abs_pos_embed — Learned absolute positions",
    what:
      "Adds a learned (S, H) positional residual to the input "
      + "before any attention block.",
    why:
      "Used by GPT-2 family and pre-RoPE architectures. Caps S at "
      + "max_position_embeddings.",
    inputs:
      "x: (B, S, H). max_position_embeddings param (e.g. 1024 for "
      + "GPT-2, 2048 for GPT-2 XL).",
    outputs:
      "y: (B, S, H) — x + learned_positions[:S].",
    normalization:
      "No norm needed inside; following attention block applies "
      + "its own pre-norm.",
  },
  brick_per_layer_embed: {
    title: "per_layer_embed — Gemma-4 per-layer scale",
    what:
      "Per-layer scaled embedding residual: adds a learned "
      + "embedding scaled by a per-layer factor (1/sqrt(layer_idx)).",
    why:
      "Gemma 4 E2B/E4B convention. Stabilises gradient flow in "
      + "very deep stacks.",
    inputs:
      "x: (B, S, H). layer_index + num_layers params. Scale factor "
      + "computed at init.",
    outputs: "y: (B, S, H).",
    normalization: "No norm; this is a pure residual add.",
  },

  // ----- Adapters -----
  adapter_merge_heads: {
    title: "merge_heads — concatenate head dimension",
    what:
      "Reshape (B, S, nh, hd) → (B, S, nh*hd). Used between "
      + "attention output and projection.",
    why:
      "Plumbing between split-head and dense projection spaces. "
      + "Necessary when an attention brick emits per-head output "
      + "but the next brick expects (B, S, H).",
    inputs: "(B, S, nh, hd).",
    outputs: "(B, S, nh*hd).",
    normalization: "No norm. Pure reshape.",
  },
  adapter_split_heads: {
    title: "split_heads — split into head dimension",
    what:
      "Reshape (B, S, nh*hd) → (B, S, nh, hd). Inverse of "
      + "merge_heads.",
    why:
      "Use before a per-head op (e.g. RoPE applied per-head) when "
      + "the upstream brick produces a flat residual.",
    inputs: "(B, S, nh*hd).",
    outputs: "(B, S, nh, hd).",
    normalization: "No norm. Pure reshape.",
  },
  adapter_transpose_bnsd: {
    title: "transpose_bnsd — BNSD ↔ BSND layout swap",
    what:
      "Swaps the seq and head axes: (B, S, nh, hd) ↔ (B, nh, S, hd).",
    why:
      "Some attention kernels expect (B, nh, S, hd); others "
      + "(B, S, nh, hd). Drop this adapter when layouts disagree.",
    inputs: "(B, S, nh, hd) or (B, nh, S, hd).",
    outputs: "Other ordering of the same dims.",
    normalization: "No norm.",
  },
  adapter_linear_bridge: {
    title: "linear_bridge — H_in → H_out projection",
    what:
      "Plain nn.Linear used to bridge mismatched residual widths.",
    why:
      "Insert between two bricks when their residual H differs "
      + "(e.g. an upstream MLA emits 2*H_residual due to "
      + "decoupled-Q convention).",
    inputs: "(B, S, H_in).",
    outputs: "(B, S, H_out).",
    normalization:
      "Optional. Insert RMSNorm after if the projection changes "
      + "scale significantly.",
  },
  adapter_rmsnorm: {
    title: "rmsnorm — Root Mean Square Norm",
    what:
      "Standard RMSNorm: divides by sqrt(mean(x²) + eps) then "
      + "scales by a learned gamma. No bias term, unlike LayerNorm.",
    why:
      "Pre-norm or post-norm gate. Most modern LLMs (Llama, Mistral, "
      + "DeepSeek, Gemma) use RMSNorm pre-attention + pre-MLP.",
    inputs: "(B, S, H).",
    outputs: "(B, S, H). Shape-preserving.",
    normalization:
      "Insert BEFORE attention/MLP (pre-norm). post-norm is rare. "
      + "eps default 1e-6.",
  },
  adapter_residual: {
    title: "residual — explicit Add",
    what:
      "Adds the brick output back to the upstream residual stream. "
      + "Implicit in most blocks; this adapter makes it explicit "
      + "for advanced topologies.",
    why:
      "Use only when you've turned off a brick's implicit residual "
      + "or you're stitching together parallel branches manually.",
    inputs: "Two tensors of the same shape.",
    outputs: "Sum of inputs.",
    normalization: "None.",
  },
};

// V7-Q11: adapter kinds also appear as nodes on the canvas; the
// BrickContextPanel looks them up under brick_<kind>. Mirror the
// adapter_* entries to brick_* so the panel doesn't show "No
// explanation" for rmsnorm / residual / merge_heads / split_heads /
// transpose_bnsd / linear_bridge nodes.
for (const adapterKind of [
  "merge_heads", "split_heads", "transpose_bnsd",
  "linear_bridge", "rmsnorm", "residual",
]) {
  const aKey = `adapter_${adapterKind}`;
  const bKey = `brick_${adapterKind}`;
  if (HELP_TOPICS[aKey] && !HELP_TOPICS[bKey]) {
    HELP_TOPICS[bKey] = HELP_TOPICS[aKey]!;
  }
}

export interface HelpIconProps {
  topic: keyof typeof HELP_TOPICS | string;
  size?: number;
}

export function HelpIcon({ topic, size = 14 }: HelpIconProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const entry = HELP_TOPICS[topic];

  return (
    <>
      <button
        type="button"
        data-testid={`help-icon-${topic}`}
        aria-label={`Help: ${entry?.title ?? topic}`}
        onClick={() => setOpen(true)}
        style={{
          width: size, height: size, padding: 0,
          borderRadius: "50%", background: "rgba(255, 255, 255, 0.08)",
          color: T.accent, border: `1px solid ${T.border}`,
          fontSize: Math.max(9, size - 4), lineHeight: `${size - 2}px`,
          cursor: "pointer", fontWeight: 700, marginLeft: 4,
          display: "inline-flex", alignItems: "center", justifyContent: "center",
        }}
      >
        ?
      </button>
      {open && <HelpModal topic={topic} onClose={() => setOpen(false)} />}
    </>
  );
}

export interface HelpModalProps {
  topic: string;
  onClose: () => void;
}

export function HelpModal({ topic, onClose }: HelpModalProps): JSX.Element {
  const entry = HELP_TOPICS[topic];

  // V7-Q12: render via createPortal(document.body) so the modal escapes
  // any transformed ancestor (React Flow canvas, palette, anything
  // with a CSS transform). Without this, position:fixed is positioned
  // relative to the nearest transformed ancestor (per CSS spec), which
  // pinned the modal inside the palette tile instead of the viewport.
  const modal = (
    <div
      data-testid="help-modal-backdrop"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        // UX-fix: backdropFilter: blur(8px) forced Chrome to recompose
        // the GPU layer on every mousemove (cursor crossing the
        // overlay → repaint loop with React Flow canvas underneath
        // → visible flicker). Plain rgba(0,0,0,0.5) keeps focus on
        // the modal without forcing per-frame backdrop recomposite.
        background: "rgba(15, 23, 42, 0.55)",
        // Promote to its own compositing layer so mousemove over the
        // backdrop doesn't invalidate the canvas paint underneath.
        transform: "translateZ(0)",
        willChange: "opacity",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 150,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        data-testid={`help-modal-${topic}`}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "rgba(30, 41, 59, 0.95)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          borderRadius: "12px",
          padding: "24px",
          width: "620px",
          maxWidth: "92vw",
          maxHeight: "88vh",
          overflowY: "auto",
          boxShadow: "0 20px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05)",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
            paddingBottom: "12px",
          }}
        >
          <h3
            data-testid="help-modal-title"
            style={{
              margin: 0,
              fontSize: "16px",
              fontWeight: "bold",
              color: "#22d3ee", // Vibrant cyan
              letterSpacing: "0.2px",
            }}
          >
            {entry?.title ?? topic}
          </h3>
          <button
            data-testid="help-modal-close"
            onClick={onClose}
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "50%",
              width: "32px",
              height: "32px",
              cursor: "pointer",
              fontSize: "20px",
              color: "#94a3b8",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              lineHeight: 1,
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(239, 68, 68, 0.2)";
              e.currentTarget.style.color = "#f87171";
              e.currentTarget.style.borderColor = "rgba(239, 68, 68, 0.3)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.05)";
              e.currentTarget.style.color = "#94a3b8";
              e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.1)";
            }}
          >
            ×
          </button>
        </header>

        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          {entry ? (
            <>
              <Section label="What" testid="help-modal-what">
                {entry.what}
              </Section>
              {(() => {
                const Diag = TENSOR_DIAGRAMS[topic];
                return Diag ? (
                  <Section label="Tensor flow"
                            testid="help-modal-diagram">
                    <Diag />
                  </Section>
                ) : null;
              })()}
              <Section label="Why" testid="help-modal-why">
                {entry.why}
              </Section>
              {entry.inputs && (
                <Section label="Inputs" testid="help-modal-inputs">
                  {entry.inputs}
                </Section>
              )}
              {entry.outputs && (
                <Section label="Outputs" testid="help-modal-outputs">
                  {entry.outputs}
                </Section>
              )}
              {entry.normalization && (
                <Section label="Normalization"
                         testid="help-modal-normalization">
                  {entry.normalization}
                </Section>
              )}
              {entry.example && (
                <Section label="Example" testid="help-modal-example">
                  <code
                    style={{
                      fontFamily: "ui-monospace, monospace",
                      fontSize: "12px",
                      background: "rgba(15, 23, 42, 0.6)",
                      color: "#a7f3d0", // Vibrant light green/mint
                      padding: "4px 8px",
                      borderRadius: "6px",
                      border: "1px solid rgba(255, 255, 255, 0.05)",
                      display: "block",
                      wordBreak: "break-all",
                    }}
                  >
                    {entry.example}
                  </code>
                </Section>
              )}
              {entry.reference && (
                <Section label="Reference" testid="help-modal-reference">
                  <a
                    href={entry.reference}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: "#38bdf8",
                      textDecoration: "underline",
                      fontSize: "13px",
                      transition: "color 0.15s ease",
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.color = "#7dd3fc";
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.color = "#38bdf8";
                    }}
                  >
                    {entry.reference}
                  </a>
                </Section>
              )}
            </>
          ) : (
            <p
              data-testid="help-modal-missing"
              style={{ color: "#94a3b8", margin: 0, fontSize: "13px" }}
            >
              (No explanation for <code>{topic}</code> yet.)
            </p>
          )}
        </div>

        <footer
          style={{
            display: "flex",
            justifyContent: "flex-end",
            borderTop: "1px solid rgba(255, 255, 255, 0.1)",
            paddingTop: "16px",
            marginTop: "8px",
          }}
        >
          <button
            data-testid="help-modal-got-it"
            onClick={onClose}
            style={{
              background: "#0891b2", // Cyan base
              color: "white",
              border: "none",
              borderRadius: "8px",
              padding: "10px 24px",
              fontSize: "14px",
              fontWeight: "bold",
              cursor: "pointer",
              transition: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)",
              boxShadow: "0 4px 12px rgba(8, 145, 178, 0.2)",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "#06b6d4";
              e.currentTarget.style.transform = "translateY(-1px)";
              e.currentTarget.style.boxShadow = "0 6px 16px rgba(34, 211, 238, 0.4)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "#0891b2";
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 4px 12px rgba(8, 145, 178, 0.2)";
            }}
          >
            Got It
          </button>
        </footer>
      </div>
    </div>
  );

  // SSR / test envs without a document fall back to inline rendering.
  if (typeof document === "undefined") return modal;
  return createPortal(modal, document.body);
}

function Section({
  label,
  testid,
  children,
}: {
  label: string;
  testid: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <div
        style={{
          color: "#38bdf8", // Premium light blue for section tags
          fontSize: "10px",
          fontWeight: "bold",
          textTransform: "uppercase",
          letterSpacing: "0.8px",
        }}
      >
        {label}
      </div>
      <div
        data-testid={testid}
        style={{
          color: "#f1f5f9", // Bright white for premium readability
          fontSize: "13px",
          lineHeight: "1.6",
        }}
      >
        {children}
      </div>
    </section>
  );
}
