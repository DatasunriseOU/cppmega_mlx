// General-purpose "?" icon + explanation modal for any in-app value.
// Topic content is centralised in `HELP_TOPICS` so writing the help is
// not gated on touching the component being explained.

import { useState } from "react";

export interface HelpTopic {
  title: string;
  what: string;
  why: string;
  example?: string;
  reference?: string;
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
};

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
          borderRadius: "50%", background: "#e5e7eb",
          color: "#374151", border: "none",
          fontSize: Math.max(9, size - 4), lineHeight: 1,
          cursor: "pointer", fontWeight: 700, marginLeft: 4,
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
  return (
    <div data-testid="help-modal-backdrop"
         role="dialog" aria-modal="true"
         onClick={onClose}
         style={{
           position: "fixed", inset: 0,
           background: "rgba(15,23,42,0.45)",
           display: "flex", alignItems: "center", justifyContent: "center",
           zIndex: 150, fontFamily: "system-ui, sans-serif",
         }}>
      <div data-testid={`help-modal-${topic}`}
           onClick={(e) => e.stopPropagation()}
           style={{
             background: "white", borderRadius: 6, padding: 20,
             width: 520, maxHeight: "80vh", overflowY: "auto",
             boxShadow: "0 10px 30px rgba(15,23,42,0.2)",
           }}>
        <header style={{ display: "flex", justifyContent: "space-between",
                          alignItems: "center", marginBottom: 12 }}>
          <h3 data-testid="help-modal-title"
              style={{ margin: 0, fontSize: 15 }}>
            {entry?.title ?? topic}
          </h3>
          <button data-testid="help-modal-close" onClick={onClose}
                  style={{ background: "none", border: "none",
                           cursor: "pointer", fontSize: 18 }}>
            ×
          </button>
        </header>
        {entry ? (
          <>
            <Section label="What" testid="help-modal-what">
              {entry.what}
            </Section>
            <Section label="Why" testid="help-modal-why">
              {entry.why}
            </Section>
            {entry.example && (
              <Section label="Example" testid="help-modal-example">
                <code style={{ fontFamily: "ui-monospace, monospace",
                               fontSize: 12,
                               background: "#f3f4f6", padding: "2px 4px",
                               borderRadius: 2 }}>
                  {entry.example}
                </code>
              </Section>
            )}
            {entry.reference && (
              <Section label="Reference" testid="help-modal-reference">
                <a href={entry.reference} target="_blank"
                   rel="noopener noreferrer"
                   style={{ color: "#2563eb" }}>
                  {entry.reference}
                </a>
              </Section>
            )}
          </>
        ) : (
          <p data-testid="help-modal-missing"
             style={{ color: "#6b7280", margin: 0 }}>
            (No explanation for <code>{topic}</code> yet.)
          </p>
        )}
      </div>
    </div>
  );
}

function Section({
  label, testid, children,
}: { label: string; testid: string; children: React.ReactNode }): JSX.Element {
  return (
    <section style={{ marginBottom: 10 }}>
      <div style={{ color: "#6b7280", fontSize: 11,
                    textTransform: "uppercase", letterSpacing: 0.5,
                    marginBottom: 2 }}>
        {label}
      </div>
      <div data-testid={testid} style={{ color: "#111827", fontSize: 13 }}>
        {children}
      </div>
    </section>
  );
}
