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
