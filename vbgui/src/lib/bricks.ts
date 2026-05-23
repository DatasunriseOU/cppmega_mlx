// Brick + adapter metadata for the React Flow palette and node renderers.
// Categories mirror cppmega_v4.fusion.compatibility._CATEGORY_BY_KIND.

export type BrickCategory =
  | "sdpa_attention"
  | "linear_attn"
  | "ssm"
  | "moe"
  | "sparse_attn"
  | "cross_attn"
  | "norm_or_proj"
  | "nonlinear_rnn"
  | "io";

export interface BrickMeta {
  kind: string;
  label: string;
  category: BrickCategory;
}

export const CATEGORY_COLORS: Record<BrickCategory, string> = {
  sdpa_attention: "#3b82f6",   // blue
  linear_attn:    "#10b981",   // green
  ssm:            "#a855f7",   // purple
  moe:            "#f59e0b",   // amber
  sparse_attn:    "#06b6d4",   // cyan
  cross_attn:     "#ec4899",   // pink
  norm_or_proj:   "#6b7280",   // gray
  nonlinear_rnn:  "#ef4444",   // red
  io:             "#6366f1",   // indigo
};

// 25 bricks (matches BLOCK_BUILDERS post GalCov Stage B).
export const BRICKS: readonly BrickMeta[] = [
  // sdpa_attention
  { kind: "attention",       label: "Attention (vanilla)", category: "sdpa_attention" },
  { kind: "gated_attention", label: "Gated Attention",     category: "sdpa_attention" },
  { kind: "mla",             label: "MLA",                 category: "sdpa_attention" },
  { kind: "mla_absorb",      label: "MLA (absorb)",        category: "sdpa_attention" },
  { kind: "mistral4_mla",    label: "Mistral4 MLA",        category: "sdpa_attention" },
  { kind: "dsv4_attention",  label: "DSv4 Attention",      category: "sdpa_attention" },
  { kind: "bailing_mla",     label: "Bailing MLA",         category: "sdpa_attention" },
  { kind: "gqa_sliding",     label: "GQA + Sliding",       category: "sdpa_attention" },
  { kind: "cca_attention",   label: "CCA Attention",       category: "sdpa_attention" },
  { kind: "gemma4_drafter",  label: "Gemma4 Drafter",      category: "sdpa_attention" },
  { kind: "nemotron_h_mtp",  label: "Nemotron-H MTP",      category: "sdpa_attention" },
  // linear_attn
  { kind: "bailing_linear",  label: "Bailing Linear",      category: "linear_attn" },
  { kind: "gdn",             label: "GDN",                 category: "linear_attn" },
  { kind: "kda",             label: "KDA",                 category: "linear_attn" },
  // ssm
  { kind: "mamba3",          label: "Mamba 3",             category: "ssm" },
  // moe
  { kind: "moe",             label: "MoE",                 category: "moe" },
  { kind: "bailing_moe",     label: "Bailing MoE",         category: "moe" },
  // sparse_attn
  { kind: "nsa",             label: "NSA",                 category: "sparse_attn" },
  { kind: "lightning_indexer", label: "Lightning Indexer", category: "sparse_attn" },
  // cross_attn
  { kind: "csa_hca",         label: "CSA / HCA Hybrid",    category: "cross_attn" },
  { kind: "engram",          label: "Engram",              category: "cross_attn" },
  // norm_or_proj
  { kind: "mlp",             label: "MLP",                 category: "norm_or_proj" },
  { kind: "abs_pos_embed",   label: "Abs Pos Embed",       category: "norm_or_proj" },
  { kind: "per_layer_embed", label: "Per-Layer Embed",     category: "norm_or_proj" },
  // nonlinear_rnn
  { kind: "mlstm",           label: "mLSTM",               category: "nonlinear_rnn" },
  // io
  { kind: "tokenizer",       label: "Tokenizer",           category: "io" },
  { kind: "detokenizer",     label: "De-Tokenizer",         category: "io" },
];

export interface AdapterMeta {
  kind: string;
  label: string;
}

// 6 adapter nodes — dashed border, ghost preview.
export const ADAPTERS: readonly AdapterMeta[] = [
  { kind: "merge_heads",    label: "Merge Heads" },
  { kind: "split_heads",    label: "Split Heads" },
  { kind: "transpose_bnsd", label: "Transpose BNSD" },
  { kind: "linear_bridge",  label: "Linear Bridge" },
  { kind: "rmsnorm",        label: "RMSNorm" },
  { kind: "residual",       label: "Residual Add" },
];

export function brickFor(kind: string): BrickMeta | undefined {
  return BRICKS.find((b) => b.kind === kind);
}

export function adapterFor(kind: string): AdapterMeta | undefined {
  return ADAPTERS.find((a) => a.kind === kind);
}

export function colorFor(category: BrickCategory): string {
  return CATEGORY_COLORS[category];
}
