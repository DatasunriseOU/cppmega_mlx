// Single source of truth for the in-canvas spec (LossSpec, OptimSpec,
// ShardingSpec, rewriter chain). The frontend mutates this via
// reducer actions; F-D wires it to anywidget traitlets.

import type { Severity } from "@/lib/types";

export type LossKind = "cross_entropy" | "mtp_weighted" | "ifim_shaped"
                     | "mhc_attn_bias" | "custom";

export interface LossState {
  kind: LossKind;
  head_outputs: string[];
  params: Record<string, number | string>;
}

export type OptimKind = "adamw" | "muon" | "muon_adamw_hybrid"
                       | "lion" | "lion8bit" | "adam8bit" | "sgd";

export type ScheduleKind = "constant" | "linear_warmup" | "cosine"
                          | "wsd" | "inv_sqrt" | "polynomial";

export interface ScheduleSpecState {
  kind: ScheduleKind;
  warmup_steps?: number;
  total_steps?: number;
  min_lr_ratio?: number;
  decay_steps?: number;
  power?: number;
}

export interface ParamGroupState {
  matcher: string;
  lr: number;
  weight_decay: number;
  betas?: [number, number];
  ns_steps?: number | null;
  schedule?: ScheduleSpecState;
}

export interface OptimState {
  kind: OptimKind;
  groups: ParamGroupState[];
  grad_clip_norm: number;
  mixed_precision: boolean;
}

export type RewriterName = "MTPRewriter" | "IFIMRewriter" | "MHCRewriter";

export interface RewriterState {
  name: RewriterName;
  params: Record<string, number | string>;
}

export type SideChannelMode = "off" | "auto" | "require" | "if_available";
export type SideChannelEmbedding = "categorical" | "numeric_bucket" | "span"
                                 | "edge_bias" | "none";
export type SideChannelFallback = "zeros" | "unknown_id" | "drop_family"
                                | "error";
export type InferenceEnrichmentSource = "none" | "prompt_only"
                                      | "parse_if_possible" | "project_index"
                                      | "auto";
export type InferenceFailPolicy = "drop_family" | "text_only" | "error";

export interface SideChannelFamilyState {
  mode: SideChannelMode;
  columns: string[];
  embedding: SideChannelEmbedding;
  dropout: number;
  residual_scale: number;
  fallback: SideChannelFallback;
  language_scope: string[];
}

export interface InferenceEnrichmentState {
  source: InferenceEnrichmentSource;
  fail_policy: InferenceFailPolicy;
  timeout_ms: number;
  cache_enabled: boolean;
}

export interface SideChannelState {
  mode: SideChannelMode;
  families: Record<string, SideChannelFamilyState>;
  inference: InferenceEnrichmentState;
}

export type TopologyFactory =
  | "h100_8x" | "h200_8x" | "a100_8x" | "b100_8x"
  | "gb10_quarter" | "tpu_v6e_8" | "tpu_v5p_4" | "m3_ultra_solo";

export interface ShardingAxis {
  axis_name: string;
  kind: string;
  degree: number;
}

export interface ShardingState {
  topology: TopologyFactory;
  axis_assignments: ShardingAxis[];
  compile_mode: "off" | "regional" | "whole_model";
  fp8_enabled: boolean;
  master_weights_fp32: boolean;
  activation_checkpointing: boolean;
}

export interface GotchaState {
  id: string;
  severity: Severity;
  message: string;
  reference?: string;
}

export interface SpecState {
  loss: LossState;
  optim: OptimState;
  rewriters: RewriterState[];
  side_channels: SideChannelState;
  sharding: ShardingState;
  gotchas: GotchaState[];
  worst_rank_bytes: number;
  device_hbm_bytes: number;
  /** H11: actual Metal peak from extras.memory_peak_bytes (last Train).
   *  Renders the "actual" half of the dual MemoryBar — left unset
   *  until a Train run completes. */
  actual_peak_bytes?: number;
  last_verify_ms: number;
  brick_count: number;
  backend_status: "connected" | "reconnecting" | "disconnected";
}

export const INITIAL_SPEC: SpecState = {
  loss: { kind: "cross_entropy", head_outputs: ["logits"], params: {} },
  optim: {
    kind: "adamw",
    groups: [{ matcher: "all", lr: 3e-4, weight_decay: 0.01,
               betas: [0.9, 0.95] }],
    grad_clip_norm: 1.0,
    mixed_precision: true,
  },
  rewriters: [],
  side_channels: {
    mode: "auto",
    families: {
      platform: {
        mode: "auto",
        columns: ["platform_ids", "source_platform_ids"],
        embedding: "categorical",
        dropout: 0.10,
        residual_scale: 1.0,
        fallback: "drop_family",
        language_scope: ["any"],
      },
      syntax: {
        mode: "if_available",
        columns: [
          "token_ast_depth",
          "token_sibling_index",
          "token_ast_node_type",
        ],
        embedding: "categorical",
        dropout: 0.25,
        residual_scale: 1.0,
        fallback: "drop_family",
        language_scope: ["any"],
      },
      structure: {
        mode: "if_available",
        columns: [
          "token_structure_ids",
          "token_dep_levels",
          "token_chunk_starts",
          "token_chunk_ends",
          "token_chunk_kinds",
          "token_chunk_dep_levels",
        ],
        embedding: "categorical",
        dropout: 0.25,
        residual_scale: 1.0,
        fallback: "drop_family",
        language_scope: ["any"],
      },
      semantic_graph: {
        mode: "if_available",
        columns: [
          "token_symbol_ids",
          "token_call_targets",
          "token_type_refs",
          "token_def_use",
          "token_call_edges",
          "token_type_edges",
        ],
        embedding: "edge_bias",
        dropout: 0.50,
        residual_scale: 1.0,
        fallback: "drop_family",
        language_scope: ["any"],
      },
      temporal_diff: {
        mode: "off",
        columns: [
          "token_change_mask_pre",
          "token_change_mask_post",
          "hunk_id_per_token",
          "edit_op_per_token",
        ],
        embedding: "categorical",
        dropout: 0.0,
        residual_scale: 1.0,
        fallback: "drop_family",
        language_scope: ["any"],
      },
    },
    inference: {
      source: "auto",
      fail_policy: "drop_family",
      timeout_ms: 500,
      cache_enabled: true,
    },
  },
  sharding: {
    topology: "h100_8x",
    axis_assignments: [{ axis_name: "dp", kind: "fsdp2", degree: 8 }],
    compile_mode: "regional",
    fp8_enabled: false,
    master_weights_fp32: false,
    activation_checkpointing: false,
  },
  gotchas: [],
  worst_rank_bytes: 0,
  device_hbm_bytes: 80 * 1024 ** 3,
  last_verify_ms: 0,
  brick_count: 0,
  backend_status: "disconnected",
};

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export type SpecAction =
  | { type: "loss.set"; loss: LossState }
  | { type: "optim.set"; optim: OptimState }
  | { type: "optim.add_group"; group: ParamGroupState }
  | { type: "optim.remove_group"; index: number }
  | { type: "rewriters.add"; rewriter: RewriterState }
  | { type: "rewriters.remove"; index: number }
  | { type: "rewriters.reorder"; from: number; to: number }
  | { type: "side_channels.set"; side_channels: SideChannelState }
  | { type: "sharding.set"; sharding: ShardingState }
  | { type: "gotchas.set"; gotchas: GotchaState[] }
  | { type: "memory.set"; worst_rank_bytes: number; device_hbm_bytes?: number }
  | { type: "memory.actual_set"; actual_peak_bytes: number | undefined }
  | { type: "verify.complete"; elapsed_ms: number; brick_count: number }
  | { type: "backend.status"; status: SpecState["backend_status"] }
  | { type: "spec.replace"; spec: SpecState };

export function specReducer(s: SpecState, a: SpecAction): SpecState {
  switch (a.type) {
    case "loss.set":   return { ...s, loss: a.loss };
    case "optim.set":  return { ...s, optim: a.optim };
    case "optim.add_group":
      return { ...s, optim: { ...s.optim, groups: [...s.optim.groups, a.group] } };
    case "optim.remove_group":
      return { ...s, optim: {
        ...s.optim,
        groups: s.optim.groups.filter((_, i) => i !== a.index),
      } };
    case "rewriters.add":
      return { ...s, rewriters: [...s.rewriters, a.rewriter] };
    case "rewriters.remove":
      return { ...s, rewriters: s.rewriters.filter((_, i) => i !== a.index) };
    case "rewriters.reorder": {
      const out = [...s.rewriters];
      const [moved] = out.splice(a.from, 1);
      out.splice(a.to, 0, moved);
      return { ...s, rewriters: out };
    }
    case "side_channels.set": return { ...s, side_channels: a.side_channels };
    case "spec.replace": return a.spec;
    case "sharding.set": return { ...s, sharding: a.sharding };
    case "gotchas.set":  return { ...s, gotchas: a.gotchas };
    case "memory.set":   return {
      ...s,
      worst_rank_bytes: a.worst_rank_bytes,
      device_hbm_bytes: a.device_hbm_bytes ?? s.device_hbm_bytes,
    };
    case "memory.actual_set": return {
      ...s, actual_peak_bytes: a.actual_peak_bytes,
    };
    case "verify.complete": return {
      ...s, last_verify_ms: a.elapsed_ms, brick_count: a.brick_count,
    };
    case "backend.status":  return { ...s, backend_status: a.status };
  }
}

// ---------------------------------------------------------------------------
// Helpers consumed by both sidebar and top bar.
// ---------------------------------------------------------------------------

export function memoryFillRatio(s: SpecState): number {
  if (s.device_hbm_bytes <= 0) return 0;
  return s.worst_rank_bytes / s.device_hbm_bytes;
}

export function memoryColor(s: SpecState): "green" | "yellow" | "red" {
  const r = memoryFillRatio(s);
  if (r < 0.7) return "green";
  if (r < 0.9) return "yellow";
  return "red";
}
