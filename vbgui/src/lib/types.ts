// Wire-format types matching cppmega_v4.jsonrpc.schema (Pydantic side).
// Keep in sync with SCHEMA_VERSION 1.1.0. When the backend bumps the
// version, regen this file (codegen lands in F-D).

export type JsonRpcVersion = "2.0";

export interface JsonRpcRequest<P = Record<string, unknown>> {
  jsonrpc: JsonRpcVersion;
  id: string | number;
  method: string;
  params?: P;
}

export interface JsonRpcError {
  code: number;
  message: string;
  data?: Record<string, unknown>;
}

export interface JsonRpcResponse<R = unknown> {
  jsonrpc: JsonRpcVersion;
  id: string | number | null;
  result?: R;
  error?: JsonRpcError;
}

export const SCHEMA_VERSION = "1.1.0";

// ---------------------------------------------------------------------------
// Domain payloads
// ---------------------------------------------------------------------------

export interface NodeSpec {
  id: string;
  kind: string;
  params?: Record<string, unknown>;
}

export interface EdgeSpec {
  src: string;
  dst: string;
}

export interface GraphSpec {
  nodes: NodeSpec[];
  edges: EdgeSpec[];
}

export type Severity = "info" | "warning" | "error";

export type SideChannelMode = "off" | "auto" | "require" | "if_available";
export type SideChannelEmbedding = "categorical" | "numeric_bucket" | "span"
                                 | "edge_bias" | "none";
export type SideChannelFallback = "zeros" | "unknown_id" | "drop_family"
                                | "error";
export type InferenceEnrichmentSource = "none" | "prompt_only"
                                      | "parse_if_possible" | "project_index"
                                      | "auto";
export type InferenceFailPolicy = "drop_family" | "text_only" | "error";

export interface FamilySpecPayload {
  mode: SideChannelMode;
  columns: string[];
  embedding: SideChannelEmbedding;
  dropout: number;
  residual_scale: number;
  fallback: SideChannelFallback;
  language_scope: string[];
}

export interface InferenceEnrichmentSpecPayload {
  source: InferenceEnrichmentSource;
  fail_policy: InferenceFailPolicy;
  timeout_ms: number;
  cache_enabled: boolean;
}

export interface SideChannelSpecPayload {
  mode: SideChannelMode;
  families: Record<string, FamilySpecPayload>;
  inference: InferenceEnrichmentSpecPayload;
}

export interface EdgeResolution {
  src: string;
  dst: string;
  shape: number[];
  matched: boolean;
  severity: Severity;
}

export interface PerBrickMemory {
  params_bytes: number;
  activations_bytes: number;
  kv_cache_bytes: number;
}

export interface VerifyResult {
  resolved: {
    edges: EdgeResolution[];
    diagnostics: { severity: Severity; component: string; message: string;
                   suggested_fix?: string | null }[];
    has_errors: boolean;
  };
  memory_per_brick: Record<string, PerBrickMemory>;
  fusion_plan: { brick_names: string[]; backend: string;
                 is_fused: boolean; estimated_savings_us: number }[];
  elapsed_ms: number;
}
