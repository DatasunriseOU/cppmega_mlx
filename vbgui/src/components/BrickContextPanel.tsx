// BrickContextPanel — per-brick parameter editor overlay (E7-5 / E7-6).
//
// Opens when the user clicks a node in FlowCanvas. Shows an Activation
// dropdown (for mlp/gated_mlp/moe), pre_norm/post_norm dropdowns
// (E7-6), and a free-form params JSON pane.

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { Tooltip } from "@/components/Tooltip";
import { ExplainModal } from "@/components/ExplainModal";
import { BRICKS, brickFor } from "@/lib/bricks";
import { computeBrickDims, fmtParamCount } from "@/lib/brickDims";
import { PathExplorer } from "./PathExplorer";

const ACTIVATION_OPTIONS = [
  "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
  "swiglu", "geglu", "reglu", "xielu",
];
const NORM_OPTIONS = ["rmsnorm", "layernorm", "none"];

function HistogramSvg({ counts }: { counts: number[] }): JSX.Element {
  const W = 220, H = 60;
  const maxC = Math.max(...counts, 1);
  const barW = W / Math.max(1, counts.length);
  return (
    <svg data-testid="brick-histogram-svg" width={W} height={H}
         style={{ marginTop: 4 }}>
      {counts.map((c, i) => {
        const h = (c / maxC) * (H - 2);
        return (
          <rect key={i}
                data-testid={`brick-histogram-bar-${i}`}
                x={i * barW} y={H - h}
                width={Math.max(1, barW - 1)} height={h}
                fill="#8b5cf6">
            <title>{`bin ${i}: ${c}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

const SUPPORTS_ACTIVATION = new Set([
  "mlp", "gated_mlp", "moe", "bailing_moe",
]);
const SUPPORTS_NORM = new Set([
  "attention", "gated_attention", "mla", "mla_absorb", "mistral4_mla",
  "dsv4_attention", "gqa_sliding", "cca_attention", "gdn", "kda",
  "mlp", "gated_mlp", "moe",
]);

const KINDS_WITHOUT_WEIGHTS = new Set([
  "residual",
  "adapter_residual",
  "merge_heads",
  "adapter_merge_heads",
  "split_heads",
  "adapter_split_heads",
  "transpose_bnsd",
  "adapter_transpose_bnsd",
]);

export interface BrickContextPanelProps {
  rpc: RpcClient | null;
  brickId: string;
  /** Current dim_env.H — used to compute approximate input/output
   *  shapes and parameter count for this brick. */
  hidden_size?: number;
  /** Tokenizer vocab size — used for the embedding_table brick. */
  vocab_size?: number;
  brickKind: string;
  params: Record<string, unknown>;
  onApply: (newParams: Record<string, unknown>) => void;
  // V7-F52 — live block swap. Fires when the user picks a different
  // same-category brick kind for this node and clicks Swap. The host
  // (App.tsx) preserves the node id + edges and just mutates kind.
  onSwapKind?: (newKind: string) => void;
  onClose: () => void;
  /** V7-H08: callback to fetch the weight histogram for this brick.
   *  Host (App.tsx) builds the full spec and calls inspect.histogram. */
  onInspectHistogram?: (brickId: string) => Promise<HistogramResult>;
  onDelete?: () => void; // Support deleting the brick
}

export interface HistogramResult {
  brick_id: string;
  buckets: number;
  bins: number[];
  counts: number[];
  min: number;
  max: number;
  mean: number;
  n_values: number;
}

const FIELD: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 6, fontSize: 12,
  marginBottom: 12,
};

export function BrickContextPanel({
  rpc, brickId, brickKind, params, hidden_size = 128,
  vocab_size = 65536, onApply, onSwapKind, onClose,
  onInspectHistogram, onDelete,
}: BrickContextPanelProps): JSX.Element {
  const [draft, setDraft] = useState<Record<string, unknown>>(params);
  const [swapTarget, setSwapTarget] = useState<string>(brickKind);
  const [explain, setExplain] = useState<{ cat: string; name: string }
                                          | null>(null);
  const [hist, setHist] = useState<HistogramResult | null>(null);
  const [histLoading, setHistLoading] = useState<boolean>(false);
  const [histError, setHistError] = useState<string | null>(null);

  useEffect(() => { setDraft(params); setSwapTarget(brickKind); },
            [params, brickId, brickKind]);

  function setField(field: string, value: unknown) {
    setDraft({ ...draft, [field]: value });
  }

  const supportsAct = SUPPORTS_ACTIVATION.has(brickKind);
  const supportsNorm = SUPPORTS_NORM.has(brickKind);
  const activation = (draft.activation as string | undefined) ?? "glu";
  const preNorm = (draft.pre_norm as string | undefined) ?? "rmsnorm";
  const postNorm = (draft.post_norm as string | undefined) ?? "none";

  return (
    <div data-testid={`brick-context-${brickId}`}
         style={{
           position: "absolute", top: 60, right: 8,
           width: 320, maxHeight: "calc(100vh - 120px)",
           background: "rgba(30, 41, 59, 0.95)",
           backdropFilter: "blur(16px)",
           border: "1px solid rgba(255, 255, 255, 0.1)",
           borderRadius: 12, padding: 16, overflowY: "auto",
           boxShadow: "0 20px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05)",
           fontFamily: "system-ui, sans-serif", zIndex: 50,
           color: "#f1f5f9",
           display: "flex",
           flexDirection: "column",
           gap: "14px",
         }}>
      <header style={{ display: "flex", justifyContent: "space-between",
                        alignItems: "center", borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
                        paddingBottom: 10 }}>
        <h4 style={{ margin: 0, fontSize: 13, fontWeight: "bold", color: "#22d3ee" }}>
          {brickId}
          <span style={{ marginLeft: 6, color: "#94a3b8",
                          fontSize: 10 }}>
            [{brickKind}]
          </span>
        </h4>
        <button
          data-testid="brick-context-close"
          onClick={onClose}
          style={{
            background: "rgba(255, 255, 255, 0.05)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "50%",
            width: "28px",
            height: "28px",
            cursor: "pointer",
            fontSize: "18px",
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

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {brickKind === "tokenizer" && (
          <div style={{ marginBottom: 10 }}>
            <PathExplorer
              rpc={rpc}
              initialPath={((draft.selected_path as string) || ".")}
              onSelect={(path, type) => {
                setDraft((d) => ({ ...d, selected_path: path, content_type: type }));
              }}
            />
          </div>
        )}

        {supportsAct && (
          <label style={FIELD}>
            <Tooltip rpc={rpc} category="activation" name={activation}
                     onInfoClick={() =>
                       setExplain({ cat: "activation", name: activation })}>
              <span style={{ color: "#94a3b8", fontWeight: 500 }}>Activation</span>
            </Tooltip>
            <select data-testid={`brick-context-${brickId}-activation`}
                    value={activation}
                    onChange={(e) => setField("activation", e.target.value)}
                    style={{
                      background: "rgba(15, 23, 42, 0.6)",
                      color: "#f1f5f9",
                      border: "1px solid rgba(255, 255, 255, 0.15)",
                      borderRadius: "6px",
                      padding: "6px 10px",
                      outline: "none",
                      fontSize: "13px",
                      cursor: "pointer",
                      width: "100%",
                    }}>
              {ACTIVATION_OPTIONS.map((a) =>
                <option key={a} value={a}>{a}</option>)}
            </select>
          </label>
        )}

        {supportsNorm && (
          <>
            <label style={FIELD}>
              <Tooltip rpc={rpc} category="norm" name={preNorm}
                       onInfoClick={() =>
                         setExplain({ cat: "norm", name: preNorm })}>
                <span style={{ color: "#94a3b8", fontWeight: 500 }}>pre_norm</span>
              </Tooltip>
              <select data-testid={`brick-context-${brickId}-pre-norm`}
                      value={preNorm}
                      onChange={(e) => setField("pre_norm", e.target.value)}
                      style={{
                        background: "rgba(15, 23, 42, 0.6)",
                        color: "#f1f5f9",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: "6px",
                        padding: "6px 10px",
                        outline: "none",
                        fontSize: "13px",
                        cursor: "pointer",
                        width: "100%",
                      }}>
                {NORM_OPTIONS.map((n) =>
                  <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label style={FIELD}>
              <Tooltip rpc={rpc} category="norm" name={postNorm}
                       onInfoClick={() =>
                         setExplain({ cat: "norm", name: postNorm })}>
                <span style={{ color: "#94a3b8", fontWeight: 500 }}>post_norm</span>
              </Tooltip>
              <select data-testid={`brick-context-${brickId}-post-norm`}
                      value={postNorm}
                      onChange={(e) => setField("post_norm", e.target.value)}
                      style={{
                        background: "rgba(15, 23, 42, 0.6)",
                        color: "#f1f5f9",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: "6px",
                        padding: "6px 10px",
                        outline: "none",
                        fontSize: "13px",
                        cursor: "pointer",
                        width: "100%",
                      }}>
                {NORM_OPTIONS.map((n) =>
                  <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          </>
        )}

        {brickKind === "attention" && (
          <fieldset style={{
            border: "1px solid rgba(34, 211, 238, 0.15)",
            borderRadius: 8,
            padding: 12,
            margin: "4px 0",
            background: "rgba(0,0,0,0.15)"
          }}>
            <legend style={{ fontSize: 10, fontWeight: "bold", color: "#22d3ee", padding: "0 6px" }}>
              ⚡ KV Cache & Compression Settings
            </legend>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 4 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11 }}>
                <span style={{ color: "#cbd5e1" }}>KV Cache Sharing</span>
                <select
                  data-testid="kv-cache-sharing-select"
                  value={(draft.kv_sharing as string) || "none"}
                  onChange={(e) => setField("kv_sharing", e.target.value)}
                  style={{
                    background: "rgba(15, 23, 42, 0.6)",
                    color: "#f1f5f9",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: 6,
                    padding: "4px 8px",
                    outline: "none",
                    cursor: "pointer",
                    fontSize: 12
                  }}
                >
                  <option value="none">None (Standard)</option>
                  <option value="grouped">Grouped-Query (GQA)</option>
                  <option value="cross_layer">Cross-Layer (CLA)</option>
                </select>
              </label>

              <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11 }}>
                <span style={{ color: "#cbd5e1" }}>Cache Compression</span>
                <select
                  data-testid="cache-compression-select"
                  value={(draft.cache_compression as string) || "none"}
                  onChange={(e) => setField("cache_compression", e.target.value)}
                  style={{
                    background: "rgba(15, 23, 42, 0.6)",
                    color: "#f1f5f9",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: 6,
                    padding: "4px 8px",
                    outline: "none",
                    cursor: "pointer",
                    fontSize: 12
                  }}
                >
                  <option value="none">None</option>
                  <option value="int4">INT4</option>
                  <option value="int8">INT8</option>
                  <option value="fp8">FP8</option>
                  <option value="turbo_3bit">TurboQuant 3-bit</option>
                  <option value="turbo_4bit">TurboQuant 4-bit</option>
                </select>
              </label>

              {draft.kv_sharing === "cross_layer" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11 }}>
                    <span style={{ color: "#cbd5e1" }}>KV Producer Layer Index</span>
                    <input
                      data-testid="kv-producer-layer-input"
                      type="number"
                      min={0}
                      value={(draft.kv_producer_layer as number) ?? 0}
                      onChange={(e) => setField("kv_producer_layer", parseInt(e.target.value) || 0)}
                      style={{
                        background: "rgba(15, 23, 42, 0.6)",
                        color: "#f1f5f9",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: 6,
                        padding: "4px 8px",
                        outline: "none",
                        fontSize: 12
                      }}
                    />
                  </label>

                  <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11 }}>
                    <span style={{ color: "#cbd5e1" }}>KV LoRA Compression Rank</span>
                    <input
                      data-testid="kv-lora-rank-input"
                      type="number"
                      min={8}
                      max={512}
                      value={(draft.kv_lora_rank as number) || 128}
                      onChange={(e) => setField("kv_lora_rank", parseInt(e.target.value) || 128)}
                      style={{
                        background: "rgba(15, 23, 42, 0.6)",
                        color: "#f1f5f9",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: 6,
                        padding: "4px 8px",
                        outline: "none",
                        fontSize: 12
                      }}
                    />
                  </label>
                </div>
              )}
            </div>
          </fieldset>
        )}

        {/* Computed dimensions & parameter count */}
        {(() => {
          const dims = computeBrickDims(
            brickKind, hidden_size, draft, vocab_size);
          return (
            <div data-testid={`brick-context-${brickId}-dims`}
                 style={{ marginTop: 8, padding: "8px 10px",
                          background: "rgba(34, 211, 238, 0.06)",
                          border: "1px solid rgba(34, 211, 238, 0.2)",
                          borderRadius: 6, fontSize: 11 }}>
              <div style={{ color: "#22d3ee", fontWeight: 700,
                            marginBottom: 4 }}>
                📐 Dimensions & params (approx)
              </div>
              <div style={{ fontFamily: "ui-monospace, monospace",
                            color: "#e8e8e8", lineHeight: 1.7 }}>
                <div>
                  <span style={{ color: "#94a3b8" }}>input: </span>
                  <span data-testid={
                    `brick-context-${brickId}-dim-input`}>
                    {dims.input}
                  </span>
                </div>
                <div>
                  <span style={{ color: "#94a3b8" }}>output: </span>
                  <span data-testid={
                    `brick-context-${brickId}-dim-output`}>
                    {dims.output}
                  </span>
                </div>
                <div>
                  <span style={{ color: "#94a3b8" }}>H (hidden): </span>
                  <span style={{ color: "#f5b841" }}>{hidden_size}</span>
                </div>
                <div>
                  <span style={{ color: "#94a3b8" }}>params: </span>
                  <span data-testid={
                    `brick-context-${brickId}-dim-params`}
                        style={{ color: "#7bc47f", fontWeight: 700 }}>
                    {fmtParamCount(dims.n_params)}
                  </span>
                </div>
              </div>
              <div style={{ color: "#94a3b8", fontSize: 10,
                            marginTop: 4, fontStyle: "italic" }}>
                {dims.formula}
              </div>
            </div>
          );
        })()}

        {/* Dynamic block parameters editor */}
        {(() => {
          const keys = Object.keys(draft).filter(k => !["activation", "pre_norm", "post_norm"].includes(k));
          if (keys.length === 0) return null;
          return (
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: 8 }}>
              <h5 style={{ margin: "10px 0 5px", fontSize: 11, fontWeight: "bold", color: "#22d3ee", borderBottom: "1px solid rgba(255, 255, 255, 0.1)", paddingBottom: 4 }}>
                🧱 Block Parameters
              </h5>
              {keys.map((key) => {
                const value = draft[key];
                const type = typeof value;
                return (
                  <label key={key} style={FIELD}>
                    <span style={{ color: "#94a3b8", fontWeight: 500, fontFamily: "monospace", fontSize: 11 }}>{key}</span>
                    {type === "number" ? (
                      <input
                        type="number"
                        value={value as number}
                        onChange={(e) => setField(key, parseFloat(e.target.value) || 0)}
                        style={{
                          background: "rgba(15, 23, 42, 0.6)",
                          color: "#f1f5f9",
                          border: "1px solid rgba(255, 255, 255, 0.15)",
                          borderRadius: "6px",
                          padding: "6px 10px",
                          outline: "none",
                          fontSize: "13px",
                          width: "100%",
                        }}
                      />
                    ) : type === "boolean" ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                        <input
                          type="checkbox"
                          checked={value as boolean}
                          onChange={(e) => setField(key, e.target.checked)}
                          style={{ cursor: "pointer" }}
                        />
                        <span style={{ color: "#cbd5e1", fontSize: 12 }}>Enabled</span>
                      </div>
                    ) : (
                      <input
                        type="text"
                        value={type === "object" ? JSON.stringify(value) : String(value)}
                        onChange={(e) => {
                          let parsed: any = e.target.value;
                          if (type === "object") {
                            try { parsed = JSON.parse(e.target.value); } catch {}
                          }
                          setField(key, parsed);
                        }}
                        style={{
                          background: "rgba(15, 23, 42, 0.6)",
                          color: "#f1f5f9",
                          border: "1px solid rgba(255, 255, 255, 0.15)",
                          borderRadius: "6px",
                          padding: "6px 10px",
                          outline: "none",
                          fontSize: "13px",
                          width: "100%",
                        }}
                      />
                    )}
                  </label>
                );
              })}
            </div>
          );
        })()}

        {onSwapKind && (() => {
          const currentMeta = brickFor(brickKind);
          const sameCategory = currentMeta
            ? BRICKS.filter((b) => b.category === currentMeta.category)
            : [];
          if (sameCategory.length <= 1) return null;
          return (
            <label style={FIELD}>
              <span style={{ color: "#94a3b8", fontWeight: 500, marginBottom: 4 }}>
                Swap to (same category)
              </span>
              <select
                data-testid={`brick-context-${brickId}-swap-target`}
                value={swapTarget}
                onChange={(e) => setSwapTarget(e.target.value)}
                style={{
                  background: "rgba(15, 23, 42, 0.6)",
                  color: "#f1f5f9",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  borderRadius: "6px",
                  padding: "6px 10px",
                  outline: "none",
                  fontSize: "13px",
                  cursor: "pointer",
                  width: "100%",
                  marginBottom: 6,
                }}>
                {sameCategory.map((b) => (
                  <option key={b.kind} value={b.kind}>{b.label}</option>
                ))}
              </select>
              <button
                data-testid={`brick-context-${brickId}-swap-apply`}
                disabled={swapTarget === brickKind}
                onClick={() => { onSwapKind(swapTarget); onClose(); }}
                style={{
                  background: swapTarget === brickKind ? "rgba(255, 255, 255, 0.05)" : "#0ea5e9",
                  color: swapTarget === brickKind ? "#64748b" : "white",
                  border: swapTarget === brickKind ? "1px solid rgba(255, 255, 255, 0.08)" : "none",
                  padding: "6px 12px",
                  borderRadius: 6,
                  fontSize: 12,
                  fontWeight: "bold",
                  cursor: swapTarget === brickKind ? "default" : "pointer",
                  transition: "all 0.15s ease",
                  width: "100%",
                }}>
                Swap kind
              </button>
            </label>
          );
        })()}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 4 }}>
        <button
          data-testid={`brick-context-${brickId}-apply`}
          onClick={() => { onApply(draft); onClose(); }}
          style={{
            background: "#0891b2",
            color: "white",
            border: "none",
            padding: "8px 14px",
            borderRadius: 6,
            cursor: "pointer",
            fontSize: 12,
            fontWeight: "bold",
            transition: "all 0.15s ease",
            width: "100%",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = "#06b6d4";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = "#0891b2";
          }}
        >
          Apply
        </button>

        {onInspectHistogram && (
          <div data-testid={`brick-context-${brickId}-histogram-block`}
               style={{ fontSize: 11 }}>
            {KINDS_WITHOUT_WEIGHTS.has(brickKind) ? (
              <button
                data-testid={`brick-context-${brickId}-histogram-disabled`}
                disabled={true}
                style={{
                  background: "rgba(255, 255, 255, 0.05)",
                  color: "#64748b",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  padding: "8px 14px",
                  borderRadius: 6,
                  fontSize: 11,
                  fontWeight: "bold",
                  width: "100%",
                  cursor: "not-allowed",
                  transition: "all 0.15s ease",
                }}
              >
                No trainable weights
              </button>
            ) : (
              <button
                data-testid={`brick-context-${brickId}-histogram-fetch`}
                disabled={histLoading}
                onClick={async () => {
                  setHistLoading(true); setHistError(null);
                  try {
                    const r = await onInspectHistogram(brickId);
                    setHist(r);
                  } catch (e) {
                    setHistError(e instanceof Error ? e.message : String(e));
                  } finally {
                    setHistLoading(false);
                  }
                }}
                style={{
                  background: "#7c3aed",
                  color: "white",
                  border: "none",
                  padding: "8px 14px",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 11,
                  fontWeight: "bold",
                  transition: "all 0.15s ease",
                  width: "100%",
                }}
              >
                {histLoading ? "Loading…" : "Inspect weight histogram"}
              </button>
            )}
            {histError && (
              <div data-testid={`brick-context-${brickId}-histogram-error`}
                   style={{ color: "#ef4444", marginTop: 4 }}>
                {histError}
              </div>
            )}
            {hist && (
              <div data-testid={`brick-context-${brickId}-histogram-result`}
                   style={{
                     marginTop: 6,
                     fontFamily: "monospace",
                     background: "rgba(15, 23, 42, 0.5)",
                     border: "1px solid rgba(255, 255, 255, 0.08)",
                     padding: 8,
                     borderRadius: 6,
                     fontSize: 10,
                   }}>
                <div data-testid={`brick-context-${brickId}-histogram-stats`} style={{ color: "#94a3b8", lineHeight: 1.4 }}>
                  n={hist.n_values} <br />
                  min={hist.min.toExponential(2)} <br />
                  max={hist.max.toExponential(2)} <br />
                  mean={hist.mean.toExponential(2)}
                </div>
                <HistogramSvg counts={hist.counts} />
              </div>
            )}
          </div>
        )}

        {onDelete && (
          <button
            data-testid="brick-context-delete"
            onClick={onDelete}
            style={{
              background: "rgba(239, 68, 68, 0.15)",
              border: "1px solid rgba(239, 68, 68, 0.3)",
              color: "#f87171",
              borderRadius: "6px",
              padding: "8px 14px",
              fontSize: "12px",
              fontWeight: "bold",
              cursor: "pointer",
              transition: "all 0.15s ease",
              width: "100%",
              marginTop: 4,
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(239, 68, 68, 0.3)";
              e.currentTarget.style.borderColor = "rgba(239, 68, 68, 0.5)";
              e.currentTarget.style.color = "#fca5a5";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(239, 68, 68, 0.15)";
              e.currentTarget.style.borderColor = "rgba(239, 68, 68, 0.3)";
              e.currentTarget.style.color = "#f87171";
            }}
          >
            🗑 Delete Brick
          </button>
        )}
      </div>

      {explain && (
        <ExplainModal rpc={rpc} category={explain.cat} name={explain.name}
                      onClose={() => setExplain(null)} />
      )}
    </div>
  );
}
