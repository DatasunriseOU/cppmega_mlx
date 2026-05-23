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
                fill="#7c3aed">
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

export interface BrickContextPanelProps {
  rpc: RpcClient | null;
  brickId: string;
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
  display: "flex", flexDirection: "column", gap: 4, fontSize: 12,
  marginBottom: 10,
};

export function BrickContextPanel({
  rpc, brickId, brickKind, params, onApply, onSwapKind, onClose,
  onInspectHistogram,
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
           width: 320, maxHeight: "calc(100vh - 200px)",
           background: "white", border: "1px solid #e5e7eb",
           borderRadius: 6, padding: 12, overflowY: "auto",
           boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
           fontFamily: "system-ui, sans-serif", zIndex: 50,
         }}>
      <header style={{ display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 10 }}>
        <h4 style={{ margin: 0, fontSize: 13 }}>
          {brickId}
          <span style={{ marginLeft: 6, color: "#6b7280",
                          fontSize: 10 }}>
            [{brickKind}]
          </span>
        </h4>
        <button data-testid="brick-context-close" onClick={onClose}>×</button>
      </header>

      {supportsAct && (
        <label style={FIELD}>
          <Tooltip rpc={rpc} category="activation" name={activation}
                   onInfoClick={() =>
                     setExplain({ cat: "activation", name: activation })}>
            <span style={{ color: "#6b7280" }}>Activation</span>
          </Tooltip>
          <select data-testid={`brick-context-${brickId}-activation`}
                  value={activation}
                  onChange={(e) => setField("activation", e.target.value)}>
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
              <span style={{ color: "#6b7280" }}>pre_norm</span>
            </Tooltip>
            <select data-testid={`brick-context-${brickId}-pre-norm`}
                    value={preNorm}
                    onChange={(e) => setField("pre_norm", e.target.value)}>
              {NORM_OPTIONS.map((n) =>
                <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label style={FIELD}>
            <Tooltip rpc={rpc} category="norm" name={postNorm}
                     onInfoClick={() =>
                       setExplain({ cat: "norm", name: postNorm })}>
              <span style={{ color: "#6b7280" }}>post_norm</span>
            </Tooltip>
            <select data-testid={`brick-context-${brickId}-post-norm`}
                    value={postNorm}
                    onChange={(e) => setField("post_norm", e.target.value)}>
              {NORM_OPTIONS.map((n) =>
                <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </>
      )}

      {onSwapKind && (() => {
        const currentMeta = brickFor(brickKind);
        const sameCategory = currentMeta
          ? BRICKS.filter((b) => b.category === currentMeta.category)
          : [];
        if (sameCategory.length <= 1) return null;
        return (
          <label style={FIELD}>
            <span style={{ color: "#6b7280" }}>
              Swap to (same category)
            </span>
            <select
              data-testid={`brick-context-${brickId}-swap-target`}
              value={swapTarget}
              onChange={(e) => setSwapTarget(e.target.value)}>
              {sameCategory.map((b) => (
                <option key={b.kind} value={b.kind}>{b.label}</option>
              ))}
            </select>
            <button
              data-testid={`brick-context-${brickId}-swap-apply`}
              disabled={swapTarget === brickKind}
              onClick={() => { onSwapKind(swapTarget); onClose(); }}
              style={{ background: swapTarget === brickKind
                          ? "#e5e7eb" : "#0ea5e9",
                        color: swapTarget === brickKind
                          ? "#9ca3af" : "white",
                        border: "none", padding: "4px 10px",
                        borderRadius: 4, fontSize: 12, marginTop: 4,
                        cursor: swapTarget === brickKind
                          ? "default" : "pointer" }}>
              Swap kind
            </button>
          </label>
        );
      })()}

      <button data-testid={`brick-context-${brickId}-apply`}
              onClick={() => { onApply(draft); onClose(); }}
              style={{ background: "#2563eb", color: "white",
                        border: "none", padding: "5px 12px",
                        borderRadius: 4, cursor: "pointer", fontSize: 12,
                        marginTop: 8 }}>
        Apply
      </button>

      {onInspectHistogram && (
        <div data-testid={`brick-context-${brickId}-histogram-block`}
             style={{ marginTop: 10, fontSize: 11 }}>
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
            style={{ background: "#7c3aed", color: "white",
                     border: "none", padding: "4px 10px",
                     borderRadius: 4, cursor: "pointer", fontSize: 11 }}>
            {histLoading ? "Loading…" : "Inspect weight histogram"}
          </button>
          {histError && (
            <div data-testid={`brick-context-${brickId}-histogram-error`}
                 style={{ color: "#dc2626", marginTop: 4 }}>
              {histError}
            </div>
          )}
          {hist && (
            <div data-testid={`brick-context-${brickId}-histogram-result`}
                 style={{ marginTop: 6, fontFamily: "monospace",
                          background: "#f9fafb", padding: 6,
                          borderRadius: 4 }}>
              <div data-testid={`brick-context-${brickId}-histogram-stats`}>
                n={hist.n_values} · min={hist.min.toExponential(2)} ·
                max={hist.max.toExponential(2)} ·
                mean={hist.mean.toExponential(2)}
              </div>
              <HistogramSvg counts={hist.counts} />
            </div>
          )}
        </div>
      )}

      {explain && (
        <ExplainModal rpc={rpc} category={explain.cat} name={explain.name}
                      onClose={() => setExplain(null)} />
      )}
    </div>
  );
}
