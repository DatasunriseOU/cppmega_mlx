import { useState } from "react";
import type { GotchaState } from "@/state/spec";

export interface AdapterStep {
  rule: string;
  description: string;
  params?: Record<string, unknown>;
}

export interface AdapterChain {
  producer: string;
  consumer: string;
  producer_shape: number[];
  consumer_shape: number[];
  chain: AdapterStep[];
  reason: string;
}

export interface GotchasTabProps {
  gotchas: GotchaState[];
  onAutoFix?: (id: string) => void;
  /** V7-K3: suggest_adapters callback — UI sends producer/consumer
   *  brick names, backend returns the adapter chain that would bridge
   *  them. Rendered inline so users can see how to fix an edge gap. */
  onSuggestAdapters?: (producer: string,
                        consumer: string) => Promise<AdapterChain>;
}

const COLOR: Record<GotchaState["severity"], string> = {
  error:   "#dc2626",
  warning: "#d97706",
  info:    "#2563eb",
};

// V7-L50: card-background tints per severity. Border-left + header
// color were already differentiated, but the body sat on a neutral
// gray for all three — so WARNING blended visually with INFO. These
// soft pastel tints make each severity readable at a glance without
// being shouty.
const BG_TINT: Record<GotchaState["severity"], string> = {
  error:   "#fee2e2",
  warning: "#fef3c7",
  info:    "#dbeafe",
};

// V7-H01: extended auto-fix coverage for the 6 most common
// validation gotchas. The App handler dispatches the corresponding
// spec mutation when the user clicks the inline Apply-fix button.
const AUTO_FIXABLE: Set<string> = new Set([
  "fsdp2_whole_compile",
  "megatron_tp_whole_compile",
  "missing_edge",
  "dim_mismatch",
  "unknown_brick",
  "bad_dtype_combo",
  "schedule_out_of_range",
  "tokenizer_mismatch",
]);

const FIX_LABELS: Record<string, string> = {
  fsdp2_whole_compile:      "Switch compile_mode → regional",
  megatron_tp_whole_compile: "Switch compile_mode → regional",
  missing_edge:             "Insert missing edge",
  dim_mismatch:             "Adjust hidden_size to nearest valid",
  unknown_brick:            "Remove unknown brick",
  bad_dtype_combo:          "Reset dtype to bf16 master",
  schedule_out_of_range:    "Clamp schedule to valid range",
  tokenizer_mismatch:       "Pick MATRIX-compatible tokenizer",
};

// V7-L49: pull a short 'file:line' chip out of the reference URL.
// Supports common shapes:
//   '/abs/path/file.py:123' → 'file.py:123'
//   'cppmega_v4/foo/bar.py:42' → 'bar.py:42'
//   'docs/plan.md#anchor' → 'plan.md#anchor'
//   'https://github.com/.../file.py#L42' → 'file.py#L42'
function parseSourceFile(ref: string | undefined): string | null {
  if (!ref) return null;
  // Strip leading path; keep last segment + optional :line / #anchor.
  const last = ref.split("/").pop() ?? ref;
  if (!last) return null;
  return last;
}

function groupBySeverity(gs: GotchaState[]): Record<string, GotchaState[]> {
  const out: Record<string, GotchaState[]> = { error: [], warning: [], info: [] };
  for (const g of gs) (out[g.severity] ??= []).push(g);
  return out;
}

export function GotchasTab({
  gotchas, onAutoFix, onSuggestAdapters,
}: GotchasTabProps): JSX.Element {
  const grouped = groupBySeverity(gotchas);
  const [adapterProducer, setAdapterProducer] = useState<string>("");
  const [adapterConsumer, setAdapterConsumer] = useState<string>("");
  const [adapterChain, setAdapterChain] = useState<AdapterChain | null>(null);
  const [adapterError, setAdapterError] = useState<string | null>(null);
  const [adapterLoading, setAdapterLoading] = useState<boolean>(false);
  return (
    <div data-testid="gotchas-tab" style={panel}>
      {onSuggestAdapters && (
        <section data-testid="gotchas-suggest-adapters-panel"
                 style={{ background: "#f3f4f6", padding: 8, borderRadius: 4,
                          fontSize: 11 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Suggest adapter chain (V7-K3)
          </div>
          <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
            <input data-testid="gotchas-suggest-adapters-producer"
                   placeholder="producer brick id"
                   value={adapterProducer}
                   onChange={(e) => setAdapterProducer(e.target.value)}
                   style={{ flex: 1, fontFamily: "monospace",
                             fontSize: 11 }} />
            <span>→</span>
            <input data-testid="gotchas-suggest-adapters-consumer"
                   placeholder="consumer brick id"
                   value={adapterConsumer}
                   onChange={(e) => setAdapterConsumer(e.target.value)}
                   style={{ flex: 1, fontFamily: "monospace",
                             fontSize: 11 }} />
            <button data-testid="gotchas-suggest-adapters-run"
                    disabled={adapterLoading || !adapterProducer
                              || !adapterConsumer}
                    onClick={async () => {
                      setAdapterLoading(true);
                      setAdapterError(null);
                      try {
                        const r = await onSuggestAdapters(
                          adapterProducer, adapterConsumer);
                        setAdapterChain(r);
                      } catch (e) {
                        setAdapterError(
                          e instanceof Error ? e.message : String(e));
                      } finally {
                        setAdapterLoading(false);
                      }
                    }}>
              {adapterLoading ? "…" : "Suggest"}
            </button>
          </div>
          {adapterError && (
            <div data-testid="gotchas-suggest-adapters-error"
                 style={{ color: "#dc2626" }}>{adapterError}</div>
          )}
          {adapterChain && (
            <div data-testid="gotchas-suggest-adapters-result"
                 style={{ fontFamily: "monospace" }}>
              <div data-testid="gotchas-suggest-adapters-shapes">
                {adapterChain.producer}{" "}
                [{adapterChain.producer_shape.join("×")}] →{" "}
                {adapterChain.consumer}{" "}
                [{adapterChain.consumer_shape.join("×")}]
              </div>
              <div data-testid="gotchas-suggest-adapters-reason"
                   style={{ color: "#6b7280", marginTop: 2 }}>
                {adapterChain.reason}
              </div>
              {adapterChain.chain.length === 0 ? (
                <div data-testid="gotchas-suggest-adapters-chain-empty"
                     style={{ color: "#16a34a", marginTop: 2 }}>
                  ✓ no adapter needed
                </div>
              ) : (
                <ol data-testid="gotchas-suggest-adapters-chain"
                    style={{ margin: "4px 0 0 18px", padding: 0 }}>
                  {adapterChain.chain.map((step, i) => (
                    <li key={i}
                        data-testid={`gotchas-suggest-adapters-step-${i}`}>
                      <strong>{step.rule}</strong>: {step.description}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
        </section>
      )}
      {gotchas.length === 0 && (
        <p style={{ color: "#9ca3af" }}>No gotchas fired.</p>
      )}
      {(["error", "warning", "info"] as const).map((sev) =>
        grouped[sev].length > 0 ? (
          <section key={sev} data-testid={`gotchas-${sev}`}>
            <h4 style={{ margin: "0 0 4px", color: COLOR[sev] }}>
              {sev.toUpperCase()}
            </h4>
            {grouped[sev].map((g) => {
              const sourceFile = parseSourceFile(g.reference);
              // V7-L48: prefer backend-provided suggested_fix over the
              // hardcoded legacy AUTO_FIXABLE/FIX_LABELS pair.
              const fixLabel = g.suggested_fix
                            ?? FIX_LABELS[g.id]
                            ?? null;
              const showFix = onAutoFix && fixLabel !== null
                            && (g.suggested_fix !== undefined
                               || AUTO_FIXABLE.has(g.id));
              return (
              <div key={g.id} data-testid={`gotcha-${g.id}`}
                   data-severity={sev}
                   style={{ background: BG_TINT[sev],
                            borderLeft: `4px solid ${COLOR[sev]}`,
                            padding: "6px 8px", marginBottom: 4,
                            borderRadius: 3 }}>
                <div style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                  <span data-testid={`gotcha-${g.id}-id`}
                        style={{ fontWeight: 600 }}>{g.id}</span>
                  <span data-testid={`gotcha-${g.id}-severity`}
                        style={{ background: COLOR[sev], color: "#fff",
                                 padding: "1px 6px", borderRadius: 9999,
                                 fontSize: 9, textTransform: "uppercase",
                                 letterSpacing: 0.4 }}>
                    {sev}
                  </span>
                </div>
                <div style={{ color: "#374151" }}>{g.message}</div>
                {sourceFile && (
                  <span data-testid={`gotcha-${g.id}-source`}
                        title={g.reference}
                        style={{ display: "inline-block", marginRight: 6,
                                 fontFamily: "monospace", fontSize: 10,
                                 color: "#374151",
                                 background: "rgba(255,255,255,0.6)",
                                 border: "1px solid #d1d5db",
                                 borderRadius: 3, padding: "0 4px" }}>
                    src: {sourceFile}
                  </span>
                )}
                {g.reference && (
                  <a data-testid={`gotcha-${g.id}-ref`}
                     href={g.reference.startsWith("http")
                       ? g.reference : `#${g.reference}`}
                     target="_blank" rel="noreferrer"
                     style={{ color: "#2563eb", fontSize: 11 }}>
                    {g.reference}
                  </a>
                )}
                {showFix && (
                  <button data-testid={`gotcha-${g.id}-autofix`}
                          onClick={() => onAutoFix!(g.id)}
                          title={fixLabel ?? "Auto-fix"}
                          style={{ display: "block", marginTop: 4 }}>
                    {fixLabel ?? "Auto-fix"}
                  </button>
                )}
                {g.suggested_fix !== undefined && !showFix && (
                  // backend wants the fix but host didn't pass onAutoFix
                  // — still surface the hint so the architect knows what
                  // would happen.
                  <span data-testid={`gotcha-${g.id}-fix-hint`}
                        style={{ display: "block", marginTop: 4,
                                 color: "#6b7280", fontSize: 11,
                                 fontStyle: "italic" }}>
                    suggested fix: {g.suggested_fix}
                  </span>
                )}
              </div>
              );
            })}
          </section>
        ) : null,
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 12, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
