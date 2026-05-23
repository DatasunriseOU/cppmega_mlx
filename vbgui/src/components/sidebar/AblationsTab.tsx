// AblationsTab — side-by-side variant comparison (E7-11 UI).
// Calls ablation.run RPC; renders a results table with mini-chart
// loss curves per variant + ranking by final loss.

import { Fragment, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";
import type { OptimState, LossState } from "@/state/spec";

type AblationAxis = "activation" | "optimizer" | "norm" | "schedule";

const VARIANTS_PER_AXIS: Record<AblationAxis, string[]> = {
  activation: ["glu", "swiglu", "gelu", "relu", "relu2", "silu",
               "mish", "geglu", "reglu"],
  optimizer:  ["adamw", "muon", "muon_adamw_hybrid", "lion",
               "lion8bit", "adam8bit", "sgd"],
  norm:       ["rmsnorm", "layernorm", "none"],
  schedule:   ["constant", "cosine", "linear_warmup", "wsd",
               "inv_sqrt", "polynomial"],
};

interface AblationVariantResult {
  variant: string;
  status: "ok" | "fail";
  losses: number[];
  elapsed_ms: number;
  weight_delta_norm: number;
  error?: Record<string, unknown> | null;
  /** H14: full train extras subtree per variant (losses, model_summary,
   *  optimizer_kind, schedule_kind, data_source, etc.). Optional so
   *  pre-H14 backend responses still parse. */
  extras?: Record<string, unknown>;
}

interface AblationResult {
  results: AblationVariantResult[];
  ranked_by_final_loss: string[];
  baseline_variant: string;
  elapsed_ms_total: number;
}

export interface AblationsTabProps {
  rpc: RpcClient | null;
  nodes: Node[];
  edges: Edge[];
  optim: OptimState;
  loss: LossState;
  hiddenSize?: number;
}

function nodesToGraph(nodes: Node[], edges: Edge[]) {
  const modelNodes = nodes.filter(
    (n) => n.type !== "tokenizer_virtual" && n.type !== "detokenizer_virtual"
  );
  const modelNodeIds = new Set(modelNodes.map((n) => n.id));
  const modelEdges = edges.filter(
    (e) => modelNodeIds.has(e.source) && modelNodeIds.has(e.target)
  );

  return {
    nodes: modelNodes.map((n) => {
      const d = n.data as { kind?: string; params?: Record<string, unknown> };
      return { id: n.id, kind: d.kind ?? "mlp", params: d.params ?? {} };
    }),
    edges: modelEdges.map((e) => ({ src: e.source, dst: e.target })),
  };
}

function MiniChart({ values }: { values: number[] }): JSX.Element {
  if (values.length === 0) return <span style={{ color: "var(--vb-text-muted)" }}>—</span>;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * 78 + 1;
    const y = 30 - ((v - min) / span) * 26 - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={80} height={30}
         style={{ background: "var(--vb-surface-2)", borderRadius: 2 }}>
      <polyline fill="none" stroke="#2563eb" strokeWidth="1.2" points={pts} />
    </svg>
  );
}

export function AblationsTab({
  rpc, nodes, edges, optim, loss, hiddenSize = 128,
}: AblationsTabProps): JSX.Element {
  const [axis, setAxis] = useState<AblationAxis>("activation");
  const [variants, setVariants] = useState<Set<string>>(new Set(["glu", "swiglu"]));
  const [numSteps, setNumSteps] = useState(5);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AblationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // H14: per-variant expand state for the full extras subtree.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  function toggleExpanded(v: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(v)) next.delete(v); else next.add(v);
      return next;
    });
  }

  function toggleVariant(v: string) {
    setVariants((prev) => {
      const next = new Set(prev);
      next.has(v) ? next.delete(v) : next.add(v);
      return next;
    });
  }

  async function run() {
    if (!rpc) { setError("rpc unavailable"); return; }
    if (nodes.length === 0) { setError("canvas is empty"); return; }
    if (variants.size < 2) { setError("pick at least 2 variants"); return; }
    setRunning(true); setError(null);
    try {
      const r = await rpc.call<AblationResult>("ablation.run", {
        base_spec: {
          graph: nodesToGraph(nodes, edges),
          dim_env: { B: 1, S: 8, H: hiddenSize, nh: 2, nkv: 1,
                     head_dim: 64, num_experts: 4, top_k: 2 },
          loss: { kind: loss.kind, head_outputs: loss.head_outputs,
                  params: loss.params },
          optim: {
            kind: optim.kind,
            groups: optim.groups.map((g) => ({
              matcher: g.matcher, lr: g.lr,
              weight_decay: g.weight_decay, betas: g.betas,
            })),
          },
        },
        ablation_axis: axis,
        variants: Array.from(variants),
        num_steps: numSteps,
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  const baseline = result?.baseline_variant ?? "";
  const baselineFinal = result?.results
    .find((r) => r.variant === baseline && r.status === "ok")
    ?.losses.at(-1);
  const ordered = result ? [
    ...result.ranked_by_final_loss
      .map((n) => result.results.find((r) => r.variant === n)!),
    ...result.results.filter((r) =>
      !result.ranked_by_final_loss.includes(r.variant)),
  ] : [];

  return (
    <div data-testid="ablations-tab" style={panel}>
      <h4 style={{ margin: 0, fontSize: 13 }}>Ablation Runner</h4>

      <label>Axis
        <select data-testid="ablation-axis"
                value={axis}
                onChange={(e) => {
                  const ax = e.target.value as AblationAxis;
                  setAxis(ax);
                  setVariants(new Set(VARIANTS_PER_AXIS[ax].slice(0, 2)));
                }}>
          {(Object.keys(VARIANTS_PER_AXIS) as AblationAxis[]).map((a) =>
            <option key={a} value={a}>{a}</option>)}
        </select>
      </label>

       <div data-testid="ablation-variants"
            style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {VARIANTS_PER_AXIS[axis].map((v) => (
          <label key={v} data-testid={`ablation-variant-${v}`}
                 style={{ display: "inline-flex", gap: 3, fontSize: 11,
                          padding: "2px 6px",
                          background: variants.has(v) ? "var(--vb-accent-soft)" : "var(--vb-surface-3)",
                          color: variants.has(v) ? "var(--vb-accent)" : "var(--vb-text)",
                          borderRadius: 3, cursor: "pointer" }}>
            <input type="checkbox" checked={variants.has(v)}
                   onChange={() => toggleVariant(v)} />
            {v}
          </label>
        ))}
      </div>

      <label style={{ fontSize: 11 }}>num_steps
        <input data-testid="ablation-num-steps" type="number"
               min={1} max={100} value={numSteps}
               onChange={(e) => setNumSteps(Number(e.target.value))}
               style={{ width: 60, marginLeft: 4 }} />
      </label>

      <button data-testid="ablation-run" onClick={run} disabled={running}
              style={{ background: "var(--vb-accent)", color: "var(--vb-accent-contrast)",
                       border: "none", padding: "5px 12px", fontWeight: "bold",
                       borderRadius: 4, cursor: "pointer", fontSize: 12 }}>
        {running ? "Running…" : "Run ablation"}
      </button>

      {error && (
        <div data-testid="ablation-error" style={{ color: "#dc2626",
                                                   fontSize: 11 }}>
          {error}
        </div>
      )}

      {result && (
        <table data-testid="ablation-results"
               style={{ width: "100%", fontSize: 11, marginTop: 6,
                        borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--vb-surface-2)" }}>
              <th style={th}></th>
              <th style={th}>Variant</th>
              <th style={th}>Final</th>
              <th style={th}>Δ</th>
              <th style={th}>Loss curve</th>
              <th style={th}>Status</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((r) => {
              const final = r.losses.at(-1);
              const delta = (baselineFinal != null && final != null)
                ? ((final - baselineFinal) / Math.max(1e-9, baselineFinal)) * 100
                : null;
              const open = expanded.has(r.variant);
              return (
                <Fragment key={r.variant}>
                  <tr data-testid={`ablation-row-${r.variant}`}
                      style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={td}>
                      <button data-testid={
                                `ablation-row-${r.variant}-expand`}
                              onClick={() => toggleExpanded(r.variant)}
                              style={{ background: "transparent",
                                       border: "none", cursor: "pointer",
                                       padding: 0 }}>
                        {open ? "▾" : "▸"}
                      </button>
                    </td>
                    <td style={td}>
                      {r.variant === baseline && (
                        <span style={{ color: "#f59e0b",
                                       marginRight: 2 }}>★</span>
                      )}
                      <code>{r.variant}</code>
                    </td>
                    <td data-testid={`ablation-final-${r.variant}`}
                        style={td}>{final?.toFixed(4) ?? "—"}</td>
                    <td style={{ ...td,
                                  color: delta == null ? "var(--vb-text-muted)"
                                         : delta > 0 ? "#dc2626"
                                                     : "#16a34a" }}>
                      {delta == null ? "—"
                        : delta === 0 ? "0%"
                        : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`}
                    </td>
                    <td style={td}><MiniChart values={r.losses} /></td>
                    <td style={{ ...td,
                                  color: r.status === "ok"
                                    ? "#16a34a" : "#dc2626" }}>
                      {r.status}
                    </td>
                  </tr>
                  {open && (
                    <tr data-testid={`ablation-row-${r.variant}-extras`}>
                      <td colSpan={6}
                          style={{ ...td, background: "var(--vb-surface-2)",
                                   padding: 8 }}>
                        <VariantExtras variant={r.variant}
                                       losses={r.losses}
                                       extras={r.extras ?? {}} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function VariantExtras({
  variant, losses, extras,
}: { variant: string; losses: number[];
     extras: Record<string, unknown> }): JSX.Element {
  // H14: render the full extras subtree per variant so the user can see
  // exactly what diverged across ablation runs (model_summary, optimizer
  // kind, schedule kind, data_source, etc.) — not just final loss.
  return (
    <dl data-testid={`ablation-row-${variant}-extras-content`}
        style={{ margin: 0, fontSize: 11, fontFamily: "monospace",
                 display: "grid",
                 gridTemplateColumns: "120px 1fr",
                 columnGap: 8, rowGap: 2 }}>
      <dt style={{ color: "var(--vb-text-muted)" }}>losses</dt>
      <dd data-testid={`ablation-row-${variant}-losses`}
          style={{ margin: 0 }}>
        [{losses.map((l) => l.toFixed(4)).join(", ")}]
      </dd>
      {Object.entries(extras).map(([k, v]) => (
        <Fragment key={k}>
          <dt style={{ color: "var(--vb-text-muted)" }}>{k}</dt>
          <dd data-testid={`ablation-row-${variant}-extras-${k}`}
              style={{ margin: 0, wordBreak: "break-all" }}>
            {v === null || v === undefined
              ? "null"
              : typeof v === "object"
                ? JSON.stringify(v)
                : String(v)}
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 8, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
const th: React.CSSProperties = {
  textAlign: "left", padding: "3px 4px", color: "var(--vb-text-muted)",
  fontSize: 10, fontWeight: 600,
};
const td: React.CSSProperties = { padding: "3px 4px" };
