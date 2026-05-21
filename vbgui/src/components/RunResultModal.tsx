// Modal that displays per-stage results from a pipeline.run response.

import { useState } from "react";

export interface StageResult {
  name: string;
  status: "ok" | "skipped" | "fail";
  elapsed_ms: number;
  warnings?: number;
  errors?: number;
  error?: { type?: string; detail?: string; [k: string]: unknown } | null;
  [k: string]: unknown;
}

export interface RunReport {
  stages: StageResult[];
  overall_status: "ok" | "fail";
  total_elapsed_ms: number;
}

export interface RunResultModalProps {
  report: RunReport | null;
  error?: string | null;
  onClose: () => void;
}

const ICONS = { ok: "✓", fail: "✗", skipped: "·" } as const;
const COLORS = {
  ok: "#10b981", fail: "#dc2626", skipped: "#9ca3af",
} as const;

export function RunResultModal({
  report, error, onClose,
}: RunResultModalProps): JSX.Element | null {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!report && !error) return null;

  return (
    <div data-testid="run-result-modal-backdrop"
         role="dialog" aria-modal="true"
         onClick={onClose}
         style={{
           position: "fixed", inset: 0, background: "rgba(15,23,42,0.45)",
           display: "flex", alignItems: "center", justifyContent: "center",
           zIndex: 50, fontFamily: "system-ui, sans-serif",
         }}>
      <div data-testid="run-result-modal"
           onClick={(e) => e.stopPropagation()}
           style={{
             background: "white", borderRadius: 6, padding: 16,
             minWidth: 540, maxWidth: 720, maxHeight: "80vh",
             overflowY: "auto",
           }}>
        <header style={{ display: "flex", justifyContent: "space-between",
                         alignItems: "center", marginBottom: 12 }}>
          <h3 data-testid="run-result-title" style={{ margin: 0, fontSize: 14 }}>
            Pipeline result{" "}
            <span data-testid="run-result-overall"
                  style={{ color: report
                    ? COLORS[report.overall_status]
                    : COLORS.fail }}>
              {report ? `· ${report.overall_status} · ` +
                        `${report.total_elapsed_ms.toFixed(1)} ms`
                      : "· error"}
            </span>
          </h3>
          <button data-testid="run-result-close" onClick={onClose}>×</button>
        </header>

        {error && (
          <div data-testid="run-result-error"
               style={{ background: "#fee2e2", padding: 8, borderRadius: 4,
                        color: "#991b1b", fontSize: 12 }}>
            {error}
          </div>
        )}

        {report && (
          <table data-testid="run-result-stages"
                 style={{ width: "100%", fontSize: 12,
                          borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                <th style={th}></th>
                <th style={th}>Stage</th>
                <th style={th}>Status</th>
                <th style={th}>ms</th>
                <th style={th}></th>
              </tr>
            </thead>
            <tbody>
              {report.stages.map((s) => {
                const open = expanded.has(s.name);
                return (
                  <>
                    <tr key={s.name}
                        data-testid={`run-result-stage-${s.name}`}
                        style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={td}>
                        <span style={{ color: COLORS[s.status],
                                       fontWeight: 700 }}>
                          {ICONS[s.status]}
                        </span>
                      </td>
                      <td style={td}>{s.name}</td>
                      <td style={{ ...td, color: COLORS[s.status] }}>
                        {s.status}
                      </td>
                      <td style={{ ...td, textAlign: "right" }}>
                        {s.elapsed_ms.toFixed(1)}
                      </td>
                      <td style={td}>
                        {s.error && (
                          <button data-testid={`run-result-expand-${s.name}`}
                                  onClick={() => toggle(expanded,
                                                       setExpanded, s.name)}>
                            {open ? "▾" : "▸"}
                          </button>
                        )}
                      </td>
                    </tr>
                    {open && s.error && (
                      <tr data-testid={`run-result-detail-${s.name}`}>
                        <td colSpan={5} style={{ ...td, background: "#f9fafb",
                                                 fontFamily: "monospace",
                                                 fontSize: 11 }}>
                          <strong>{s.error.type ?? "Error"}</strong>:{" "}
                          {s.error.detail ?? JSON.stringify(s.error)}
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function toggle(set: Set<string>, setter: (s: Set<string>) => void,
                key: string): void {
  const next = new Set(set);
  next.has(key) ? next.delete(key) : next.add(key);
  setter(next);
}

const th: React.CSSProperties = { textAlign: "left", padding: "4px 6px",
                                  color: "#6b7280", fontWeight: 600 };
const td: React.CSSProperties = { padding: "4px 6px" };
