// Modal that displays per-stage results from a pipeline.run response.

import { Fragment, useEffect, useRef, useState } from "react";
import { LossChart } from "@/components/LossChart";
import { GradAttnPanel } from "@/components/GradAttnPanel";
import { ErrorDetailsPanel,
         type ErrorDetails } from "@/components/ErrorDetailsPanel";
import { TrainExtrasOverlay,
         type TrainExtras } from "@/components/TrainExtrasOverlay";

export interface StageResult {
  name: string;
  status: "ok" | "skipped" | "fail" | "cancelled";
  elapsed_ms: number;
  warnings?: number;
  errors?: number;
  error?: { type?: string; detail?: string; [k: string]: unknown } | null;
  [k: string]: unknown;
}

export interface RunReport {
  stages: StageResult[];
  overall_status: "ok" | "fail" | "cancelled";
  total_elapsed_ms: number;
}

export interface RunResultModalProps {
  report: RunReport | null;
  // V7-L46: error can be a legacy string or a rich ErrorDetails
  // shape carrying field-level Pydantic errors / traceback.
  error?: string | ErrorDetails | null;
  onClose: () => void;
}

function normalizeError(
  e: string | ErrorDetails | null | undefined,
): ErrorDetails | null {
  if (e == null) return null;
  return typeof e === "string" ? { message: e } : e;
}

const ICONS = { ok: "✓", fail: "✗", skipped: "·", cancelled: "!" } as const;
const COLORS = {
  ok: "#10b981", fail: "#dc2626", skipped: "#9ca3af", cancelled: "#f59e0b",
} as const;

// V3-4: keys excluded from the visible extras dl because they're
// redundant with the row's status / error rendering.
const EXTRAS_RESERVED = new Set<string>([
  "name", "status", "elapsed_ms", "warnings", "errors", "error",
]);

function extrasOf(s: StageResult): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(s)) {
    if (!EXTRAS_RESERVED.has(k)) out[k] = v;
  }
  return out;
}

function hasContent(s: StageResult): boolean {
  return s.error != null || Object.keys(extrasOf(s)).length > 0;
}

function StageExtras({
  stage, extras,
}: { stage: string; extras: Record<string, unknown> }): JSX.Element {
  return (
    <dl data-testid={`run-result-extras-${stage}`}
        style={{ margin: 0, fontSize: 11, fontFamily: "monospace" }}>
      {Object.entries(extras).map(([k, v]) => (
        <ExtrasEntry key={k} stage={stage} k={k} v={v} />
      ))}
    </dl>
  );
}

function ExtrasEntry({
  stage, k, v,
}: { stage: string; k: string; v: unknown }): JSX.Element {
  const base = `run-result-extras-${stage}-${k}`;
  if (Array.isArray(v)) {
    return (
      <div style={{ display: "flex", gap: 8 }}>
        <dt style={{ color: "#6b7280", minWidth: 140 }}>{k}</dt>
        <dd style={{ margin: 0 }}>
          <ol data-testid={base}
              style={{ margin: 0, padding: "0 0 0 16px",
                       display: "flex", gap: 6, flexWrap: "wrap",
                       listStyle: "none" }}>
            {v.map((item, i) => (
              <li key={i} data-testid={`${base}-${i}`}>
                {typeof item === "object" && item !== null
                  ? JSON.stringify(item)
                  : String(item)}
              </li>
            ))}
          </ol>
        </dd>
      </div>
    );
  }
  if (v !== null && typeof v === "object") {
    // G01: recurse so nested arrays/objects (e.g. extras.mtp.betas) get
    // their own per-index testids instead of being JSON-stringified into
    // a single dd. The recursion uses the same `${base}-${sk}` testid
    // prefix the test framework already expects.
    return (
      <div style={{ display: "flex", gap: 8 }}>
        <dt style={{ color: "#6b7280", minWidth: 140 }}>{k}</dt>
        <dd style={{ margin: 0 }}>
          <dl data-testid={base}
              style={{ margin: 0, paddingLeft: 8 }}>
            {Object.entries(v as Record<string, unknown>).map(([sk, sv]) => (
              <ExtrasEntry key={sk} stage={stage} k={`${k}-${sk}`} v={sv} />
            ))}
          </dl>
        </dd>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", gap: 8 }}>
      <dt style={{ color: "#6b7280", minWidth: 140 }}>{k}</dt>
      <dd data-testid={base} style={{ margin: 0 }}>
        {v === null || v === undefined ? "null" : String(v)}
      </dd>
    </div>
  );
}

export function RunResultModal({
  report, error, onClose,
}: RunResultModalProps): JSX.Element | null {
  const normalizedError = normalizeError(error);
  // V7-L47: pre-expand the first failing stage so the user doesn't
  // have to scroll the modal AND click to discover the failure.
  const firstFailed = report?.stages?.find((s) => s.status === "fail");
  const [expanded, setExpanded] = useState<Set<string>>(
    () => firstFailed ? new Set([firstFailed.name]) : new Set());
  const failedRowRef = useRef<HTMLTableRowElement | null>(null);
  // When `report` swaps to a different failing run, re-expand the
  // new failed stage too.
  useEffect(() => {
    if (firstFailed) {
      setExpanded((prev) => prev.has(firstFailed.name)
        ? prev : new Set([...prev, firstFailed.name]));
      if (failedRowRef.current
          && typeof failedRowRef.current.scrollIntoView === "function") {
        failedRowRef.current.scrollIntoView({
          block: "center", behavior: "auto",
        });
      }
    }
  }, [firstFailed]);
  if (!report && !normalizedError) return null;

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

        {normalizedError && (
          <div data-testid="run-result-error">
            <ErrorDetailsPanel error={normalizedError} />
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
                const extras = extrasOf(s);
                const hasExtras = Object.keys(extras).length > 0;
                const isFirstFailed = firstFailed?.name === s.name;
                return (
                  <Fragment key={s.name}>
                    <tr data-testid={`run-result-stage-${s.name}`}
                        ref={isFirstFailed ? failedRowRef : undefined}
                        data-first-failed={isFirstFailed
                                             ? "true" : undefined}
                        style={{
                          borderBottom: "1px solid #f3f4f6",
                          // V7-L47: visual highlight on the first
                          // failing row so the user sees it without
                          // scrolling.
                          background: isFirstFailed
                            ? "#fef2f2" : undefined,
                          outline: isFirstFailed
                            ? "2px solid #fca5a5" : undefined,
                          outlineOffset: isFirstFailed ? -2 : undefined,
                        }}>
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
                        {hasContent(s) && (
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
                    {open && hasExtras && (
                      <tr data-testid={`run-result-extras-row-${s.name}`}>
                        <td colSpan={5} style={{ ...td, background: "#f9fafb",
                                                 padding: "6px 12px" }}>
                          {s.name === "train" && Array.isArray(extras.losses)
                            && extras.losses.length > 0 && (
                            <div data-testid="run-result-loss-chart-wrap"
                                 style={{ marginBottom: 8 }}>
                              <LossChart
                                losses={(extras.losses as unknown[])
                                  .map(Number)
                                  .filter((n) => Number.isFinite(n))}
                              />
                            </div>
                          )}
                          {s.name === "train" && (
                            <TrainExtrasOverlay
                              extras={extras as TrainExtras}
                            />
                          )}
                          {s.name === "train" && (
                            <div data-testid="run-result-grad-attn-wrap"
                                 style={{ marginBottom: 8 }}>
                              <GradAttnPanel
                                gradNorms={
                                  extras.per_brick_grad_norms as
                                    Record<string, number> | undefined}
                                attnHeadMeans={
                                  extras.attn_head_means as
                                    Record<string, number[]> | undefined}
                              />
                            </div>
                          )}
                          <StageExtras stage={s.name} extras={extras} />
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
