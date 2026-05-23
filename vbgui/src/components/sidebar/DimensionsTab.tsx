// DimensionsTab — table of every (brick, parameter) inference row
// from the latest verify response. Source 'auto' badges in blue; click
// a row → emit highlight event so the canvas selects that node.

import { useState } from "react";

export interface InferenceEntryClient {
  brick: string;
  param: string;
  value: unknown;
  source: "user" | "auto";
  reason: string;
}

export interface DimensionsTabProps {
  log: InferenceEntryClient[];
  onHighlight?: (brick: string) => void;
  /** G19: callback when user clicks Apply on an auto-inferred row.
   *  Parent dispatches the spec mutation so subsequent verify hides
   *  the suggestion (closing the loop). */
  onApply?: (entry: InferenceEntryClient) => void;
}

export function DimensionsTab({
  log, onHighlight, onApply,
}: DimensionsTabProps): JSX.Element {
  const [filterSource, setFilterSource] =
    useState<"all" | "user" | "auto">("all");
  const [filterBrick, setFilterBrick] = useState<string>("");

  const visible = log.filter((e) => {
    if (filterSource !== "all" && e.source !== filterSource) return false;
    if (filterBrick && !e.brick.toLowerCase().includes(filterBrick.toLowerCase()))
      return false;
    return true;
  });

  return (
    <div data-testid="dimensions-tab" style={{ padding: 12, fontSize: 12 }}>
      <header style={{ marginBottom: 8 }}>
        <h4 style={{ margin: 0, fontSize: 13 }}>Inferred Dimensions</h4>
        <div style={{ color: "#6b7280", marginTop: 2 }}>
          {visible.length} of {log.length} entries
        </div>
      </header>

      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <label style={{ display: "flex", flexDirection: "column" }}>
          <span style={{ color: "#6b7280", fontSize: 11 }}>Source</span>
          <select data-testid="dimensions-filter-source"
                  value={filterSource}
                  onChange={(e) =>
                    setFilterSource(e.target.value as typeof filterSource)}>
            <option value="all">all</option>
            <option value="auto">auto</option>
            <option value="user">user</option>
          </select>
        </label>
        <label style={{ display: "flex", flexDirection: "column", flex: 1 }}>
          <span style={{ color: "#6b7280", fontSize: 11 }}>Brick</span>
          <input data-testid="dimensions-filter-brick"
                 placeholder="filter…"
                 value={filterBrick}
                 onChange={(e) => setFilterBrick(e.target.value)} />
        </label>
      </div>

      <table data-testid="dimensions-table"
             style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ background: "#f9fafb" }}>
            <th style={th}>Brick</th>
            <th style={th}>Param</th>
            <th style={th}>Value</th>
            <th style={th}>Source</th>
            <th style={th}>Reason</th>
            <th style={th}></th>
          </tr>
        </thead>
        <tbody>
          {visible.map((e, i) => (
            <tr key={`${e.brick}.${e.param}.${i}`}
                data-testid={`dim-row-${e.brick}-${e.param}`}
                onClick={() => onHighlight?.(e.brick)}
                style={{ cursor: onHighlight ? "pointer" : "default",
                          borderBottom: "1px solid #f3f4f6" }}>
              <td style={td}><code>{e.brick}</code></td>
              <td style={td}>{e.param}</td>
              <td style={td}><code>{String(e.value)}</code></td>
              <td style={td}>
                <span data-testid={`dim-source-${e.brick}-${e.param}`}
                      style={{
                        background: e.source === "auto"
                          ? "#dbeafe" : "#f3f4f6",
                        color: e.source === "auto"
                          ? "#1e40af" : "#374151",
                        padding: "1px 6px", borderRadius: 3,
                        fontSize: 10, textTransform: "uppercase",
                      }}>{e.source}</span>
              </td>
              <td style={{ ...td, color: "#6b7280", fontSize: 11 }}>
                {e.reason}
              </td>
              <td style={td}>
                {onApply && e.source === "auto" && (
                  <button data-testid={`dim-row-${e.brick}-${e.param}-apply`}
                          onClick={(ev) => { ev.stopPropagation();
                                             onApply(e); }}
                          style={{ fontSize: 10, padding: "1px 6px" }}>
                    Apply
                  </button>
                )}
              </td>
            </tr>
          ))}
          {visible.length === 0 && (
            <tr><td colSpan={5} style={{ ...td, color: "#9ca3af",
                                          fontStyle: "italic" }}>
              No matching entries.
            </td></tr>
          )}
        </tbody>
      </table>

      {/* V7-M36 — inference_steps flow trace: same entries, but rendered
          as an ordered sequence the architect can walk step by step.
          Numbered prefix + colorised source so it reads like a small
          stack trace of how each parameter was resolved. */}
      <details data-testid="dimensions-flow-trace"
               style={{ marginTop: 12 }}>
        <summary style={{ cursor: "pointer", fontWeight: 600,
                          color: "#374151" }}>
          Flow trace ({log.length} step{log.length === 1 ? "" : "s"})
        </summary>
        <ol data-testid="dimensions-flow-trace-list"
            style={{ margin: "6px 0 0 0", paddingLeft: 24,
                     fontFamily: "ui-monospace, monospace",
                     fontSize: 11, color: "#111827" }}>
          {log.map((e, i) => (
            <li key={`flow-${e.brick}-${e.param}-${i}`}
                data-testid={`flow-step-${i}`}
                data-brick={e.brick}
                data-param={e.param}
                data-source={e.source}
                style={{ marginBottom: 2,
                         color: e.source === "auto"
                           ? "#1e40af" : "#111827" }}>
              <strong>{e.brick}.{e.param}</strong> ={" "}
              <code>{String(e.value)}</code>{" "}
              <em style={{ color: "#6b7280" }}>
                ({e.source}; {e.reason})
              </em>
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}

const th: React.CSSProperties = {
  textAlign: "left", padding: "4px 6px",
  color: "#6b7280", fontSize: 11, fontWeight: 600,
  borderBottom: "1px solid #e5e7eb",
};
const td: React.CSSProperties = { padding: "4px 6px" };
