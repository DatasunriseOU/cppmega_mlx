// V7-F51 — insert a new brick into the *middle* of an existing edge.
// HTML5 drag-drop with hit-testing of an SVG edge target is brittle in
// Playwright, so we expose a button-driven variant of the same
// operation: pick a brick kind, pick an edge (src → dst), click
// Insert; the host removes the original edge, adds the new node
// between src/dst, and wires two replacement edges.

import { useState } from "react";
import { BRICKS } from "@/lib/bricks";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface EdgePair {
  source: string;
  target: string;
}

export interface InsertIntoEdgeBarProps {
  edges: readonly EdgePair[];
  onInsert: (kind: string, edge: EdgePair) => void;
  sidebar?: boolean;
}

const DEFAULT_KIND = "mlstm";

export function InsertIntoEdgeBar({
  edges, onInsert, sidebar = false,
}: InsertIntoEdgeBarProps): JSX.Element {
  const [kind, setKind] = useState<string>(DEFAULT_KIND);
  const [edgeKey, setEdgeKey] = useState<string>(() =>
    edges.length > 0 ? edgeId(edges[0]) : "");

  const selectedEdge = edges.find((e) => edgeId(e) === edgeKey);
  const insertable = !!selectedEdge && !!kind;

  if (sidebar) {
    return (
      <div data-testid="insert-edge-bar"
           style={{ display: "flex", flexDirection: "column", gap: 8,
                    fontFamily: T.font, fontSize: 12 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontWeight: 600, color: T.textSecondary }}>Insert Into Edge</span>
          <HelpIcon topic="insert_into_edge" />
        </div>
        <label style={{ color: T.textSecondary, display: "flex", flexDirection: "column", gap: 4 }}>
          brick type
          <select data-testid="insert-edge-brick-kind"
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                  style={{
                    width: "100%",
                    color: T.text,
                    background: T.surface3,
                    border: `1px solid ${T.border}`,
                    borderRadius: 4,
                    padding: "3px 6px",
                  }}>
            {BRICKS.map((b) => (
              <option key={b.kind} value={b.kind}>
                {b.label} [{b.kind}]
              </option>
            ))}
          </select>
        </label>
        <label style={{ color: T.textSecondary, display: "flex", flexDirection: "column", gap: 4 }}>
          target edge
          <select data-testid="insert-edge-target"
                  value={edgeKey}
                  onChange={(e) => setEdgeKey(e.target.value)}
                  style={{
                    width: "100%",
                    color: T.text,
                    background: T.surface3,
                    border: `1px solid ${T.border}`,
                    borderRadius: 4,
                    padding: "3px 6px",
                  }}>
            {edges.length === 0 && (
              <option value="" disabled>(no edges yet)</option>
            )}
            {edges.map((e) => (
              <option key={edgeId(e)} value={edgeId(e)}>
                {e.source} → {e.target}
              </option>
            ))}
          </select>
        </label>
        <button data-testid="insert-edge-go"
                disabled={!insertable}
                onClick={() => {
                  if (selectedEdge && kind) onInsert(kind, selectedEdge);
                }}
                style={{ padding: "6px 12px",
                         background: insertable ? T.accent : T.surface3,
                         color: insertable ? "#fff" : T.textMuted,
                         border: "none", borderRadius: 4,
                         width: "100%",
                         fontWeight: "bold",
                         cursor: insertable ? "pointer" : "default" }}>
          Insert
        </button>
      </div>
    );
  }

  return (
    <div data-testid="insert-edge-bar"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: T.surface,
                  borderBottom: `1px solid ${T.border}`,
                  fontFamily: T.font, fontSize: 12 }}>
      <strong style={{ color: T.accent }}>Insert into edge</strong>
      <HelpIcon topic="insert_into_edge" />
      <label style={{ color: T.textSecondary, display: "flex", alignItems: "center", gap: 4 }}>
        brick
        <select data-testid="insert-edge-brick-kind"
                value={kind}
                onChange={(e) => setKind(e.target.value)}
                style={{
                  marginLeft: 4,
                  width: 180,
                  color: T.text,
                  background: T.surface3,
                  border: `1px solid ${T.border}`,
                }}>
          {BRICKS.map((b) => (
            <option key={b.kind} value={b.kind}>
              {b.label} [{b.kind}]
            </option>
          ))}
        </select>
      </label>
      <label style={{ color: T.textSecondary, display: "flex", alignItems: "center", gap: 4 }}>
        between
        <select data-testid="insert-edge-target"
                value={edgeKey}
                onChange={(e) => setEdgeKey(e.target.value)}
                style={{
                  marginLeft: 4,
                  width: 240,
                  color: T.text,
                  background: T.surface3,
                  border: `1px solid ${T.border}`,
                }}>
          {edges.length === 0 && (
            <option value="" disabled>(no edges yet)</option>
          )}
          {edges.map((e) => (
            <option key={edgeId(e)} value={edgeId(e)}>
              {e.source} → {e.target}
            </option>
          ))}
        </select>
      </label>
      <button data-testid="insert-edge-go"
              disabled={!insertable}
              onClick={() => {
                if (selectedEdge && kind) onInsert(kind, selectedEdge);
              }}
              style={{ padding: "2px 10px",
                       background: insertable ? T.accent : T.surface3,
                       color: insertable ? "#0f172a" : T.textMuted,
                       border: `1px solid ${T.border}`, borderRadius: 4,
                       cursor: insertable ? "pointer" : "default" }}>
        Insert
      </button>
    </div>
  );
}

function edgeId(e: EdgePair): string {
  return `${e.source}->${e.target}`;
}
