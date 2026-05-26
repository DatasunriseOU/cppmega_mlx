/**
 * LossGhostNode — read-only canvas node that visualises the current
 * spec.loss kind + params. Anchored to spec.loss.head_outputs[0] via
 * a dashed edge. Lets users see "what gets applied" when they click
 * Apply in the Loss sidebar tab.
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { T } from "@/theme";

export interface LossGhostNodeData {
  kind: string;
  params?: Record<string, number | string>;
}

export function LossGhostNode({ data, id }: NodeProps): JSX.Element {
  const d = data as unknown as LossGhostNodeData;
  const targetPosition = (data as any)?.targetPosition ?? Position.Left;
  const paramSummary = d.params
    ? Object.entries(d.params)
        .map(([k, v]) => `${k}=${v}`).join(", ")
    : "";
  return (
    <div
      role="group"
      aria-label={`loss ${d.kind}`}
      data-testid={`loss-ghost-${id}`}
      style={{
        minWidth: 180,
        padding: "10px 12px",
        background: "rgba(250, 204, 21, 0.07)",
        border: "2px dashed #facc15",
        borderRadius: 10,
        fontFamily: T.font,
        color: T.text,
      }}
    >
      <Handle type="target" position={targetPosition} />
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span aria-hidden="true"
              style={{ fontSize: 16, color: "#facc15" }}>L</span>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}
               data-testid="loss-ghost-kind">
            Loss · {d.kind}
          </div>
          {paramSummary && (
            <div style={{ color: T.textMuted, fontSize: 11,
                          fontFamily: T.fontMono, marginTop: 2 }}
                 data-testid="loss-ghost-params">
              {paramSummary}
            </div>
          )}
        </div>
      </div>
      <footer style={{ marginTop: 8, paddingTop: 6,
                       borderTop: `1px solid ${T.borderSoft}`,
                       color: T.textMuted, fontSize: 10 }}>
        synthetic — applied via Sidebar → Loss
      </footer>
    </div>
  );
}
