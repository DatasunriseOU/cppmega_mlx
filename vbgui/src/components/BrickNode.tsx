import { Handle, Position, type NodeProps } from "@xyflow/react";
import { brickFor, colorFor } from "@/lib/bricks";

export interface BrickNodeData {
  kind: string;
  name?: string;
  params?: Record<string, unknown>;
  shape?: number[];
  memory_mb?: number;
  // Side-channel availability hint surfaced from the resolver. When set
  // to false, the node renders a small "missing" badge.
  side_channels_ok?: boolean;
}

export function BrickNode({ data, id }: NodeProps): JSX.Element {
  const d = data as unknown as BrickNodeData;
  const meta = brickFor(d.kind);
  const color = meta ? colorFor(meta.category) : "#9ca3af";
  return (
    <div
      role="group"
      aria-label={`brick ${d.kind}`}
      data-testid={`brick-node-${id}`}
      style={{
        borderLeft: `4px solid ${color}`,
        background: "#fff",
        borderRadius: 6,
        padding: "8px 12px",
        minWidth: 160,
        fontFamily: "system-ui, sans-serif",
        fontSize: 12,
        boxShadow: "0 1px 2px rgba(0,0,0,0.08)",
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontWeight: 600 }}>{meta?.label ?? d.kind}</div>
      <div style={{ color: "#6b7280", fontSize: 10 }}>{d.kind}</div>

      {d.shape && (
        <div data-testid="brick-shape" style={{ marginTop: 4, color: "#374151" }}>
          shape: [{d.shape.join(", ")}]
        </div>
      )}

      {typeof d.memory_mb === "number" && (
        <div data-testid="brick-memory-bar"
             style={{ marginTop: 4, color: "#374151" }}>
          mem: {d.memory_mb.toFixed(1)} MB
        </div>
      )}

      {d.side_channels_ok === false && (
        <div data-testid="brick-side-channel-warn"
             style={{ marginTop: 4, color: "#b91c1c", fontWeight: 600 }}>
          ⚠ side-channel missing
        </div>
      )}

      <Handle type="source" position={Position.Right} />
    </div>
  );
}
