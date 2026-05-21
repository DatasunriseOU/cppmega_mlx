import { Handle, Position, type NodeProps } from "@xyflow/react";
import { adapterFor } from "@/lib/bricks";

export interface AdapterNodeData {
  kind: string;
  ghost?: boolean;
}

export function AdapterNode({ data, id }: NodeProps): JSX.Element {
  const d = data as unknown as AdapterNodeData;
  const meta = adapterFor(d.kind);
  return (
    <div
      role="group"
      aria-label={`adapter ${d.kind}`}
      data-testid={`adapter-node-${id}`}
      style={{
        border: "2px dashed #9ca3af",
        background: d.ghost ? "rgba(255,255,255,0.5)" : "#f9fafb",
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 140,
        fontFamily: "system-ui, sans-serif",
        fontSize: 11,
        fontStyle: "italic",
        color: "#374151",
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontWeight: 600 }}>{meta?.label ?? d.kind}</div>
      <div style={{ color: "#9ca3af", fontSize: 10 }}>adapter</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
