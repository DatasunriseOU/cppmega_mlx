import { Handle, Position, type NodeProps } from "@xyflow/react";
import { adapterFor } from "@/lib/bricks";
import { T, accentVar } from "@/theme";

export interface AdapterNodeData {
  kind: string;
  ghost?: boolean;
}

export function AdapterNode({ data, id, selected }: NodeProps): JSX.Element {
  const d = data as unknown as AdapterNodeData;
  const meta = adapterFor(d.kind);
  const accent = T.accent; // adapters share the cyan "operation" accent

  return (
    <div
      role="group"
      aria-label={`adapter ${d.kind}`}
      data-testid={`adapter-node-${id}`}
      className={`vb-node${selected ? " vb-node-selected" : ""}`}
      style={{
        ...accentVar(accent),
        minWidth: 148,
        padding: "9px 12px",
        fontFamily: T.font,
        fontSize: 11.5,
        color: T.text,
        background: d.ghost ? "rgba(23, 26, 40, 0.55)" : undefined,
        borderStyle: "dashed",
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span aria-hidden="true"
              style={{ color: accent, fontSize: 13 }}>⇄</span>
        <div>
          <div style={{ fontWeight: 600 }}>{meta?.label ?? d.kind}</div>
          <div style={{ color: T.textMuted, fontSize: 10, marginTop: 1,
                        textTransform: "uppercase", letterSpacing: 0.4 }}>
            adapter
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
