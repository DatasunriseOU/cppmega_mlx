import { Handle, Position, type NodeProps } from "@xyflow/react";
import { brickFor } from "@/lib/bricks";
import { T, accentForCategory, accentVar, CATEGORY_ICON } from "@/theme";

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

export function BrickNode({ data, id, selected }: NodeProps): JSX.Element {
  const d = data as unknown as BrickNodeData;
  const meta = brickFor(d.kind);
  const accent = accentForCategory(meta?.category);
  const glyph = meta ? CATEGORY_ICON[meta.category] : "◇";
  const categoryLabel = meta?.category.replace(/_/g, " ") ?? "node";

  return (
    <div
      role="group"
      aria-label={`brick ${d.kind}`}
      data-testid={`brick-node-${id}`}
      className={`vb-node${selected ? " vb-node-selected" : ""}`}
      style={{
        ...accentVar(accent),
        minWidth: 188,
        padding: "12px 14px 10px",
        fontFamily: T.font,
        color: T.text,
      }}
    >
      <Handle type="target" position={Position.Left} />

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="vb-chip" aria-hidden="true" style={{ fontSize: 15 }}>
          {glyph}
        </span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: 13.5, lineHeight: 1.2,
                        whiteSpace: "nowrap", overflow: "hidden",
                        textOverflow: "ellipsis" }}>
            {meta?.label ?? d.kind}
          </div>
          <div style={{ color: T.textMuted, fontSize: 11, marginTop: 2,
                        fontFamily: T.fontMono }}>
            #{d.name ?? id}
          </div>
        </div>
      </div>

      {(d.shape || typeof d.memory_mb === "number") && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6,
                      marginTop: 10 }}>
          {d.shape && (
            <span data-testid="brick-shape" style={statPill}>
              [{d.shape.join(", ")}]
            </span>
          )}
          {typeof d.memory_mb === "number" && (
            <span data-testid="brick-memory-bar" style={statPill}>
              {d.memory_mb.toFixed(1)} MB
            </span>
          )}
        </div>
      )}

      {d.side_channels_ok === false && (
        <div data-testid="brick-side-channel-warn"
             style={{ marginTop: 8, color: T.warning, fontSize: 11,
                      fontWeight: 600, display: "flex", alignItems: "center",
                      gap: 5 }}>
          <span aria-hidden="true">⚠</span> side-channel missing
        </div>
      )}

      <footer style={{ display: "flex", alignItems: "center", gap: 6,
                       marginTop: 10, paddingTop: 8,
                       borderTop: `1px solid ${T.borderSoft}`,
                       color: T.textSecondary, fontSize: 11 }}>
        <span style={{ color: accent, fontWeight: 600,
                       textTransform: "capitalize" }}>
          {categoryLabel}
        </span>
        <button
          type="button"
          title="Configure parameters"
          data-testid={`brick-node-${id}-settings-btn`}
          style={{
            marginLeft: "auto",
            background: "none",
            border: "none",
            color: T.textMuted,
            cursor: "pointer",
            fontSize: "14px",
            padding: "2px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "4px",
            transition: "all 0.15s ease",
            outline: "none",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.color = "#22d3ee"; // Neocyan highlight
            e.currentTarget.style.transform = "scale(1.2)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.color = T.textMuted;
            e.currentTarget.style.transform = "scale(1)";
          }}
        >
          ⚙
        </button>
      </footer>

      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const statPill: React.CSSProperties = {
  background: "var(--vb-surface-3)",
  border: `1px solid ${T.border}`,
  borderRadius: T.radiusSm,
  padding: "2px 7px",
  fontSize: 10.5,
  fontFamily: T.fontMono,
  color: T.textSecondary,
};
