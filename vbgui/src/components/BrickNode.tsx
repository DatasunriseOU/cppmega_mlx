import { Handle, Position, type NodeProps } from "@xyflow/react";
import { brickFor } from "@/lib/bricks";
import { T, accentForCategory, accentVar, CATEGORY_ICON } from "@/theme";
import { HelpIcon } from "@/components/HelpIcon";

export interface BrickNodeData {
  kind: string;
  name?: string;
  params?: Record<string, unknown>;
  shape?: number[];
  memory_mb?: number;
  // Side-channel availability hint surfaced from the resolver. When set
  // to false, the node renders a small "missing" badge.
  side_channels_ok?: boolean;

  // Debugger additions
  debuggerMode?: boolean;
  isActiveNode?: boolean;
  isWeightUpdated?: boolean;
  gradNorm?: number;
}

export function BrickNode({ data, id, selected }: NodeProps): JSX.Element {
  const d = data as unknown as BrickNodeData;
  const targetPosition = (data as any)?.targetPosition ?? Position.Left;
  const sourcePosition = (data as any)?.sourcePosition ?? Position.Right;
  const meta = brickFor(d.kind);
  const accent = accentForCategory(meta?.category);
  const glyph = meta ? CATEGORY_ICON[meta.category] : "◇";
  const categoryLabel = meta?.category.replace(/_/g, " ") ?? "node";

  const isAttention = meta?.category === "sdpa_attention";
  const isMlp = d.kind === "mlp" || d.kind === "linear_bridge";
  const isMoe = meta?.category === "moe";
  const isEmbed = d.kind === "abs_pos_embed" || d.kind === "per_layer_embed";

  const weightPulseStyle: React.CSSProperties = d.isWeightUpdated
    ? {
        border: "2px solid var(--vb-warning)",
        boxShadow: "0 0 25px rgba(251, 191, 36, 0.6)",
      }
    : {};

  const debuggerGlowStyle: React.CSSProperties = d.debuggerMode && d.isActiveNode
    ? {
        border: `2px solid ${accent}`,
        boxShadow: `0 0 16px ${accent}80`,
      }
    : {};

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
        ...weightPulseStyle,
        ...debuggerGlowStyle,
      }}
    >
      <Handle type="target" position={targetPosition} />

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

      {(d.shape || typeof d.memory_mb === "number" || typeof d.gradNorm === "number") && (
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
          {typeof d.gradNorm === "number" && (
            <span
              data-testid="brick-grad-norm"
              style={{
                ...statPill,
                color: "var(--vb-success)",
                borderColor: "rgba(52, 211, 153, 0.4)",
                boxShadow: "0 0 8px rgba(52, 211, 153, 0.2)",
                fontWeight: 600,
              }}
            >
              ‖g‖: {d.gradNorm.toFixed(4)}
            </span>
          )}
        </div>
      )}

      {/* RENDER DIAGRAMMATIC BLOCK PARAMETERS IN SIMULATION MODE */}
      {d.debuggerMode && (
        <div style={{ marginTop: 10, borderTop: `1px solid ${T.borderSoft}`, paddingTop: 10 }}>
          {isAttention && (
            <div>
              <div style={{ fontSize: 9, color: T.textMuted, marginBottom: 4, fontWeight: 600, textTransform: "uppercase" }}>
                Attention Heads (Q, K, V tracks)
              </div>
              <div style={{ display: "flex", gap: 3, height: 24, background: "var(--vb-surface-3)", padding: 4, borderRadius: "var(--vb-radius-sm)", border: "1px solid var(--vb-border)" }}>
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} style={{
                    flex: 1,
                    borderRadius: 2,
                    background: d.isActiveNode ? accent : "var(--vb-border-strong)",
                    opacity: d.isActiveNode ? (0.3 + (i % 3) * 0.25) : 0.4,
                  }} />
                ))}
              </div>
            </div>
          )}

          {isMlp && (
            <div>
              <div style={{ fontSize: 9, color: T.textMuted, marginBottom: 4, fontWeight: 600, textTransform: "uppercase" }}>
                MLP Widening Hourglass
              </div>
              <div style={{ display: "flex", justifyContent: "center", background: "var(--vb-surface-3)", padding: "4px 8px", borderRadius: "var(--vb-radius-sm)", border: "1px solid var(--vb-border)" }}>
                <svg width="120" height="24" viewBox="0 0 120 24" fill="none">
                  <polygon
                    points="5,3 25,0 25,24 5,21"
                    fill={d.isActiveNode ? "var(--vb-accent-soft)" : "var(--vb-surface-2)"}
                    stroke={d.isActiveNode ? "var(--vb-accent)" : "var(--vb-border)"}
                    strokeWidth="1"
                  />
                  <polygon
                    points="25,0 95,4 95,20 25,24"
                    fill={d.isActiveNode ? "rgba(52, 211, 153, 0.12)" : "var(--vb-surface-2)"}
                    stroke={d.isActiveNode ? "var(--vb-success)" : "var(--vb-border)"}
                    strokeWidth="1"
                  />
                  <polygon
                    points="95,4 115,3 115,21 95,20"
                    fill={d.isActiveNode ? "var(--vb-accent-soft)" : "var(--vb-surface-2)"}
                    stroke={d.isActiveNode ? "var(--vb-accent)" : "var(--vb-border)"}
                    strokeWidth="1"
                  />
                </svg>
              </div>
            </div>
          )}

          {isMoe && (
            <div>
              <div style={{ fontSize: 9, color: T.textMuted, marginBottom: 4, fontWeight: 600, textTransform: "uppercase" }}>
                Expert Routing (Top-2 Active)
              </div>
              <div style={{ display: "flex", gap: 3 }}>
                {["E1", "E2", "E3", "E4"].map((exp, i) => {
                  const isExpActive = !!d.isActiveNode && (i === 0 || i === 2);
                  return (
                    <div key={exp} style={{
                      flex: 1,
                      height: 20,
                      borderRadius: "var(--vb-radius-sm)",
                      background: isExpActive ? "rgba(245, 158, 11, 0.16)" : "var(--vb-surface-3)",
                      border: `1px solid ${isExpActive ? "var(--vb-cat-moe)" : "var(--vb-border)"}`,
                      color: isExpActive ? "var(--vb-cat-moe)" : T.textMuted,
                      fontSize: 8.5,
                      fontWeight: 700,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}>
                      {exp}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {isEmbed && (
            <div>
              <div style={{ fontSize: 9, color: T.textMuted, marginBottom: 4, fontWeight: 600, textTransform: "uppercase" }}>
                Positional Wave Grid
              </div>
              <div style={{ display: "flex", gap: 3, height: 24, background: "var(--vb-surface-3)", padding: 4, borderRadius: "var(--vb-radius-sm)", border: "1px solid var(--vb-border)" }}>
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} style={{
                    flex: 1,
                    borderRadius: 1,
                    background: d.isActiveNode ? accent : "var(--vb-border-strong)",
                    opacity: d.isActiveNode ? 0.8 : 0.3,
                    transform: d.isActiveNode ? `scaleY(${0.6 + Math.sin(i) * 0.4})` : undefined,
                  }} />
                ))}
              </div>
            </div>
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
                       textTransform: "capitalize", display: "inline-flex",
                       alignItems: "center", gap: 4 }}>
          {categoryLabel}
          <HelpIcon topic={`brick_${d.kind}`} />
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

      <Handle type="source" position={sourcePosition} />
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
