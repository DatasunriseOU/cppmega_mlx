import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useState } from "react";
import { T } from "@/theme";

export interface BlockGroupNodeData {
  label: string;
  repeats: number;
  block_specs?: any[];
  onUnpack?: (groupId: string, count: number) => void;
}

export function BlockGroupNode({ data, id, selected }: NodeProps): JSX.Element {
  const d = data as unknown as BlockGroupNodeData;
  const targetPosition = (data as any)?.targetPosition ?? Position.Left;
  const sourcePosition = (data as any)?.sourcePosition ?? Position.Right;
  const [showUnpackModal, setShowUnpackModal] = useState(false);
  const [unpackCount, setUnpackCount] = useState(Math.min(4, d.repeats));

  const handleUnpackClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowUnpackModal(true);
  };

  const handleConfirmUnpack = (e: React.MouseEvent, count: number) => {
    e.stopPropagation();
    if (d.onUnpack) {
      d.onUnpack(id, count);
    }
    setShowUnpackModal(false);
  };

  const handleCancelUnpack = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowUnpackModal(false);
  };

  // Glassmorphic neon-purple styled card for block groups
  return (
    <div
      role="group"
      aria-label={`folded block group ${d.label}`}
      data-testid={`block-group-node-${id}`}
      className={`vb-node${selected ? " vb-node-selected" : ""}`}
      style={{
        ["--vb-node-accent" as any]: "var(--vb-cat-ssm)", // Purple/ssm neon accent color
        minWidth: 230,
        padding: "16px 18px",
        fontFamily: T.font,
        color: T.text,
        background: "rgba(31, 35, 51, 0.45)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(168, 85, 247, 0.25)",
        borderRadius: "var(--vb-radius-lg)",
        boxShadow: "0 8px 32px 0 rgba(168, 85, 247, 0.15), 0 2px 10px rgba(0, 0, 0, 0.35)",
        position: "relative",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
      }}
    >
      <Handle
        type="target"
        position={targetPosition}
        style={{ background: "var(--vb-cat-ssm)", width: 7, height: 7, border: "none" }}
      />

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              fontSize: 18,
              color: "var(--vb-cat-ssm)",
              background: "rgba(168, 85, 247, 0.15)",
              padding: "4px 8px",
              borderRadius: "var(--vb-radius-md)",
              fontWeight: "bold",
            }}
          >
            ⧉
          </span>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 10, color: "var(--vb-cat-ssm)", fontWeight: "bold", textTransform: "uppercase", letterSpacing: 0.5 }}>
              Folded Block Group
            </div>
            <div style={{ fontWeight: 600, fontSize: 14, lineHeight: 1.2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", marginTop: 2 }}>
              {d.label}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: 10, marginTop: 4 }}>
          <span style={{ fontSize: 12, color: T.textSecondary, fontFamily: T.fontMono }}>
            × {d.repeats} layers
          </span>
          
          <button
            onClick={handleUnpackClick}
            data-testid={`unpack-btn-${id}`}
            style={{
              padding: "4px 10px",
              background: "rgba(168, 85, 247, 0.15)",
              border: "1px solid var(--vb-cat-ssm)",
              borderRadius: "var(--vb-radius-sm)",
              color: "var(--vb-text)",
              fontSize: 10,
              fontWeight: "bold",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "var(--vb-cat-ssm)";
              e.currentTarget.style.color = "#0f172a";
              e.currentTarget.style.boxShadow = "0 0 12px rgba(168, 85, 247, 0.6)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(168, 85, 247, 0.15)";
              e.currentTarget.style.color = "var(--vb-text)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            ✦ Unpack
          </button>
        </div>
      </div>

      {/* Beautiful Glassmorphic Inline Unpack Modal/Slider */}
      {showUnpackModal && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 50,
            background: "rgba(10, 11, 20, 0.92)",
            borderRadius: "inherit",
            padding: 12,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            animation: "fadeIn 0.15s ease-out",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <style>{`
            @keyframes fadeIn {
              from { opacity: 0; transform: scale(0.95); }
              to { opacity: 1; transform: scale(1); }
            }
          `}</style>

          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontSize: 11, fontWeight: "bold", color: "var(--vb-cat-ssm)" }}>Unpack Layers</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12, marginTop: 4 }}>
              <span style={{ color: T.textSecondary }}>Count to unpack:</span>
              <span style={{ fontWeight: "bold", color: "var(--vb-accent)", fontFamily: T.fontMono }}>{unpackCount} / {d.repeats}</span>
            </div>
            
            <input
              type="range"
              min={1}
              max={d.repeats}
              value={unpackCount}
              onChange={(e) => setUnpackCount(parseInt(e.target.value))}
              style={{
                width: "100%",
                marginTop: 6,
                accentColor: "var(--vb-cat-ssm)",
                cursor: "pointer",
              }}
            />
          </div>

          <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
            <button
              onClick={(e) => handleConfirmUnpack(e, unpackCount)}
              data-testid={`confirm-unpack-n-${id}`}
              style={{
                flex: 1,
                padding: "4px 2px",
                fontSize: 9,
                fontWeight: "bold",
                background: "var(--vb-accent-soft)",
                border: "1px solid var(--vb-accent)",
                color: "var(--vb-text)",
                borderRadius: 4,
              }}
              onMouseOver={(e) => { e.currentTarget.style.background = "var(--vb-accent)"; e.currentTarget.style.color = "#0f172a"; }}
              onMouseOut={(e) => { e.currentTarget.style.background = "var(--vb-accent-soft)"; e.currentTarget.style.color = "var(--vb-text)"; }}
            >
              Unpack {unpackCount}
            </button>
            <button
              onClick={(e) => handleConfirmUnpack(e, d.repeats)}
              data-testid={`confirm-unpack-all-${id}`}
              style={{
                flex: 1,
                padding: "4px 2px",
                fontSize: 9,
                fontWeight: "bold",
                background: "rgba(168, 85, 247, 0.15)",
                border: "1px solid var(--vb-cat-ssm)",
                color: "var(--vb-text)",
                borderRadius: 4,
              }}
              onMouseOver={(e) => { e.currentTarget.style.background = "var(--vb-cat-ssm)"; e.currentTarget.style.color = "#0f172a"; }}
              onMouseOut={(e) => { e.currentTarget.style.background = "rgba(168, 85, 247, 0.15)"; e.currentTarget.style.color = "var(--vb-text)"; }}
            >
              Unpack All
            </button>
            <button
              onClick={handleCancelUnpack}
              style={{
                padding: "4px 6px",
                fontSize: 9,
                fontWeight: "bold",
                background: "transparent",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                color: T.textSecondary,
                borderRadius: 4,
              }}
              onMouseOver={(e) => e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.3)"}
              onMouseOut={(e) => e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.1)"}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <Handle
        type="source"
        position={sourcePosition}
        style={{ background: "var(--vb-cat-ssm)", width: 7, height: 7, border: "none" }}
      />
    </div>
  );
}
