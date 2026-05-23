import type { RewriterName, RewriterState } from "@/state/spec";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface RewritersTabProps {
  rewriters: RewriterState[];
  onAdd: (r: RewriterState) => void;
  onRemove: (i: number) => void;
  onReorder: (from: number, to: number) => void;
  onApply?: () => void;
}

const ADDABLE: RewriterName[] = ["MTPRewriter", "IFIMRewriter", "MHCRewriter"];

const DEFAULT_PARAMS: Record<RewriterName, Record<string, number>> = {
  MTPRewriter: { k: 2, beta: 0.6 },
  IFIMRewriter: { lambda_fim: 0.1 },
  MHCRewriter: { N: 2, lambda_mhc: 0.05 },
};

export function RewritersTab({
  rewriters, onAdd, onRemove, onReorder, onApply,
}: RewritersTabProps): JSX.Element {
  return (
    <div data-testid="rewriters-tab" style={{ ...panel, background: T.surface, color: T.text }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 4 }}>
        {ADDABLE.map((name) => (
          <button key={name} data-testid={`rewriter-add-${name}`}
                  onClick={() => onAdd({ name, params: { ...DEFAULT_PARAMS[name] } })}
                  style={{
                    background: T.surface3,
                    border: `1px solid ${T.border}`,
                    color: T.text,
                    padding: "6px 10px",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "11px",
                    fontWeight: 600,
                    transition: "all 0.15s ease",
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.background = T.surface2; }}
                  onMouseOut={(e) => { e.currentTarget.style.background = T.surface3; }}>
            + {name}
          </button>
        ))}
      </div>

      <ul data-testid="rewriter-chain"
          style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        {rewriters.map((r, i) => (
          <li key={i} data-testid={`rewriter-chip-${i}`}
              style={{ background: T.surface3,
                       border: `1px solid ${T.border}`,
                       borderRadius: 6,
                       color: T.text,
                       padding: "6px 10px",
                       display: "flex", gap: 6, alignItems: "center" }}>
            <button data-testid={`rewriter-up-${i}`}
                    disabled={i === 0}
                    onClick={() => onReorder(i, i - 1)}
                    style={{
                      background: "rgba(255, 255, 255, 0.05)",
                      border: `1px solid ${T.border}`,
                      color: i === 0 ? T.textMuted : T.textSecondary,
                      cursor: i === 0 ? "default" : "pointer",
                      borderRadius: 4,
                      padding: "2px 6px",
                      fontSize: 10,
                    }}>▲</button>
            <button data-testid={`rewriter-down-${i}`}
                    disabled={i === rewriters.length - 1}
                    onClick={() => onReorder(i, i + 1)}
                    style={{
                      background: "rgba(255, 255, 255, 0.05)",
                      border: `1px solid ${T.border}`,
                      color: i === rewriters.length - 1 ? T.textMuted : T.textSecondary,
                      cursor: i === rewriters.length - 1 ? "default" : "pointer",
                      borderRadius: 4,
                      padding: "2px 6px",
                      fontSize: 10,
                    }}>▼</button>
            <span style={{ flex: 1, fontFamily: "monospace", fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "inline-flex", alignItems: "center", gap: 4 }}>
              {r.name}({Object.entries(r.params)
                .map(([k, v]) => `${k}=${v}`).join(", ")})
              <HelpIcon topic={r.name === "MTPRewriter" ? "rewriter_k" : r.name === "IFIMRewriter" ? "rewriter_lambda" : "rewriter_window"} />
            </span>
            <button data-testid={`rewriter-remove-${i}`}
                    onClick={() => onRemove(i)}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: T.textSecondary,
                      cursor: "pointer",
                      fontSize: 14,
                      lineHeight: 1,
                      padding: "0 4px",
                    }}
                    onMouseOver={(e) => { e.currentTarget.style.color = T.danger; }}
                    onMouseOut={(e) => { e.currentTarget.style.color = T.textSecondary; }}>×</button>
          </li>
        ))}
      </ul>

      {onApply && (
        <button data-testid="rewriter-apply" onClick={onApply}
                style={{
                  background: T.accent,
                  color: T.accentContrast,
                  border: "none",
                  borderRadius: "6px",
                  padding: "8px 16px",
                  fontWeight: "bold",
                  cursor: "pointer",
                  fontSize: 12,
                  marginTop: 6,
                  transition: "all 0.15s ease",
                }}
                onMouseOver={(e) => { e.currentTarget.style.filter = "brightness(1.1)"; }}
                onMouseOut={(e) => { e.currentTarget.style.filter = "none"; }}>
          Apply chain
        </button>
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 12, padding: 16,
  fontFamily: T.font, fontSize: 12,
};
