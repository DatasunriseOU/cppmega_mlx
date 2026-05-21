import type { RewriterName, RewriterState } from "@/state/spec";

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
    <div data-testid="rewriters-tab" style={panel}>
      <div>
        {ADDABLE.map((name) => (
          <button key={name} data-testid={`rewriter-add-${name}`}
                  onClick={() => onAdd({ name, params: { ...DEFAULT_PARAMS[name] } })}>
            + {name}
          </button>
        ))}
      </div>

      <ul data-testid="rewriter-chain"
          style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {rewriters.map((r, i) => (
          <li key={i} data-testid={`rewriter-chip-${i}`}
              style={{ background: "#eef2ff", borderRadius: 4,
                       padding: "4px 8px", margin: "4px 0",
                       display: "flex", gap: 6, alignItems: "center" }}>
            <button data-testid={`rewriter-up-${i}`}
                    disabled={i === 0}
                    onClick={() => onReorder(i, i - 1)}>▲</button>
            <button data-testid={`rewriter-down-${i}`}
                    disabled={i === rewriters.length - 1}
                    onClick={() => onReorder(i, i + 1)}>▼</button>
            <span style={{ flex: 1 }}>
              {r.name}({Object.entries(r.params)
                .map(([k, v]) => `${k}=${v}`).join(", ")})
            </span>
            <button data-testid={`rewriter-remove-${i}`}
                    onClick={() => onRemove(i)}>×</button>
          </li>
        ))}
      </ul>

      {onApply && (
        <button data-testid="rewriter-apply" onClick={onApply}>
          Apply chain
        </button>
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 8, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
