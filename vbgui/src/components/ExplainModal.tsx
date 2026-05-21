// ExplainModal — full ExplainEntry display with "Apply recommended"
// button for optimizer/schedule entries.

import { useCatalog } from "@/hooks/useCatalog";
import type { RpcClient } from "@/lib/rpc";

export interface ExplainModalProps {
  rpc: RpcClient | null;
  category: string;
  name: string;
  onClose: () => void;
  onApplyRecommended?: (params: Record<string, unknown>) => void;
}

const SECTION: React.CSSProperties = { marginBottom: 10 };
const LABEL: React.CSSProperties = {
  color: "#6b7280", fontSize: 11, textTransform: "uppercase",
  letterSpacing: 0.5, marginBottom: 2,
};

function paramsTable(params: Record<string, unknown>): JSX.Element {
  const entries = Object.entries(params);
  if (entries.length === 0) {
    return <em style={{ color: "#9ca3af" }}>none specified</em>;
  }
  return (
    <table style={{ fontSize: 12, borderCollapse: "collapse" }}>
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <td style={{ padding: "1px 8px 1px 0", color: "#374151",
                          fontFamily: "monospace" }}>{k}</td>
            <td style={{ padding: "1px 0", color: "#111827",
                          fontFamily: "monospace" }}>
              {typeof v === "object" ? JSON.stringify(v) : String(v)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ExplainModal({
  rpc, category, name, onClose, onApplyRecommended,
}: ExplainModalProps): JSX.Element {
  const { entry, loading, error } = useCatalog(rpc, category, name, true);

  return (
    <div data-testid="explain-modal-backdrop"
         role="dialog" aria-modal="true"
         onClick={onClose}
         style={{
           position: "fixed", inset: 0, background: "rgba(15,23,42,0.45)",
           display: "flex", alignItems: "center", justifyContent: "center",
           zIndex: 100, fontFamily: "system-ui, sans-serif",
         }}>
      <div data-testid="explain-modal"
           onClick={(e) => e.stopPropagation()}
           style={{
             background: "white", borderRadius: 6, padding: 20,
             width: 560, maxHeight: "85vh", overflowY: "auto",
           }}>
        <header style={{ display: "flex", justifyContent: "space-between",
                          alignItems: "center", marginBottom: 12 }}>
          <h3 data-testid="explain-modal-title" style={{ margin: 0,
                                                          fontSize: 16 }}>
            {entry?.name ?? name}
            <span style={{ marginLeft: 8, color: "#6b7280",
                            fontSize: 11, fontWeight: 400 }}>
              [{category}]
            </span>
          </h3>
          <button data-testid="explain-modal-close" onClick={onClose}>×</button>
        </header>

        {loading && <div>Loading…</div>}
        {error && (
          <div data-testid="explain-modal-error"
               style={{ color: "#991b1b", background: "#fee2e2",
                         padding: 8, borderRadius: 4, fontSize: 12 }}>
            {error}
          </div>
        )}
        {entry && (
          <>
            <section style={SECTION}>
              <div style={LABEL}>Summary</div>
              <div data-testid="explain-modal-summary">{entry.summary}</div>
            </section>

            <section style={SECTION}>
              <div style={LABEL}>When to use</div>
              <div data-testid="explain-modal-when-to-use">
                {entry.when_to_use}
              </div>
            </section>

            <section style={SECTION}>
              <div style={LABEL}>When to avoid</div>
              <div data-testid="explain-modal-when-to-avoid">
                {entry.when_to_avoid}
              </div>
            </section>

            <section style={SECTION}>
              <div style={LABEL}>Recommended params</div>
              <div data-testid="explain-modal-recommended">
                {paramsTable(entry.recommended_params)}
              </div>
            </section>

            {entry.gotchas.length > 0 && (
              <section style={SECTION}>
                <div style={LABEL}>Gotchas</div>
                <ul data-testid="explain-modal-gotchas"
                    style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
                  {entry.gotchas.map((g, i) => (
                    <li key={i} style={{ color: "#b45309" }}>{g}</li>
                  ))}
                </ul>
              </section>
            )}

            {entry.paper_url && (
              <section style={SECTION}>
                <div style={LABEL}>Reference</div>
                <a data-testid="explain-modal-paper"
                   href={entry.paper_url} target="_blank"
                   rel="noopener noreferrer"
                   style={{ color: "#2563eb" }}>
                  {entry.paper_ref ?? entry.paper_url}
                </a>
              </section>
            )}
            {!entry.paper_url && entry.paper_ref && (
              <section style={SECTION}>
                <div style={LABEL}>Reference</div>
                <span style={{ color: "#374151" }}>{entry.paper_ref}</span>
              </section>
            )}

            {onApplyRecommended &&
             Object.keys(entry.recommended_params).length > 0 && (
              <div style={{ marginTop: 16 }}>
                <button
                  data-testid="explain-modal-apply"
                  onClick={() => {
                    onApplyRecommended(entry.recommended_params);
                    onClose();
                  }}
                  style={{ background: "#2563eb", color: "white",
                           border: "none", padding: "6px 12px",
                           borderRadius: 4, cursor: "pointer" }}>
                  Apply recommended params
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
