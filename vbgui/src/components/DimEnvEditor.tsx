// Minimal dim_env editor surfaced above the canvas. Unlocks the F56b
// visual-warning flow (mismatch surfaces a banner in the canvas) and
// the F53 dimension-scaling sweep (sweeps H ∈ {64,128,256,512} by
// calling onApply per H).

import { useState } from "react";

export interface DimEnvEditorProps {
  value: Record<string, number>;
  onApply: (next: Record<string, number>) => void;
}

const EDITABLE_KEYS = ["H", "nh", "head_dim", "B", "S"] as const;
type EditableKey = typeof EDITABLE_KEYS[number];

export function DimEnvEditor({ value, onApply }: DimEnvEditorProps): JSX.Element {
  const [draft, setDraft] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const k of EDITABLE_KEYS) out[k] = String(value[k] ?? "");
    return out;
  });

  const mismatch = (() => {
    const H = Number(draft.H);
    const nh = Number(draft.nh);
    const hd = Number(draft.head_dim);
    if (!Number.isFinite(H) || !Number.isFinite(nh) ||
        !Number.isFinite(hd)) return null;
    if (nh * hd === H) return null;
    return `nh*head_dim = ${nh * hd} ≠ H = ${H}`;
  })();

  return (
    <div data-testid="dim-env-editor"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: "#f9fafb",
                  borderBottom: "1px solid #e5e7eb",
                  fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
      <strong data-testid="dim-env-editor-label">dim_env:</strong>
      {EDITABLE_KEYS.map((k: EditableKey) => (
        <label key={k} style={{ display: "inline-flex", gap: 4 }}>
          {k}
          <input
            data-testid={`dim-env-${k}`}
            type="number"
            value={draft[k]}
            onChange={(e) => setDraft({ ...draft, [k]: e.target.value })}
            style={{ width: 64 }}
          />
        </label>
      ))}
      <button
        data-testid="dim-env-apply"
        onClick={() => {
          const next: Record<string, number> = { ...value };
          for (const k of EDITABLE_KEYS) {
            const n = Number(draft[k]);
            if (Number.isFinite(n)) next[k] = n;
          }
          onApply(next);
        }}
        style={{ padding: "2px 8px" }}
      >
        Apply
      </button>
      {mismatch && (
        <span data-testid="dim-env-inline-mismatch"
              style={{ color: "#92400e", marginLeft: 8 }}>
          ⚠ {mismatch}
        </span>
      )}
    </div>
  );
}
