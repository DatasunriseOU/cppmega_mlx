import type { GotchaState } from "@/state/spec";

export interface GotchasTabProps {
  gotchas: GotchaState[];
  onAutoFix?: (id: string) => void;
}

const COLOR: Record<GotchaState["severity"], string> = {
  error:   "#dc2626",
  warning: "#d97706",
  info:    "#2563eb",
};

// V7-H01: extended auto-fix coverage for the 6 most common
// validation gotchas. The App handler dispatches the corresponding
// spec mutation when the user clicks the inline Apply-fix button.
const AUTO_FIXABLE: Set<string> = new Set([
  "fsdp2_whole_compile",
  "megatron_tp_whole_compile",
  "missing_edge",
  "dim_mismatch",
  "unknown_brick",
  "bad_dtype_combo",
  "schedule_out_of_range",
  "tokenizer_mismatch",
]);

const FIX_LABELS: Record<string, string> = {
  fsdp2_whole_compile:      "Switch compile_mode → regional",
  megatron_tp_whole_compile: "Switch compile_mode → regional",
  missing_edge:             "Insert missing edge",
  dim_mismatch:             "Adjust hidden_size to nearest valid",
  unknown_brick:            "Remove unknown brick",
  bad_dtype_combo:          "Reset dtype to bf16 master",
  schedule_out_of_range:    "Clamp schedule to valid range",
  tokenizer_mismatch:       "Pick MATRIX-compatible tokenizer",
};

function groupBySeverity(gs: GotchaState[]): Record<string, GotchaState[]> {
  const out: Record<string, GotchaState[]> = { error: [], warning: [], info: [] };
  for (const g of gs) (out[g.severity] ??= []).push(g);
  return out;
}

export function GotchasTab({ gotchas, onAutoFix }: GotchasTabProps): JSX.Element {
  const grouped = groupBySeverity(gotchas);
  return (
    <div data-testid="gotchas-tab" style={panel}>
      {gotchas.length === 0 && (
        <p style={{ color: "#9ca3af" }}>No gotchas fired.</p>
      )}
      {(["error", "warning", "info"] as const).map((sev) =>
        grouped[sev].length > 0 ? (
          <section key={sev} data-testid={`gotchas-${sev}`}>
            <h4 style={{ margin: "0 0 4px", color: COLOR[sev] }}>
              {sev.toUpperCase()}
            </h4>
            {grouped[sev].map((g) => (
              <div key={g.id} data-testid={`gotcha-${g.id}`}
                   style={{ background: "#f9fafb",
                            borderLeft: `4px solid ${COLOR[sev]}`,
                            padding: "6px 8px", marginBottom: 4,
                            borderRadius: 3 }}>
                <div style={{ fontWeight: 600 }}>{g.id}</div>
                <div style={{ color: "#374151" }}>{g.message}</div>
                {g.reference && (
                  <a data-testid={`gotcha-${g.id}-ref`}
                     href={g.reference}
                     target="_blank" rel="noreferrer"
                     style={{ color: "#2563eb", fontSize: 11 }}>
                    {g.reference}
                  </a>
                )}
                {onAutoFix && AUTO_FIXABLE.has(g.id) && (
                  <button data-testid={`gotcha-${g.id}-autofix`}
                          onClick={() => onAutoFix(g.id)}
                          title={FIX_LABELS[g.id] ?? "Auto-fix"}>
                    {FIX_LABELS[g.id] ?? "Auto-fix"}
                  </button>
                )}
              </div>
            ))}
          </section>
        ) : null,
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 12, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
