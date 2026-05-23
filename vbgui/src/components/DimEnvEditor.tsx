// Minimal dim_env editor surfaced above the canvas. Unlocks the F56b
// visual-warning flow (mismatch surfaces a banner in the canvas) and
// the F53 dimension-scaling sweep (sweeps H ∈ {64,128,256,512} by
// calling onApply per H).

import { useState } from "react";
import { HelpIcon } from "@/components/HelpIcon";

export interface DimEnvEditorProps {
  value: Record<string, number>;
  onApply: (next: Record<string, number>) => void;
}

const EDITABLE_KEYS = ["H", "nh", "head_dim", "B", "S"] as const;
type EditableKey = typeof EDITABLE_KEYS[number];

// V7-P5: full-scale presets so the architect doesn't have to type
// llama3_8b H=4096 by hand. Each preset snaps every editable key
// to a self-consistent combination (nh*head_dim == H) so it lands
// without the F56b warning.
export const DIM_ENV_PRESETS: Record<string, Record<string, number>> = {
  mini:      { B: 1, S: 64,   H: 128,  nh: 2,  nkv: 1,  head_dim: 64 },
  dev_128:   { B: 1, S: 512,  H: 128,  nh: 2,  nkv: 1,  head_dim: 64 },
  small_512: { B: 1, S: 1024, H: 512,  nh: 8,  nkv: 2,  head_dim: 64 },
  medium_1k: { B: 1, S: 2048, H: 1024, nh: 16, nkv: 4,  head_dim: 64 },
  large_2k:  { B: 1, S: 2048, H: 2048, nh: 32, nkv: 8,  head_dim: 64 },
  llama3_8b: { B: 1, S: 4096, H: 4096, nh: 32, nkv: 8,  head_dim: 128 },
  llama3_70b:{ B: 1, S: 4096, H: 8192, nh: 64, nkv: 8,  head_dim: 128 },
};

export function DimEnvEditor({ value, onApply }: DimEnvEditorProps): JSX.Element {
  const [draft, setDraft] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const k of EDITABLE_KEYS) out[k] = String(value[k] ?? "");
    return out;
  });

  const parsed = (() => {
    const H = Number(draft.H);
    const nh = Number(draft.nh);
    const hd = Number(draft.head_dim);
    if (!Number.isFinite(H) || !Number.isFinite(nh) ||
        !Number.isFinite(hd)) return null;
    return { H, nh, hd };
  })();
  const mismatch = parsed && parsed.nh * parsed.hd !== parsed.H
    ? `nh*head_dim = ${parsed.nh * parsed.hd} ≠ H = ${parsed.H}`
    : null;
  // Two single-knob fixes that snap to consistency: change H to
  // nh*head_dim, or change head_dim to H/nh when H is cleanly
  // divisible by nh. The architect can also accept the mismatch
  // consciously (the codebase supports decoupled Q via projection).
  const fixSetH = mismatch && parsed
    ? { H: parsed.nh * parsed.hd } : null;
  const fixSetHeadDim = mismatch && parsed && parsed.nh > 0
                      && parsed.H % parsed.nh === 0
    ? { head_dim: parsed.H / parsed.nh } : null;

  function applyDraft(overrides: Record<string, number> = {}) {
    const next: Record<string, number> = { ...value };
    for (const k of EDITABLE_KEYS) {
      const n = Number(draft[k]);
      if (Number.isFinite(n)) next[k] = n;
    }
    Object.assign(next, overrides);
    // Reflect the override back into the visible draft so the user
    // sees the suggestion they accepted.
    if (Object.keys(overrides).length > 0) {
      setDraft({
        ...draft,
        ...Object.fromEntries(
          Object.entries(overrides).map(([k, v]) => [k, String(v)])),
      });
    }
    onApply(next);
  }

  return (
    <div data-testid="dim-env-editor"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: "#f9fafb",
                  borderBottom: "1px solid #e5e7eb",
                  fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
      <strong data-testid="dim-env-editor-label">dim_env:</strong>
      <label style={{ display: "inline-flex", alignItems: "center",
                       gap: 4 }}>
        scale
        <select data-testid="dim-env-preset"
                defaultValue=""
                onChange={(e) => {
                  const k = e.target.value;
                  if (!k) return;
                  const preset = DIM_ENV_PRESETS[k];
                  if (preset) {
                    setDraft(Object.fromEntries(
                      Object.entries(preset).map(
                        ([kk, vv]) => [kk, String(vv)])) as never);
                    applyDraft(preset);
                  }
                  e.target.value = "";  // reset to placeholder
                }}>
          <option value="">choose…</option>
          {Object.keys(DIM_ENV_PRESETS).map((p) => (
            <option key={p} value={p}
                    data-testid={`dim-env-preset-opt-${p}`}>
              {p}
            </option>
          ))}
        </select>
      </label>
      {EDITABLE_KEYS.map((k: EditableKey) => (
        <label key={k} style={{ display: "inline-flex", alignItems: "center",
                                 gap: 4 }}>
          {k}
          <HelpIcon topic={`dim_env_${k}`} />
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
        onClick={() => applyDraft()}
        style={{ padding: "2px 8px",
                 background: mismatch ? "#fef3c7" : undefined,
                 borderColor: mismatch ? "#d97706" : undefined }}
        title={mismatch
          ? "dim_env is inconsistent — see the suggestions on the right"
          : "Push dim_env to verify"}
      >
        Apply
      </button>
      {mismatch && (
        <span data-testid="dim-env-inline-mismatch"
              style={{ color: "#92400e", marginLeft: 8,
                       display: "inline-flex", alignItems: "center" }}>
          ⚠ {mismatch}
          <HelpIcon topic="symbolic_dim_mismatch" />
        </span>
      )}
      {fixSetH && (
        <button data-testid="dim-env-fix-set-H"
                onClick={() => applyDraft(fixSetH)}
                title={`Snap H to ${fixSetH.H} (= nh*head_dim)`}
                style={{ padding: "2px 8px", background: "#dbeafe",
                         border: "1px solid #2563eb",
                         color: "#1e40af" }}>
          Snap H → {fixSetH.H}
        </button>
      )}
      {fixSetHeadDim && (
        <button data-testid="dim-env-fix-set-head_dim"
                onClick={() => applyDraft(fixSetHeadDim)}
                title={`Snap head_dim to ${fixSetHeadDim.head_dim} (= H/nh)`}
                style={{ padding: "2px 8px", background: "#dbeafe",
                         border: "1px solid #2563eb",
                         color: "#1e40af" }}>
          Snap head_dim → {fixSetHeadDim.head_dim}
        </button>
      )}
    </div>
  );
}
