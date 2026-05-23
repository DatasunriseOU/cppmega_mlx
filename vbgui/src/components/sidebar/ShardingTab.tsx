import type { ShardingState, ShardingAxis } from "@/state/spec";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface ShardingProposalView {
  strategy_name: string;
  fits: boolean;
  estimated_per_rank_bytes: number;
  reason: string;
  axis_assignments?: ShardingAxis[];
}

export interface ShardingTabProps {
  sharding: ShardingState;
  proposals: ShardingProposalView[];
  onAccept: (idx: number) => void;
  onChange: (s: ShardingState) => void;
}

const PARALLEL_KINDS = ["dp", "fsdp1", "fsdp2", "zero1", "zero2",
                        "tp", "sp", "ep", "pp", "pp_vpp"];

const NEW_AXIS: ShardingAxis = { axis_name: "tp", kind: "tp", degree: 2 };

function formatBytes(n: number): string {
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function ShardingTab({
  sharding, proposals, onAccept, onChange,
}: ShardingTabProps): JSX.Element {
  return (
    <div data-testid="sharding-tab" style={panel}>
      <section data-testid="sharding-proposals" style={secStyle}>
        <h4 style={hd}>Proposals</h4>
        {proposals.length === 0 && <p style={{ color: T.textMuted, fontStyle: "italic" }}>
          Run verify to populate proposals.
        </p>}
        {proposals.map((p, i) => (
          <div key={i} data-testid={`sharding-proposal-${i}`}
               style={{ background: T.surface2, border: `1px solid ${T.border}`,
                        padding: "8px 10px", marginBottom: 6, borderRadius: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong style={{ color: T.text }}>{p.strategy_name}</strong>
              <span style={{ color: p.fits ? T.success : T.danger, fontWeight: "bold" }}>
                {p.fits ? "fits" : "OOM"} · {formatBytes(p.estimated_per_rank_bytes)}
              </span>
            </div>
            <div style={{ color: T.textSecondary, fontSize: 11, marginTop: 4 }}>{p.reason}</div>
            <button data-testid={`sharding-accept-${i}`}
                    onClick={() => onAccept(i)}
                    style={acceptBtnStyle}>Accept</button>
          </div>
        ))}
      </section>

      <section style={secStyle}>
        <h4 style={hd}>Custom axes</h4>
        <table data-testid="sharding-axes"
               style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${T.border}` }}>
              <th style={th}>axis</th>
              <th style={th}>kind</th>
              <th style={th}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  degree
                  <HelpIcon topic="sharding_degree" />
                </span>
              </th>
              <th style={th}></th>
            </tr>
          </thead>
          <tbody>
            {sharding.axis_assignments.map((a, i) => (
              <tr key={i} data-testid={`sharding-axis-${i}`} style={{ borderBottom: `1px solid ${T.borderSoft}` }}>
                <td style={td}><input data-testid={`sharding-axis-${i}-name`}
                           value={a.axis_name}
                           onChange={(e) => onChange(updateAxis(
                             sharding, i, { axis_name: e.target.value }))}
                           style={tableInputStyle} /></td>
                <td style={td}>
                  <select data-testid={`sharding-axis-${i}-kind`} value={a.kind}
                          onChange={(e) => onChange(updateAxis(
                            sharding, i, { kind: e.target.value }))}
                          style={tableInputStyle}>
                    {PARALLEL_KINDS.map((k) => <option key={k}>{k}</option>)}
                  </select>
                </td>
                <td style={td}>
                  <input data-testid={`sharding-axis-${i}-degree`} type="number"
                         min={1} value={a.degree}
                         onChange={(e) => onChange(updateAxis(
                           sharding, i, { degree: Number(e.target.value) }))}
                         style={tableInputStyle} />
                </td>
                <td style={{ ...td, textAlign: "right" }}>
                  <button data-testid={`sharding-axis-${i}-remove`}
                          onClick={() => onChange({
                            ...sharding,
                            axis_assignments: sharding.axis_assignments
                              .filter((_, idx) => idx !== i),
                          })}
                          style={removeBtnStyle}>×</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button data-testid="sharding-add-axis"
                onClick={() => onChange({
                  ...sharding,
                  axis_assignments: [...sharding.axis_assignments,
                                     { ...NEW_AXIS }],
                })}
                style={addBtnStyle}>+ Add axis</button>
      </section>

      <section style={secStyle}>
        <h4 style={hd}>Toggles</h4>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            ["master_weights_fp32", sharding.master_weights_fp32],
            ["fp8_enabled",         sharding.fp8_enabled],
            ["activation_checkpointing", sharding.activation_checkpointing],
          ].map(([k, v]) => (
            <label key={k as string} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
              <input type="checkbox" data-testid={`sharding-toggle-${k}`}
                     checked={v as boolean}
                     onChange={(e) => onChange({ ...sharding,
                                                 [k as string]: e.target.checked })}
                     style={{ cursor: "pointer", width: 14, height: 14 }} />
              <span style={{ fontSize: 11, color: T.text, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>{k}</span>
            </label>
          ))}
        </div>
      </section>
    </div>
  );
}

function updateAxis(s: ShardingState, i: number,
                    patch: Partial<ShardingAxis>): ShardingState {
  return {
    ...s,
    axis_assignments: s.axis_assignments.map((a, idx) =>
      idx === i ? { ...a, ...patch } : a),
  };
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 16, padding: 16,
  fontFamily: T.font, fontSize: 12,
  background: T.surface, color: T.text,
};

const hd: React.CSSProperties = {
  margin: "0 0 8px 0",
  fontSize: 12,
  fontWeight: "bold",
  color: T.accent,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const secStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: 6,
  padding: 12,
};

const acceptBtnStyle: React.CSSProperties = {
  background: T.accent,
  color: T.accentContrast,
  border: "none",
  borderRadius: "var(--vb-radius-sm)",
  padding: "4px 8px",
  fontSize: 11,
  fontWeight: "bold",
  cursor: "pointer",
  marginTop: 6,
};

const tableInputStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "4px 6px",
  fontSize: 11,
  fontFamily: T.font,
  width: "100%",
  outline: "none",
};

const addBtnStyle: React.CSSProperties = {
  background: T.surface,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "6px 10px",
  fontSize: 11,
  fontWeight: "bold",
  cursor: "pointer",
  marginTop: 8,
};

const removeBtnStyle: React.CSSProperties = {
  background: "transparent",
  border: "none",
  color: T.danger,
  fontSize: 16,
  cursor: "pointer",
  padding: "0 4px",
};

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "4px 6px",
  color: T.textSecondary,
  fontWeight: 600,
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const td: React.CSSProperties = {
  padding: "4px 6px",
  verticalAlign: "middle",
};
