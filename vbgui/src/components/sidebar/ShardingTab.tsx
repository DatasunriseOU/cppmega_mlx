import type { ShardingState, ShardingAxis } from "@/state/spec";

export interface ShardingProposalView {
  strategy_name: string;
  fits: boolean;
  estimated_per_rank_bytes: number;
  reason: string;
  /** H01: backend proposals carry the full axis_assignments + topology +
   *  compile_mode. Accept must mutate spec.sharding to use them, not
   *  just re-verify with the old spec. */
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
      <section data-testid="sharding-proposals">
        <h4 style={hd}>Proposals</h4>
        {proposals.length === 0 && <p style={{ color: "#9ca3af" }}>
          Run verify to populate proposals.
        </p>}
        {proposals.map((p, i) => (
          <div key={i} data-testid={`sharding-proposal-${i}`}
               style={{ background: "#f9fafb", border: "1px solid #e5e7eb",
                        padding: 8, marginBottom: 6, borderRadius: 4 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong>{p.strategy_name}</strong>
              <span style={{ color: p.fits ? "#10b981" : "#dc2626" }}>
                {p.fits ? "fits" : "OOM"} · {formatBytes(p.estimated_per_rank_bytes)}
              </span>
            </div>
            <div style={{ color: "#6b7280", fontSize: 11 }}>{p.reason}</div>
            <button data-testid={`sharding-accept-${i}`}
                    onClick={() => onAccept(i)}>Accept</button>
          </div>
        ))}
      </section>

      <section>
        <h4 style={hd}>Custom axes</h4>
        <table data-testid="sharding-axes"
               style={{ width: "100%", fontSize: 11 }}>
          <thead><tr><th>axis</th><th>kind</th><th>degree</th><th></th></tr></thead>
          <tbody>
            {sharding.axis_assignments.map((a, i) => (
              <tr key={i} data-testid={`sharding-axis-${i}`}>
                <td><input data-testid={`sharding-axis-${i}-name`}
                           value={a.axis_name}
                           onChange={(e) => onChange(updateAxis(
                             sharding, i, { axis_name: e.target.value }))} /></td>
                <td>
                  <select data-testid={`sharding-axis-${i}-kind`} value={a.kind}
                          onChange={(e) => onChange(updateAxis(
                            sharding, i, { kind: e.target.value }))}>
                    {PARALLEL_KINDS.map((k) => <option key={k}>{k}</option>)}
                  </select>
                </td>
                <td>
                  <input data-testid={`sharding-axis-${i}-degree`} type="number"
                         min={1} value={a.degree}
                         onChange={(e) => onChange(updateAxis(
                           sharding, i, { degree: Number(e.target.value) }))} />
                </td>
                <td>
                  <button data-testid={`sharding-axis-${i}-remove`}
                          onClick={() => onChange({
                            ...sharding,
                            axis_assignments: sharding.axis_assignments
                              .filter((_, idx) => idx !== i),
                          })}>×</button>
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
                })}>+ Add axis</button>
      </section>

      <section>
        <h4 style={hd}>Toggles</h4>
        {[
          ["master_weights_fp32", sharding.master_weights_fp32],
          ["fp8_enabled",         sharding.fp8_enabled],
          ["activation_checkpointing", sharding.activation_checkpointing],
        ].map(([k, v]) => (
          <label key={k as string} style={{ display: "block" }}>
            <input type="checkbox" data-testid={`sharding-toggle-${k}`}
                   checked={v as boolean}
                   onChange={(e) => onChange({ ...sharding,
                                               [k as string]: e.target.checked })} />
            {k}
          </label>
        ))}
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
  display: "flex", flexDirection: "column", gap: 12, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
const hd: React.CSSProperties = { margin: "0 0 6px", fontSize: 12 };
