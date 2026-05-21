import { useState } from "react";
import type { OptimKind, OptimState, ParamGroupState } from "@/state/spec";

export interface OptimTabProps {
  optim: OptimState;
  onApply: (next: OptimState) => void;
}

const KINDS: OptimKind[] = ["adamw", "muon", "muon_adamw_hybrid", "sgd"];

const DEFAULT_NEW_GROUP: ParamGroupState = {
  matcher: "regex:.*", lr: 1e-4, weight_decay: 0.0,
};

export function OptimTab({ optim, onApply }: OptimTabProps): JSX.Element {
  const [draft, setDraft] = useState<OptimState>(optim);

  function updateGroup(i: number, patch: Partial<ParamGroupState>) {
    setDraft({
      ...draft,
      groups: draft.groups.map((g, idx) => (idx === i ? { ...g, ...patch } : g)),
    });
  }
  function addGroup() {
    setDraft({ ...draft, groups: [...draft.groups, { ...DEFAULT_NEW_GROUP }] });
  }
  function removeGroup(i: number) {
    setDraft({ ...draft, groups: draft.groups.filter((_, idx) => idx !== i) });
  }

  return (
    <div data-testid="optim-tab" style={panel}>
      <label>Kind
        <select
          data-testid="optim-kind"
          value={draft.kind}
          onChange={(e) =>
            setDraft({ ...draft, kind: e.target.value as OptimKind })}
        >
          {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>

      <table data-testid="optim-groups"
             style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th>matcher</th><th>lr</th><th>wd</th><th></th>
          </tr>
        </thead>
        <tbody>
          {draft.groups.map((g, i) => (
            <tr key={i} data-testid={`optim-group-${i}`}>
              <td>
                <input data-testid={`optim-group-${i}-matcher`}
                       value={g.matcher}
                       onChange={(e) => updateGroup(i, { matcher: e.target.value })} />
              </td>
              <td>
                <input data-testid={`optim-group-${i}-lr`} type="number"
                       step={1e-5} value={g.lr}
                       onChange={(e) => updateGroup(i, { lr: Number(e.target.value) })} />
              </td>
              <td>
                <input data-testid={`optim-group-${i}-wd`} type="number"
                       step={0.01} value={g.weight_decay}
                       onChange={(e) => updateGroup(i, {
                         weight_decay: Number(e.target.value),
                       })} />
              </td>
              <td>
                <button data-testid={`optim-group-${i}-remove`}
                        onClick={() => removeGroup(i)}>×</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <button data-testid="optim-add-group" onClick={addGroup}>+ Add group</button>

      <label>grad_clip_norm
        <input data-testid="optim-clip" type="number" step={0.1}
               value={draft.grad_clip_norm}
               onChange={(e) =>
                 setDraft({ ...draft,
                            grad_clip_norm: Number(e.target.value) })} />
      </label>

      <label>
        <input data-testid="optim-mp" type="checkbox"
               checked={draft.mixed_precision}
               onChange={(e) =>
                 setDraft({ ...draft, mixed_precision: e.target.checked })} />
        mixed_precision
      </label>

      <button data-testid="optim-apply" onClick={() => onApply(draft)}>
        Apply
      </button>
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 8, padding: 12,
  fontFamily: "system-ui, sans-serif", fontSize: 12,
};
