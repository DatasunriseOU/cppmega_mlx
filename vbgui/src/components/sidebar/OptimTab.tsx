import { useState } from "react";
import type { OptimKind, OptimState, ParamGroupState,
              ScheduleSpecState } from "@/state/spec";
import { ScheduleEditor } from "@/components/ScheduleEditor";
import { Tooltip } from "@/components/Tooltip";
import { ExplainModal } from "@/components/ExplainModal";
import { AutoGroupButton } from "@/components/AutoGroupButton";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";

export interface OptimTabProps {
  optim: OptimState;
  onApply: (next: OptimState) => void;
  /** Optional RPC client for tooltip + Apply-recommended integration.
   *  When omitted the tooltip surface is rendered but inert. */
  rpc?: RpcClient | null;
  /** Canvas state for the Auto-group button. When omitted the button
   *  is rendered but disabled. */
  graphNodes?: Node[];
  graphEdges?: Edge[];
}

const KINDS: OptimKind[] = [
  "adamw", "muon", "muon_adamw_hybrid",
  "lion", "lion8bit", "adam8bit",
  "sgd",
];

// Recommended lr per kind — surfaces in tooltip + can auto-populate
// the first group's lr when the kind changes (E7-10 will wire this).
export const RECOMMENDED_LR: Record<OptimKind, number> = {
  adamw:             3e-4,
  muon:              1e-2,
  muon_adamw_hybrid: 1e-2,
  lion:              1e-4,
  lion8bit:          1e-4,
  adam8bit:          3e-4,
  sgd:               1e-2,
};

const DEFAULT_NEW_GROUP: ParamGroupState = {
  matcher: "regex:.*", lr: 1e-4, weight_decay: 0.0,
};

export function OptimTab({
  optim, onApply, rpc, graphNodes, graphEdges,
}: OptimTabProps): JSX.Element {
  const [draft, setDraft] = useState<OptimState>(optim);
  const [expandedSchedules, setExpandedSchedules] =
    useState<Set<number>>(new Set());
  const [explainKind, setExplainKind] = useState<OptimKind | null>(null);
  const [autoGroupBanner, setAutoGroupBanner] = useState<string | null>(null);

  function applyRecommendedToKind(params: Record<string, unknown>) {
    // The first group governs lr; we copy lr/weight_decay/betas from
    // the recommended map when present.
    const lr = typeof params.lr === "number" ? params.lr : draft.groups[0].lr;
    const wd = typeof params.weight_decay === "number"
      ? params.weight_decay : draft.groups[0].weight_decay;
    const betas = Array.isArray(params.betas) && params.betas.length === 2
      ? (params.betas as [number, number])
      : draft.groups[0].betas;
    setDraft({
      ...draft,
      groups: draft.groups.map((g, i) =>
        i === 0 ? { ...g, lr, weight_decay: wd, betas } : g),
    });
  }

  function updateGroup(i: number, patch: Partial<ParamGroupState>) {
    setDraft({
      ...draft,
      groups: draft.groups.map((g, idx) => (idx === i ? { ...g, ...patch } : g)),
    });
  }
  function setSchedule(i: number, schedule: ScheduleSpecState | undefined) {
    updateGroup(i, { schedule });
  }
  function toggleSchedule(i: number) {
    setExpandedSchedules((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
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
      <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <Tooltip rpc={rpc ?? null} category="optimizer" name={draft.kind}
                 onInfoClick={() => setExplainKind(draft.kind)}
                 testId="optim-kind-tooltip">
          <span>Kind</span>
        </Tooltip>
        <select
          data-testid="optim-kind"
          value={draft.kind}
          onChange={(e) =>
            setDraft({ ...draft, kind: e.target.value as OptimKind })}
        >
          {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>

      {explainKind && (
        <ExplainModal rpc={rpc ?? null} category="optimizer"
                      name={explainKind}
                      onClose={() => setExplainKind(null)}
                      onApplyRecommended={applyRecommendedToKind} />
      )}

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
                <button data-testid={`optim-group-${i}-schedule-toggle`}
                        onClick={() => toggleSchedule(i)}
                        title="Edit LR schedule">
                  {expandedSchedules.has(i) || g.schedule ? "⏲▾" : "⏲"}
                </button>
                <button data-testid={`optim-group-${i}-remove`}
                        onClick={() => removeGroup(i)}>×</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {draft.groups.map((g, i) => (
        (expandedSchedules.has(i) || g.schedule) && (
          <ScheduleEditor key={`sched-${i}`}
                          index={i}
                          baseLr={g.lr}
                          value={g.schedule}
                          onChange={(s) => setSchedule(i, s)} />
        )
      ))}

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button data-testid="optim-add-group" onClick={addGroup}>
          + Add group
        </button>
        <AutoGroupButton
          rpc={rpc ?? null}
          optimKind={draft.kind}
          nodes={graphNodes ?? []}
          edges={graphEdges ?? []}
          onApply={(groups, banner) => {
            setDraft({ ...draft, groups });
            setAutoGroupBanner(banner);
          }}
        />
      </div>
      {autoGroupBanner && (
        <pre data-testid="optim-auto-group-banner"
             style={{ background: "#eff6ff", color: "#1e40af",
                      padding: 6, borderRadius: 4, fontSize: 11,
                      whiteSpace: "pre-wrap", marginTop: 4 }}>
          {autoGroupBanner}
        </pre>
      )}

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
