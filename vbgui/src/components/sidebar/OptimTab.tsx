import { useState, useEffect } from "react";
import type { OptimKind, OptimState, ParamGroupState,
              ScheduleSpecState } from "@/state/spec";
import { ScheduleEditor } from "@/components/ScheduleEditor";
import { Tooltip } from "@/components/Tooltip";
import { ExplainModal } from "@/components/ExplainModal";
import { AutoGroupButton } from "@/components/AutoGroupButton";
import { HelpIcon } from "@/components/HelpIcon";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";
import { T } from "@/theme";

export interface OptimTabProps {
  optim: OptimState;
  onApply: (next: OptimState) => void;
  rpc?: RpcClient | null;
  graphNodes?: Node[];
  graphEdges?: Edge[];
  lastRunScheduleKind?: string | null;
}

const KINDS: OptimKind[] = [
  "adamw", "muon", "muon_adamw_hybrid",
  "lion", "lion8bit", "adam8bit",
  "sgd",
];

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
  lastRunScheduleKind = null,
}: OptimTabProps): JSX.Element {
  const [draft, setDraft] = useState<OptimState>(optim);
  const [expandedSchedules, setExpandedSchedules] =
    useState<Set<number>>(new Set());
  const [explainKind, setExplainKind] = useState<OptimKind | null>(null);
  const [autoGroupBanner, setAutoGroupBanner] = useState<string | null>(null);

  useEffect(() => {
    setDraft(optim);
  }, [optim]);

  function applyRecommendedToKind(params: Record<string, unknown>) {
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
      <label style={labelStyle}>
        <span style={labelTitle}>
          <Tooltip rpc={rpc ?? null} category="optimizer" name={draft.kind}
                   onInfoClick={() => setExplainKind(draft.kind)}
                   testId="optim-kind-tooltip">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              Kind
              <HelpIcon topic="optim_kind" />
            </span>
          </Tooltip>
        </span>
        <select
          data-testid="optim-kind"
          value={draft.kind}
          onChange={(e) =>
            setDraft({ ...draft, kind: e.target.value as OptimKind })}
          style={inputStyle}
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
             style={{ width: "100%", fontSize: 11, borderCollapse: "collapse", marginTop: 8 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${T.border}` }}>
            <th style={{ ...th, display: "inline-flex", alignItems: "center", gap: 4 }}>
              matcher
              <HelpIcon topic="optim_group_matcher" />
            </th>
            <th style={th}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                lr
                <HelpIcon topic="optim_group_lr" />
              </span>
            </th>
            <th style={th}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                wd
                <HelpIcon topic="optim_group_wd" />
              </span>
            </th>
            <th style={th}></th>
          </tr>
        </thead>
        <tbody>
          {draft.groups.map((g, i) => (
            <tr key={i} data-testid={`optim-group-${i}`} style={{ borderBottom: `1px solid ${T.borderSoft}` }}>
              <td style={td}>
                <input data-testid={`optim-group-${i}-matcher`}
                       value={g.matcher}
                       onChange={(e) => updateGroup(i, { matcher: e.target.value })}
                       style={tableInputStyle} />
              </td>
              <td style={td}>
                <input data-testid={`optim-group-${i}-lr`} type="number"
                       step={1e-5} value={g.lr}
                       onChange={(e) => updateGroup(i, { lr: Number(e.target.value) })}
                       style={tableInputStyle} />
              </td>
              <td style={td}>
                <input data-testid={`optim-group-${i}-wd`} type="number"
                       step={0.01} value={g.weight_decay}
                       onChange={(e) => updateGroup(i, {
                         weight_decay: Number(e.target.value),
                       })}
                       style={tableInputStyle} />
              </td>
              <td style={{ ...td, display: "flex", gap: 4, justifyContent: "flex-end", alignItems: "center" }}>
                <button data-testid={`optim-group-${i}-schedule-toggle`}
                        onClick={() => toggleSchedule(i)}
                        title="Edit LR schedule"
                        style={miniButtonStyle}>
                  {expandedSchedules.has(i) || g.schedule ? "⏲▾" : "⏲"}
                </button>
                <button data-testid={`optim-group-${i}-remove`}
                        onClick={() => removeGroup(i)}
                        style={miniRemoveStyle}>×</button>
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
                          onChange={(s) => setSchedule(i, s)}
                          lastRunScheduleKind={lastRunScheduleKind} />
        )
      ))}

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
        <button data-testid="optim-add-group" onClick={addGroup} style={secButtonStyle}>
          + Add group
        </button>
        <AutoGroupButton
          rpc={rpc ?? null}
          optimKind={draft.kind}
          nodes={graphNodes ?? []}
          edges={graphEdges ?? []}
          currentGroups={draft.groups}
          onApply={(groups, banner) => {
            setDraft({ ...draft, groups });
            setAutoGroupBanner(banner);
          }}
        />
      </div>
      {autoGroupBanner && (
        <pre data-testid="optim-auto-group-banner"
             style={{ background: T.surface3, color: T.accent,
                      border: `1px solid ${T.border}`,
                      padding: 8, borderRadius: 6, fontSize: 11,
                      whiteSpace: "pre-wrap", marginTop: 8 }}>
          {autoGroupBanner}
        </pre>
      )}

      <label style={labelStyle}>
        <span style={labelTitle}>
          grad_clip_norm
          <HelpIcon topic="optim_grad_clip" />
        </span>
        <input data-testid="optim-clip" type="number" step={0.1}
               value={draft.grad_clip_norm}
               onChange={(e) =>
                 setDraft({ ...draft,
                            grad_clip_norm: Number(e.target.value) })}
               style={inputStyle} />
      </label>

      <label style={{ ...labelStyle, flexDirection: "row", alignItems: "center", gap: 8, cursor: "pointer" }}>
        <input data-testid="optim-mp" type="checkbox"
               checked={draft.mixed_precision}
               onChange={(e) =>
                 setDraft({ ...draft, mixed_precision: e.target.checked })}
               style={{ cursor: "pointer", width: 14, height: 14 }} />
        <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", display: "inline-flex", alignItems: "center", gap: 4 }}>
          mixed_precision
          <HelpIcon topic="optim_mixed_precision" />
        </span>
      </label>

      <button data-testid="optim-apply" onClick={() => onApply(draft)} style={buttonStyle}>
        Apply
      </button>
    </div>
  );
}

const panel: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 12, padding: 16,
  fontFamily: T.font, fontSize: 12,
  background: T.surface, color: T.text,
};

const labelStyle: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 6,
  color: T.textSecondary,
};

const labelTitle: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 4,
  fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.05em",
};

const inputStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "8px 10px",
  fontSize: 12,
  fontFamily: T.font,
  outline: "none",
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

const buttonStyle: React.CSSProperties = {
  background: T.accent,
  color: T.accentContrast,
  border: "none",
  borderRadius: "var(--vb-radius-sm)",
  padding: "10px 14px",
  fontSize: 12,
  fontWeight: "bold",
  cursor: "pointer",
  marginTop: 10,
};

const secButtonStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "8px 12px",
  fontSize: 12,
  fontWeight: "bold",
  cursor: "pointer",
};

const miniButtonStyle: React.CSSProperties = {
  background: T.surface3,
  border: `1px solid ${T.border}`,
  borderRadius: "var(--vb-radius-sm)",
  color: T.text,
  padding: "2px 6px",
  fontSize: 11,
  cursor: "pointer",
};

const miniRemoveStyle: React.CSSProperties = {
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
