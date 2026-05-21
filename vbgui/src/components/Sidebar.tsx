import { useState } from "react";
import { LossTab } from "./sidebar/LossTab";
import { OptimTab } from "./sidebar/OptimTab";
import { RewritersTab } from "./sidebar/RewritersTab";
import { ShardingTab, type ShardingProposalView } from "./sidebar/ShardingTab";
import { GotchasTab } from "./sidebar/GotchasTab";
import { DimensionsTab,
         type InferenceEntryClient } from "./sidebar/DimensionsTab";
import type {
  GotchaState, LossState, OptimState, RewriterState, ShardingState,
} from "@/state/spec";

export type SidebarTab = "loss" | "optim" | "rewriters" | "sharding"
                       | "gotchas" | "dimensions";

export interface SidebarProps {
  loss: LossState;
  optim: OptimState;
  rewriters: RewriterState[];
  sharding: ShardingState;
  gotchas: GotchaState[];
  proposals: ShardingProposalView[];
  onLossApply:    (l: LossState) => void;
  onOptimApply:   (o: OptimState) => void;
  onRewriterAdd:  (r: RewriterState) => void;
  onRewriterRemove: (i: number) => void;
  onRewriterReorder: (from: number, to: number) => void;
  onRewriterApply?: () => void;
  onShardingChange: (s: ShardingState) => void;
  onShardingAccept: (idx: number) => void;
  onGotchaAutoFix?: (id: string) => void;
  /** Optional rpc client — enables tooltip/explain integration in
   *  OptimTab and downstream tabs. App passes useRpc() here. */
  rpc?: import("@/lib/rpc").RpcClient | null;
  /** Optional canvas state — required for OptimTab Auto-group button. */
  graphNodes?: import("@xyflow/react").Node[];
  graphEdges?: import("@xyflow/react").Edge[];
  /** E7-2: inferred dimensions populated from verify response. */
  inferenceLog?: InferenceEntryClient[];
  onHighlightBrick?: (brick: string) => void;
}

const TAB_LABELS: { key: SidebarTab; label: string }[] = [
  { key: "loss",       label: "Loss" },
  { key: "optim",      label: "Optim" },
  { key: "rewriters",  label: "Rewriters" },
  { key: "sharding",   label: "Sharding" },
  { key: "gotchas",    label: "Gotchas" },
  { key: "dimensions", label: "Dimensions" },
];

export function Sidebar(p: SidebarProps): JSX.Element {
  const [active, setActive] = useState<SidebarTab>("loss");
  return (
    <aside data-testid="sidebar"
           style={{ width: 320, background: "#fff",
                    borderLeft: "1px solid #e5e7eb",
                    display: "flex", flexDirection: "column",
                    fontFamily: "system-ui, sans-serif" }}>
      <nav role="tablist" data-testid="sidebar-tabs"
           style={{ display: "flex", borderBottom: "1px solid #e5e7eb" }}>
        {TAB_LABELS.map((t) => (
          <button key={t.key}
                  role="tab"
                  aria-selected={active === t.key}
                  data-testid={`sidebar-tab-${t.key}`}
                  onClick={() => setActive(t.key)}
                  style={{
                    flex: 1, padding: "8px 4px", border: "none",
                    background: active === t.key ? "#f3f4f6" : "transparent",
                    cursor: "pointer", fontSize: 12,
                    borderBottom: active === t.key
                      ? "2px solid #2563eb" : "2px solid transparent",
                  }}>
            {t.label}
          </button>
        ))}
      </nav>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {active === "loss"      && <LossTab loss={p.loss} onApply={p.onLossApply} />}
        {active === "optim"     && <OptimTab optim={p.optim}
                                              onApply={p.onOptimApply}
                                              rpc={p.rpc ?? null}
                                              graphNodes={p.graphNodes}
                                              graphEdges={p.graphEdges} />}
        {active === "rewriters" && (
          <RewritersTab rewriters={p.rewriters}
                        onAdd={p.onRewriterAdd}
                        onRemove={p.onRewriterRemove}
                        onReorder={p.onRewriterReorder}
                        onApply={p.onRewriterApply} />
        )}
        {active === "sharding"  && (
          <ShardingTab sharding={p.sharding} proposals={p.proposals}
                       onAccept={p.onShardingAccept}
                       onChange={p.onShardingChange} />
        )}
        {active === "gotchas"   && (
          <GotchasTab gotchas={p.gotchas} onAutoFix={p.onGotchaAutoFix} />
        )}
        {active === "dimensions" && (
          <DimensionsTab log={p.inferenceLog ?? []}
                          onHighlight={p.onHighlightBrick} />
        )}
      </div>
    </aside>
  );
}
