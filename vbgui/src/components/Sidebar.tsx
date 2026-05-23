import { useState } from "react";
import { T } from "@/theme";
import { LossTab } from "./sidebar/LossTab";
import { OptimTab } from "./sidebar/OptimTab";
import { RewritersTab } from "./sidebar/RewritersTab";
import { ShardingTab, type ShardingProposalView } from "./sidebar/ShardingTab";
import { GotchasTab } from "./sidebar/GotchasTab";
import { DimensionsTab,
         type InferenceEntryClient } from "./sidebar/DimensionsTab";
import { AblationsTab } from "./sidebar/AblationsTab";
import { SideChannelsTab } from "./sidebar/SideChannelsTab";
import { MemoryMatrixTab } from "./sidebar/MemoryMatrixTab";
import { TrainOpsTab } from "./sidebar/TrainOpsTab";
import type { TrainOptions } from "@/components/TrainOptionsPanel";
import type {
  GotchaState, LossState, OptimState, RewriterState, ShardingState,
  SideChannelState,
} from "@/state/spec";

export type SidebarTab = "loss" | "optim" | "rewriters" | "sharding"
                       | "gotchas" | "dimensions" | "ablations"
                       | "side_channels" | "memory" | "trainops";

export interface SidebarProps {
  loss: LossState;
  optim: OptimState;
  rewriters: RewriterState[];
  sideChannels: SideChannelState;
  availableSideChannels: string[];
  selectedTrainSideChannels: string[];
  sharding: ShardingState;
  gotchas: GotchaState[];
  proposals: ShardingProposalView[];
  onLossApply:    (l: LossState) => void;
  onOptimApply:   (o: OptimState) => void;
  onRewriterAdd:  (r: RewriterState) => void;
  onRewriterRemove: (i: number) => void;
  onRewriterReorder: (from: number, to: number) => void;
  onRewriterApply?: () => void;
  onSideChannelsApply: (s: SideChannelState) => void;
  onTrainSideChannelsChange: (channels: string[]) => void;
  onShardingChange: (s: ShardingState) => void;
  onShardingAccept: (idx: number) => void;
  onGotchaAutoFix?: (id: string) => void;
  /** V7-K3: GotchasTab forwards suggest_adapters calls up to App. */
  onSuggestAdapters?: (producer: string, consumer: string) =>
    Promise<import("./sidebar/GotchasTab").AdapterChain>;
  /** Optional rpc client — enables tooltip/explain integration in
   *  OptimTab and downstream tabs. App passes useRpc() here. */
  rpc?: import("@/lib/rpc").RpcClient | null;
  /** Optional canvas state — required for OptimTab Auto-group button. */
  graphNodes?: import("@xyflow/react").Node[];
  graphEdges?: import("@xyflow/react").Edge[];
  /** E7-2: inferred dimensions populated from verify response. */
  inferenceLog?: InferenceEntryClient[];
  onHighlightBrick?: (brick: string) => void;
  /** H07: parent dispatches the per-brick params mutation when the
   *  user clicks Apply on an auto-inferred row in DimensionsTab. */
  onDimensionsApply?: (entry: InferenceEntryClient) => void;
  tokenizerSource?: string | null;
  /** V7-H45: extras.schedule_kind from the most-recent train; threaded
   *  through to OptimTab → ScheduleEditor. */
  lastRunScheduleKind?: string | null;
  /** V8-R03: the current VerifyParams payload used as input to the
   *  memory.matrix RPC. The Memory tab refetches whenever this changes. */
  verifySpec?: unknown;
  /** UX#2: dim_env editor lives inside Dimensions tab. */
  dimEnv?: Record<string, number>;
  onDimEnvApply?: (next: Record<string, number>) => void;
  /** UX#3: train ops + warm-start picker live in a dedicated tab. */
  trainOptions?: TrainOptions;
  onTrainOptionsChange?: (next: TrainOptions) => void;
  trainRunHistory?: readonly string[];
  selectedWarmStartRunId?: string | null;
  onWarmStartSelect?: (runId: string | null) => void;
  // Tab state lifting
  activeTab?: SidebarTab;
  onTabChange?: (tab: SidebarTab) => void;
  // Splicing
  onParallelCompose?: (nodes: any[], edges: any[]) => void;
  onInsertIntoEdge?: (kind: string, edge: any) => void;
  onTransplant?: (kind: string, params: Record<string, unknown>) => void;
}

const TAB_LABELS: { key: SidebarTab; label: string }[] = [
  { key: "loss",       label: "Loss" },
  { key: "optim",      label: "Optim" },
  { key: "rewriters",  label: "Rewriters" },
  { key: "side_channels", label: "Side Ch." },
  { key: "sharding",   label: "Sharding" },
  { key: "gotchas",    label: "Gotchas" },
  { key: "dimensions", label: "Dimensions" },
  { key: "trainops",   label: "Train Ops" },
  { key: "ablations",  label: "Ablations" },
  { key: "memory",     label: "Memory" },
];

export function Sidebar(p: SidebarProps): JSX.Element {
  const [localActive, setLocalActive] = useState<SidebarTab>("loss");
  const active = p.activeTab ?? localActive;
  const setActive = (tab: SidebarTab) => {
    setLocalActive(tab);
    p.onTabChange?.(tab);
  };

  return (
    <aside data-testid="sidebar"
           style={{ width: 320, background: T.surface,
                    borderLeft: `1px solid ${T.border}`,
                    color: T.text,
                    display: "flex", flexDirection: "column",
                    fontFamily: "system-ui, sans-serif" }}>
      <nav role="tablist" data-testid="sidebar-tabs"
           style={{ display: "flex", flexWrap: "wrap",
                    borderBottom: `1px solid ${T.border}` }}>
        {TAB_LABELS.map((t) => (
          <button key={t.key}
                  role="tab"
                  aria-selected={active === t.key}
                  data-testid={`sidebar-tab-${t.key}`}
                  onClick={() => setActive(t.key)}
                  style={{
                    flex: "1 0 33%", padding: "8px 4px", border: "none",
                    background: active === t.key ? T.surface2 : "transparent",
                    color: active === t.key ? T.text : T.textSecondary,
                    cursor: "pointer", fontSize: 12,
                    borderBottom: active === t.key
                      ? `2px solid ${T.accent}` : "2px solid transparent",
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
                                              graphEdges={p.graphEdges}
                                              lastRunScheduleKind={
                                                p.lastRunScheduleKind} />}
        {active === "rewriters" && (
          <RewritersTab rewriters={p.rewriters}
                        onAdd={p.onRewriterAdd}
                        onRemove={p.onRewriterRemove}
                        onReorder={p.onRewriterReorder}
                        onApply={p.onRewriterApply} />
        )}
        {active === "side_channels" && (
          <SideChannelsTab sideChannels={p.sideChannels}
                           availableChannels={p.availableSideChannels}
                           selectedTrainChannels={p.selectedTrainSideChannels}
                           gotchas={p.gotchas}
                           rpc={p.rpc ?? null}
                           tokenizerSource={p.tokenizerSource ?? null}
                           onApply={p.onSideChannelsApply}
                           onTrainChannelsChange={
                             p.onTrainSideChannelsChange} />
        )}
        {active === "sharding"  && (
          <ShardingTab sharding={p.sharding} proposals={p.proposals}
                       onAccept={p.onShardingAccept}
                       onChange={p.onShardingChange} />
        )}
        {active === "gotchas"   && (
          <GotchasTab gotchas={p.gotchas}
                       onAutoFix={p.onGotchaAutoFix}
                       onSuggestAdapters={p.onSuggestAdapters} />
        )}
        {active === "dimensions" && (
          <DimensionsTab log={p.inferenceLog ?? []}
                          onHighlight={p.onHighlightBrick}
                          onApply={p.onDimensionsApply}
                          dimEnv={p.dimEnv}
                          onDimEnvApply={p.onDimEnvApply} />
        )}
        {active === "trainops" && p.trainOptions
                              && p.onTrainOptionsChange && (
          <TrainOpsTab
            trainOptions={p.trainOptions}
            onTrainOptionsChange={p.onTrainOptionsChange}
            history={p.trainRunHistory ?? []}
            selectedWarmStart={p.selectedWarmStartRunId ?? null}
            onWarmStartSelect={p.onWarmStartSelect ?? (() => {})}
            graphEdges={p.graphEdges ?? []}
            rpc={p.rpc ?? null}
            onParallelCompose={p.onParallelCompose}
            onInsertIntoEdge={p.onInsertIntoEdge}
            onTransplant={p.onTransplant} />
        )}
        {active === "ablations" && (
          <AblationsTab rpc={p.rpc ?? null}
                         nodes={p.graphNodes ?? []}
                         edges={p.graphEdges ?? []}
                         optim={p.optim}
                         loss={p.loss} />
        )}
        {active === "memory" && p.rpc && p.verifySpec !== undefined && (
          <MemoryMatrixTab rpc={p.rpc}
                            specPayload={p.verifySpec} />
        )}
      </div>
    </aside>
  );
}
