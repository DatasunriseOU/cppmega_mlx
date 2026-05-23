// UX#3: TrainOpsTab — sidebar home for the K3-K8 train options and
// the warm-start run-history picker. Previously these lived as two
// full-width strips above the canvas; they're now in their own
// sidebar tab where they belong.

import { TrainOptionsPanel,
         type TrainOptions } from "@/components/TrainOptionsPanel";
import { RunHistoryPicker } from "@/components/RunHistoryPicker";
import { ParallelComposeBar } from "@/components/ParallelComposeBar";
import { InsertIntoEdgeBar } from "@/components/InsertIntoEdgeBar";
import { TransplantBar } from "@/components/TransplantBar";
import { T } from "@/theme";

export interface TrainOpsTabProps {
  trainOptions: TrainOptions;
  onTrainOptionsChange: (next: TrainOptions) => void;
  history: readonly string[];
  selectedWarmStart: string | null;
  onWarmStartSelect: (runId: string | null) => void;

  // Splicing properties
  graphEdges: any[];
  rpc: any;
  onParallelCompose?: (nodes: any[], edges: any[]) => void;
  onInsertIntoEdge?: (kind: string, edge: any) => void;
  onTransplant?: (kind: string, params: Record<string, unknown>) => void;
}

const PRESET_LIST = ["mini", "dev_128", "small_512", "medium_1k", "large_2k", "llama3_8b", "llama3_70b"] as const;

export function TrainOpsTab({
  trainOptions, onTrainOptionsChange,
  history, selectedWarmStart, onWarmStartSelect,
  graphEdges, rpc,
  onParallelCompose, onInsertIntoEdge, onTransplant,
}: TrainOpsTabProps): JSX.Element {
  return (
    <div data-testid="train-ops-tab"
         style={{ display: "flex", flexDirection: "column", gap: 0,
                  fontSize: 12, background: T.surface, color: T.text }}>
      <TrainOptionsPanel
        value={trainOptions}
        onChange={onTrainOptionsChange}
      />
      
      <div style={{ borderTop: `1px solid ${T.border}`, padding: "12px 12px 4px 12px" }}>
        <RunHistoryPicker
          history={history}
          selected={selectedWarmStart}
          onSelect={onWarmStartSelect}
        />
      </div>

      {/* Canvas splicing section */}
      <div style={{ borderTop: `1px solid ${T.border}`, padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        <h4 style={{ margin: "0 0 2px 0", fontSize: 13, color: T.accent, fontWeight: "bold" }}>Canvas Splicing</h4>
        
        {onParallelCompose && (
          <div style={{ background: T.surface3, border: `1px solid ${T.border}`, borderRadius: 6, padding: 10 }}>
            <ParallelComposeBar onCompose={onParallelCompose} />
          </div>
        )}
        
        {onInsertIntoEdge && (
          <div style={{ background: T.surface3, border: `1px solid ${T.border}`, borderRadius: 6, padding: 10 }}>
            <InsertIntoEdgeBar edges={graphEdges} onInsert={onInsertIntoEdge} />
          </div>
        )}
        
        {onTransplant && (
          <div style={{ background: T.surface3, border: `1px solid ${T.border}`, borderRadius: 6, padding: 10 }}>
            <TransplantBar rpc={rpc} presets={PRESET_LIST} onTransplant={onTransplant} />
          </div>
        )}
      </div>
    </div>
  );
}
