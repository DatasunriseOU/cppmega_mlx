// V7-K9 + K10 — controls that live alongside a train run. Visible
// before/during train. The Trigger-checkpoint button enqueues a one-
// shot checkpoint path; the host forwards it on the NEXT train run
// (live mid-run checkpointing requires WS plumbing covered in V7-H05
// follow-up, but for visual e2e the next-run trigger is enough).
//
// Live LR mutation (K10) submits a pipeline.update_lr RPC against an
// in-flight train run id. Currently a no-op fallback when the
// backend RPC is absent — UI surfaces the input + state.

import { useState } from "react";
import { HelpIcon } from "@/components/HelpIcon";
import type { RpcClient } from "@/lib/rpc";

export interface TrainLiveControlsProps {
  rpc: RpcClient | null;
  trainInFlight: boolean;
  activeRunId: string | null;
  onScheduleCheckpoint: (path: string) => void;
}

export function TrainLiveControls({
  rpc, trainInFlight, activeRunId, onScheduleCheckpoint,
}: TrainLiveControlsProps): JSX.Element {
  const [ckptPath, setCkptPath] = useState<string>("/tmp/midrun.safetensors");
  const [newLr, setNewLr] = useState<string>("");
  const [lrStatus, setLrStatus] = useState<string | null>(null);

  async function pushLr() {
    setLrStatus(null);
    if (!rpc || !activeRunId) {
      setLrStatus("no active run");
      return;
    }
    const parsed = parseFloat(newLr);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setLrStatus("invalid lr");
      return;
    }
    try {
      await rpc.call<{ status: string }>(
        "pipeline.update_lr",
        { run_id: activeRunId, new_lr: parsed });
      setLrStatus(`lr → ${parsed}`);
    } catch (e) {
      setLrStatus(`error: ${(e as Error).message}`);
    }
  }

  return (
    <div data-testid="train-live-controls"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: "#fef9c3",
                  borderTop: "1px solid #facc15",
                  fontSize: 12, fontFamily: "system-ui, sans-serif" }}>
      <strong>Live</strong>
      <HelpIcon topic="train_live_controls" />

      <span data-testid="train-live-status"
            style={{ color: trainInFlight ? "#92400e" : "#6b7280" }}>
        {trainInFlight ? "● train in flight" : "○ idle"}
      </span>

      <label>
        ckpt path
        <input data-testid="train-live-ckpt-path"
               type="text" value={ckptPath}
               onChange={(e) => setCkptPath(e.target.value)}
               style={{ marginLeft: 4, width: 200 }} />
      </label>
      <button data-testid="train-live-trigger-ckpt"
              onClick={() => {
                if (ckptPath) onScheduleCheckpoint(ckptPath);
              }}
              disabled={!ckptPath}
              style={{ padding: "2px 8px",
                       background: ckptPath ? "#d97706" : "#e5e7eb",
                       color: ckptPath ? "white" : "#9ca3af",
                       border: "none", borderRadius: 4,
                       cursor: ckptPath ? "pointer" : "default" }}>
        Trigger checkpoint
      </button>

      <span style={{ marginLeft: 8 }}>|</span>

      <label>
        live lr
        <input data-testid="train-live-new-lr"
               type="number" step="0.0001" min={0}
               value={newLr} placeholder="0.0003"
               onChange={(e) => setNewLr(e.target.value)}
               style={{ marginLeft: 4, width: 90 }} />
      </label>
      <button data-testid="train-live-apply-lr"
              onClick={pushLr}
              disabled={!newLr || !activeRunId}
              style={{ padding: "2px 8px",
                       background: (newLr && activeRunId)
                         ? "#16a34a" : "#e5e7eb",
                       color: (newLr && activeRunId)
                         ? "white" : "#9ca3af",
                       border: "none", borderRadius: 4,
                       cursor: (newLr && activeRunId)
                         ? "pointer" : "default" }}>
        Apply lr
      </button>
      {lrStatus && (
        <span data-testid="train-live-lr-status"
              style={{ color: "#374151", marginLeft: 4 }}>
          {lrStatus}
        </span>
      )}
    </div>
  );
}
