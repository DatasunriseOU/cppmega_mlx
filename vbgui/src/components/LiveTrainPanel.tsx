// V7-L37..L41: live training panel.
//
//   L37: LossChart subscribed to liveTrainEvents — sparkline draws as
//        steps arrive, not only on pipeline.run resolve.
//   L38: overflow_per_step marker on the sparkline timeline.
//   L39: dead-man-switch — when no event arrives for > stallSeconds,
//        the strip shows ⚠ "stalled X.Xs".
//   L40: finish:'ok' frame produces a toast.
//   L41: WS reconnect — see useLiveTrainEvents below; the panel itself
//        just consumes the array passed by App.tsx.

import { useEffect, useState } from "react";
import { LossChart } from "./LossChart";
import { T } from "@/theme";

export interface LiveTrainEvent {
  step: number;
  loss: number;
  lr?: number | null;
  overflow?: boolean;
  mem_mb?: number | null;
  ts?: number | null;
  grad_norms?: Record<string, number>;
  expert_load?: number[] | null;
  dataset_progress?: {
    progress_percent?: number;
    token_offset?: number;
    download_speed?: string | null;
  } | null;
  generated_text?: string | null;
  output_token?: string | null;
  mtp_logits?: any;
}

export interface LiveTrainPanelProps {
  events: LiveTrainEvent[];
  trainInFlight: boolean;
  finishToast?: boolean;
  reconnectAttempts?: number;
  stallSeconds?: number;       // default 8
  onDismissToast?: () => void;
}

export function LiveTrainPanel({
  events, trainInFlight, finishToast = false,
  reconnectAttempts = 0, stallSeconds = 8, onDismissToast,
}: LiveTrainPanelProps): JSX.Element | null {
  // V7-L39: poll wall-clock so the dead-man-switch can fire even when
  // events stop arriving. Tick every second while a run is active.
  const [now, setNow] = useState<number>(() => Date.now() / 1000);
  useEffect(() => {
    if (!trainInFlight) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [trainInFlight]);

  if (!trainInFlight && events.length === 0 && !finishToast) return null;

  const last = events[events.length - 1];
  const lastTs = last?.ts ?? null;
  const stalledFor = (lastTs && trainInFlight)
    ? Math.max(0, now - lastTs)
    : 0;
  const isStalled = stalledFor > stallSeconds;
  const overflowSteps = events
    .filter((e) => e.overflow)
    .map((e) => e.step);
  const losses = events.map((e) => e.loss);

  return (
    <div data-testid="live-train-panel"
         style={{ position: "fixed", bottom: 36, right: 12,
                  background: T.surface, border: `1px solid ${T.border}`,
                  borderRadius: 8, padding: 12, fontSize: 11,
                  fontFamily: T.fontMono,
                  color: T.text,
                  boxShadow: T.shadowPanel,
                  zIndex: 30, minWidth: 360 }}>
      <div data-testid="live-train-panel-header"
           style={{ display: "flex", alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 4 }}>
        <div style={{ fontWeight: 600, color: T.text }}>
          live train · step {events.length}
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {reconnectAttempts > 0 && (
            <span data-testid="live-train-panel-reconnect"
                  style={{ color: T.warning }}>
              ↻ reconnect #{reconnectAttempts}
            </span>
          )}
          {isStalled && (
            <span data-testid="live-train-panel-stalled"
                  style={{ color: T.danger }}>
              ⚠ stalled {stalledFor.toFixed(1)}s
            </span>
          )}
        </div>
      </div>

      {events.length > 0 ? (
        <div data-testid="live-train-panel-chart-wrap"
             style={{ marginBottom: 4 }}>
          <LossChart losses={losses}
                      overflowSteps={overflowSteps}
                      width={340} height={120}
                      testidPrefix="live-train-chart" />
        </div>
      ) : (
        <div data-testid="live-train-panel-empty"
             style={{ color: T.textSecondary, padding: 8 }}>
          waiting for first event…
        </div>
      )}

      {last && (
        <div data-testid="live-train-panel-pill"
             style={{ display: "flex", gap: 8, flexWrap: "wrap", color: T.textSecondary }}>
          <span data-testid="live-train-panel-last-loss">
            loss <strong style={{ color: T.accent }}>{last.loss.toFixed(4)}</strong>
          </span>
          <span data-testid="live-train-panel-last-lr">
            lr {last.lr != null ? last.lr.toExponential(2) : "?"}
          </span>
          {last.mem_mb != null && (
            <span data-testid="live-train-panel-last-mem">
              mem {last.mem_mb.toFixed(1)}MB
            </span>
          )}
          {/* V7-H34: per-step per-brick grad-norm summary. */}
          {last.grad_norms && Object.keys(last.grad_norms).length > 0 && (
            <span data-testid="live-train-panel-last-grad-norms">
              ‖g‖ {Object.keys(last.grad_norms).length}b ·
              max {Math.max(...Object.values(last.grad_norms)).toFixed(3)}
            </span>
          )}
          {/* V7-H36: per-step expert-load mini-bar. */}
          {last.expert_load && last.expert_load.length > 0 && (
            <span data-testid="live-train-panel-last-expert-load"
                  style={{ display: "inline-flex", gap: 2,
                           alignItems: "center" }}>
              experts:
              {last.expert_load.map((load, i) => (
                <span key={i}
                      data-testid={`live-train-panel-expert-${i}`}
                      title={`expert ${i}: ${load.toFixed(3)}`}
                      style={{ display: "inline-block",
                               width: 8,
                               height: Math.max(2, Math.min(14,
                                 load * 20)),
                               background: load > 0.5 ? T.danger
                                          : load > 0.2 ? T.warning
                                                       : T.success }} />
              ))}
            </span>
          )}
          {last.overflow && (
            <span data-testid="live-train-panel-last-overflow"
                  style={{ color: T.danger }}>
              ⚠ scaler overflow
            </span>
          )}
        </div>
      )}

      {finishToast && (
        <div data-testid="live-train-panel-toast"
             style={{ marginTop: 6, padding: "6px 10px",
                      background: "rgba(52, 211, 153, 0.12)", color: T.success,
                      border: `1px solid rgba(52, 211, 153, 0.25)`,
                      borderRadius: 4, display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center" }}>
          <span>✓ train done</span>
          {onDismissToast && (
            <button data-testid="live-train-panel-toast-dismiss"
                    onClick={onDismissToast}
                    style={{ background: "transparent", border: "none",
                             color: T.success, cursor: "pointer", fontSize: 14 }}>
              ×
            </button>
          )}
        </div>
      )}
    </div>
  );
}
