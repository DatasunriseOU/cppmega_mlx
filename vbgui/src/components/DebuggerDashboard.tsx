import { T } from "@/theme";
import type { Token } from "@/hooks/useNeuralDebugger";

export interface DebuggerDashboardProps {
  activeStep: number;
  maxStep: number;
  direction: "forward" | "backward";
  prompt: string;
  tokens: Token[];
  isPlaying: boolean;
  setIsPlaying: (val: boolean) => void;
  lr: number;
  setLr: (val: number) => void;
  lossVal: number;
  isWeightUpdated: boolean;
  onStepForward: () => void;
  onStepBackward: () => void;
  onReset: () => void;
  onFullTrainStep?: () => void;
  activeNodeLabel?: string;
}

export function DebuggerDashboard({
  activeStep,
  maxStep,
  direction,
  isPlaying,
  setIsPlaying,
  lr,
  setLr,
  lossVal,
  isWeightUpdated,
  onStepForward,
  onStepBackward,
  onReset,
  onFullTrainStep,
  activeNodeLabel,
}: DebuggerDashboardProps): JSX.Element {
  const currentStepName =
    activeStep === -1
      ? "Segmenting prompt into tokens"
      : activeStep === maxStep
      ? "Forward pass completed. Backpropagation ready."
      : activeStep === maxStep - 1
      ? "Computing Loss & Logits"
      : activeNodeLabel
      ? `Passing activation through ${activeNodeLabel}`
      : `Step ${activeStep} in progress`;

  return (
    <div
      role="region"
      aria-label="neural debugger control dashboard"
      data-testid="debugger-dashboard"
      style={{
        position: "absolute",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        width: "90%",
        maxWidth: 880,
        background: "var(--vb-surface-glass)",
        backdropFilter: "blur(12px)",
        border: "1px solid var(--vb-border-strong)",
        borderRadius: "var(--vb-radius-xl)",
        padding: "16px 20px",
        boxShadow: "var(--vb-shadow-pop)",
        display: "flex",
        flexDirection: "column",
        gap: 12,
        zIndex: 1000,
        transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>🐛</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: T.text, display: "flex", alignItems: "center", gap: 6 }}>
              Neural Debugger &amp; Simulator
              <span
                style={{
                  fontSize: 10,
                  padding: "2px 8px",
                  borderRadius: "var(--vb-radius-pill)",
                  background:
                    direction === "forward" ? "rgba(34, 211, 238, 0.16)" : "rgba(245, 158, 11, 0.16)",
                  color: direction === "forward" ? "var(--vb-accent)" : "var(--vb-cat-moe)",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                {direction === "forward" ? "Forward Pass" : "Backpropagation"}
              </span>
            </div>
            <div style={{ fontSize: 12, color: T.textSecondary, marginTop: 2 }}>
              {currentStepName}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 9, color: T.textMuted, textTransform: "uppercase", fontWeight: 700 }}>
              Learning Rate (lr)
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
              <input
                type="number"
                step="0.0001"
                className="nodrag nopan"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
                style={{
                  width: 70,
                  background: "var(--vb-surface-3)",
                  border: "1px solid var(--vb-border)",
                  borderRadius: "var(--vb-radius-sm)",
                  color: T.text,
                  fontSize: 11,
                  fontFamily: T.fontMono,
                  padding: "2px 4px",
                  outline: "none",
                  textAlign: "center",
                }}
              />
            </div>
          </div>

          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 9, color: T.textMuted, textTransform: "uppercase", fontWeight: 700 }}>
              Loss Value
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: T.fontMono, color: "var(--vb-accent)", marginTop: 2 }}>
              {lossVal.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12, borderTop: `1px solid var(--vb-border-soft)`, paddingTop: 12 }}>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            data-testid="debugger-btn-reset"
            title="Reset Simulation"
            onClick={onReset}
            style={actionBtnStyle}
          >
            ⏮ Reset
          </button>
          <button
            type="button"
            data-testid="debugger-btn-step-bwd"
            title="Step Backward"
            onClick={onStepBackward}
            style={actionBtnStyle}
          >
            ◀ Step Bwd
          </button>
          <button
            type="button"
            data-testid="debugger-btn-play"
            title={isPlaying ? "Pause Simulation" : "Play Simulation"}
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              ...actionBtnStyle,
              background: isPlaying ? "rgba(239, 68, 68, 0.16)" : "var(--vb-accent-soft)",
              color: isPlaying ? "var(--vb-danger)" : "var(--vb-accent)",
              border: `1px solid ${isPlaying ? "var(--vb-danger)" : "var(--vb-accent)"}`,
            }}
          >
            {isPlaying ? "⏸ Pause" : "⏯ Play"}
          </button>
          <button
            type="button"
            data-testid="debugger-btn-step-fwd"
            title="Step Forward"
            onClick={onStepForward}
            style={actionBtnStyle}
          >
            ▶ Step Fwd
          </button>
          {onFullTrainStep && (
            <button
              type="button"
              data-testid="debugger-btn-full-train"
              title="Animate entire Train step (fwd + bwd + weight update)"
              onClick={onFullTrainStep}
              style={{
                ...actionBtnStyle,
                background: "rgba(245, 158, 11, 0.16)",
                border: "1px solid var(--vb-cat-moe)",
                color: "var(--vb-cat-moe)",
                fontWeight: 700,
              }}
            >
              🚀 Full Train Step
            </button>
          )}
        </div>

        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8 }}>
          {isWeightUpdated && (
            <div
              data-testid="debugger-weight-update-pulse"
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: "var(--vb-warning)",
                background: "rgba(251, 191, 36, 0.15)",
                border: "1px solid var(--vb-warning)",
                borderRadius: "var(--vb-radius-pill)",
                padding: "2px 10px",
                animation: "pulse 1s infinite alternate",
              }}
            >
              ✨ Optimizer weight update: W = W - lr * dW
            </div>
          )}
          <span style={{ fontSize: 11, color: T.textMuted, fontFamily: T.fontMono }}>
            Step Index: {activeStep} / {maxStep}
          </span>
        </div>
      </div>
    </div>
  );
}

const actionBtnStyle: React.CSSProperties = {
  background: "var(--vb-surface-3)",
  border: "1px solid var(--vb-border)",
  color: T.text,
  borderRadius: "var(--vb-radius-sm)",
  padding: "6px 12px",
  fontSize: 12,
  fontWeight: 600,
  cursor: "pointer",
  outline: "none",
  transition: "all 0.15s ease",
};
