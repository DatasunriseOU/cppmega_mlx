import type { SpecState } from "@/state/spec";

export interface BottomStripProps {
  state: SpecState;
  fusedRegionCount?: number;
  onHelpToggle?: () => void;
}

const STATUS_COLOR: Record<SpecState["backend_status"], string> = {
  connected:    "#10b981",
  reconnecting: "#d97706",
  disconnected: "#dc2626",
};

const STATUS_LABEL: Record<SpecState["backend_status"], string> = {
  connected:    "Backend connected",
  reconnecting: "Reconnecting…",
  disconnected: "Disconnected",
};

export function BottomStrip({
  state, fusedRegionCount = 0, onHelpToggle,
}: BottomStripProps): JSX.Element {
  return (
    <footer data-testid="bottom-strip"
            style={{ height: 32, display: "flex", alignItems: "center",
                     gap: 16, padding: "0 12px",
                     borderTop: "1px solid #e5e7eb",
                     background: "#f9fafb",
                     fontFamily: "system-ui, sans-serif", fontSize: 11 }}>
      {state.backend_status === "reconnecting" && (
        <span data-testid="bottom-strip-reconnecting"
              style={{ display: "none" }}>reconnecting</span>
      )}
      <span data-testid="backend-status"
            style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <span data-testid="backend-status-dot"
              style={{ width: 8, height: 8, borderRadius: "50%",
                       background: STATUS_COLOR[state.backend_status],
                       animation: state.backend_status === "reconnecting"
                         ? "pulse 1.2s infinite" : "none" }} />
        {STATUS_LABEL[state.backend_status]}
      </span>
      <span data-testid="verify-latency">
        Verify: {state.last_verify_ms.toFixed(1)}ms
      </span>
      <span data-testid="brick-count">
        {state.brick_count} bricks, {fusedRegionCount} fused regions
      </span>
      <span style={{ flex: 1 }} />
      <button data-testid="help-toggle" onClick={onHelpToggle}>?</button>
    </footer>
  );
}
