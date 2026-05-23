// V7-H42: live gen.run token stream display.
//
// Mirror of LiveTrainPanel for the generation path:
//   * rolling tail of last N tokens (default 64),
//   * dead-man-switch when no event for > stallSeconds,
//   * finish:'ok' toast.
//
// Consumes useGenerateStream's events array.

import { useEffect, useState } from "react";
import type { GenTokenEvent } from "@/hooks/useGenerateStream";

export interface LiveGenPanelProps {
  events: GenTokenEvent[];
  genInFlight: boolean;
  finishToast?: boolean;
  reconnectAttempts?: number;
  stallSeconds?: number;
  onDismissToast?: () => void;
  /** Window of latest tokens to render. */
  tail?: number;
}

export function LiveGenPanel({
  events, genInFlight, finishToast = false,
  reconnectAttempts = 0, stallSeconds = 8, onDismissToast,
  tail = 64,
}: LiveGenPanelProps): JSX.Element {
  // Dead-man-switch tick — refreshes every second while gen is active
  // so "stalled X.Xs" updates without a fresh event.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!genInFlight) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [genInFlight]);

  const last = events.length > 0 ? events[events.length - 1] : null;
  const window = events.slice(-tail);
  const stalled = last && genInFlight
    ? ((Date.now() / 1000)
        - ((last as { ts?: number }).ts ?? 0))
    : 0;

  if (!genInFlight && events.length === 0 && !finishToast) {
    return <></>;
  }

  return (
    <div data-testid="live-gen-panel"
         style={{ padding: 8, border: "1px solid var(--vb-border)",
                  borderRadius: 4, background: "var(--vb-surface-2)",
                  fontFamily: "system-ui, sans-serif", fontSize: 11 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    marginBottom: 4 }}>
        <strong>live gen</strong>
        {reconnectAttempts > 0 && (
          <span data-testid="live-gen-panel-reconnects"
                style={{ color: "#d97706" }}>
            reconnects: {reconnectAttempts}
          </span>
        )}
      </div>
      {events.length === 0 ? (
        <div data-testid="live-gen-panel-empty" style={{ color: "var(--vb-text-muted)" }}>
          waiting for first token…
        </div>
      ) : (
        <>
          <div data-testid="live-gen-panel-pill"
               style={{ display: "flex", gap: 8, flexWrap: "wrap",
                        marginBottom: 4 }}>
            <span data-testid="live-gen-panel-token-count">
              tokens {events.length}
            </span>
            <span data-testid="live-gen-panel-last-token">
              last id {last?.token_id ?? "?"}
            </span>
            {stalled > stallSeconds && (
              <span data-testid="live-gen-panel-stalled"
                    style={{ color: "#dc2626" }}>
                ⚠ stalled {stalled.toFixed(1)}s
              </span>
            )}
          </div>
          <div data-testid="live-gen-panel-tail"
               style={{ fontFamily: "monospace", color: "var(--vb-text-secondary)",
                        whiteSpace: "pre-wrap", maxHeight: 80,
                        overflowY: "auto" }}>
            {window.map((e) => e.token_id).join(" ")}
          </div>
        </>
      )}
      {finishToast && (
        <div data-testid="live-gen-panel-toast"
             style={{ marginTop: 6, padding: 6,
                      background: "#dcfce7", color: "#166534",
                      borderRadius: 4, display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center" }}>
          <span>✓ gen done</span>
          {onDismissToast && (
            <button data-testid="live-gen-panel-toast-dismiss"
                    onClick={onDismissToast}
                    style={{ background: "transparent", border: "none",
                             color: "#166534", cursor: "pointer" }}>
              ×
            </button>
          )}
        </div>
      )}
    </div>
  );
}
