import { memoryColor, type SpecState } from "@/state/spec";

export interface MemoryBarProps {
  state: SpecState;
  /** UX-redesign #6: compact pill mode (100×18px) for TopBar, freeing
   *  the horizontal space the legacy flex:1 bar greedily occupied. */
  compact?: boolean;
  /** UX-redesign #6: per-rank bytes, used when topology.world_size > 1
   *  so the bar renders N vertical mini-strips with per-rank
   *  breakdown — single-bar mode is misleading on a 8-GPU cluster. */
  perRankBytes?: readonly number[];
}

const COLOR_HEX: Record<"green" | "yellow" | "red", string> = {
  green: "#10b981", yellow: "#d97706", red: "#dc2626",
};

function formatGB(n: number): string {
  if (n > 0 && n < 1024 ** 3) {
    return `${(n / 1024 ** 2).toFixed(2)} MB`;
  }
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function rankColor(r: number): string {
  if (r < 0.5) return COLOR_HEX.green;
  if (r < 0.8) return COLOR_HEX.yellow;
  return COLOR_HEX.red;
}

export function MemoryBar({
  state, compact = false, perRankBytes,
}: MemoryBarProps): JSX.Element {
  const ratio = state.device_hbm_bytes > 0
    ? Math.min(1, state.worst_rank_bytes / state.device_hbm_bytes)
    : 0;
  const color = COLOR_HEX[memoryColor(state)];
  const estimate = state.worst_rank_bytes;
  const actual = state.actual_peak_bytes;
  const ranks = perRankBytes && perRankBytes.length > 1
    ? perRankBytes : null;

  const perRankTip = ranks
    ? "\nper-rank: "
      + ranks.map((b, i) => `r${i}=${formatGB(b)}`).join(" ")
    : "";
  const tooltip = (actual != null
    ? `estimate ${formatGB(estimate)} · actual ${formatGB(actual)} / ` +
      `${formatGB(state.device_hbm_bytes)}`
    : `${formatGB(estimate)} / ${formatGB(state.device_hbm_bytes)}`)
    + perRankTip;

  // UX-redesign #6 compact + cluster: N vertical mini-bars (one per
  // rank, 8 px wide each). Total ≤ ranks*9 px → 8 ranks = 72 px.
  if (compact && ranks) {
    return (
      <div data-testid="memory-bar"
           data-mode="compact-cluster"
           title={tooltip}
           style={{ display: "inline-flex", gap: 1, alignItems: "flex-end",
                    height: 18, padding: "0 4px",
                    background: "var(--vb-surface-3)", borderRadius: 4 }}>
        {ranks.map((b, i) => {
          const r = state.device_hbm_bytes > 0
            ? Math.min(1, b / state.device_hbm_bytes) : 0;
          return (
            <span key={i}
                  data-testid={`memory-bar-rank-${i}`}
                  data-bytes={b}
                  style={{ width: 8,
                           height: `${Math.max(2, r * 16)}px`,
                           background: rankColor(r),
                           borderRadius: 1 }} />
          );
        })}
        <span data-testid="memory-bar-cluster-label"
              style={{ marginLeft: 6, fontSize: 10,
                       fontFamily: "monospace",
                       color: "var(--vb-text-secondary)" }}>
          {ranks.length}× max {formatGB(Math.max(...ranks))}
        </span>
      </div>
    );
  }

  // UX-redesign #6 compact single-device: 100×18px pill, no flex.
  if (compact) {
    return (
      <div data-testid="memory-bar"
           data-mode="compact"
           title={tooltip}
           style={{ display: "inline-flex", alignItems: "center",
                    width: 100, height: 18, background: "var(--vb-surface-3)",
                    borderRadius: 4, overflow: "hidden",
                    position: "relative" }}>
        <div data-testid="memory-bar-fill"
             style={{ position: "absolute", left: 0, top: 0,
                      bottom: 0, width: `${ratio * 100}%`,
                      background: color, transition: "width 200ms" }} />
        <span data-testid="memory-bar-estimate"
              data-bytes={estimate}
              style={{ position: "relative", margin: "0 auto",
                       fontSize: 10, fontFamily: "monospace",
                       color: "var(--vb-text)" }}>
          {formatGB(estimate)}
        </span>
        {actual != null && (
          <span data-testid="memory-bar-actual" data-bytes={actual}
                style={{ display: "none" }}>
            {formatGB(actual)}
          </span>
        )}
      </div>
    );
  }

  // Legacy flex:1 horizontal bar — kept verbatim so existing tests +
  // non-TopBar consumers still work.
  return (
    <div data-testid="memory-bar" title={tooltip}
         style={{ flex: 1, height: 24, background: "var(--vb-surface-3)",
                  borderRadius: 4, overflow: "hidden",
                  position: "relative" }}>
      <div data-testid="memory-bar-fill"
           style={{ width: `${ratio * 100}%`, height: "100%",
                    background: color, transition: "width 200ms" }} />
      <div style={{ position: "absolute", inset: 0,
                    display: "flex", alignItems: "center",
                    justifyContent: "center", gap: 6,
                    fontSize: 11, fontFamily: "system-ui, sans-serif",
                    color: "var(--vb-text)" }}>
        <span data-testid="memory-bar-estimate"
              data-bytes={estimate}>
          est {formatGB(estimate)}
        </span>
        {actual != null && (
          <span data-testid="memory-bar-actual"
                data-bytes={actual}
                style={{ opacity: 0.85 }}>
            · act {formatGB(actual)}
          </span>
        )}
        <span style={{ opacity: 0.75 }}>
          / {formatGB(state.device_hbm_bytes)}
        </span>
      </div>
    </div>
  );
}
