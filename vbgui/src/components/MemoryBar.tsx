import { memoryColor, type SpecState } from "@/state/spec";

export interface MemoryBarProps {
  state: SpecState;
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

export function MemoryBar({ state }: MemoryBarProps): JSX.Element {
  const ratio = state.device_hbm_bytes > 0
    ? Math.min(1, state.worst_rank_bytes / state.device_hbm_bytes)
    : 0;
  const color = COLOR_HEX[memoryColor(state)];
  const estimate = state.worst_rank_bytes;
  const actual = state.actual_peak_bytes;
  // H11: dual readout — estimate is always shown (from verify);
  // actual lights up once Train completes and dispatches
  // extras.memory_peak_bytes via memory.actual_set. We render BOTH
  // values as siblings inside the bar so existing memory-bar /
  // memory-bar-fill testids keep working.
  const tooltip = actual != null
    ? `estimate ${formatGB(estimate)} · actual ${formatGB(actual)} / ` +
      `${formatGB(state.device_hbm_bytes)}`
    : `${formatGB(estimate)} / ${formatGB(state.device_hbm_bytes)}`;
  return (
    <div data-testid="memory-bar" title={tooltip}
         style={{ flex: 1, height: 24, background: "#e5e7eb",
                  borderRadius: 4, overflow: "hidden",
                  position: "relative" }}>
      <div data-testid="memory-bar-fill"
           style={{ width: `${ratio * 100}%`, height: "100%",
                    background: color, transition: "width 200ms" }} />
      <div style={{ position: "absolute", inset: 0,
                    display: "flex", alignItems: "center",
                    justifyContent: "center", gap: 6,
                    fontSize: 11, fontFamily: "system-ui, sans-serif",
                    color: ratio > 0.5 ? "white" : "#111827" }}>
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
