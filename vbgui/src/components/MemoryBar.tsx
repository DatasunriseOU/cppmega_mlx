import { memoryColor, type SpecState } from "@/state/spec";

export interface MemoryBarProps {
  state: SpecState;
}

const COLOR_HEX: Record<"green" | "yellow" | "red", string> = {
  green: "#10b981", yellow: "#d97706", red: "#dc2626",
};

function formatGB(n: number): string {
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function MemoryBar({ state }: MemoryBarProps): JSX.Element {
  const ratio = state.device_hbm_bytes > 0
    ? Math.min(1, state.worst_rank_bytes / state.device_hbm_bytes)
    : 0;
  const color = COLOR_HEX[memoryColor(state)];
  const tooltip = `${formatGB(state.worst_rank_bytes)} / ${formatGB(state.device_hbm_bytes)}`;
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
                    justifyContent: "center",
                    fontSize: 11, fontFamily: "system-ui, sans-serif",
                    color: ratio > 0.5 ? "white" : "#111827" }}>
        {tooltip}
      </div>
    </div>
  );
}
