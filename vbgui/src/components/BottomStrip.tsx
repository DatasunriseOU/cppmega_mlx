import type { SpecState } from "@/state/spec";
import type { CacheStats } from "@/hooks/useCacheStats";
import { formatHitRate } from "@/hooks/useCacheStats";

export interface BottomStripProps {
  state: SpecState;
  fusedRegionCount?: number;
  onHelpToggle?: () => void;
  /** V7-H48: backend git sha + boot timestamp from the heartbeat. */
  backendBuildId?: string | null;
  activeDevice?: string;
  /** V7-I07: JsonRPC LRU cache snapshot from /cache/stats. */
  cacheStats?: CacheStats | null;
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

function cacheHitRateColor(rate: number): string {
  if (!Number.isFinite(rate)) return "#9ca3af";
  if (rate >= 0.75) return "#10b981"; // green
  if (rate >= 0.40) return "#d97706"; // amber
  return "#dc2626"; // red
}

export function BottomStrip({
  state, fusedRegionCount = 0, onHelpToggle,
  backendBuildId = null, activeDevice,
  cacheStats = null,
}: BottomStripProps): JSX.Element {
  return (
    <footer data-testid="bottom-strip"
            style={{ height: 32, display: "flex", alignItems: "center",
                     gap: 16, padding: "0 14px",
                     borderTop: "1px solid var(--vb-border)",
                     background: "var(--vb-surface)",
                     color: "var(--vb-text-secondary)",
                     fontFamily: "var(--vb-font)", fontSize: 11 }}>
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
      {/* V7-I07: JsonRPC LRU cache hit-rate dashboard chip. */}
      <span data-testid="cache-stats"
            title={cacheStats
              ? `LRU cache: hits=${cacheStats.hits} misses=${cacheStats.misses} `
                + `evictions=${cacheStats.evictions} `
                + `size=${cacheStats.size}/${cacheStats.capacity}`
              : "LRU cache stats unavailable"}
            style={{ display: "inline-flex", alignItems: "center",
                     gap: 4, fontFamily: "monospace" }}>
        <span style={{
          width: 6, height: 6, borderRadius: "50%",
          background: cacheHitRateColor(
            cacheStats?.hit_rate ?? Number.NaN),
        }} />
        cache{" "}
        <span data-testid="cache-stats-hit-rate"
              data-hit-rate={cacheStats?.hit_rate ?? ""}
              data-hits={cacheStats?.hits ?? ""}
              data-misses={cacheStats?.misses ?? ""}
              data-evictions={cacheStats?.evictions ?? ""}
              data-size={cacheStats?.size ?? ""}
              data-capacity={cacheStats?.capacity ?? ""}>
          {cacheStats ? formatHitRate(cacheStats.hit_rate) : "—"}
        </span>
        <span data-testid="cache-stats-size" style={{ opacity: 0.7 }}>
          {cacheStats
            ? ` ${cacheStats.size}/${cacheStats.capacity}`
            : ""}
        </span>
      </span>
      {activeDevice && (
        <span data-testid="platform-indicator"
              style={{ display: "inline-flex", alignItems: "center", gap: 4,
                       color: "var(--vb-info)", fontWeight: 600, textTransform: "uppercase" }}>
          Platform: {activeDevice}
        </span>
      )}
      {backendBuildId && (
        <span data-testid="backend-build-id"
              title={`Backend build id: ${backendBuildId}`}
              style={{ fontFamily: "var(--vb-font-mono)", color: "var(--vb-text-muted)" }}>
          build {backendBuildId}
        </span>
      )}
      <span style={{ flex: 1 }} />
      <button data-testid="help-toggle" onClick={onHelpToggle}>?</button>
    </footer>
  );
}
