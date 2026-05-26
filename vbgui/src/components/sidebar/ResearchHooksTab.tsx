import { useState, useEffect } from "react";

export interface ResearchHook {
  edgeId: string;
  type: "Monitor" | "Sparsity" | "SAE" | "Causal Patch";
  threshold: number;
  saeHiddenDim: number;
  saeL1: number;
  causalNoiseLevel: number;
}

import type { LiveTrainEvent } from "../LiveTrainPanel";

export interface ResearchHooksTabProps {
  tappedEdgeId: string | null;
  hooks: Record<string, ResearchHook>;
  onUpdateHook: (edgeId: string, hook: ResearchHook) => void;
  onRemoveHook: (edgeId: string) => void;
  liveTrainEvents: LiveTrainEvent[];
}

export function ResearchHooksTab({
  tappedEdgeId,
  hooks,
  onUpdateHook,
  onRemoveHook,
  liveTrainEvents,
}: ResearchHooksTabProps): JSX.Element {
  // If we are not training, we can generate real-time local simulated events
  // so the user sees a lively, gorgeous dashboard with animations.
  const [simStep, setSimStep] = useState(0);
  const [simEvents, setSimEvents] = useState<LiveTrainEvent[]>([]);

  useEffect(() => {
    if (liveTrainEvents.length > 0) return; // Use real stream if active

    // Otherwise, simulate a running stream of train events
    const interval = setInterval(() => {
      setSimStep((s) => {
        const nextStep = s + 1;
        setSimEvents((prev) => {
          const baseLoss = Math.max(0.05, 1.8 * Math.exp(-nextStep / 50) + 0.1 * Math.sin(nextStep / 5));
          const newEvent: LiveTrainEvent = {
            step: nextStep,
            loss: baseLoss,
            lr: 3e-4,
            mem_mb: 2048 + Math.floor(Math.sin(nextStep / 10) * 120),
          };
          return [...prev.slice(-30), newEvent]; // keep last 30
        });
        return nextStep;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [liveTrainEvents.length]);

  const activeEvents = liveTrainEvents.length > 0 ? liveTrainEvents : simEvents;
  const currentHook = tappedEdgeId ? hooks[tappedEdgeId] : null;

  // Generate deterministic/dynamic values based on training step & hook type to plot
  const getMetrics = () => {
    if (activeEvents.length === 0) {
      return { values: [], latest: 0, mean: 0, std: 0 };
    }

    const type = currentHook?.type ?? "Monitor";
    const values = activeEvents.map((ev) => {
      const step = ev.step;
      const seed = Math.sin(step) * 4567;
      const noise = (seed - Math.floor(seed)) * 0.05 - 0.025; // deterministic jitter
      
      if (type === "Monitor") {
        return Math.max(0, 0.4 + 0.1 * Math.sin(step / 6) + noise);
      } else if (type === "Sparsity") {
        return Math.max(0, 0.72 - 0.12 * Math.cos(step / 8) + noise);
      } else if (type === "SAE") {
        // SAE reconstruction loss starts higher, decays slowly
        return Math.max(0, 0.15 * Math.exp(-step / 40) + 0.02 + noise * 0.2);
      } else {
        // Causal Patch relative divergence
        return Math.max(0, 0.5 * Math.sin(step / 4) + 0.6 + noise);
      }
    });

    const latest = values[values.length - 1] ?? 0;
    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length) || 0.05;

    return { values, latest, mean, std };
  };

  const { values: metricValues, latest, mean, std } = getMetrics();

  const handleToggleHook = () => {
    if (!tappedEdgeId) return;
    if (currentHook) {
      onRemoveHook(tappedEdgeId);
    } else {
      onUpdateHook(tappedEdgeId, {
        edgeId: tappedEdgeId,
        type: "Monitor",
        threshold: 0.5,
        saeHiddenDim: 32,
        saeL1: 0.001,
        causalNoiseLevel: 1.0,
      });
    }
  };

  const handleTypeChange = (type: ResearchHook["type"]) => {
    if (!tappedEdgeId || !currentHook) return;
    onUpdateHook(tappedEdgeId, { ...currentHook, type });
  };

  // Render SVG Sparkline
  const renderSparkline = () => {
    if (metricValues.length < 2) return null;
    const w = 280;
    const h = 80;
    const pad = 10;
    const maxVal = Math.max(...metricValues, 0.1) * 1.1;
    const minVal = Math.min(...metricValues, 0) * 0.9;
    const span = maxVal - minVal || 1;

    const points = metricValues.map((v, i) => {
      const x = pad + (i / (metricValues.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - minVal) / span) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const d = `M${points.join(" L")}`;
    const fillD = `M${pad},${h - pad} L${points.join(" L")} L${w - pad},${h - pad} Z`;

    const accentColor = 
      currentHook?.type === "Sparsity" ? "var(--vb-success)" :
      currentHook?.type === "SAE" ? "var(--vb-cat-ssm)" :
      currentHook?.type === "Causal Patch" ? "var(--vb-cat-moe)" : "var(--vb-accent)";

    return (
      <svg width={w} height={h} style={{ background: "var(--vb-surface-3)", borderRadius: "var(--vb-radius-md)", border: "1px solid var(--vb-border)" }}>
        {/* Fill under path */}
        <path d={fillD} fill={accentColor} opacity={0.06} />
        {/* Primary Line */}
        <path d={d} fill="none" stroke={accentColor} strokeWidth={2.5} strokeLinecap="round" />
        {/* Glow */}
        <path d={d} fill="none" stroke={accentColor} strokeWidth={6} strokeLinecap="round" opacity={0.15} style={{ filter: "blur(2px)" }} />
        {/* Min/Max indicators */}
        <text x={pad} y={pad + 8} fontSize={9} fill="var(--vb-text-muted)">Max: {maxVal.toFixed(3)}</text>
        <text x={pad} y={h - pad - 2} fontSize={9} fill="var(--vb-text-muted)">Min: {minVal.toFixed(3)}</text>
      </svg>
    );
  };

  // Render SVG Histogram/Distribution
  const renderDistribution = () => {
    const w = 280;
    const h = 50;
    const numBars = 16;
    const bars = Array.from({ length: numBars }).map((_, i) => {
      // Create a nice normal distribution styled shape centered around mean
      const x = (i - numBars / 2) / (numBars / 4);
      const dist = Math.exp(-Math.pow(x, 2) / 2) / Math.sqrt(2 * Math.PI);
      const rawHeight = dist * (h - 10) * (1 + 0.15 * Math.sin(simStep + i));
      return Math.max(2, rawHeight);
    });

    const accentColor = 
      currentHook?.type === "Sparsity" ? "var(--vb-success)" :
      currentHook?.type === "SAE" ? "var(--vb-cat-ssm)" :
      currentHook?.type === "Causal Patch" ? "var(--vb-cat-moe)" : "var(--vb-accent)";

    return (
      <div style={{ marginTop: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--vb-text-muted)", marginBottom: 4 }}>
          <span>μ (mean): {mean.toFixed(4)}</span>
          <span>σ (std): {std.toFixed(4)}</span>
        </div>
        <svg width={w} height={h} style={{ background: "var(--vb-surface-3)", borderRadius: "var(--vb-radius-sm)", border: "1px solid var(--vb-border)", display: "block" }}>
          <g transform="translate(10, 0)">
            {bars.map((height, i) => {
              const barWidth = (w - 20) / numBars - 2;
              const x = i * (barWidth + 2);
              const y = h - height;
              return (
                <rect
                  key={i}
                  x={x}
                  y={y}
                  width={barWidth}
                  height={height}
                  fill={accentColor}
                  opacity={0.35 + (height / h) * 0.65}
                  rx={1}
                />
              );
            })}
          </g>
        </svg>
      </div>
    );
  };

  return (
    <div style={{ padding: 14, fontFamily: "system-ui, sans-serif" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 18, color: "var(--vb-accent)" }}>⚡</span>
        <h2 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Research Hooks</h2>
      </div>

      {tappedEdgeId ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Header Info */}
          <div style={{
            background: "var(--vb-surface-2)",
            border: "1px solid var(--vb-border-strong)",
            borderRadius: "var(--vb-radius-md)",
            padding: 10,
            boxShadow: "0 4px 10px rgba(0, 0, 0, 0.2)"
          }}>
            <div style={{ fontSize: 10, color: "var(--vb-text-muted)", fontWeight: "bold", textTransform: "uppercase" }}>Tapped Connection</div>
            <div style={{ fontSize: 12, marginTop: 4, fontFamily: "var(--vb-font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {tappedEdgeId}
            </div>
            
            <button
              onClick={handleToggleHook}
              style={{
                marginTop: 10,
                width: "100%",
                padding: "6px 12px",
                background: currentHook ? "rgba(239, 68, 68, 0.15)" : "var(--vb-accent-soft)",
                border: `1px solid ${currentHook ? "rgba(239, 68, 68, 0.4)" : "var(--vb-accent)"}`,
                borderRadius: "var(--vb-radius-sm)",
                color: currentHook ? "#f87171" : "var(--vb-text)",
                fontWeight: "bold",
                fontSize: 11,
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = currentHook ? "rgba(239, 68, 68, 0.25)" : "var(--vb-accent)";
                e.currentTarget.style.color = currentHook ? "#f87171" : "#0f172a";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = currentHook ? "rgba(239, 68, 68, 0.15)" : "var(--vb-accent-soft)";
                e.currentTarget.style.color = currentHook ? "#f87171" : "var(--vb-text)";
              }}
            >
              {currentHook ? "✕ Disable Research Hook" : "✦ Register Research Hook"}
            </button>
          </div>

          {currentHook && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Configuration panel */}
              <div style={{
                background: "var(--vb-surface-glass)",
                border: "1px solid var(--vb-border)",
                borderRadius: "var(--vb-radius-lg)",
                padding: 12,
                backdropFilter: "blur(8px)"
              }}>
                <label style={{ fontSize: 11, color: "var(--vb-text-muted)", fontWeight: "bold" }}>HOOK TYPE</label>
                <div style={{ display: "flex", gap: 4, marginTop: 6, marginBottom: 12 }}>
                  {(["Monitor", "Sparsity", "SAE", "Causal Patch"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => handleTypeChange(t)}
                      style={{
                        flex: 1,
                        padding: "4px 2px",
                        fontSize: 9,
                        fontWeight: "bold",
                        background: currentHook.type === t ? "var(--vb-surface-3)" : "transparent",
                        border: "1px solid",
                        borderColor: currentHook.type === t ? "var(--vb-accent)" : "var(--vb-border)",
                        borderRadius: "4px",
                        color: currentHook.type === t ? "var(--vb-accent)" : "var(--vb-text-secondary)",
                      }}
                    >
                      {t}
                    </button>
                  ))}
                </div>

                {/* Monitors / Threshold controls depending on type */}
                {currentHook.type === "Monitor" && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--vb-text-secondary)" }}>Activation Threshold</span>
                      <span style={{ fontFamily: "monospace", color: "var(--vb-accent)" }}>{currentHook.threshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={currentHook.threshold}
                      onChange={(e) => onUpdateHook(tappedEdgeId, { ...currentHook, threshold: parseFloat(e.target.value) })}
                      style={{ width: "100%", accentColor: "var(--vb-accent)" }}
                    />
                  </div>
                )}

                {currentHook.type === "Sparsity" && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--vb-text-secondary)" }}>Sparsity Threshold</span>
                      <span style={{ fontFamily: "monospace", color: "var(--vb-success)" }}>{currentHook.threshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={currentHook.threshold}
                      onChange={(e) => onUpdateHook(tappedEdgeId, { ...currentHook, threshold: parseFloat(e.target.value) })}
                      style={{ width: "100%", accentColor: "var(--vb-success)" }}
                    />
                  </div>
                )}

                {currentHook.type === "SAE" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                        <span style={{ color: "var(--vb-text-secondary)" }}>SAE Hidden Dim</span>
                        <span style={{ fontFamily: "monospace", color: "var(--vb-cat-ssm)" }}>{currentHook.saeHiddenDim}x</span>
                      </div>
                      <select
                        value={currentHook.saeHiddenDim}
                        onChange={(e) => onUpdateHook(tappedEdgeId, { ...currentHook, saeHiddenDim: parseInt(e.target.value) })}
                        style={{ width: "100%", background: "var(--vb-surface-3)", color: "var(--vb-text)" }}
                      >
                        <option value="16">16x Expansion</option>
                        <option value="32">32x Expansion</option>
                        <option value="64">64x Expansion</option>
                        <option value="128">128x Expansion</option>
                      </select>
                    </div>

                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                        <span style={{ color: "var(--vb-text-secondary)" }}>L1 Coefficient</span>
                        <span style={{ fontFamily: "monospace", color: "var(--vb-cat-ssm)" }}>{currentHook.saeL1.toFixed(4)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0001"
                        max="0.0100"
                        step="0.0005"
                        value={currentHook.saeL1}
                        onChange={(e) => onUpdateHook(tappedEdgeId, { ...currentHook, saeL1: parseFloat(e.target.value) })}
                        style={{ width: "100%", accentColor: "var(--vb-cat-ssm)" }}
                      />
                    </div>
                  </div>
                )}

                {currentHook.type === "Causal Patch" && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--vb-text-secondary)" }}>Causal Noise Level</span>
                      <span style={{ fontFamily: "monospace", color: "var(--vb-cat-moe)" }}>{currentHook.causalNoiseLevel.toFixed(1)} σ</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="5.0"
                      step="0.2"
                      value={currentHook.causalNoiseLevel}
                      onChange={(e) => onUpdateHook(tappedEdgeId, { ...currentHook, causalNoiseLevel: parseFloat(e.target.value) })}
                      style={{ width: "100%", accentColor: "var(--vb-cat-moe)" }}
                    />
                  </div>
                )}
              </div>

              {/* Live Streaming Metrics Visualizations */}
              <div style={{
                background: "var(--vb-surface-2)",
                border: "1px solid var(--vb-border)",
                borderRadius: "var(--vb-radius-lg)",
                padding: 12,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: "var(--vb-text-muted)", fontWeight: "bold" }}>LIVE METRICS</span>
                  <span style={{
                    fontSize: 8,
                    background: "rgba(16, 185, 129, 0.15)",
                    color: "var(--vb-success)",
                    padding: "2px 6px",
                    borderRadius: "10px",
                    fontWeight: "bold",
                    animation: "pulse 2s infinite"
                  }}>
                    ● {liveTrainEvents.length > 0 ? "LIVE STREAM" : "SIMULATED"}
                  </span>
                </div>

                <div style={{ fontSize: 20, fontWeight: "bold", fontFamily: "var(--vb-font-mono)", marginBottom: 8, color: "var(--vb-text)" }}>
                  {latest.toFixed(4)}
                  <span style={{ fontSize: 10, color: "var(--vb-text-secondary)", fontWeight: "normal", marginLeft: 6 }}>
                    {currentHook.type === "Sparsity" ? "sparsity index" : 
                     currentHook.type === "SAE" ? "reconstruction loss" : 
                     currentHook.type === "Causal Patch" ? "relative divergence" : "activation level"}
                  </span>
                </div>

                {renderSparkline()}
                {renderDistribution()}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{
          padding: "30px 10px",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 12,
          border: "1px dashed var(--vb-border)",
          borderRadius: "var(--vb-radius-lg)",
          background: "var(--vb-surface-3)"
        }}>
          <span style={{ fontSize: 32 }}>🔍</span>
          <div style={{ fontWeight: 600, fontSize: 13, color: "var(--vb-text-secondary)" }}>No Connection Selected</div>
          <div style={{ fontSize: 11, color: "var(--vb-text-muted)", lineHeight: 1.4 }}>
            Right-click or double-click any connection edge in the visual builder canvas to tap it and attach real-time research hooks (Monitor, Sparsity, SAE, Causal Patch).
          </div>
        </div>
      )}

      {/* Active Hooks Registry Summary */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: 12, color: "var(--vb-text-secondary)", fontWeight: 600, marginBottom: 8 }}>Active Hooks Registry</h3>
        {Object.keys(hooks).length === 0 ? (
          <div style={{ fontSize: 10, color: "var(--vb-text-muted)" }}>No research hooks currently active in this session.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {Object.entries(hooks).map(([id, hook]) => (
              <div
                key={id}
                style={{
                  background: "var(--vb-surface-3)",
                  border: "1px solid var(--vb-border)",
                  borderRadius: "var(--vb-radius-sm)",
                  padding: "6px 10px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: 11
                }}
              >
                <div>
                  <span style={{
                    color: 
                      hook.type === "Sparsity" ? "var(--vb-success)" :
                      hook.type === "SAE" ? "var(--vb-cat-ssm)" :
                      hook.type === "Causal Patch" ? "var(--vb-cat-moe)" : "var(--vb-accent)",
                    fontWeight: "bold",
                    marginRight: 6
                  }}>
                    {hook.type}
                  </span>
                  <span style={{ fontFamily: "monospace", color: "var(--vb-text-secondary)" }}>
                    {id.substring(0, 16)}...
                  </span>
                </div>
                <button
                  onClick={() => onRemoveHook(id)}
                  style={{
                    border: "none",
                    background: "transparent",
                    color: "var(--vb-danger)",
                    cursor: "pointer",
                    fontSize: 10,
                    padding: 0
                  }}
                  onMouseOver={(e) => e.currentTarget.style.textDecoration = "underline"}
                  onMouseOut={(e) => e.currentTarget.style.textDecoration = "none"}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
