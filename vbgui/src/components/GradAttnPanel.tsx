// V7-H07: per-brick grad-norm + attention-head heatmap visualisation.
// Reads extras.per_brick_grad_norms (dict<str, float>) and
// extras.attn_head_means (dict<str, float[]>) from stage_train and
// renders two color-mapped panels:
//   1. Horizontal bar chart of per-brick grad-norm magnitudes,
//      log-scaled, colored by relative magnitude.
//   2. Per-attention-block grid: one row per attn brick, one cell per
//      head, colored by head's mean attention weight.
//
// No external charting deps — pure SVG so Playwright can assert on
// rect counts + fill attributes.

import type { JSX } from "react";

export interface GradAttnPanelProps {
  gradNorms?: Record<string, number>;
  attnHeadMeans?: Record<string, number[]>;
}

function colorRamp(t: number): string {
  // t ∈ [0,1] → blue→red gradient.
  const r = Math.round(255 * Math.min(1, Math.max(0, t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1 - t)));
  return `rgb(${r}, 60, ${b})`;
}

export function GradAttnPanel({
  gradNorms, attnHeadMeans,
}: GradAttnPanelProps): JSX.Element | null {
  const hasGrads = gradNorms && Object.keys(gradNorms).length > 0;
  const hasAttn = attnHeadMeans && Object.keys(attnHeadMeans).length > 0;
  if (!hasGrads && !hasAttn) return null;

  return (
    <div data-testid="grad-attn-panel"
         style={{ border: "1px solid #e5e7eb", borderRadius: 4,
                  padding: 8, background: "#f9fafb", fontSize: 11,
                  fontFamily: "monospace" }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>
        per-brick grad + attention map (V7-H07)
      </div>

      {hasGrads && (
        <div data-testid="grad-attn-panel-grads"
             style={{ marginBottom: 8 }}>
          <div style={{ color: "#6b7280", marginBottom: 2 }}>
            grad-norm by brick
          </div>
          <GradBars norms={gradNorms!} />
        </div>
      )}

      {hasAttn && (
        <div data-testid="grad-attn-panel-attn">
          <div style={{ color: "#6b7280", marginBottom: 2 }}>
            attention-head mean weight
          </div>
          <AttnHeatmap means={attnHeadMeans!} />
        </div>
      )}
    </div>
  );
}

function GradBars({ norms }: { norms: Record<string, number> }):
  JSX.Element {
  const entries = Object.entries(norms);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);
  const W = 220;
  const rowH = 14;
  const labelW = 90;
  return (
    <svg data-testid="grad-attn-panel-grads-svg"
         width={W + labelW} height={entries.length * rowH}>
      {entries.map(([k, v], i) => {
        const t = max > 0 ? v / max : 0;
        const w = Math.max(2, t * W);
        return (
          <g key={k} transform={`translate(0, ${i * rowH})`}>
            <text x={0} y={rowH - 3}
                  data-testid={`grad-attn-panel-grad-label-${i}`}
                  style={{ fontSize: 10 }}>
              {k.length > 12 ? k.slice(0, 12) + "…" : k}
            </text>
            <rect x={labelW} y={2} width={w} height={rowH - 4}
                  fill={colorRamp(t)}
                  data-testid={`grad-attn-panel-grad-bar-${i}`}>
              <title>{`${k}: ${v.toExponential(3)}`}</title>
            </rect>
            <text x={labelW + w + 4} y={rowH - 3}
                  style={{ fontSize: 9, fill: "#374151" }}>
              {v.toExponential(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function AttnHeatmap({ means }: { means: Record<string, number[]> }):
  JSX.Element {
  const entries = Object.entries(means);
  const cellW = 14, cellH = 14;
  const labelW = 90;
  const maxHeads = Math.max(1, ...entries.map(([, v]) => v.length));
  const W = labelW + maxHeads * cellW;
  return (
    <svg data-testid="grad-attn-panel-attn-svg"
         width={W} height={entries.length * cellH}>
      {entries.map(([brick, heads], r) => (
        <g key={brick} transform={`translate(0, ${r * cellH})`}>
          <text x={0} y={cellH - 3}
                data-testid={`grad-attn-panel-attn-label-${r}`}
                style={{ fontSize: 10 }}>
            {brick.length > 12 ? brick.slice(0, 12) + "…" : brick}
          </text>
          {heads.map((v, h) => {
            // Heads are softmax-mean values in [0, 1].
            const t = Math.min(1, Math.max(0, v));
            return (
              <rect key={h}
                    data-testid={`grad-attn-panel-attn-cell-${r}-${h}`}
                    x={labelW + h * cellW} y={1}
                    width={cellW - 1} height={cellH - 2}
                    fill={colorRamp(t)}>
                <title>{`${brick} h${h}: ${v.toFixed(4)}`}</title>
              </rect>
            );
          })}
        </g>
      ))}
    </svg>
  );
}
