import { T } from "@/theme";

export interface LossSeries {
  label: string;
  values: number[];
  color?: string;
}

export interface LossChartProps {
  // Single primary curve (loss). Extra overlays via `series`.
  losses: number[];
  series?: readonly LossSeries[];
  width?: number;
  height?: number;
  testidPrefix?: string;
  /** V7-L38: per-step overflow markers (step indices). Rendered as
   *  red vertical bars at the matching x positions on the primary
   *  loss curve so the user can see scaler overflow events without
   *  scanning extras text. */
  overflowSteps?: readonly number[];
}

const DEFAULT_COLOR = "#22d3ee";
const PALETTE = ["#22d3ee", "#34d399", "#fbbf24", "#ec4899", "#a855f7"];

function pathFor(values: number[], w: number, h: number,
                 yMin: number, yMax: number, pad: number): string {
  if (values.length === 0) return "";
  const span = yMax - yMin || 1;
  const innerW = w - 2 * pad;
  const innerH = h - 2 * pad;
  return values
    .map((v, i) => {
      const x = pad + (values.length === 1
        ? innerW / 2
        : (i / (values.length - 1)) * innerW);
      const y = pad + innerH - ((v - yMin) / span) * innerH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function LossChart({
  losses, series = [], width = 360, height = 140, testidPrefix = "chart",
  overflowSteps = [],
}: LossChartProps): JSX.Element {
  // Whether the primary "loss" series has any data. When it doesn't,
  // we still keep overlay series visible but treat them as named
  // overlays (label-suffixed testids), not as the bare primary line.
  const primaryHasData = losses.length > 0;
  const allSeries: LossSeries[] = [
    ...(primaryHasData
      ? [{ label: "loss", values: losses, color: DEFAULT_COLOR }]
      : []),
    ...series.map((s, i) => ({
      ...s,
      color: s.color ?? PALETTE[(i + 1) % PALETTE.length],
    })),
  ].filter((s) => s.values.length > 0);

  const pad = 24;
  const allValues = allSeries.flatMap((s) => s.values);
  const yMin = allValues.length ? Math.min(...allValues) : 0;
  const yMax = allValues.length ? Math.max(...allValues) : 1;
  const xMaxStep = Math.max(0, ...allSeries.map((s) => s.values.length - 1));

  return (
    <div data-testid={testidPrefix} style={{ fontFamily: T.font }}>
      <svg
        data-testid={`${testidPrefix}-svg`}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="training loss chart"
        style={{ background: T.surface2, border: `1px solid ${T.border}`,
                 borderRadius: 4 }}
      >
        {/* axes */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad}
              stroke={T.borderStrong} strokeWidth={1} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad}
              stroke={T.borderStrong} strokeWidth={1} />

        {/* y-axis labels */}
        <text data-testid={`${testidPrefix}-axis-y-label-max`}
              x={4} y={pad + 4} fontSize={10} fill={T.textSecondary}>
          {yMax.toFixed(3)}
        </text>
        <text data-testid={`${testidPrefix}-axis-y-label-min`}
              x={4} y={height - pad} fontSize={10} fill={T.textSecondary}>
          {yMin.toFixed(3)}
        </text>

        {/* x-axis labels */}
        <text data-testid={`${testidPrefix}-axis-x-label-0`}
              x={pad} y={height - 4} fontSize={10} fill={T.textSecondary}>
          0
        </text>
        <text data-testid={`${testidPrefix}-axis-x-label-last`}
              x={width - pad - 8} y={height - 4} fontSize={10} fill={T.textSecondary}>
          {xMaxStep}
        </text>

        {/* V7-L38: overflow markers — red vertical bar at each step
            index. Drawn before the curves so the line + points sit
            on top. Only meaningful when the primary loss series has
            enough points to map step→x. */}
        {overflowSteps.length > 0 && losses.length > 0 && (() => {
          const innerW = width - 2 * pad;
          return overflowSteps.map((step, i) => {
            if (step < 0 || step >= losses.length) return null;
            const x = pad + (losses.length === 1
              ? innerW / 2
              : (step / (losses.length - 1)) * innerW);
            return (
              <line key={i}
                    data-testid={`${testidPrefix}-overflow-${step}`}
                    x1={x} y1={pad} x2={x} y2={height - pad}
                    stroke="#dc2626" strokeWidth={2}
                    strokeDasharray="4 2"
                    opacity={0.7}>
                <title>overflow at step {step}</title>
              </line>
            );
          });
        })()}
        {allSeries.map((s, sIdx) => {
          const d = pathFor(s.values, width, height, yMin, yMax, pad);
          // Only the canonical primary "loss" series at index 0 gets
          // the bare testid; everything else is label-suffixed so
          // sweep-style overlays don't collide.
          const isPrimary = sIdx === 0 && s.label === "loss";
          const lineTid = isPrimary
            ? `${testidPrefix}-line`
            : `${testidPrefix}-line-${s.label}`;
          return (
            <g key={s.label} data-testid={`${testidPrefix}-series-${s.label}`}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2}
                    data-testid={lineTid} />
              {s.values.map((v, i) => {
                const innerW = width - 2 * pad;
                const innerH = height - 2 * pad;
                const span = yMax - yMin || 1;
                const x = pad + (s.values.length === 1
                  ? innerW / 2
                  : (i / (s.values.length - 1)) * innerW);
                const y = pad + innerH - ((v - yMin) / span) * innerH;
                const pointTid = isPrimary
                  ? `${testidPrefix}-point-${i}`
                  : `${testidPrefix}-point-${s.label}-${i}`;
                return (
                  <circle key={i} cx={x} cy={y} r={3} fill={s.color}
                          data-testid={pointTid}
                          data-loss-value={v.toFixed(6)} />
                );
              })}
            </g>
          );
        })}
      </svg>
      {allSeries.length > 1 && (
        <ul data-testid={`${testidPrefix}-legend`}
            style={{ display: "flex", gap: 12, margin: "4px 0 0 0",
                     padding: 0, listStyle: "none", fontSize: 11 }}>
          {allSeries.map((s) => (
            <li key={s.label}
                data-testid={`${testidPrefix}-legend-${s.label}`}
                style={{ display: "inline-flex", alignItems: "center",
                         gap: 4, color: "#374151" }}>
              <span style={{ width: 10, height: 10, background: s.color,
                             display: "inline-block", borderRadius: 2 }} />
              {s.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
