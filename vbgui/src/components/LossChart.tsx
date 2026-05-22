// Inline SVG loss / lr line chart used by RunResultModal (and any future
// live-training surface). No external charting deps so e2e can assert on
// visible coordinates and the rendered <path> element itself, not on the
// flat text list of extras.losses-N items.

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
}

const DEFAULT_COLOR = "#2563eb";
const PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#ec4899", "#a855f7"];

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
}: LossChartProps): JSX.Element {
  const allSeries: LossSeries[] = [
    { label: "loss", values: losses, color: DEFAULT_COLOR },
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
    <div data-testid={testidPrefix} style={{ fontFamily: "system-ui, sans-serif" }}>
      <svg
        data-testid={`${testidPrefix}-svg`}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="training loss chart"
        style={{ background: "#ffffff", border: "1px solid #e5e7eb",
                 borderRadius: 4 }}
      >
        {/* axes */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad}
              stroke="#9ca3af" strokeWidth={1} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad}
              stroke="#9ca3af" strokeWidth={1} />

        {/* y-axis labels */}
        <text data-testid={`${testidPrefix}-axis-y-label-max`}
              x={4} y={pad + 4} fontSize={10} fill="#6b7280">
          {yMax.toFixed(3)}
        </text>
        <text data-testid={`${testidPrefix}-axis-y-label-min`}
              x={4} y={height - pad} fontSize={10} fill="#6b7280">
          {yMin.toFixed(3)}
        </text>

        {/* x-axis labels */}
        <text data-testid={`${testidPrefix}-axis-x-label-0`}
              x={pad} y={height - 4} fontSize={10} fill="#6b7280">
          0
        </text>
        <text data-testid={`${testidPrefix}-axis-x-label-last`}
              x={width - pad - 8} y={height - 4} fontSize={10} fill="#6b7280">
          {xMaxStep}
        </text>

        {allSeries.map((s, sIdx) => {
          const d = pathFor(s.values, width, height, yMin, yMax, pad);
          const isPrimary = sIdx === 0;
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
