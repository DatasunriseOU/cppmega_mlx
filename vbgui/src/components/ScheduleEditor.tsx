// Schedule editor — per-ParamGroup LR schedule picker with a tiny
// SVG sparkline preview. Mirrors cppmega_v4/buildspec/schedules.py.

import type { ScheduleKind, ScheduleSpecState } from "@/state/spec";

const SCHEDULE_KINDS: ScheduleKind[] = [
  "constant", "linear_warmup", "cosine", "wsd", "inv_sqrt", "polynomial",
];

export interface ScheduleEditorProps {
  index: number;
  baseLr: number;
  value?: ScheduleSpecState;
  onChange: (next: ScheduleSpecState | undefined) => void;
  /** V7-H45: schedule_kind actually executed by the most-recent train
   *  run (from extras.schedule_kind). When set + differs from the
   *  currently-selected kind, surface a hint so the user can tell
   *  what the backend really used (e.g. fallback to 'constant' when
   *  total_steps was missing). */
  lastRunScheduleKind?: string | null;
}

const FIELD: React.CSSProperties = {
  display: "inline-flex", flexDirection: "column", marginRight: 8,
  fontSize: 11,
};

/** Sample N points along the schedule for a sparkline preview.
 *  Pure JS mirror of ScheduleSpec.sample() in Python. */
function sampleSchedule(s: ScheduleSpecState, baseLr: number,
                       nPoints = 50): number[] {
  const warmup = s.warmup_steps ?? 0;
  const total = s.total_steps;
  const floor = baseLr * (s.min_lr_ratio ?? 0.1);
  const decay = s.decay_steps ?? 0;
  const power = s.power ?? 2.0;

  let horizon: number;
  if (total != null) horizon = total;
  else if (warmup > 0) horizon = Math.max(warmup * 2, nPoints);
  else horizon = nPoints;
  const stepSize = Math.max(1, Math.floor(horizon / nPoints));

  const out: number[] = [];
  for (let i = 0; i < nPoints; i++) {
    const step = i * stepSize;
    let lr: number;
    if (s.kind === "constant") {
      lr = baseLr;
    } else if (s.kind === "linear_warmup") {
      lr = step < warmup ? baseLr * (step / Math.max(1, warmup)) : baseLr;
    } else if (s.kind === "cosine" && total != null) {
      if (step < warmup) lr = baseLr * (step / Math.max(1, warmup));
      else {
        const p = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
        const c = 0.5 * (1 + Math.cos(Math.PI * p));
        lr = floor + (baseLr - floor) * c;
      }
    } else if (s.kind === "wsd" && total != null && decay >= 1) {
      const steadyEnd = total - decay;
      if (step < warmup) lr = baseLr * (step / Math.max(1, warmup));
      else if (step < steadyEnd) lr = baseLr;
      else {
        const p = Math.min(1, (step - steadyEnd) / Math.max(1, decay));
        lr = baseLr + (floor - baseLr) * p;
      }
    } else if (s.kind === "inv_sqrt") {
      const scale = Math.max(1, warmup);
      lr = step < warmup
        ? baseLr * (step / Math.max(1, warmup))
        : baseLr * Math.sqrt(scale / Math.max(1, step));
    } else if (s.kind === "polynomial" && total != null) {
      if (step < warmup) lr = baseLr * (step / Math.max(1, warmup));
      else {
        const p = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
        lr = floor + (baseLr - floor) * Math.pow(1 - p, power);
      }
    } else {
      lr = baseLr;
    }
    out.push(lr);
  }
  return out;
}

/** SVG mini-sparkline (120×30 px). */
function Sparkline({ values }: { values: number[] }): JSX.Element {
  const max = Math.max(...values, 1e-12);
  const min = Math.min(...values);
  const span = max - min || 1;
  const width = 120;
  const height = 30;
  const pts = values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * (width - 2) + 1;
    const y = height - 2 - ((v - min) / span) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={width} height={height}
         data-testid="schedule-sparkline"
         style={{ background: "#f9fafb", border: "1px solid #e5e7eb",
                  borderRadius: 3 }}>
      <polyline fill="none" stroke="#2563eb" strokeWidth="1.5"
                points={pts} />
    </svg>
  );
}

export function ScheduleEditor({
  index, baseLr, value, onChange, lastRunScheduleKind = null,
}: ScheduleEditorProps): JSX.Element {
  const kind = value?.kind ?? "constant";

  function setField<K extends keyof ScheduleSpecState>(
    field: K, val: ScheduleSpecState[K],
  ) {
    onChange({ ...(value ?? { kind: "constant" }), [field]: val });
  }
  function setKind(k: ScheduleKind) {
    if (k === "constant") onChange(undefined);
    else onChange({ ...(value ?? { kind: "constant" }), kind: k });
  }

  const sampled = value && kind !== "constant"
    ? sampleSchedule(value, baseLr) : null;

  return (
    <div data-testid={`schedule-editor-${index}`}
         style={{ marginTop: 4, padding: 6,
                  border: "1px dashed #d1d5db", borderRadius: 4 }}>
      <label style={FIELD}>
        <span style={{ color: "#6b7280" }}>Schedule</span>
        <select data-testid={`schedule-kind-${index}`}
                value={kind}
                onChange={(e) => setKind(e.target.value as ScheduleKind)}>
          {SCHEDULE_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>
      {/* V7-H45: echo backend's actual schedule_kind from the most-
          recent train run. Surfaces silent fallbacks (e.g. the user
          picked 'cosine' but missed total_steps so backend ran
          'constant'). */}
      {lastRunScheduleKind && (
        <span data-testid={`schedule-last-run-${index}`}
              style={{ marginLeft: 4, fontSize: 10,
                       padding: "1px 4px", borderRadius: 3,
                       background: lastRunScheduleKind === kind
                                  ? "#d1fae5" : "#fef3c7",
                       color: lastRunScheduleKind === kind
                                  ? "#065f46" : "#92400e" }}>
          last run: {lastRunScheduleKind}
          {lastRunScheduleKind !== kind && " (≠ selected)"}
        </span>
      )}
      {kind !== "constant" && (
        <label style={FIELD}>
          <span style={{ color: "#6b7280" }}>warmup_steps</span>
          <input data-testid={`schedule-warmup-${index}`}
                 type="number" min={0} step={1} style={{ width: 70 }}
                 value={value?.warmup_steps ?? 0}
                 onChange={(e) =>
                   setField("warmup_steps", Number(e.target.value))} />
        </label>
      )}
      {(kind === "cosine" || kind === "wsd" || kind === "polynomial") && (
        <label style={FIELD}>
          <span style={{ color: "#6b7280" }}>total_steps</span>
          <input data-testid={`schedule-total-${index}`}
                 type="number" min={1} step={1} style={{ width: 80 }}
                 value={value?.total_steps ?? 100}
                 onChange={(e) =>
                   setField("total_steps", Number(e.target.value))} />
        </label>
      )}
      {(kind === "cosine" || kind === "wsd" || kind === "polynomial") && (
        <label style={FIELD}>
          <span style={{ color: "#6b7280" }}>min_lr_ratio</span>
          <input data-testid={`schedule-min-ratio-${index}`}
                 type="number" min={0} max={1} step={0.01}
                 style={{ width: 65 }}
                 value={value?.min_lr_ratio ?? 0.1}
                 onChange={(e) =>
                   setField("min_lr_ratio", Number(e.target.value))} />
        </label>
      )}
      {kind === "wsd" && (
        <label style={FIELD}>
          <span style={{ color: "#6b7280" }}>decay_steps</span>
          <input data-testid={`schedule-decay-${index}`}
                 type="number" min={1} step={1} style={{ width: 70 }}
                 value={value?.decay_steps ?? 100}
                 onChange={(e) =>
                   setField("decay_steps", Number(e.target.value))} />
        </label>
      )}
      {kind === "polynomial" && (
        <label style={FIELD}>
          <span style={{ color: "#6b7280" }}>power</span>
          <input data-testid={`schedule-power-${index}`}
                 type="number" min={0.1} step={0.1} style={{ width: 60 }}
                 value={value?.power ?? 2.0}
                 onChange={(e) =>
                   setField("power", Number(e.target.value))} />
        </label>
      )}
      {sampled && (
        <div style={{ marginTop: 4 }}>
          <Sparkline values={sampled} />
        </div>
      )}
    </div>
  );
}
