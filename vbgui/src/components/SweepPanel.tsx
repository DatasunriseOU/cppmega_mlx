// V7-F53 dimension-scaling sweep panel. Runs the same preset four
// times with H ∈ {64, 128, 256, 512}, captures losses per H, and
// renders a multi-line LossChart overlay so the architect can see at
// a glance how loss scales with hidden size.

import { useState } from "react";
import { LossChart, type LossSeries } from "@/components/LossChart";
import { HelpIcon } from "@/components/HelpIcon";

export interface SweepRunner {
  // Runs a 2-step train at the given H and returns the per-step loss
  // trajectory. Caller is responsible for spinning up a real RPC
  // pipeline; SweepPanel is unit-testable with a fake runner.
  (H: number): Promise<number[]>;
}

export interface SweepPanelProps {
  runner: SweepRunner;
  hSizes?: readonly number[];
}

const DEFAULT_H = [64, 128, 256, 512] as const;

export function SweepPanel({
  runner, hSizes = DEFAULT_H,
}: SweepPanelProps): JSX.Element {
  const [results, setResults] = useState<Map<number, number[]>>(new Map());
  const [running, setRunning] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runSweep() {
    setError(null);
    setResults(new Map());
    for (const H of hSizes) {
      setRunning(H);
      try {
        const losses = await runner(H);
        setResults((prev) => new Map(prev).set(H, losses));
      } catch (e) {
        setError(`H=${H} failed: ${(e as Error).message}`);
        break;
      }
    }
    setRunning(null);
  }

  const series: LossSeries[] = hSizes
    .filter((H) => results.has(H))
    .map((H) => ({
      label: `H${H}`,
      values: results.get(H) ?? [],
    }));

  return (
    <div data-testid="sweep-panel"
         style={{ padding: 12, fontFamily: "system-ui, sans-serif",
                  fontSize: 12, display: "flex", flexDirection: "column",
                  gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>
          Mini → full dimension scaling sweep
        </h3>
        <HelpIcon topic="dim_env_H" />
        <button data-testid="scaling-sweep-run"
                onClick={runSweep}
                disabled={running !== null}
                style={{ padding: "4px 10px",
                         background: running !== null ? "#e5e7eb" : "#2563eb",
                         color: running !== null ? "var(--vb-text-muted)" : "white",
                         border: "none", borderRadius: 4,
                         cursor: running !== null ? "wait" : "pointer" }}>
          {running !== null
            ? `Running H=${running}…`
            : `Run sweep ${hSizes.join(", ")}`}
        </button>
        {running !== null && (
          <span data-testid="sweep-progress"
                style={{ color: "var(--vb-text-muted)" }}>
            {results.size}/{hSizes.length} complete
          </span>
        )}
      </div>
      {error && (
        <div data-testid="sweep-error"
             style={{ color: "#991b1b", background: "#fee2e2",
                      padding: 8, borderRadius: 4 }}>
          {error}
        </div>
      )}
      {results.size > 0 && (
        <LossChart
          losses={[]}
          series={series}
          width={520} height={200}
          testidPrefix="sweep-chart"
        />
      )}
    </div>
  );
}
