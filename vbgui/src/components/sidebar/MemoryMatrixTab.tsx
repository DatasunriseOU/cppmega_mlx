/**
 * V8-R03 MemoryMatrixTab — sidebar tab that renders a topology ×
 * precision grid of memory-fit verdicts for the current canvas spec.
 *
 * Calls `memory.matrix` whenever the spec hash changes; each cell is
 * coloured green if fits=true, red otherwise, with a tooltip showing
 * the per-component breakdown (weights / grads / optimizer / activ /
 * kv-cache / edge handoff).
 */

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

interface MemoryMatrixCell {
  topology: string;
  precision: string;
  bytes: number;
  device_hbm_bytes: number;
  fits: boolean;
  headroom: number;
  breakdown: Record<string, number>;
}

interface MemoryMatrixResult {
  cells: MemoryMatrixCell[];
  topologies: string[];
  precisions: string[];
}

export interface MemoryMatrixTabProps {
  rpc: RpcClient;
  /** Stringified VerifyParams payload — when this changes we refetch. */
  specPayload: unknown;
  topologies?: string[];
  precisions?: string[];
}

const ONE_GB = 1_073_741_824;
function fmtBytes(n: number): string {
  if (n >= ONE_GB) return `${(n / ONE_GB).toFixed(2)} GB`;
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(0)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export function MemoryMatrixTab({
  rpc, specPayload, topologies, precisions,
}: MemoryMatrixTabProps): JSX.Element {
  const [matrix, setMatrix] = useState<MemoryMatrixResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const r = await rpc.call<MemoryMatrixResult>(
          "memory.matrix",
          { spec: specPayload, topologies, precisions },
        );
        if (!cancelled) setMatrix(r);
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [rpc, specPayload, topologies, precisions]);

  if (loading && !matrix) {
    return (
      <div data-testid="memory-matrix-loading" style={panel}>loading…</div>
    );
  }
  if (err) {
    return (
      <div data-testid="memory-matrix-error" style={{ ...panel,
           color: "#b91c1c" }}>{err}</div>
    );
  }
  if (!matrix) {
    return <div data-testid="memory-matrix-empty" style={panel}>—</div>;
  }

  return (
    <div data-testid="memory-matrix" style={panel}>
      <h3 style={{ margin: "0 0 8px", fontSize: 14 }}>
        Memory matrix (topology × precision)
      </h3>
      <table style={{ borderCollapse: "collapse", fontSize: 11,
                      tableLayout: "fixed", width: "100%" }}>
        <thead>
          <tr>
            <th style={th}></th>
            {matrix.precisions.map((p) => (
              <th key={p} style={th} data-testid={`memory-matrix-col-${p}`}>
                {p}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.topologies.map((t) => (
            <tr key={t}>
              <th style={{ ...th, textAlign: "right" }}
                  data-testid={`memory-matrix-row-${t}`}>
                {t}
              </th>
              {matrix.precisions.map((p) => {
                const cell = matrix.cells.find(
                  (c) => c.topology === t && c.precision === p);
                if (!cell) {
                  return (
                    <td key={p} style={tdMissing}
                        data-testid={`memory-matrix-cell-${t}-${p}`}>
                      —
                    </td>
                  );
                }
                const bg = cell.fits ? "rgba(52, 211, 153, 0.16)" : "rgba(248, 113, 113, 0.16)";
                const color = cell.fits ? "var(--vb-success)" : "var(--vb-danger)";
                const title = (
                  `${fmtBytes(cell.bytes)} of ${fmtBytes(cell.device_hbm_bytes)}` +
                  ` (headroom ${(cell.headroom * 100).toFixed(0)}%)\n` +
                  Object.entries(cell.breakdown)
                    .map(([k, v]) => `  ${k}: ${fmtBytes(v)}`)
                    .join("\n")
                );
                return (
                  <td key={p}
                      data-testid={`memory-matrix-cell-${t}-${p}`}
                      title={title}
                      style={{ ...td, background: bg, color, borderColor: cell.fits ? "var(--vb-success)" : "var(--vb-danger)" }}>
                    <span data-testid={`memory-matrix-cell-bytes-${t}-${p}`}>
                      {fmtBytes(cell.bytes)}
                    </span>
                    <br />
                    <span
                      data-testid={`memory-matrix-cell-fits-${t}-${p}`}
                      data-fits={cell.fits}>
                      {cell.fits ? "fits" : "over"}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {loading && (
        <p data-testid="memory-matrix-refreshing" style={{ fontSize: 10,
           color: "var(--vb-text-muted)", margin: "4px 0 0" }}>
          refreshing…
        </p>
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  padding: 12, fontFamily: "system-ui, sans-serif", fontSize: 12,
  display: "flex", flexDirection: "column", gap: 6,
};

const th: React.CSSProperties = {
  padding: "4px 6px", fontWeight: 600, color: "var(--vb-text-secondary)",
  borderBottom: "1px solid var(--vb-border)", textAlign: "center",
};

const td: React.CSSProperties = {
  padding: "6px 8px", textAlign: "center", borderRadius: 4,
  border: "1px solid var(--vb-border)",
};

const tdMissing: React.CSSProperties = {
  ...td, background: "var(--vb-surface-2)", color: "var(--vb-text-muted)",
};
