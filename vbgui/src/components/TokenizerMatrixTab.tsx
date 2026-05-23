// V7-F55 — preset × tokenizer compatibility matrix.
// Each cell shows a status pill driven by a real tokenizer.encode_visualize
// roundtrip. Cells start in 'idle'; user clicks "Probe all" to fill
// the grid (sequential, ~5-50ms each) or clicks individual cells to
// fetch one. Click any populated cell to expand the inline panel with
// the first 10 token ids it produced.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { HelpIcon } from "@/components/HelpIcon";

const PROBE_TEXT = "def hello():\n    print('hello, world')\n";

export interface TokenizerMatrixTabProps {
  rpc: RpcClient | null;
  presets: readonly string[];
  tokenizers: readonly string[];
}

type Status = "idle" | "ok" | "incompat" | "error";

interface CellState {
  status: Status;
  token_count?: number;
  bytes_per_token_avg?: number;
  first_token_ids?: number[];
  error?: string;
}

function cellKey(preset: string, tokenizer: string): string {
  return `${preset}::${tokenizer}`;
}

const PILL: Record<Status, { bg: string; fg: string; label: string }> = {
  idle:     { bg: "var(--vb-surface-3)",
              fg: "var(--vb-text-muted)",  label: "—" },
  ok:       { bg: "rgba(52, 211, 153, 0.16)",
              fg: "var(--vb-success)",     label: "ok" },
  incompat: { bg: "rgba(251, 191, 36, 0.16)",
              fg: "var(--vb-warning)",     label: "incompat" },
  error:    { bg: "rgba(248, 113, 113, 0.16)",
              fg: "var(--vb-danger)",      label: "error" },
};

export function TokenizerMatrixTab({
  rpc, presets, tokenizers,
}: TokenizerMatrixTabProps): JSX.Element {
  const [cells, setCells] = useState<Map<string, CellState>>(new Map());
  const [running, setRunning] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);

  async function probeOne(preset: string, tokenizer: string) {
    setCells((prev) => {
      const next = new Map(prev);
      next.set(cellKey(preset, tokenizer), { status: "idle" });
      return next;
    });
    try {
      if (!rpc) throw new Error("rpc unavailable");
      const r = await rpc.call<{
        tokens: { id: number }[];
        token_count: number;
        bytes_per_token_avg: number;
      }>("tokenizer.encode_visualize", {
        tokenizer_source: tokenizer,
        text: PROBE_TEXT,
      });
      const status: Status = r.token_count > 0 ? "ok" : "incompat";
      const ids = r.tokens.slice(0, 10).map((t) => t.id);
      setCells((prev) => {
        const next = new Map(prev);
        next.set(cellKey(preset, tokenizer), {
          status,
          token_count: r.token_count,
          bytes_per_token_avg: r.bytes_per_token_avg,
          first_token_ids: ids,
        });
        return next;
      });
    } catch (e) {
      setCells((prev) => {
        const next = new Map(prev);
        next.set(cellKey(preset, tokenizer), {
          status: "error", error: (e as Error).message,
        });
        return next;
      });
    }
  }

  async function probeAll() {
    setRunning(true);
    for (const p of presets) {
      for (const t of tokenizers) {
        await probeOne(p, t);
      }
    }
    setRunning(false);
  }

  return (
    <div data-testid="tokenizer-matrix-tab"
         style={{ padding: 12, fontFamily: "system-ui, sans-serif",
                  fontSize: 12, overflowY: "auto", flex: 1 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8,
                    marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>
          Tokenizer × preset compatibility matrix
        </h3>
        <HelpIcon topic="tokenizer_matrix" />
        <button data-testid="tokmatrix-probe-all"
                disabled={running}
                onClick={probeAll}
                style={{ padding: "4px 10px", border: "none",
                         borderRadius: 4,
                         background: running ? "#e5e7eb" : "#2563eb",
                         color: running ? "#6b7280" : "white",
                         cursor: running ? "wait" : "pointer" }}>
          {running ? "Probing…" : "Probe all cells"}
        </button>
      </div>
      <table data-testid="tokenizer-matrix-table"
             style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
            <th style={th}>preset \ tokenizer</th>
            {tokenizers.map((t) => (
              <th key={t} style={th}
                  title={t}
                  data-testid={`tokmatrix-col-${tokFileLabel(t)}`}>
                {tokFileLabel(t)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {presets.map((p) => (
            <tr key={p} data-testid={`tokmatrix-row-${p}`}
                style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ ...td, fontWeight: 600 }}>{p}</td>
              {tokenizers.map((t) => {
                const key = cellKey(p, t);
                const c = cells.get(key) ?? { status: "idle" as Status };
                const pill = PILL[c.status];
                const isExpanded = expanded === key;
                const tokLabel = tokFileLabel(t);
                return (
                  <td key={t} style={td}
                      data-testid={`tokmatrix-${p}-${tokLabel}`}
                      data-status={c.status}>
                    <button
                      data-testid={`tokmatrix-${p}-${tokLabel}-pill`}
                      onClick={() => {
                        if (c.status === "idle") probeOne(p, t);
                        else setExpanded(isExpanded ? null : key);
                      }}
                      style={{ background: pill.bg, color: pill.fg,
                               border: "none", padding: "2px 8px",
                               borderRadius: 9999, cursor: "pointer",
                               fontWeight: 600, fontSize: 11 }}>
                      {pill.label}
                    </button>
                    {isExpanded && c.first_token_ids && (
                      <div data-testid={`tokmatrix-${p}-${tokLabel}-expand`}
                           style={{ marginTop: 4, fontFamily: "monospace",
                                    fontSize: 11, color: "#374151" }}>
                        ids: [{c.first_token_ids.join(", ")}]
                        <br />
                        count: {c.token_count}, bpt: {c.bytes_per_token_avg?.toFixed(2)}
                      </div>
                    )}
                    {isExpanded && c.error && (
                      <div data-testid={`tokmatrix-${p}-${tokLabel}-expand-error`}
                           style={{ marginTop: 4, color: "#991b1b",
                                    fontSize: 11 }}>
                        {c.error}
                      </div>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function tokFileLabel(path: string): string {
  // Strip directory + extension so cell testids stay readable.
  const base = path.split("/").pop() ?? path;
  return base.replace(/\.json$/, "");
}

const th: React.CSSProperties = {
  textAlign: "left", padding: "4px 6px",
  color: "#374151", fontWeight: 600,
};
const td: React.CSSProperties = { padding: "4px 6px",
                                  verticalAlign: "top" };
