import { useCallback, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export interface PreviewRow {
  row_index: number;
  tokens: number[];
  channels: Record<string, unknown>;
}

export interface PreviewParquetResult {
  rows: PreviewRow[];
  token_column: string;
  available_channels: string[];
  bytes_per_token_avg: number;
  bytes_per_token_p95: number;
  bytes_per_token_max: number;
  total_rows: number;
  elapsed_ms: number;
}

export interface DataInspectorProps {
  rpc: RpcClient;
  initialPath?: string;
  pageSize?: number;
}

const CHANNEL_COLORS = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca",
                        "#ddd6fe", "#fed7aa", "#fbcfe8", "#cffafe"];

function colorForChannel(idx: number): string {
  return CHANNEL_COLORS[idx % CHANNEL_COLORS.length];
}

function isScalarRibbon(v: unknown): v is number {
  return typeof v === "number";
}

function isArrayRibbon(v: unknown): v is unknown[] {
  return Array.isArray(v);
}

export function DataInspector({
  rpc, initialPath = "", pageSize = 16,
}: DataInspectorProps): JSX.Element {
  const [path, setPath] = useState(initialPath);
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<PreviewParquetResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState<Set<string>>(new Set());

  const load = useCallback(async (nextOffset: number) => {
    if (!path) return;
    try {
      const r = await rpc.call<PreviewParquetResult>(
        "data.preview_parquet",
        { path, offset: nextOffset, limit: pageSize },
      );
      setResult((prev) => {
        // Reset channel toggles when the underlying schema changes
        // (different path OR different available_channels set).
        const schemaChanged =
          !prev ||
          prev.token_column !== r.token_column ||
          prev.available_channels.length !== r.available_channels.length ||
          prev.available_channels.some((c, i) => c !== r.available_channels[i]);
        if (schemaChanged) setEnabled(new Set(r.available_channels));
        return r;
      });
      setOffset(nextOffset);
      setError(null);
    } catch (e) {
      setError(String(e));
      setResult(null);
    }
  }, [path, pageSize, rpc]);

  const toggleChannel = useCallback((ch: string) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      next.has(ch) ? next.delete(ch) : next.add(ch);
      return next;
    });
  }, []);

  const totalPages = result ? Math.ceil(result.total_rows / pageSize) : 0;
  const currentPage = pageSize > 0 ? Math.floor(offset / pageSize) : 0;

  return (
    <div data-testid="data-inspector"
         style={{ display: "flex", flexDirection: "column",
                  height: "100%", padding: 12, gap: 8,
                  fontFamily: "system-ui, sans-serif" }}>
      <header style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Training Data Inspector</h3>
        <input data-testid="data-path"
               type="text" placeholder="/path/to/shard.parquet"
               value={path}
               onChange={(e) => setPath(e.target.value)}
               style={{ flex: 1, fontFamily: "monospace", fontSize: 11 }} />
        <button data-testid="data-load" onClick={() => load(0)}>
          Load
        </button>
      </header>

      {error && (
        <div data-testid="data-error"
             style={{ color: "#b91c1c", fontSize: 11 }}>
          {error}
        </div>
      )}

      {result && (
        <>
          <div data-testid="data-metrics"
               style={{ fontSize: 11, color: "#374151" }}>
            {result.total_rows} rows · token col = {result.token_column} ·
            bytes/tok avg {result.bytes_per_token_avg.toFixed(2)}
            {" "}p95 {result.bytes_per_token_p95.toFixed(2)}
            {" "}max {result.bytes_per_token_max}
          </div>

          <div data-testid="data-channels"
               style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {result.available_channels.map((ch, i) => (
              <label key={ch}
                     data-testid={`data-channel-toggle-${ch}`}
                     style={{ display: "inline-flex", gap: 4,
                              alignItems: "center",
                              padding: "2px 6px",
                              background: colorForChannel(i),
                              borderRadius: 3,
                              cursor: "pointer", fontSize: 11 }}>
                <input type="checkbox"
                       checked={enabled.has(ch)}
                       onChange={() => toggleChannel(ch)} />
                {ch}
              </label>
            ))}
          </div>

          <div data-testid="data-rows"
               style={{ flex: 1, overflowY: "auto",
                        border: "1px solid #e5e7eb",
                        borderRadius: 4 }}>
            {result.rows.map((row) => (
              <div key={row.row_index}
                   data-testid={`data-row-${row.row_index}`}
                   style={{ padding: 6, borderBottom: "1px solid #f3f4f6",
                            fontFamily: "monospace", fontSize: 11 }}>
                <div style={{ color: "#6b7280" }}>row #{row.row_index}</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 2,
                              padding: "2px 0" }}>
                  {row.tokens.map((tok, i) => (
                    <span key={i}
                          data-testid={`data-row-${row.row_index}-tok-${i}`}
                          style={{ background: "#e0e7ff",
                                   padding: "0 4px", borderRadius: 2 }}>
                      {tok}
                    </span>
                  ))}
                </div>
                {result.available_channels.filter((c) => enabled.has(c))
                                          .map((ch, ci) => (
                  <Ribbon key={ch} channel={ch} colorIdx={ci}
                          value={row.channels[ch]}
                          tokenCount={row.tokens.length}
                          rowIndex={row.row_index} />
                ))}
              </div>
            ))}
          </div>

          <footer data-testid="data-pagination"
                  style={{ display: "flex", gap: 8, fontSize: 11,
                           alignItems: "center" }}>
            <button data-testid="data-prev"
                    disabled={offset === 0}
                    onClick={() => load(Math.max(0, offset - pageSize))}>
              ←
            </button>
            <span>page {currentPage + 1} / {totalPages}</span>
            <button data-testid="data-next"
                    disabled={offset + pageSize >= result.total_rows}
                    onClick={() => load(offset + pageSize)}>
              →
            </button>
          </footer>
        </>
      )}
    </div>
  );
}


interface RibbonProps {
  channel: string;
  colorIdx: number;
  value: unknown;
  tokenCount: number;
  rowIndex: number;
}

function Ribbon({ channel, colorIdx, value, tokenCount, rowIndex }: RibbonProps): JSX.Element {
  const color = colorForChannel(colorIdx);
  if (isArrayRibbon(value)) {
    // Per-token strip — same length as token row when consistent
    return (
      <div data-testid={`data-ribbon-${rowIndex}-${channel}`}
           style={{ display: "flex", gap: 2, padding: "2px 0",
                    fontSize: 10, color: "#374151" }}>
        <span style={{ width: 96, color: "#6b7280" }}>{channel}:</span>
        {value.slice(0, tokenCount).map((v, i) => (
          <span key={i}
                style={{ background: color, padding: "0 3px",
                         borderRadius: 2, minWidth: 8, textAlign: "center" }}>
            {String(v)}
          </span>
        ))}
      </div>
    );
  }
  if (isScalarRibbon(value)) {
    return (
      <div data-testid={`data-ribbon-${rowIndex}-${channel}`}
           style={{ fontSize: 10, color: "#374151" }}>
        <span style={{ color: "#6b7280" }}>{channel}: </span>
        <span style={{ background: color, padding: "0 4px",
                       borderRadius: 2 }}>{value}</span>
      </div>
    );
  }
  return (
    <div data-testid={`data-ribbon-${rowIndex}-${channel}`}
         style={{ fontSize: 10, color: "#6b7280" }}>
      {channel}: <em>{JSON.stringify(value)}</em>
    </div>
  );
}
