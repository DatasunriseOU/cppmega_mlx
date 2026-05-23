import { useCallback, useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { HFQuickStartModal } from "@/components/HFQuickStartModal";

export interface PreviewRow {
  row_index: number;
  tokens: number[];
  channels: Record<string, unknown>;
}

export interface PreviewParquetResult {
  rows: PreviewRow[];
  token_column: string;
  available_channels: string[];
  side_channel_families?: Record<string, SideChannelFamilyCoverage>;
  edge_distributions?: Record<string, EdgeDistributionPreview>;
  shards?: ShardPreview[];
  bytes_per_token_avg: number;
  bytes_per_token_p95: number;
  bytes_per_token_max: number;
  total_rows: number;
  elapsed_ms: number;
  /** V7-G04: corpus_stats sidecar emitted by clang_enriched_to_parquet
   *  (compute_corpus_stats output). Null for legacy shards. */
  corpus_stats?: CorpusStats | null;
}

export interface CorpusStats {
  token_coverage_pct?: number;
  doc_length_p50?: number;
  doc_length_p90?: number;
  doc_length_p99?: number;
  doc_length_histogram?: Array<[number, number]>;
  vocab_usage_topk?: Array<[number, number]>;
  long_tail_count?: number;
  n_docs?: number;
}

export interface SideChannelFamilyCoverage {
  family: string;
  status: string;
  columns: string[];
  missing_columns: string[];
  dropped_columns: string[];
  token_alignment: string;
  graph_remapping: string;
  provenance: string;
  non_null_ratio: number;
}

export interface EdgeDistributionPreview {
  column: string;
  edge_count: number;
  row_count: number;
  non_empty_rows: number;
  min_node_id: number | null;
  max_node_id: number | null;
  distinct_node_count: number;
  per_row_min: number;
  per_row_avg: number;
  per_row_max: number;
  synthetic_0_to_7_only: boolean;
  sample_edges: Array<{ from: number; to: number }>;
}

export interface ShardPreview {
  index: number;
  path: string;
  byte_size: number;
  row_count: number;
}

export interface DataInspectorProps {
  rpc: RpcClient;
  initialPath?: string;
  pageSize?: number;
  /** V4-1: callback when user picks the loaded parquet for training.
   *  App stores the path and forwards via stage_options.train.parquet_path. */
  onUseForTrain?: (
    parquetPath: string,
    tokenizerPath: string | null,
    shardPaths?: string[],
  ) => void;
  /** V4-1: current path App is using for training (drives button label). */
  trainParquetPath?: string | null;
  onAvailableChannelsChange?: (channels: string[]) => void;
}

const CHANNEL_COLORS = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca",
                        "#ddd6fe", "#fed7aa", "#fbcfe8", "#cffafe"];

function colorForChannel(idx: number): string {
  return CHANNEL_COLORS[idx % CHANNEL_COLORS.length];
}

function familyColor(status: string): string {
  if (status === "present" || status === "derived") return "#166534";
  if (status === "partial" || status === "dropped") return "#92400e";
  return "#991b1b";
}

function isScalarRibbon(v: unknown): v is number {
  return typeof v === "number";
}

function isArrayRibbon(v: unknown): v is unknown[] {
  return Array.isArray(v);
}

interface RoundtripRowResult {
  row_idx: number;
  matches: boolean;
  byte_diff: number;
  decoded_preview: string;
  original_bytes: number;
  decoded_bytes: number;
}
interface RoundtripCheckResult {
  rows: RoundtripRowResult[];
  pass_rate: number;
  tokenizer_capability: string;
  has_original_text: boolean;
}

export function DataInspector({
  rpc, initialPath = "", pageSize = 16,
  onUseForTrain, trainParquetPath, onAvailableChannelsChange,
}: DataInspectorProps): JSX.Element {
  const [path, setPath] = useState(initialPath);
  const [hfModalOpen, setHfModalOpen] = useState(false);
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<PreviewParquetResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState<Set<string>>(new Set());
  const [tokenizerSource, setTokenizerSource] = useState("");
  const [roundtrip, setRoundtrip] =
    useState<Map<number, RoundtripRowResult>>(new Map());

  const load = useCallback(async (nextOffset: number) => {
    if (!path) return;
    try {
      const r = await rpc.call<PreviewParquetResult>(
        "data.preview_parquet",
        { path, offset: nextOffset, limit: pageSize },
      );
      onAvailableChannelsChange?.(r.available_channels);
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
  const useForTrain = useCallback(() => {
    if (!onUseForTrain) return;
    const tokenizerPath = tokenizerSource || null;
    const shardPaths = result?.shards && result.shards.length > 1
      ? result.shards.map((shard) => shard.path)
      : undefined;
    if (shardPaths) {
      onUseForTrain(path, tokenizerPath, shardPaths);
    } else {
      onUseForTrain(path, tokenizerPath);
    }
  }, [onUseForTrain, path, result, tokenizerSource]);

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
        {/* E-AUDIT-01: file upload picker. POSTs to /upload/parquet,
            auto-populates the path field with the returned absolute
            path under /tmp/vbgui_uploads/<uuid>.parquet. */}
        <input data-testid="data-inspector-file-upload"
               type="file" accept=".parquet"
               onChange={async (ev) => {
                 const f = ev.target.files?.[0];
                 if (!f) return;
                 const fd = new FormData();
                 fd.append("file", f);
                 const baseUrl = (
                   (import.meta.env.VITE_BACKEND_URL as string | undefined)
                   ?? "http://127.0.0.1:8765");
                 try {
                   const res = await fetch(
                     `${baseUrl}/upload/parquet`,
                     { method: "POST", body: fd });
                   if (!res.ok) throw new Error(`HTTP ${res.status}`);
                   const body = await res.json() as { path: string };
                   setPath(body.path);
                 } catch (err) {
                   // Surface to the parent error pill — kept inline so
                   // the upload picker has its own testable error testid.
                   const msg = err instanceof Error
                     ? err.message : String(err);
                   const errEl = document.querySelector(
                     "[data-testid='data-inspector-file-upload-error']");
                   if (errEl) errEl.textContent = msg;
                 }
               }}
               style={{ fontSize: 11 }} />
        <span data-testid="data-inspector-file-upload-error"
              style={{ color: "#b91c1c", fontSize: 10 }} />
        <button data-testid="data-load" onClick={() => load(0)}>
          Load
        </button>
        <button data-testid="data-use-for-train"
                disabled={!result || !onUseForTrain}
                title={trainParquetPath === path
                  ? "Currently used for training"
                  : "Send this parquet (and tokenizer if set) to stage_train"}
                onClick={useForTrain}
                style={{
                  background: trainParquetPath === path
                    ? "#dcfce7" : undefined,
                  color: trainParquetPath === path
                    ? "#166534" : undefined,
                }}>
          {trainParquetPath === path ? "✓ Training" : "Use for training"}
        </button>
        <button data-testid="hf-quickstart-modal-open"
                onClick={() => setHfModalOpen(true)}
                style={{ background: "#eef2ff", color: "#3730a3" }}>
          HF quickstart
        </button>
      </header>
      <HFQuickStartModal
        rpc={rpc}
        open={hfModalOpen}
        onClose={() => setHfModalOpen(false)}
        onResult={(parquetPath) => setPath(parquetPath)}
      />
      <header style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#6b7280" }}>Tokenizer:</span>
        <input data-testid="data-tokenizer-path"
               type="text" placeholder="/path/to/tokenizer.json (optional)"
               value={tokenizerSource}
               onChange={(e) => setTokenizerSource(e.target.value)}
               style={{ flex: 1, fontFamily: "monospace", fontSize: 11 }} />
        <button data-testid="data-roundtrip"
                disabled={!path || !tokenizerSource}
                onClick={async () => {
                  try {
                    const r = await rpc.call<RoundtripCheckResult>(
                      "data.roundtrip_check",
                      { parquet_path: path, tokenizer_source: tokenizerSource,
                        max_rows: pageSize });
                    const m = new Map<number, RoundtripRowResult>();
                    for (const row of r.rows) m.set(row.row_idx, row);
                    setRoundtrip(m);
                  } catch (e) { setError(String(e)); }
                }}>
          Check roundtrip
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

          {result.corpus_stats && (
            <div data-testid="data-corpus-stats"
                 style={{ border: "1px solid #d1d5db", borderRadius: 4,
                          padding: 8, fontSize: 11,
                          background: "#f9fafb",
                          fontFamily: "monospace" }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                corpus stats (V7-G04)
              </div>
              <div data-testid="data-corpus-stats-token-coverage">
                token coverage:{" "}
                {result.corpus_stats.token_coverage_pct?.toFixed(2)}%
              </div>
              <div data-testid="data-corpus-stats-doc-length">
                doc length p50/p90/p99:{" "}
                {result.corpus_stats.doc_length_p50 ?? "?"}/
                {result.corpus_stats.doc_length_p90 ?? "?"}/
                {result.corpus_stats.doc_length_p99 ?? "?"}
              </div>
              <div data-testid="data-corpus-stats-n-docs">
                docs: {result.corpus_stats.n_docs ?? 0}
              </div>
              <div data-testid="data-corpus-stats-long-tail">
                long-tail tokens (≤1 use):{" "}
                {result.corpus_stats.long_tail_count ?? 0}
              </div>
            </div>
          )}
          {result.shards && result.shards.length > 0 && (
            <div data-testid="data-shards"
                 style={{ display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                          gap: 6 }}>
              {result.shards.map((shard) => (
                <div key={`${shard.index}-${shard.path}`}
                     data-testid={`data-shard-${shard.index}`}
                     style={{ border: "1px solid #e5e7eb", borderRadius: 4,
                              padding: 6, fontSize: 11 }}>
                  <div style={{ display: "flex", justifyContent: "space-between",
                                gap: 6 }}>
                    <strong>shard {shard.index + 1}/{result.shards?.length ?? 1}</strong>
                    <span>{shard.row_count} rows</span>
                  </div>
                  <div style={{ color: "#6b7280", fontFamily: "monospace",
                                overflowWrap: "anywhere" }}>
                    {shard.path}
                  </div>
                  <div style={{ color: "#6b7280" }}>
                    {shard.byte_size} bytes
                  </div>
                </div>
              ))}
            </div>
          )}

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

          {result.side_channel_families &&
            Object.keys(result.side_channel_families).length > 0 && (
            <div data-testid="data-family-coverage"
                 style={{ display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                          gap: 6 }}>
              {Object.entries(result.side_channel_families).map(([name, fam]) => (
                <div key={name} data-testid={`data-family-${name}`}
                     style={{ border: "1px solid #e5e7eb", borderRadius: 4,
                              padding: 6, fontSize: 11 }}>
                  <div style={{ display: "flex", justifyContent: "space-between",
                                gap: 6 }}>
                    <strong>{name}</strong>
                    <span data-testid={`data-family-${name}-status`}
                          style={{ color: familyColor(fam.status) }}>
                      {fam.status}
                    </span>
                  </div>
                  <div data-testid={`data-family-${name}-alignment`}
                       style={{ color: "#6b7280" }}>
                    align={fam.token_alignment} · graph={fam.graph_remapping}
                  </div>
                  <div data-testid={`data-family-${name}-provenance`}
                       style={{ color: "#6b7280" }}>
                    provenance={fam.provenance} · non-null {fam.non_null_ratio.toFixed(2)}
                  </div>
                  {fam.missing_columns.length > 0 && (
                    <div data-testid={`data-family-${name}-missing`}
                         style={{ color: "#92400e" }}>
                      missing: {fam.missing_columns.join(", ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {result.edge_distributions &&
            Object.keys(result.edge_distributions).length > 0 && (
            <div data-testid="data-edge-distributions"
                 style={{ display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                          gap: 6 }}>
              {Object.entries(result.edge_distributions).map(([name, dist]) => (
                <div key={name} data-testid={`data-edge-distribution-${name}`}
                     style={{ border: "1px solid #e5e7eb", borderRadius: 4,
                              padding: 6, fontSize: 11 }}>
                  <div style={{ display: "flex", justifyContent: "space-between",
                                gap: 6 }}>
                    <strong>{name}</strong>
                    <span style={{ color: dist.synthetic_0_to_7_only
                      ? "#92400e" : "#166534" }}>
                      {dist.synthetic_0_to_7_only ? "synthetic 0..7" : "real"}
                    </span>
                  </div>
                  <div style={{ color: "#374151" }}>
                    edges {dist.edge_count} · rows {dist.non_empty_rows}/{dist.row_count}
                  </div>
                  <div style={{ color: "#6b7280" }}>
                    ids {dist.min_node_id ?? "n/a"}..max {dist.max_node_id ?? "n/a"} ·
                    distinct {dist.distinct_node_count}
                  </div>
                  <div style={{ color: "#6b7280" }}>
                    per-row {dist.per_row_min}/{dist.per_row_avg.toFixed(2)}/{dist.per_row_max}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div data-testid="data-rows"
               style={{ flex: 1, overflowY: "auto",
                        border: "1px solid #e5e7eb",
                        borderRadius: 4 }}>
            {result.rows.map((row) => (
              <div key={row.row_index}
                   data-testid={`data-row-${row.row_index}`}
                   style={{ padding: 6, borderBottom: "1px solid #f3f4f6",
                            fontFamily: "monospace", fontSize: 11 }}>
                <div style={{ color: "#6b7280",
                              display: "flex", gap: 6, alignItems: "center" }}>
                  <span>row #{row.row_index}</span>
                  {roundtrip.has(row.row_index) && (() => {
                    const rt = roundtrip.get(row.row_index)!;
                    return (
                      <span data-testid={`data-roundtrip-${row.row_index}`}
                            title={`byte_diff=${rt.byte_diff} · decoded="${rt.decoded_preview}"`}
                            style={{ padding: "1px 6px", borderRadius: 3,
                                     fontSize: 10, fontWeight: 600,
                                     background: rt.matches ? "#dcfce7" : "#fee2e2",
                                     color: rt.matches ? "#166534" : "#991b1b" }}>
                        Roundtrip {rt.matches ? "OK" : "FAIL"}
                      </span>
                    );
                  })()}
                </div>
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
