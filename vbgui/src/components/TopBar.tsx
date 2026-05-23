import { useEffect, useState } from "react";
import { MemoryBar } from "./MemoryBar";
import { CheckpointHistoryDropdown } from "./CheckpointHistoryDropdown";
import type { SpecState, TopologyFactory } from "@/state/spec";
import type { RpcClient } from "@/lib/rpc";

export type RunMode = "smoke" | "full" | "train";

/** V7-C03: shape returned by ckpt.inspect RPC. */
export interface CkptInspectInfo {
  exists: boolean;
  has_metadata?: boolean;
  cppmega_version?: string | null;
  arch_hash?: string | null;
  opt_kind?: string | null;
  opt_lr?: number | null;
  global_step?: number | null;
  error?: string | null;
}

/** V7-D06: per-dtype cost row from dtype.cost_estimate RPC. */
export interface DtypeCostRow {
  dtype: "fp32" | "bf16" | "fp16";
  supported: boolean;
  fwd_ms: number | null;
  fwdbwd_ms: number | null;
  fwd_ms_per_token: number | null;
  fwdbwd_ms_per_token: number | null;
  cast_overhead_ms: number | null;
  error: string | null;
}

export interface TopBarProps {
  state: SpecState;
  projectName: string;
  presets: readonly string[];
  topologies: readonly TopologyFactory[];
  onProjectNameChange: (name: string) => void;
  onPresetDrop: (name: string) => void;
  onTopologyChange: (t: TopologyFactory) => void;
  onCompileModeChange: (m: SpecState["sharding"]["compile_mode"]) => void;
  onRunPipeline: (mode: RunMode,
    opts?: { num_steps?: number; warm_start?: boolean;
      checkpoint_save_path?: string;
      checkpoint_load_path?: string;
      inference_probe_text?: string;
      master_dtype?: "fp32" | "bf16" | "fp16" | "auto";
      fim_enabled?: boolean;
      // V7-Q03.3: checkpoint write/load flags.
      compress?: "none" | "weights-int8" | "opt-fp16" | "both";
      ckpt_strict?: boolean;
      opt_state_strict?: boolean;
    }) => void;
  /** V7-Q03.2: optional rpc handle for ckpt.list_history dropdown.
   *  When provided, TopBar renders a "history" picker next to the
   *  ckpt-load-path input. */
  rpc?: RpcClient | null;
  /** H02: toggle callbacks. */
  onMixedPrecisionChange?: (enabled: boolean) => void;
  onFp8EnabledChange?: (enabled: boolean) => void;
  /** V7-H03 undo/redo controls. Buttons render only when callbacks
   *  provided so unit tests for default TopBar are unaffected. */
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  /** H03: cancel the currently in-flight Train run. */
  trainInFlight?: boolean;
  trainRunId?: string | null;
  onCancelTrain?: () => void;
  /** V7-H06: pause / resume the currently in-flight Train run. */
  trainPaused?: boolean;
  onPauseTrain?: () => void;
  onResumeTrain?: () => void;
  /** V7-H06b: surfaces UI-side "abort RPC fired, waiting for backend
   * to confirm running=false" so the user sees the cancel is in
   * progress instead of an instant disabled-button flip. */
  trainAborting?: boolean;
  /** V3-8/V3-9: when present, Train button is rendered disabled with
   *  reason exposed via data-testid='top-bar-train-disabled-reason'. */
  trainDisabled?: { reason: string } | null;
  /** V4-1: parquet + tokenizer paths picked in Data/Tokenizer tabs.
   *  Drives the data-testid='train-data-source' indicator next to
   *  Train so the user can tell whether training will use real tokens
   *  or fall back to synthetic. */
  trainParquetPath?: string | null;
  trainTokenizerPath?: string | null;
  /** G11: save/load callbacks. */
  onSaveSpec?: () => void;
  onLoadSpec?: (file: File) => void;
  /** V7-C03: callback to inspect a checkpoint path's metadata before
   *  warm-start. UI fires this on debounced change of ckpt-load-path
   *  and renders arch_hash / opt_kind / version inline. */
  onInspectCheckpoint?: (path: string) => Promise<CkptInspectInfo>;
  /** V7-D06: callback to fetch per-dtype cost estimate. UI fires once
   *  when the Train menu first opens and caches the table to render
   *  ms/token next to each option in the master_dtype dropdown. */
  onDtypeCostEstimate?: () => Promise<{ rows: DtypeCostRow[] }>;
  /** V7-K2: callback to fire probe.run — Contract Probe (capability
   *  check + alt-config suggester). When set, the run-pipeline menu
   *  exposes a 'Run Probe' button. */
  onRunProbe?: () => Promise<ProbeRunInfo>;
  filterByPlatform?: boolean;
  onFilterByPlatformChange?: (filter: boolean) => void;
  activeDevice?: string;
}

/** V7-K2: shape returned by probe.run RPC. */
export interface ProbeRunInfo {
  schema_version: string;
  is_clean: boolean;
  elapsed_ms: number;
  // ProbeRunResult is extra='allow' — additional fields land here.
  [key: string]: unknown;
}

export function TopBar(p: TopBarProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [trainNumSteps, setTrainNumSteps] = useState<number>(2);
  const [warmStart, setWarmStart] = useState<boolean>(false);
  const [ckptSavePath, setCkptSavePath] = useState<string>("");
  const [ckptLoadPath, setCkptLoadPath] = useState<string>("");
  const [probeText, setProbeText] = useState<string>("");
  // "auto" defers to spec.optim.mixed_precision (H02) — the explicit
  // fp32/bf16/fp16 options override that for H23.
  const [masterDtype, setMasterDtype] =
    useState<"fp32" | "bf16" | "fp16" | "auto">("auto");
  // V7-G05: FIM (Fill-In-Middle) data path toggle. When on, stage_train
  // surfaces extras.train.fim_active + fim_ratio so the UI honest-closure
  // shows the FIM math actually fired.
  const [fimEnabled, setFimEnabled] = useState<boolean>(false);
  // V7-Q03.3: checkpoint write/load knobs. Default "none" / unchecked
  // preserves prior behaviour; advanced users opt in.
  const [compress, setCompress] =
    useState<"none" | "weights-int8" | "opt-fp16" | "both">("none");
  const [ckptStrict, setCkptStrict] = useState<boolean>(false);
  const [optStateStrict, setOptStateStrict] = useState<boolean>(false);
  // V7-C03: cached metadata for the current ckpt-load-path. Populated
  // by a 300ms-debounced ckpt.inspect call so the user sees arch_hash /
  // opt_kind / version before clicking Train with warm-start.
  const [ckptInfo, setCkptInfo] = useState<CkptInspectInfo | null>(null);
  const [ckptInfoLoading, setCkptInfoLoading] = useState<boolean>(false);
  // V7-D06: cached dtype cost table. Fetched once when the menu opens
  // so the dtype dropdown can render ms/token alongside each option.
  const [dtypeCosts, setDtypeCosts] = useState<DtypeCostRow[] | null>(null);
  const [dtypeCostsLoading, setDtypeCostsLoading] =
    useState<boolean>(false);
  // V7-K2: cached probe.run result rendered inside the menu.
  const [probeInfo, setProbeInfo] = useState<ProbeRunInfo | null>(null);
  const [probeLoading, setProbeLoading] = useState<boolean>(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  useEffect(() => {
    if (!open || !p.onDtypeCostEstimate || dtypeCosts !== null
        || dtypeCostsLoading) return;
    let cancelled = false;
    setDtypeCostsLoading(true);
    (async () => {
      try {
        const res = await p.onDtypeCostEstimate!();
        if (!cancelled) setDtypeCosts(res.rows);
      } catch {
        if (!cancelled) setDtypeCosts([]);
      } finally {
        if (!cancelled) setDtypeCostsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [open, p.onDtypeCostEstimate, dtypeCosts, dtypeCostsLoading]);

  const dtypeCostLabel = (dt: "fp32" | "bf16" | "fp16"): string => {
    if (!dtypeCosts) return "";
    const row = dtypeCosts.find((r) => r.dtype === dt);
    if (!row || !row.supported || row.fwdbwd_ms_per_token == null) return "";
    return ` · ${row.fwdbwd_ms_per_token.toFixed(4)} ms/tok`;
  };
  useEffect(() => {
    if (!p.onInspectCheckpoint) return;
    const path = ckptLoadPath.trim();
    if (!path) { setCkptInfo(null); return; }
    let cancelled = false;
    setCkptInfoLoading(true);
    const t = setTimeout(async () => {
      try {
        const info = await p.onInspectCheckpoint!(path);
        if (!cancelled) { setCkptInfo(info); setCkptInfoLoading(false); }
      } catch (err) {
        if (!cancelled) {
          setCkptInfo({ exists: false,
                        error: err instanceof Error ? err.message : String(err) });
          setCkptInfoLoading(false);
        }
      }
    }, 300);
    return () => { cancelled = true; clearTimeout(t); };
  }, [ckptLoadPath, p.onInspectCheckpoint]);
  return (
    <header data-testid="top-bar"
            style={{ height: 56, display: "flex", alignItems: "center",
                     gap: 10, padding: "0 14px",
                     background: "var(--vb-surface)",
                     borderBottom: "1px solid var(--vb-border)",
                     color: "var(--vb-text)",
                     fontFamily: "var(--vb-font)", fontSize: 12 }}>
      <input data-testid="project-name"
             value={p.projectName}
             onChange={(e) => p.onProjectNameChange(e.target.value)}
             style={{ width: 160, fontWeight: 600 }} />

      <select data-testid="preset-launcher" defaultValue=""
              onChange={(e) => { p.onPresetDrop(e.target.value);
                                 e.currentTarget.value = ""; }}>
        <option value="" disabled>Preset…</option>
        {p.presets.map((n) => <option key={n} value={n}>{n}</option>)}
      </select>

      <select data-testid="topology-selector"
              value={p.state.sharding.topology}
              onChange={(e) =>
                p.onTopologyChange(e.target.value as TopologyFactory)}>
        {p.topologies.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>

      {p.onFilterByPlatformChange !== undefined && (
        <label style={{ fontSize: 10, display: "flex", gap: 3,
                        alignItems: "center", whiteSpace: "nowrap" }}>
          <input data-testid="top-bar-filter-by-platform" type="checkbox"
                 checked={!!p.filterByPlatform}
                 onChange={(e) =>
                   p.onFilterByPlatformChange?.(e.target.checked)} />
          filter_platform ({p.activeDevice ?? "..."})
        </label>
      )}

      <select data-testid="compile-mode"
              value={p.state.sharding.compile_mode}
              onChange={(e) =>
                p.onCompileModeChange(
                  e.target.value as SpecState["sharding"]["compile_mode"])}>
        <option value="off">compile: off</option>
        <option value="regional">compile: regional</option>
        <option value="whole_model">compile: whole_model ⚠</option>
      </select>

      {/* UX-redesign #6: compact, topology-aware MemoryBar. world_size
          derives from spec.sharding.axis_assignments degrees product
          so on h100:8 (or any FSDP/TP/PP combo) the bar splits into
          N per-rank mini-strips instead of one misleading total. */}
      <MemoryBar state={p.state} compact={true}
                 perRankBytes={(() => {
                   const axes = p.state.sharding?.axis_assignments ?? [];
                   const ws = axes.reduce(
                     (acc, a) => acc * Math.max(1, a.degree | 0), 1);
                   if (ws <= 1) return undefined;
                   // Backend reports only worst_rank_bytes today; replicate
                   // across ranks so the visual cluster cardinality is
                   // honest even when per-rank breakdown isn't ready yet.
                   return Array.from({ length: ws },
                     () => p.state.worst_rank_bytes);
                 })()} />

      {p.onMixedPrecisionChange && (
        <label style={{ fontSize: 10, display: "flex", gap: 3,
                        alignItems: "center" }}>
          <input data-testid="top-bar-mixed-precision" type="checkbox"
                 checked={!!p.state.optim.mixed_precision}
                 onChange={(e) =>
                   p.onMixedPrecisionChange?.(e.target.checked)} />
          mixed_precision
        </label>
      )}
      {p.onFp8EnabledChange && (
        <label style={{ fontSize: 10, display: "flex", gap: 3,
                        alignItems: "center" }}>
          <input data-testid="top-bar-fp8-enabled" type="checkbox"
                 checked={!!p.state.sharding.fp8_enabled}
                 onChange={(e) =>
                   p.onFp8EnabledChange?.(e.target.checked)} />
          fp8
        </label>
      )}

      {p.onUndo && (
        <button data-testid="top-bar-undo"
                onClick={p.onUndo}
                disabled={!p.canUndo}
                title="Undo (Cmd/Ctrl+Z)">↶</button>
      )}
      {p.onRedo && (
        <button data-testid="top-bar-redo"
                onClick={p.onRedo}
                disabled={!p.canRedo}
                title="Redo (Shift+Cmd/Ctrl+Z)">↷</button>
      )}
      {p.onSaveSpec && (
        <button data-testid="spec-save" onClick={p.onSaveSpec}>Save</button>
      )}
      {p.onLoadSpec && (
        <>
          <input data-testid="spec-load-input"
                 type="file" accept=".json"
                 style={{ display: "none" }}
                 onChange={(e) => {
                   const f = e.target.files?.[0];
                   if (f && p.onLoadSpec) p.onLoadSpec(f);
                   e.currentTarget.value = "";
                 }} />
          <button data-testid="spec-load"
                  onClick={() => {
                    const el = document.querySelector<HTMLInputElement>(
                      "[data-testid='spec-load-input']");
                    el?.click();
                  }}>Load</button>
        </>
      )}

      <span data-testid="train-data-source"
            style={{ fontSize: 10,
                     color: p.trainParquetPath ? "#16a34a" : "#9ca3af",
                     fontFamily: "monospace" }}>
        {p.trainParquetPath
          ? `parquet: ${basename(p.trainParquetPath)}` +
            (p.trainTokenizerPath
              ? ` · tok: ${basename(p.trainTokenizerPath)}` : "")
          : "synthetic"}
      </span>

      <span data-testid="top-bar-train-status"
            style={{ fontSize: 10,
                     color: p.trainAborting ? "#b91c1c"
                          : p.trainPaused   ? "#7c3aed"
                          : p.trainInFlight ? "#d97706"
                                            : "#9ca3af",
                     fontFamily: "monospace" }}>
        {p.trainAborting ? "aborting…"
          : p.trainPaused ? "paused"
          : p.trainInFlight ? "training" : "idle"}
      </span>
      <div style={{ position: "relative" }}>
        <button data-testid="run-pipeline"
                onClick={() => p.onRunPipeline("smoke")}
                style={{ background: "var(--vb-accent)",
                         color: "var(--vb-accent-contrast)",
                         border: "1px solid var(--vb-accent-strong)",
                         fontWeight: 600,
                         boxShadow: "0 0 14px var(--vb-accent-soft)" }}>
          ▶ Smoke
        </button>
        <button data-testid="run-pipeline-cancel"
                onClick={() => p.onCancelTrain?.()}
                disabled={!p.trainInFlight || !p.trainRunId
                          || p.trainAborting}
                title={p.trainAborting ? "Abort pending — waiting for backend"
                      : p.trainInFlight ? "Cancel Train"
                      : "No Train run active"}
                style={{ marginLeft: 4 }}>
          {p.trainAborting ? "Aborting…" : "Cancel"}
        </button>
        {p.onPauseTrain && p.onResumeTrain && (
          <button data-testid="run-pipeline-pause"
                  onClick={() => p.trainPaused
                    ? p.onResumeTrain?.()
                    : p.onPauseTrain?.()}
                  disabled={!p.trainInFlight || !p.trainRunId}
                  title={p.trainPaused ? "Resume Train" : "Pause Train"}
                  style={{ marginLeft: 4 }}>
            {p.trainPaused ? "Resume" : "Pause"}
          </button>
        )}
        <button data-testid="run-pipeline-toggle"
                onClick={() => setOpen((x) => !x)}>▾</button>
        {open && (
          <div data-testid="run-pipeline-menu"
               style={{ position: "absolute", top: "100%", right: 0,
                        marginTop: 6,
                        background: "var(--vb-surface-2)",
                        border: "1px solid var(--vb-border)",
                        borderRadius: "var(--vb-radius-lg)",
                        boxShadow: "var(--vb-shadow-pop)",
                        overflow: "hidden",
                        zIndex: 10 }}>
            <button data-testid="run-pipeline-full"
                    onClick={() => { setOpen(false); p.onRunPipeline("full"); }}
                    style={menuItem}>Full validate</button>
            {p.onRunProbe && (
              <div style={{ padding: "6px 12px", display: "flex",
                            flexDirection: "column", gap: 4 }}>
                <button data-testid="run-probe"
                        disabled={probeLoading}
                        onClick={async () => {
                          setProbeLoading(true);
                          setProbeError(null);
                          try {
                            const r = await p.onRunProbe!();
                            setProbeInfo(r);
                          } catch (e) {
                            setProbeError(
                              e instanceof Error ? e.message : String(e));
                          } finally {
                            setProbeLoading(false);
                          }
                        }}
                        style={{ ...menuItem, fontSize: 11 }}>
                  {probeLoading ? "probing…" : "Run Probe (capability)"}
                </button>
                {probeError && (
                  <div data-testid="run-probe-error"
                       style={{ color: "#dc2626", fontSize: 10 }}>
                    {probeError}
                  </div>
                )}
                {probeInfo && (
                  <div data-testid="run-probe-result"
                       style={{ fontSize: 10, fontFamily: "var(--vb-font-mono)",
                                background: "var(--vb-surface-3)", padding: 4,
                                borderRadius: 4 }}>
                    <div data-testid="run-probe-result-clean">
                      {probeInfo.is_clean ? "✓ clean" : "⚠ issues"}
                      {" · "}
                      v{probeInfo.schema_version}
                      {" · "}
                      {probeInfo.elapsed_ms.toFixed(1)}ms
                    </div>
                  </div>
                )}
              </div>
            )}
            <div style={{ padding: "6px 12px", display: "flex",
                          alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 11, color: "#6b7280" }}>
                train steps:</span>
              <input data-testid="train-num-steps"
                     type="number" min={1} max={512}
                     value={trainNumSteps}
                     onChange={(e) =>
                       setTrainNumSteps(Math.max(1, parseInt(
                         e.target.value || "1", 10)))}
                     style={{ width: 50 }} />
            </div>
            <label style={{ padding: "6px 12px", display: "flex",
                            alignItems: "center", gap: 6, fontSize: 11,
                            color: "#374151" }}>
              <input data-testid="train-warm-start" type="checkbox"
                     checked={warmStart}
                     onChange={(e) => setWarmStart(e.target.checked)} />
              warm-start (continue from last run)
            </label>
            <div style={{ padding: "6px 12px", display: "flex",
                          flexDirection: "column", gap: 4, fontSize: 11,
                          color: "#374151" }}>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <span style={{ width: 78, color: "#6b7280" }}>ckpt save:</span>
                <input data-testid="train-checkpoint-save-path" type="text"
                       placeholder="/tmp/ckpt.safetensors"
                       value={ckptSavePath}
                       onChange={(e) => setCkptSavePath(e.target.value)}
                       style={{ width: 200 }} />
              </label>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <span style={{ width: 78, color: "#6b7280" }}>ckpt load:</span>
                <input data-testid="train-checkpoint-load-path" type="text"
                       placeholder="/tmp/prev.safetensors"
                       value={ckptLoadPath}
                       onChange={(e) => setCkptLoadPath(e.target.value)}
                       style={{ width: 200 }} />
                {/* V7-Q03.2: history picker → fills ckpt-load-path. */}
                {p.rpc && (
                  <CheckpointHistoryDropdown
                    rpc={p.rpc}
                    directory="."
                    onSelect={(path: string) => setCkptLoadPath(path)}
                  />
                )}
              </label>
              {/* V7-Q03.3: compress + strict toggles. */}
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <span style={{ width: 78, color: "#6b7280" }}>compress:</span>
                <select data-testid="train-opt-compress"
                        value={compress}
                        onChange={(e) =>
                          setCompress(e.target.value as typeof compress)}
                        style={{ width: 132 }}>
                  <option value="none">none</option>
                  <option value="weights-int8">weights-int8</option>
                  <option value="opt-fp16">opt-fp16</option>
                  <option value="both">both</option>
                </select>
              </label>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <input data-testid="train-opt-ckpt-strict" type="checkbox"
                       checked={ckptStrict}
                       onChange={(e) => setCkptStrict(e.target.checked)} />
                <span style={{ color: "#6b7280" }}>
                  ckpt_strict (arch-hash must match)
                </span>
              </label>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <input data-testid="train-opt-opt-state-strict"
                       type="checkbox"
                       checked={optStateStrict}
                       onChange={(e) =>
                         setOptStateStrict(e.target.checked)} />
                <span style={{ color: "#6b7280" }}>
                  opt_state_strict (skip on shape diff)
                </span>
              </label>
              {ckptLoadPath && (
                <div data-testid="ckpt-inspect-block"
                     style={{ padding: "2px 0 0 84px", fontSize: 10,
                              color: "#374151", fontFamily: "monospace",
                              lineHeight: 1.4 }}>
                  {ckptInfoLoading && (
                    <span data-testid="ckpt-inspect-loading"
                          style={{ color: "#9ca3af" }}>
                      inspecting…
                    </span>
                  )}
                  {!ckptInfoLoading && ckptInfo && !ckptInfo.exists && (
                    <span data-testid="ckpt-inspect-missing"
                          style={{ color: "#dc2626" }}>
                      file not found
                    </span>
                  )}
                  {!ckptInfoLoading && ckptInfo?.exists
                    && !ckptInfo.has_metadata && (
                    <span data-testid="ckpt-inspect-no-metadata"
                          style={{ color: "#d97706" }}>
                      no cppmega metadata
                      {ckptInfo.error ? ` (${ckptInfo.error})` : ""}
                    </span>
                  )}
                  {!ckptInfoLoading && ckptInfo?.has_metadata && (
                    <div data-testid="ckpt-inspect-info"
                         style={{ display: "flex", flexDirection: "column",
                                  gap: 1 }}>
                      <span data-testid="ckpt-inspect-version">
                        v: {ckptInfo.cppmega_version ?? "?"}
                      </span>
                      <span data-testid="ckpt-inspect-arch-hash"
                            title={ckptInfo.arch_hash ?? ""}>
                        arch: {ckptInfo.arch_hash
                          ? ckptInfo.arch_hash.slice(0, 12) + "…"
                          : "?"}
                      </span>
                      <span data-testid="ckpt-inspect-opt-kind">
                        opt: {ckptInfo.opt_kind ?? "?"}
                        {ckptInfo.opt_lr !== null
                          && ckptInfo.opt_lr !== undefined
                          ? ` · lr=${ckptInfo.opt_lr}` : ""}
                      </span>
                      <span data-testid="ckpt-inspect-step">
                        step: {ckptInfo.global_step ?? "?"}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
            <label style={{ padding: "6px 12px", display: "flex",
                            alignItems: "center", gap: 6, fontSize: 11,
                            color: "#374151" }}>
              <span style={{ color: "#6b7280" }}>master_dtype:</span>
              <select data-testid="top-bar-precision-mode"
                      value={masterDtype}
                      onChange={(e) =>
                        setMasterDtype(e.target.value as
                          "fp32" | "bf16" | "fp16" | "auto")}>
                <option value="auto">auto (mixed_precision)</option>
                <option value="fp32">fp32{dtypeCostLabel("fp32")}</option>
                <option value="bf16">bf16{dtypeCostLabel("bf16")}</option>
                <option value="fp16">fp16{dtypeCostLabel("fp16")}</option>
              </select>
              {dtypeCostsLoading && (
                <span data-testid="dtype-cost-loading"
                      style={{ fontSize: 10, color: "#9ca3af" }}>
                  measuring…
                </span>
              )}
              {!dtypeCostsLoading && dtypeCosts && dtypeCosts.length > 0 && (
                <span data-testid="dtype-cost-summary"
                      style={{ fontSize: 10, color: "#6b7280",
                               fontFamily: "monospace" }}>
                  measured
                </span>
              )}
            </label>
            <label style={{ padding: "6px 12px", display: "flex",
                            alignItems: "center", gap: 6, fontSize: 11,
                            color: "#374151" }}>
              <input data-testid="train-fim-enabled" type="checkbox"
                     checked={fimEnabled}
                     onChange={(e) => setFimEnabled(e.target.checked)} />
              FIM (Fill-In-Middle) data path
            </label>
            <label style={{ padding: "6px 12px", display: "flex",
                            flexDirection: "column", gap: 3, fontSize: 11,
                            color: "#374151" }}>
              <span style={{ color: "#6b7280" }}>probe text:</span>
              <textarea data-testid="train-probe-text"
                        placeholder="Optional: encode this for inference probe"
                        value={probeText}
                        onChange={(e) => setProbeText(e.target.value)}
                        rows={2}
                        style={{ width: 280, fontFamily: "monospace",
                                 fontSize: 11 }} />
            </label>
            <button data-testid="run-pipeline-train"
                    onClick={() => { setOpen(false);
                                     p.onRunPipeline("train",
                                       { num_steps: trainNumSteps,
                                         warm_start: warmStart,
                                         checkpoint_save_path:
                                           ckptSavePath || undefined,
                                         checkpoint_load_path:
                                           ckptLoadPath || undefined,
                                         inference_probe_text:
                                           probeText || undefined,
                                         master_dtype: masterDtype,
                                         fim_enabled: fimEnabled,
                                         compress,
                                         ckpt_strict: ckptStrict,
                                         opt_state_strict: optStateStrict,
                                       }); }}
                    // H22: disable while a Train is already running so
                    // double-clicks don't spawn a parallel pipeline.
                    disabled={!!p.trainDisabled || !!p.trainInFlight}
                    title={p.trainDisabled?.reason
                      ?? (p.trainInFlight ? "Training in progress" : "")}
                    style={{ ...menuItem,
                             opacity: (p.trainDisabled || p.trainInFlight)
                               ? 0.5 : 1,
                             cursor: (p.trainDisabled || p.trainInFlight)
                               ? "not-allowed" : "pointer" }}>
              {p.trainInFlight ? "Training…" : "Train"}
              {p.trainDisabled && (
                <span data-testid="top-bar-train-disabled-reason"
                      style={{ display: "block", fontSize: 10,
                               color: "#dc2626" }}>
                  {p.trainDisabled.reason}
                </span>
              )}
            </button>
          </div>
        )}
      </div>
    </header>
  );
}

function basename(p: string): string {
  const slash = p.lastIndexOf("/");
  return slash >= 0 ? p.slice(slash + 1) : p;
}

const menuItem: React.CSSProperties = {
  display: "block", padding: "7px 14px", border: "none", borderRadius: 0,
  background: "transparent", color: "var(--vb-text)",
  cursor: "pointer", textAlign: "left", width: "100%",
};
