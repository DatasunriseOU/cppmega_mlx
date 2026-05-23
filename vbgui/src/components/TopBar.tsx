import { useEffect, useState } from "react";
import { MemoryBar } from "./MemoryBar";
import type { SpecState, TopologyFactory } from "@/state/spec";

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
    }) => void;
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
                     gap: 12, padding: "0 12px",
                     borderBottom: "1px solid #e5e7eb",
                     fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
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

      <select data-testid="compile-mode"
              value={p.state.sharding.compile_mode}
              onChange={(e) =>
                p.onCompileModeChange(
                  e.target.value as SpecState["sharding"]["compile_mode"])}>
        <option value="off">compile: off</option>
        <option value="regional">compile: regional</option>
        <option value="whole_model">compile: whole_model ⚠</option>
      </select>

      <MemoryBar state={p.state} />

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
            style={{ fontSize: 10, color: p.trainInFlight ? "#d97706"
                                                          : "#9ca3af",
                     fontFamily: "monospace" }}>
        {p.trainInFlight ? "training" : "idle"}
      </span>
      <div style={{ position: "relative" }}>
        <button data-testid="run-pipeline"
                onClick={() => p.onRunPipeline("smoke")}>
          Smoke
        </button>
        <button data-testid="run-pipeline-cancel"
                onClick={() => p.onCancelTrain?.()}
                disabled={!p.trainInFlight || !p.trainRunId}
                title={p.trainInFlight ? "Cancel Train" : "No Train run active"}
                style={{ marginLeft: 4 }}>
          Cancel
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
                        background: "white", border: "1px solid #e5e7eb",
                        boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
                        zIndex: 10 }}>
            <button data-testid="run-pipeline-full"
                    onClick={() => { setOpen(false); p.onRunPipeline("full"); }}
                    style={menuItem}>Full validate</button>
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
                                         fim_enabled: fimEnabled }); }}
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
  display: "block", padding: "6px 12px", border: "none",
  background: "white", cursor: "pointer", textAlign: "left", width: "100%",
};
