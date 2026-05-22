import { useState } from "react";
import { MemoryBar } from "./MemoryBar";
import type { SpecState, TopologyFactory } from "@/state/spec";

export type RunMode = "smoke" | "full" | "train";

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
                <option value="fp32">fp32</option>
                <option value="bf16">bf16</option>
                <option value="fp16">fp16</option>
              </select>
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
                                         master_dtype: masterDtype }); }}
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
