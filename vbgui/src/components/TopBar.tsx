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
    opts?: { num_steps?: number; side_channels?: string[] }) => void;
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
  // V4-10: side-channel toggles for the train run. Off by default;
  // when on, App.handleRunPipeline forwards a synthetic int list to
  // backend opts.side_channels so stage_train can record observation.
  const [scDocIds, setScDocIds] = useState<boolean>(false);
  const [scTokenIds, setScTokenIds] = useState<boolean>(false);
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

      <div style={{ position: "relative" }}>
        <button data-testid="run-pipeline"
                onClick={() => p.onRunPipeline("smoke")}>
          Smoke
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
            <div style={{ padding: "0 12px 6px", display: "flex",
                          alignItems: "center", gap: 10, fontSize: 11 }}>
              <span style={{ color: "#6b7280" }}>side-channels:</span>
              <label style={{ display: "flex", gap: 3, alignItems: "center" }}>
                <input data-testid="train-side-channel-doc_ids"
                       type="checkbox" checked={scDocIds}
                       onChange={(e) => setScDocIds(e.target.checked)} />
                doc_ids
              </label>
              <label style={{ display: "flex", gap: 3, alignItems: "center" }}>
                <input data-testid="train-side-channel-token_ids"
                       type="checkbox" checked={scTokenIds}
                       onChange={(e) => setScTokenIds(e.target.checked)} />
                token_ids
              </label>
            </div>
            <button data-testid="run-pipeline-train"
                    onClick={() => { setOpen(false);
                                     const sc: string[] = [];
                                     if (scDocIds) sc.push("doc_ids");
                                     if (scTokenIds) sc.push("token_ids");
                                     p.onRunPipeline("train",
                                       { num_steps: trainNumSteps,
                                         side_channels: sc }); }}
                    disabled={!!p.trainDisabled}
                    title={p.trainDisabled?.reason ?? ""}
                    style={{ ...menuItem,
                             opacity: p.trainDisabled ? 0.5 : 1,
                             cursor: p.trainDisabled ? "not-allowed"
                                                     : "pointer" }}>
              Train
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
