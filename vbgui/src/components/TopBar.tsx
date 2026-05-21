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
  onRunPipeline: (mode: RunMode) => void;
}

export function TopBar(p: TopBarProps): JSX.Element {
  const [open, setOpen] = useState(false);
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
            <button data-testid="run-pipeline-train"
                    onClick={() => { setOpen(false); p.onRunPipeline("train"); }}
                    style={menuItem}>Train</button>
          </div>
        )}
      </div>
    </header>
  );
}

const menuItem: React.CSSProperties = {
  display: "block", padding: "6px 12px", border: "none",
  background: "white", cursor: "pointer", textAlign: "left", width: "100%",
};
