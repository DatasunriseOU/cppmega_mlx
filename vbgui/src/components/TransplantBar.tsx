// V7-F54 — cross-preset brick transplant. A small bar above the
// canvas where the architect picks a source preset and one of its
// bricks, then clicks Import to drop that brick (kind + params) onto
// the current canvas. Edges are not transplanted — the user wires
// them after import.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { HelpIcon } from "@/components/HelpIcon";

interface BrickSpec {
  kind: string;
  name?: string;
  params?: Record<string, unknown>;
}

export interface TransplantBarProps {
  rpc: RpcClient | null;
  presets: readonly string[];
  onTransplant: (kind: string, params: Record<string, unknown>) => void;
}

export function TransplantBar({
  rpc, presets, onTransplant,
}: TransplantBarProps): JSX.Element {
  const [sourcePreset, setSourcePreset] = useState<string>(
    presets[0] ?? "");
  const [bricks, setBricks] = useState<BrickSpec[]>([]);
  const [selectedBrick, setSelectedBrick] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadPreset(preset: string) {
    setLoading(true);
    setError(null);
    setBricks([]);
    setSelectedBrick("");
    try {
      if (!rpc) throw new Error("rpc unavailable");
      const r = await rpc.call<{ specs: BrickSpec[] }>(
        "build_preset_specs",
        { preset_name: preset, hidden_size: 128 });
      setBricks(r.specs);
      if (r.specs.length > 0) {
        setSelectedBrick(r.specs[0].name ?? r.specs[0].kind);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const candidate = bricks.find(
    (b) => (b.name ?? b.kind) === selectedBrick);

  return (
    <div data-testid="transplant-bar"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: "#eef2ff",
                  borderBottom: "1px solid #c7d2fe",
                  fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
      <strong>Transplant</strong>
      <HelpIcon topic="brick_transplant" />
      <label>
        from
        <select data-testid="transplant-source-preset"
                value={sourcePreset}
                onChange={(e) => {
                  setSourcePreset(e.target.value);
                  setBricks([]);
                  setSelectedBrick("");
                }}
                style={{ marginLeft: 4, width: 180 }}>
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </label>
      <button data-testid="transplant-load-source"
              onClick={() => loadPreset(sourcePreset)}
              disabled={loading || !sourcePreset}
              style={{ padding: "2px 8px" }}>
        {loading ? "Loading…" : "Load"}
      </button>
      {bricks.length > 0 && (
        <>
          <label>
            brick
            <select data-testid="transplant-source-brick"
                    value={selectedBrick}
                    onChange={(e) => setSelectedBrick(e.target.value)}
                    style={{ marginLeft: 4, width: 200 }}>
              {bricks.map((b) => {
                const key = b.name ?? b.kind;
                return (
                  <option key={key} value={key}>
                    {key} [{b.kind}]
                  </option>
                );
              })}
            </select>
          </label>
          <div data-testid="transplant-draggable-list"
               style={{ display: "flex", gap: 6, alignItems: "center", marginLeft: 12 }}>
            <span style={{ color: "#4b5563", fontWeight: 600 }}>Drag:</span>
            {bricks.map((b) => {
              const key = b.name ?? b.kind;
              return (
                <div
                  key={key}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData("application/x-cppmega-transplant-kind", b.kind);
                    e.dataTransfer.setData("application/x-cppmega-transplant-params", JSON.stringify(b.params ?? {}));
                    e.dataTransfer.effectAllowed = "copy";
                  }}
                  data-testid={`transplant-drag-brick-${key}`}
                  style={{
                    padding: "3px 8px",
                    background: "#fff",
                    border: "1px solid #a5b4fc",
                    borderRadius: 4,
                    cursor: "grab",
                    fontSize: 10,
                    fontWeight: 600,
                    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
                  }}
                >
                  {key}
                </div>
              );
            })}
          </div>
        </>
      )}
      <button data-testid="transplant-import"
              onClick={() => {
                if (candidate) {
                  onTransplant(candidate.kind, candidate.params ?? {});
                }
              }}
              disabled={!candidate}
              style={{ padding: "2px 10px",
                       background: candidate ? "#4f46e5" : "#e5e7eb",
                       color: candidate ? "white" : "#9ca3af",
                       border: "none", borderRadius: 4,
                       cursor: candidate ? "pointer" : "default" }}>
        Import →
      </button>
      {error && (
        <span data-testid="transplant-error"
              style={{ color: "#991b1b" }}>
          {error}
        </span>
      )}
    </div>
  );
}
