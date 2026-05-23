// V7-F54 — cross-preset brick transplant. A small bar above the
// canvas where the architect picks a source preset and one of its
// bricks, then clicks Import to drop that brick (kind + params) onto
// the current canvas. Edges are not transplanted — the user wires
// them after import.

import { useState } from "react";
import type { RpcClient } from "@/lib/rpc";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

interface BrickSpec {
  kind: string;
  name?: string;
  params?: Record<string, unknown>;
}

export interface TransplantBarProps {
  rpc: RpcClient | null;
  presets: readonly string[];
  onTransplant: (kind: string, params: Record<string, unknown>) => void;
  sidebar?: boolean;
}

export function TransplantBar({
  rpc, presets, onTransplant, sidebar = false,
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

  if (sidebar) {
    return (
      <div data-testid="transplant-bar"
           style={{ display: "flex", flexDirection: "column", gap: 8,
                    fontFamily: T.font, fontSize: 12 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontWeight: 600, color: T.textSecondary }}>Transplant Brick</span>
          <HelpIcon topic="brick_transplant" />
        </div>
        <div style={{ display: "flex", gap: 6, width: "100%" }}>
          <label style={{ color: T.textSecondary, display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
            source preset
            <select data-testid="transplant-source-preset"
                    value={sourcePreset}
                    onChange={(e) => {
                      setSourcePreset(e.target.value);
                      setBricks([]);
                      setSelectedBrick("");
                    }}
                    style={{
                      width: "100%",
                      color: T.text,
                      background: T.surface3,
                      border: `1px solid ${T.border}`,
                      borderRadius: 4,
                      padding: "3px 6px",
                    }}>
              {presets.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <button data-testid="transplant-load-source"
                  onClick={() => loadPreset(sourcePreset)}
                  disabled={loading || !sourcePreset}
                  style={{
                    padding: "2px 10px",
                    color: T.text,
                    background: T.surface3,
                    border: `1px solid ${T.border}`,
                    borderRadius: 4,
                    alignSelf: "flex-end",
                    height: 26,
                    cursor: "pointer"
                  }}>
            {loading ? "…" : "Load"}
          </button>
        </div>
        {bricks.length > 0 && (
          <label style={{ color: T.textSecondary, display: "flex", flexDirection: "column", gap: 4 }}>
            source brick
            <select data-testid="transplant-source-brick"
                    value={selectedBrick}
                    onChange={(e) => setSelectedBrick(e.target.value)}
                    style={{
                      width: "100%",
                      color: T.text,
                      background: T.surface3,
                      border: `1px solid ${T.border}`,
                      borderRadius: 4,
                      padding: "3px 6px",
                    }}>
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
        )}
        <button data-testid="transplant-import"
                onClick={() => {
                  if (candidate) {
                    onTransplant(candidate.kind, candidate.params ?? {});
                  }
                }}
                disabled={!candidate}
                style={{ padding: "6px 12px",
                         background: candidate ? T.accent : T.surface3,
                         color: candidate ? "#fff" : T.textMuted,
                         border: "none", borderRadius: 4,
                         width: "100%",
                         fontWeight: "bold",
                         cursor: candidate ? "pointer" : "default" }}>
          Import
        </button>
        {error && (
          <span data-testid="transplant-error"
                style={{ color: T.danger }}>
            {error}
          </span>
        )}
      </div>
    );
  }

  return (
    <div data-testid="transplant-bar"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: T.surface,
                  borderBottom: `1px solid ${T.border}`,
                  fontFamily: T.font, fontSize: 12 }}>
      <strong style={{ color: T.accent }}>Transplant</strong>
      <HelpIcon topic="brick_transplant" />
      <label style={{ color: T.textSecondary, display: "flex", alignItems: "center", gap: 4 }}>
        from
        <select data-testid="transplant-source-preset"
                value={sourcePreset}
                onChange={(e) => {
                  setSourcePreset(e.target.value);
                  setBricks([]);
                  setSelectedBrick("");
                }}
                style={{
                  marginLeft: 4,
                  width: 180,
                  color: T.text,
                  background: T.surface3,
                  border: `1px solid ${T.border}`,
                }}>
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </label>
      <button data-testid="transplant-load-source"
              onClick={() => loadPreset(sourcePreset)}
              disabled={loading || !sourcePreset}
              style={{
                padding: "2px 8px",
                color: T.text,
                background: T.surface3,
                border: `1px solid ${T.border}`,
              }}>
        {loading ? "Loading…" : "Load"}
      </button>
      {bricks.length > 0 && (
        <>
          <label style={{ color: T.textSecondary, display: "flex", alignItems: "center", gap: 4 }}>
            brick
            <select data-testid="transplant-source-brick"
                    value={selectedBrick}
                    onChange={(e) => setSelectedBrick(e.target.value)}
                    style={{
                      marginLeft: 4,
                      width: 200,
                      color: T.text,
                      background: T.surface3,
                      border: `1px solid ${T.border}`,
                    }}>
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
            <span style={{ color: T.text, fontWeight: 700 }}>Drag:</span>
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
                    background: T.surface3,
                    border: `1px solid ${T.accent}`,
                    color: T.accent,
                    borderRadius: 4,
                    cursor: "grab",
                    fontSize: 10,
                    fontWeight: 700,
                    boxShadow: T.shadowPanel,
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
                       background: candidate ? T.accent : T.surface3,
                       color: candidate ? "#0f172a" : T.textMuted,
                       border: `1px solid ${T.border}`, borderRadius: 4,
                       cursor: candidate ? "pointer" : "default" }}>
        Import →
      </button>
      {error && (
        <span data-testid="transplant-error"
              style={{ color: T.danger }}>
          {error}
        </span>
      )}
    </div>
  );
}
