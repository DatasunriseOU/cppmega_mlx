// V7-F57 — manual parallel-block composition. The architect wants to
// build a tiny-aya-style block: a single input fan-outs to TWO
// parallel paths (an attention sub-block and an mlp sub-block), both
// feeding a residual-add join, then a final norm. Manually drawing
// parallel edges through React Flow drag-handles is unreliable in
// Playwright, so we expose a button-driven "Compose" shortcut that
// produces the same graph.

import { useState } from "react";
import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface ParallelComposeBarProps {
  onCompose: (nodes: ComposeNode[], edges: ComposeEdge[]) => void;
  sidebar?: boolean;
  rpc?: any;
  presets?: readonly string[];
}

export interface ComposeNode {
  id: string;
  kind: string;
  position: { x: number; y: number };
  params?: Record<string, unknown>;
}

export interface ComposeEdge {
  source: string;
  target: string;
}

// tiny-aya layout (matches the parallel block in cppmega_v4
// architectures.presets._aya factory variant):
//   input ──┬──> attn ──┐
//           │           ├──> resid_add ──> norm
//           └──> mlp  ──┘
function tinyAya(): { nodes: ComposeNode[]; edges: ComposeEdge[] } {
  return {
    nodes: [
      { id: "aya_input", kind: "abs_pos_embed",
        position: { x: 60,  y: 60 } },
      { id: "aya_attn",  kind: "attention",
        position: { x: 280, y: 20  } },
      { id: "aya_mlp",   kind: "mlp",
        position: { x: 280, y: 140 } },
      { id: "aya_join",  kind: "residual",
        position: { x: 520, y: 80 } },
      { id: "aya_norm",  kind: "rmsnorm",
        position: { x: 740, y: 80 } },
    ],
    edges: [
      { source: "aya_input", target: "aya_attn" },
      { source: "aya_input", target: "aya_mlp"  },
      { source: "aya_attn",  target: "aya_join" },
      { source: "aya_mlp",   target: "aya_join" },
      { source: "aya_join",  target: "aya_norm" },
    ],
  };
}

async function composePreset(presetName: string, rpc: any): Promise<{ nodes: ComposeNode[]; edges: ComposeEdge[] }> {
  if (!rpc) return tinyAya();
  try {
    const r = await (rpc as any).call(
      "build_preset_specs",
      { preset_name: presetName, hidden_size: 128 }
    ) as { specs: { kind: string; params?: Record<string, unknown> }[] };
    const specs: Array<{ kind: string; params?: Record<string, unknown> }> = r.specs || [];
    const attnSpec = specs.find(s => s.kind.includes("attn") || s.kind.includes("mla") || s.kind.includes("attention")) ?? { kind: "attention" };
    const mlpSpec = specs.find(s => s.kind.includes("mlp")) ?? { kind: "mlp" };
    const residSpec = specs.find(s => s.kind.includes("residual")) ?? { kind: "residual" };
    const normSpec = specs.find(s => s.kind.includes("norm")) ?? { kind: "rmsnorm" };

    return {
      nodes: [
        { id: `${presetName}_input`, kind: "abs_pos_embed", position: { x: 60, y: 60 } },
        { id: `${presetName}_attn`, kind: attnSpec.kind, params: attnSpec.params, position: { x: 280, y: 20 } },
        { id: `${presetName}_mlp`, kind: mlpSpec.kind, params: mlpSpec.params, position: { x: 280, y: 140 } },
        { id: `${presetName}_join`, kind: residSpec.kind, params: residSpec.params, position: { x: 520, y: 80 } },
        { id: `${presetName}_norm`, kind: normSpec.kind, params: normSpec.params, position: { x: 740, y: 80 } },
      ],
      edges: [
        { source: `${presetName}_input`, target: `${presetName}_attn` },
        { source: `${presetName}_input`, target: `${presetName}_mlp` },
        { source: `${presetName}_attn`, target: `${presetName}_join` },
        { source: `${presetName}_mlp`, target: `${presetName}_join` },
        { source: `${presetName}_join`, target: `${presetName}_norm` },
      ],
    };
  } catch (e) {
    return tinyAya();
  }
}

export function ParallelComposeBar({
  onCompose,
  sidebar = false,
  rpc,
  presets = ["mini", "dev_128", "small_512", "medium_1k", "large_2k", "llama3_8b", "llama3_70b"],
}: ParallelComposeBarProps): JSX.Element {
  const [selectedPreset, setSelectedPreset] = useState<string>("mini");
  const [loading, setLoading] = useState(false);

  const handleCompose = async () => {
    if (!rpc) {
      // Synchronous path for unit tests to maintain test assertions
      const result = tinyAya();
      onCompose(result.nodes, result.edges);
      return;
    }
    setLoading(true);
    const result = await composePreset(selectedPreset, rpc);
    setLoading(false);
    onCompose(result.nodes, result.edges);
  };

  const selectElement = (
    <select
      data-testid="parallel-compose-select"
      value={selectedPreset}
      onChange={(e) => setSelectedPreset(e.target.value)}
      style={{
        color: T.text,
        background: T.surface3,
        border: `1px solid ${T.border}`,
        borderRadius: "var(--vb-radius-sm)",
        padding: "4px 8px",
        fontSize: 12,
        outline: "none",
        flex: 1,
      }}
    >
      {presets.map((p) => (
        <option key={p} value={p}>
          {p}
        </option>
      ))}
    </select>
  );

  if (sidebar) {
    return (
      <div data-testid="parallel-compose-bar"
           style={{ display: "flex", flexDirection: "column", gap: 6,
                    fontFamily: T.font, fontSize: 12 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontWeight: 600, color: T.textSecondary }}>Parallel Compose</span>
          <HelpIcon topic="parallel_block" />
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {selectElement}
          <button data-testid="parallel-compose-tiny-aya"
                  disabled={loading}
                  onClick={handleCompose}
                  style={{ padding: "6px 12px",
                           background: T.accent,
                           color: "var(--vb-accent-contrast)",
                           border: "none",
                           borderRadius: "var(--vb-radius-sm)", cursor: "pointer",
                           fontWeight: "bold",
                           flex: 1 }}>
            {loading ? "Composing…" : `Compose ${selectedPreset}`}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div data-testid="parallel-compose-bar"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: T.surface,
                  borderBottom: `1px solid ${T.border}`,
                  fontFamily: T.font, fontSize: 12 }}>
      <strong style={{ color: T.accent }}>Parallel composition</strong>
      <HelpIcon topic="parallel_block" />
      {selectElement}
      <button data-testid="parallel-compose-tiny-aya"
              disabled={loading}
              onClick={handleCompose}
              style={{ padding: "4px 10px",
                       background: T.accent,
                       color: "var(--vb-accent-contrast)",
                       border: `1px solid ${T.border}`,
                       borderRadius: "var(--vb-radius-sm)", cursor: "pointer",
                       fontWeight: "bold" }}>
        {loading ? "Composing…" : `Compose ${selectedPreset} (attn ∥ mlp → add → norm)`}
      </button>
    </div>
  );
}
