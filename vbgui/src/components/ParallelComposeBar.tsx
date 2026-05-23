// V7-F57 — manual parallel-block composition. The architect wants to
// build a tiny-aya-style block: a single input fan-outs to TWO
// parallel paths (an attention sub-block and an mlp sub-block), both
// feeding a residual-add join, then a final norm. Manually drawing
// parallel edges through React Flow drag-handles is unreliable in
// Playwright, so we expose a button-driven "Compose tiny-aya"
// shortcut that produces the same graph.

import { HelpIcon } from "@/components/HelpIcon";

export interface ParallelComposeBarProps {
  onCompose: (nodes: ComposeNode[], edges: ComposeEdge[]) => void;
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

export function ParallelComposeBar({
  onCompose,
}: ParallelComposeBarProps): JSX.Element {
  return (
    <div data-testid="parallel-compose-bar"
         style={{ display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", background: "#fdf4ff",
                  borderBottom: "1px solid #e9d5ff",
                  fontFamily: "system-ui, sans-serif", fontSize: 12 }}>
      <strong>Parallel composition</strong>
      <HelpIcon topic="parallel_block" />
      <button data-testid="parallel-compose-tiny-aya"
              onClick={() => {
                const { nodes, edges } = tinyAya();
                onCompose(nodes, edges);
              }}
              style={{ padding: "2px 10px", background: "#7e22ce",
                       color: "white", border: "none",
                       borderRadius: 4, cursor: "pointer" }}>
        Compose tiny-aya (attn ∥ mlp → add → norm)
      </button>
    </div>
  );
}
