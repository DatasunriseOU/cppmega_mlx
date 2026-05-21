// Auto-layout helper backed by ELK.js. Runs synchronously in the main
// thread for now; the F-C ticket moves it to a Web Worker once we have
// observable canvas-size to justify the worker round-trip.

import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";

const elk = new ELK();

const DEFAULT_OPTS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.layered.spacing.nodeNodeBetweenLayers": "60",
  "elk.spacing.nodeNode": "40",
};

const DEFAULT_NODE_W = 180;
const DEFAULT_NODE_H = 80;

export async function layoutFlow(
  nodes: Node[],
  edges: Edge[],
  options: Record<string, string> = {},
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const graph: ElkNode = {
    id: "root",
    layoutOptions: { ...DEFAULT_OPTS, ...options },
    children: nodes.map((n) => ({
      id: n.id,
      width: (n.width as number | undefined) ?? DEFAULT_NODE_W,
      height: (n.height as number | undefined) ?? DEFAULT_NODE_H,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };
  const result = await elk.layout(graph);
  const out = nodes.map((n) => {
    const placed = result.children?.find((c) => c.id === n.id);
    return placed
      ? { ...n, position: { x: placed.x ?? 0, y: placed.y ?? 0 } }
      : n;
  });
  return { nodes: out, edges };
}
