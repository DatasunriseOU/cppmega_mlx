// Auto-layout helper backed by ELK.js (layered / Sugiyama). This replaces the
// hand-rolled boustrophedon "snake" layout: ELK performs real layer
// assignment + crossing minimization, which is what produces an optimal
// left-to-right topological layout with minimal edge crossings — including for
// residual DAGs where a skip edge jumps over several layers.

import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import { Position, type Edge, type Node } from "@xyflow/react";

const elk = new ELK();

const DEFAULT_OPTS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  // Spacing tuned so wide folded-group nodes and tall tokenizer nodes don't
  // crowd their neighbours.
  "elk.layered.spacing.nodeNodeBetweenLayers": "90",
  "elk.spacing.nodeNode": "64",
  "elk.spacing.edgeNode": "28",
  "elk.spacing.edgeEdge": "16",
  // Crossing minimization + tidy node placement.
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  // Orthogonal routing gives clean right-angle channels and emits bend points
  // we render directly (see MidpointEdge), instead of long beziers sweeping
  // across the whole graph.
  "elk.edgeRouting": "ORTHOGONAL",
};

const DEFAULT_NODE_W = 240;
const DEFAULT_NODE_H = 120;

export interface LayoutFlowOpts {
  /** Per-node size; falls back to measured dims then defaults. */
  sizeOf?: (n: Node) => { w: number; h: number };
  /** Extra ELK layout options (override defaults). */
  options?: Record<string, string>;
}

export async function layoutFlow(
  nodes: Node[],
  edges: Edge[],
  opts: LayoutFlowOpts = {},
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const { sizeOf, options = {} } = opts;

  const size = (n: Node): { w: number; h: number } => {
    if (sizeOf) return sizeOf(n);
    const m = (n as any).measured as { width?: number; height?: number } | undefined;
    if (m?.width && m?.height) return { w: m.width, h: m.height };
    return { w: DEFAULT_NODE_W, h: DEFAULT_NODE_H };
  };

  const layoutOptions = { ...DEFAULT_OPTS, ...options };
  const direction = layoutOptions["elk.direction"] ?? "RIGHT";
  const horizontal = direction === "RIGHT" || direction === "LEFT";

  const graph: ElkNode = {
    id: "root",
    layoutOptions,
    children: nodes.map((n) => {
      const { w, h } = size(n);
      // FREE ports let ELK attach an edge on whichever side (left/right)
      // reduces crossings, instead of forcing every input west / output east.
      return { id: n.id, width: w, height: h, layoutOptions: { "elk.portConstraints": "FREE" } };
    }),
    edges: edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const result = await elk.layout(graph);

  // Placed geometry, so we can tell which side of each node an edge attaches to.
  const geo = new Map<string, { x: number; y: number; w: number; h: number }>();
  (result.children ?? []).forEach((c: any) => {
    geo.set(c.id, { x: c.x ?? 0, y: c.y ?? 0, w: c.width ?? DEFAULT_NODE_W, h: c.height ?? DEFAULT_NODE_H });
  });

  // Tally, per node, whether ELK attached its inputs / outputs on the left or
  // right side (from the routed section endpoints). Standard forward edges stay
  // left-in / right-out; ELK only flips a side when it reduces crossings.
  const inSide = new Map<string, { l: number; r: number }>();
  const outSide = new Map<string, { l: number; r: number }>();
  const bump = (m: Map<string, { l: number; r: number }>, id: string, right: boolean) => {
    const o = m.get(id) ?? { l: 0, r: 0 };
    if (right) o.r++; else o.l++;
    m.set(id, o);
  };
  const bendsById = new Map<
    string,
    { pts: { x: number; y: number }[]; srcRight: boolean; tgtRight: boolean }
  >();
  (result.edges ?? []).forEach((e: any) => {
    const sec = e.sections?.[0];
    if (!sec) return;
    const src = e.sources?.[0];
    const tgt = e.targets?.[0];
    const sg = src ? geo.get(src) : undefined;
    const tg = tgt ? geo.get(tgt) : undefined;
    const srcRight = !!(sg && sec.startPoint && sec.startPoint.x >= sg.x + sg.w / 2);
    const tgtRight = !!(tg && sec.endPoint && sec.endPoint.x >= tg.x + tg.w / 2);
    if (sg && sec.startPoint) bump(outSide, src, srcRight);
    if (tg && sec.endPoint) bump(inSide, tgt, tgtRight);

    // Capture orthogonal bend points for the edge renderer.
    const pts: { x: number; y: number }[] = [];
    if (sec.startPoint) pts.push({ x: sec.startPoint.x, y: sec.startPoint.y });
    (sec.bendPoints ?? []).forEach((b: any) => pts.push({ x: b.x, y: b.y }));
    if (sec.endPoint) pts.push({ x: sec.endPoint.x, y: sec.endPoint.y });
    if (pts.length > 0) bendsById.set(e.id, { pts, srcRight, tgtRight });
  });

  // Decide each node's single in/out handle side by majority of ELK's port
  // sides. Default stays input-left / output-right; a side only flips when ELK
  // attached most of that node's edges on the other side to reduce crossings.
  const nodeSourceRight = new Map<string, boolean>();
  const nodeTargetRight = new Map<string, boolean>();
  const out = nodes.map((n) => {
    const placed = result.children?.find((c) => c.id === n.id);
    let targetPosition: Position;
    let sourcePosition: Position;
    if (horizontal) {
      const ti = inSide.get(n.id);
      const so = outSide.get(n.id);
      const tRight = !!(ti && ti.r > ti.l);
      const sRight = !(so && so.l > so.r); // default right unless left dominates
      nodeTargetRight.set(n.id, tRight);
      nodeSourceRight.set(n.id, sRight);
      targetPosition = tRight ? Position.Right : Position.Left;
      sourcePosition = sRight ? Position.Right : Position.Left;
    } else {
      targetPosition = Position.Top;
      sourcePosition = Position.Bottom;
    }
    if (!placed) return n;
    return {
      ...n,
      position: { x: placed.x ?? 0, y: placed.y ?? 0 },
      data: { ...(n.data as any), targetPosition, sourcePosition },
    };
  });

  // Attach orthogonal bend points only when both endpoints' ELK port sides
  // agree with the node handle sides we chose — otherwise the rendered polyline
  // would dangle off to the wrong side, so we fall back to a clean bezier.
  const outEdges = edges.map((e) => {
    const rec = bendsById.get(e.id);
    if (!rec) return { ...e, data: { ...(e.data as any), elkBends: undefined } };
    const ok =
      rec.srcRight === (nodeSourceRight.get(e.source) ?? false) &&
      rec.tgtRight === (nodeTargetRight.get(e.target) ?? false);
    return { ...e, data: { ...(e.data as any), elkBends: ok ? rec.pts : undefined } };
  });

  return { nodes: out, edges: outEdges };
}
