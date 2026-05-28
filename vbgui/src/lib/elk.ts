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
  /** Available canvas width; when set, the layout wraps into multiple rows
   *  (boustrophedon snake) if the single-row layout would overflow it. */
  canvasWidth?: number;
  /** Available canvas height; combined with canvasWidth, drives ELK's
   *  aspectRatio hint so the layout doesn't run off the bottom of the canvas
   *  on graphs with many parallel residual branches. */
  canvasHeight?: number;
}

interface WrapAssignment { x: number; y: number; reverse: boolean }

// When ELK has already produced multiple horizontal "rows" of its own
// (e.g. repetitive residual blocks stacked vertically), flip every other row
// to flow right-to-left so cross-row connections become short vertical hops
// instead of long beziers sweeping across the canvas. Detects rows by
// y-gap clustering of the placed nodes.
function snakifyExistingRows(
  children: { id: string; x: number; y: number; w: number; h: number }[],
): Map<string, WrapAssignment> | null {
  if (children.length < 2) return null;
  const byY = [...children].sort((a, b) => a.y - b.y);
  // Gap-based clustering: a new row starts when the gap from the prior row's
  // bottom exceeds ~80px (well above intra-row residual offsets).
  const rows: typeof byY[] = [];
  for (const c of byY) {
    const last = rows[rows.length - 1];
    if (last) {
      const lastMaxY = Math.max(...last.map((p) => p.y + p.h));
      if (c.y - lastMaxY > 80) rows.push([c]); else last.push(c);
    } else rows.push([c]);
  }
  if (rows.length < 2) return null; // single row, no snake to make

  const out = new Map<string, WrapAssignment>();
  // Parity counts only multi-node rows. Singleton rows (a lone rmsnorm or
  // residual_add between bigger rows) inherit the prior row's direction so
  // their handle sides line up with the chain instead of flipping in place.
  let majorIdx = 0;
  let lastReverse = false;
  rows.forEach((row) => {
    if (row.length < 2) {
      // Singleton row — keep ELK position, follow prior multi-node row's direction.
      row.forEach((c) => out.set(c.id, { x: c.x, y: c.y, reverse: lastReverse }));
      return;
    }
    const reverse = majorIdx % 2 === 1;
    majorIdx++;
    lastReverse = reverse;
    if (!reverse) {
      row.forEach((c) => out.set(c.id, { x: c.x, y: c.y, reverse: false }));
      return;
    }
    const minX = Math.min(...row.map((c) => c.x));
    const maxR = Math.max(...row.map((c) => c.x + c.w));
    row.forEach((c) => {
      const mirroredX = maxR + minX - (c.x + c.w);
      out.set(c.id, { x: mirroredX, y: c.y, reverse: true });
    });
  });
  return out;
}

// Pack the layered ELK result into multiple rows so a long forward chain
// fills the visible canvas instead of running off the right edge. Odd rows
// flow right-to-left (with handle sides flipped) — the boustrophedon "snake"
// the user explicitly wants at the row level (not the per-node level).
function computeRowWrap(
  children: { id: string; x: number; y: number; w: number; h: number }[],
  canvasWidth: number,
): Map<string, WrapAssignment> | null {
  if (children.length === 0) return null;
  const minX = Math.min(...children.map((c) => c.x));
  const maxR = Math.max(...children.map((c) => c.x + c.w));
  const totalWidth = maxR - minX;
  const budget = Math.max(900, canvasWidth - 80);
  if (totalWidth <= budget) return null; // fits in one row, no snake needed

  // Cluster into ELK layers by x position (within ~30px = same column).
  const sorted = [...children].sort((a, b) => a.x - b.x);
  type Child = typeof sorted[number];
  const layers: Child[][] = [];
  const TOL = 30;
  for (const c of sorted) {
    const last = layers[layers.length - 1];
    if (last && Math.abs(c.x - last[0].x) < TOL) last.push(c);
    else layers.push([c]);
  }

  interface LayerInfo {
    nodes: Child[]; minX: number; width: number; minY: number; height: number;
  }
  const lis: LayerInfo[] = layers.map((layer) => {
    const lx = Math.min(...layer.map((c) => c.x));
    const lr = Math.max(...layer.map((c) => c.x + c.w));
    const ly = Math.min(...layer.map((c) => c.y));
    const lb = Math.max(...layer.map((c) => c.y + c.h));
    return { nodes: layer, minX: lx, width: lr - lx, minY: ly, height: lb - ly };
  });

  // Greedy: pack layers into rows fitting `budget`.
  const H_GAP = 60;
  const V_GAP = 110;
  const rows: { layers: LayerInfo[]; width: number; height: number }[] = [];
  let cur: LayerInfo[] = [];
  let curW = 0;
  for (const li of lis) {
    const next = cur.length === 0 ? li.width : curW + H_GAP + li.width;
    if (cur.length > 0 && next > budget) {
      rows.push({ layers: cur, width: curW, height: Math.max(...cur.map((l) => l.height)) });
      cur = [li]; curW = li.width;
    } else {
      cur.push(li); curW = next;
    }
  }
  if (cur.length) rows.push({ layers: cur, width: curW, height: Math.max(...cur.map((l) => l.height)) });
  if (rows.length <= 1) return null; // didn't actually need to wrap

  const out = new Map<string, WrapAssignment>();
  let yCursor = 0;
  rows.forEach((row, ri) => {
    const reverse = ri % 2 === 1;
    let xCursor = 0;
    const offs: number[] = [];
    row.layers.forEach((li, idx) => {
      offs.push(xCursor);
      xCursor += li.width + (idx < row.layers.length - 1 ? H_GAP : 0);
    });
    row.layers.forEach((li, idx) => {
      const rowOffX = reverse ? row.width - offs[idx] - li.width : offs[idx];
      li.nodes.forEach((c) => {
        out.set(c.id, {
          x: rowOffX + (c.x - li.minX),
          y: yCursor + (c.y - li.minY),
          reverse,
        });
      });
    });
    yCursor += row.height + V_GAP;
  });
  return out;
}

export async function layoutFlow(
  nodes: Node[],
  edges: Edge[],
  opts: LayoutFlowOpts = {},
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const { sizeOf, options = {}, canvasWidth, canvasHeight } = opts;

  const size = (n: Node): { w: number; h: number } => {
    if (sizeOf) return sizeOf(n);
    const m = (n as any).measured as { width?: number; height?: number } | undefined;
    if (m?.width && m?.height) return { w: m.width, h: m.height };
    return { w: DEFAULT_NODE_W, h: DEFAULT_NODE_H };
  };

  // Pass the canvas aspect to ELK so it stops piling parallel branches into a
  // very tall, thin column when there's plenty of horizontal room.
  const aspectHint: Record<string, string> = {};
  if (canvasWidth && canvasHeight && canvasWidth > 200 && canvasHeight > 200) {
    aspectHint["elk.aspectRatio"] = (canvasWidth / canvasHeight).toFixed(2);
  }
  const layoutOptions = { ...DEFAULT_OPTS, ...aspectHint, ...options };
  const direction = layoutOptions["elk.direction"] ?? "RIGHT";
  const horizontal = direction === "RIGHT" || direction === "LEFT";

  const graph: ElkNode = {
    id: "root",
    layoutOptions,
    children: nodes.map((n) => {
      const { w, h } = size(n);
      // FIXED_SIDE: every input goes on the west handle, every output on the
      // east handle. This keeps the layered output compact horizontally (no
      // vertical port-stacking) so wide graphs stay wide — and we provide the
      // "right-to-left when needed" freedom at the row level instead, via the
      // snake wrap below.
      return { id: n.id, width: w, height: h };
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

  // If the single-row ELK layout would overflow the visible canvas, wrap it
  // into multiple rows (snake). When wrapped, each row's reverse flag drives
  // both x mirroring and handle sides; ELK's orthogonal bends no longer match
  // the relocated nodes, so we drop them and let edges render as clean beziers.
  const placedChildren: { id: string; x: number; y: number; w: number; h: number }[] = [];
  (result.children ?? []).forEach((c: any) => {
    placedChildren.push({
      id: c.id, x: c.x ?? 0, y: c.y ?? 0,
      w: c.width ?? DEFAULT_NODE_W, h: c.height ?? DEFAULT_NODE_H,
    });
  });
  // Two-stage wrap:
  //   1) If ELK already stacked the graph into multiple rows (e.g. repetitive
  //      residual blocks), just snake-ify them — flip every other row R-to-L
  //      so cross-row links are short vertical hops.
  //   2) Otherwise, if the single-row layout would overflow canvas width,
  //      pack it into rows and snake.
  const snake = snakifyExistingRows(placedChildren);
  const wrap = snake ?? (canvasWidth ? computeRowWrap(placedChildren, canvasWidth) : null);
  const wrapped = wrap !== null;

  // Decide each node's single in/out handle side by majority of ELK's port
  // sides. Default stays input-left / output-right; a side only flips when ELK
  // attached most of that node's edges on the other side to reduce crossings.
  // When the layout is wrapped, the row direction dictates the side instead.
  const nodeSourceRight = new Map<string, boolean>();
  const nodeTargetRight = new Map<string, boolean>();
  const out = nodes.map((n) => {
    const placed = result.children?.find((c) => c.id === n.id);
    let targetPosition: Position;
    let sourcePosition: Position;
    let posX = placed?.x ?? 0;
    let posY = placed?.y ?? 0;
    const wa = wrap?.get(n.id);
    if (wa) {
      posX = wa.x;
      posY = wa.y;
      // Reverse row flows right→left: input on the right, output on the left.
      targetPosition = wa.reverse ? Position.Right : Position.Left;
      sourcePosition = wa.reverse ? Position.Left : Position.Right;
      nodeTargetRight.set(n.id, wa.reverse);
      nodeSourceRight.set(n.id, !wa.reverse);
    } else if (horizontal) {
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
      position: { x: posX, y: posY },
      data: { ...(n.data as any), targetPosition, sourcePosition },
    };
  });

  // Attach orthogonal bend points only when (a) the layout wasn't row-wrapped
  // and (b) both endpoint port sides agree with the chosen handle sides.
  // Otherwise the rendered polyline would dangle off to the wrong side or
  // point at stale coordinates, so we fall back to a clean bezier.
  const outEdges = edges.map((e) => {
    if (wrapped) return { ...e, data: { ...(e.data as any), elkBends: undefined } };
    const rec = bendsById.get(e.id);
    if (!rec) return { ...e, data: { ...(e.data as any), elkBends: undefined } };
    const ok =
      rec.srcRight === (nodeSourceRight.get(e.source) ?? false) &&
      rec.tgtRight === (nodeTargetRight.get(e.target) ?? false);
    return { ...e, data: { ...(e.data as any), elkBends: ok ? rec.pts : undefined } };
  });

  return { nodes: out, edges: outEdges };
}
