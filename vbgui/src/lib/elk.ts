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
  // Deterministic layout across runs — otherwise row assignment / crossing
  // minimization can flip between page loads, making routing unreproducible.
  "elk.randomSeed": "1",
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
  if (rows.length < 2) return null;

  const out = new Map<string, WrapAssignment>();
  // Every row contributes to parity (singletons too) so the snake stays
  // strictly alternating. Each row is also SHIFTED in x so its chain-entry
  // side aligns with the previous row's chain-exit side — that's what turns
  // the inter-row link from a long bezier sweep into a short vertical hop.
  let prevExitX: number | null = null;
  rows.forEach((row, ri) => {
    const origMinX = Math.min(...row.map((c) => c.x));
    const origMaxR = Math.max(...row.map((c) => c.x + c.w));
    const reverse = ri % 2 === 1;

    // Entry side of this row (in ELK coords, before shift):
    //   normal row → leftmost x (origMinX)
    //   mirrored row → rightmost x after mirror (still origMaxR by construction)
    const entryX = reverse ? origMaxR : origMinX;
    const shift = prevExitX !== null ? prevExitX - entryX : 0;

    if (!reverse) {
      row.forEach((c) => out.set(c.id, { x: c.x + shift, y: c.y, reverse: false }));
      prevExitX = origMaxR + shift;
    } else {
      row.forEach((c) => {
        const mirroredX = origMaxR + origMinX - (c.x + c.w);
        out.set(c.id, { x: mirroredX + shift, y: c.y, reverse: true });
      });
      prevExitX = origMinX + shift;
    }
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
  // Each row gets a horizontal shift so its chain-entry side lines up with
  // the previous row's chain-exit side — turning the snake's inter-row link
  // into a short vertical hop instead of a long diagonal bezier.
  let prevExitX: number | null = null;
  rows.forEach((row, ri) => {
    const reverse = ri % 2 === 1;
    let xCursor = 0;
    const offs: number[] = [];
    row.layers.forEach((li, idx) => {
      offs.push(xCursor);
      xCursor += li.width + (idx < row.layers.length - 1 ? H_GAP : 0);
    });
    // Determine this row's entry x (in row-local coords, before global shift).
    // Normal row: entry at left (x=0). Reverse row: entry at right (x=row.width).
    const localEntryX = reverse ? row.width : 0;
    const shift = prevExitX !== null ? prevExitX - localEntryX : 0;

    row.layers.forEach((li, idx) => {
      const rowOffX = reverse ? row.width - offs[idx] - li.width : offs[idx];
      li.nodes.forEach((c) => {
        out.set(c.id, {
          x: rowOffX + (c.x - li.minX) + shift,
          y: yCursor + (c.y - li.minY),
          reverse,
        });
      });
    });
    // Exit side x (in row-local coords) + shift = next row's prevExitX.
    const localExitX = reverse ? 0 : row.width;
    prevExitX = localExitX + shift;
    yCursor += row.height + V_GAP;
  });
  return out;
}

// ─── Orthogonal obstacle-avoiding edge router ──────────────────────────────
// After final node positions are known (post-snake), route each edge as an
// axis-aligned polyline that never crosses a node body. A* over a Hanan grid
// (the X/Y lines induced by node boundaries + handle points) with a turn
// penalty so routes prefer few bends. This is what stops connectors from
// cutting straight through node boxes once the snake has moved nodes around.

interface RRect { id: string; x: number; y: number; w: number; h: number }
export type HandleSideName = "top" | "right" | "bottom" | "left";
interface SideCand { side: HandleSideName; x: number; y: number }
interface RouteReq {
  id: string; source: string; target: string;
  // Candidate attachment points (one per side) for each endpoint. The router
  // picks the (srcSide, tgtSide) pair that yields the cleanest box-free route.
  srcCands: SideCand[]; tgtCands: SideCand[];
}
interface RouteResult {
  pts: { x: number; y: number }[];
  srcSide: HandleSideName;
  tgtSide: HandleSideName;
}

const ROUTE_MARGIN = 16; // clearance kept between routes and node bodies
const ROUTE_TURN_PENALTY = 60; // cost (px-equivalent) of each 90° bend
// Cost knobs for side selection (tuned against probe_routing.py):
const ROUTE_OCCUPANCY_PENALTY = 46; // per edge already using a node's side — spreads fan-in/out across sides
const ROUTE_FACING_PENALTY = 100000; // huge: never exit/enter a side pointing away from the other end
const ROUTE_SIDE_PRIOR_BONUS = 12; // mild: prefer right-out / left-in so forward chains stay canonical
const ROUTE_ALIGNMENT_BONUS = 80; // reward sides whose normal aligns with the direction to the other end:
// an edge approaching from below enters the BOTTOM handle, from above the TOP, etc. (geometric intuition).
// 80 > one occupancy hit (46) so alignment wins, but a 2nd edge into a busy side still spreads off.

// Outward normal of a side (screen coords: +y is down).
const SIDE_NORMAL: Record<HandleSideName, { x: number; y: number }> = {
  top: { x: 0, y: -1 }, right: { x: 1, y: 0 }, bottom: { x: 0, y: 1 }, left: { x: -1, y: 0 },
};
// Deterministic side ordering for tie-breaks.
const SIDE_ORDER: HandleSideName[] = ["top", "right", "bottom", "left"];

function candidatesFor(r: RRect): SideCand[] {
  return [
    { side: "top", x: r.x + r.w / 2, y: r.y },
    { side: "right", x: r.x + r.w, y: r.y + r.h / 2 },
    { side: "bottom", x: r.x + r.w / 2, y: r.y + r.h },
    { side: "left", x: r.x, y: r.y + r.h / 2 },
  ];
}

function uniqSorted(vals: number[], tol: number): number[] {
  const s = [...vals].sort((a, b) => a - b);
  const out: number[] = [];
  for (const v of s) {
    if (out.length === 0 || v - out[out.length - 1] > tol) out.push(v);
  }
  return out;
}

function nearestIndex(arr: number[], v: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - v);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// Minimal binary min-heap: push(priority, value) / pop() → value with min priority.
class MinHeap {
  private k: number[] = [];
  private v: number[] = [];
  get size(): number { return this.k.length; }
  push(key: number, val: number): void {
    const { k, v } = this;
    k.push(key); v.push(val);
    let i = k.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= k[i]) break;
      [k[p], k[i]] = [k[i], k[p]];
      [v[p], v[i]] = [v[i], v[p]];
      i = p;
    }
  }
  pop(): number {
    const { k, v } = this;
    const top = v[0];
    const lk = k.pop()!;
    const lv = v.pop()!;
    if (k.length) {
      k[0] = lk; v[0] = lv;
      let i = 0;
      const n = k.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let m = i;
        if (l < n && k[l] < k[m]) m = l;
        if (r < n && k[r] < k[m]) m = r;
        if (m === i) break;
        [k[m], k[i]] = [k[i], k[m]];
        [v[m], v[i]] = [v[i], v[m]];
        i = m;
      }
    }
    return top;
  }
}

// Route a single edge between explicit endpoints through the shared Hanan grid
// (xs, ys). Returns the simplified orthogonal polyline (flow coords) or null if
// no path found. Map-based A* so calling it many times per edge (one per
// candidate side pair) stays cheap.
function routeOne(
  sx: number, sy: number, tx: number, ty: number,
  xs: number[],
  ys: number[],
  inflated: number[][], // [x0, x1, y0, y1] per obstacle
): { x: number; y: number }[] | null {
  const nx = xs.length;
  const ny = ys.length;
  const si = nearestIndex(xs, sx);
  const sj = nearestIndex(ys, sy);
  const ti = nearestIndex(xs, tx);
  const tj = nearestIndex(ys, ty);
  const eps = 0.5;
  // Obstacles (already pre-filtered to exclude this edge's endpoints).
  const obst = inflated;

  // Segment passability: horizontal seg (i..i+1) at row j; vertical seg
  // (j..j+1) at col i. Blocked if it pierces an obstacle's strict interior.
  const passH = (i: number, j: number): boolean => {
    const y = ys[j];
    const x1 = xs[i];
    const x2 = xs[i + 1];
    for (const b of obst) {
      if (y > b[2] + eps && y < b[3] - eps && x2 > b[0] + eps && x1 < b[1] - eps) return false;
    }
    return true;
  };
  const passV = (i: number, j: number): boolean => {
    const x = xs[i];
    const y1 = ys[j];
    const y2 = ys[j + 1];
    for (const b of obst) {
      if (x > b[0] + eps && x < b[1] - eps && y2 > b[2] + eps && y1 < b[3] - eps) return false;
    }
    return true;
  };

  // Sparse A* (state = (j*nx+i)*3 + dir; dir: 0 none, 1 H, 2 V). Map-based so
  // we don't allocate an nx*ny*3 array per call — routeOne runs up to 16× per
  // edge during side selection.
  const g = new Map<number, number>();
  const prev = new Map<number, number>();
  const closed = new Set<number>();
  const hh = (i: number, j: number): number =>
    Math.abs(xs[i] - xs[ti]) + Math.abs(ys[j] - ys[tj]);

  const startState = (sj * nx + si) * 3;
  g.set(startState, 0);
  const heap = new MinHeap();
  heap.push(hh(si, sj), startState);

  let goalState = -1;
  let iter = 0;
  const CAP = 400000;
  while (heap.size && iter++ < CAP) {
    const s = heap.pop();
    if (closed.has(s)) continue;
    closed.add(s);
    const dir = s % 3;
    const cell = (s - dir) / 3;
    const i = cell % nx;
    const j = (cell - i) / nx;
    if (i === ti && j === tj) { goalState = s; break; }
    const gc = g.get(s)!;
    const relax = (ni: number, nj: number, ndir: number, len: number): void => {
      const ns = (nj * nx + ni) * 3 + ndir;
      const turn = dir !== 0 && dir !== ndir ? ROUTE_TURN_PENALTY : 0;
      const ng = gc + len + turn;
      if (ng < (g.get(ns) ?? Infinity)) { g.set(ns, ng); prev.set(ns, s); heap.push(ng + hh(ni, nj), ns); }
    };
    if (i + 1 < nx && passH(i, j)) relax(i + 1, j, 1, xs[i + 1] - xs[i]);
    if (i - 1 >= 0 && passH(i - 1, j)) relax(i - 1, j, 1, xs[i] - xs[i - 1]);
    if (j + 1 < ny && passV(i, j)) relax(i, j + 1, 2, ys[j + 1] - ys[j]);
    if (j - 1 >= 0 && passV(i, j - 1)) relax(i, j - 1, 2, ys[j] - ys[j - 1]);
  }
  if (goalState < 0) return null;

  // Reconstruct cell path, drop dir, dedupe, simplify collinear runs.
  const cells: number[] = [];
  let s: number | undefined = goalState;
  while (s !== undefined) { const dir = s % 3; cells.push((s - dir) / 3); s = prev.get(s); }
  cells.reverse();
  const raw = cells.map((c) => { const i = c % nx; return { x: xs[i], y: ys[(c - i) / nx] }; });
  const dd: { x: number; y: number }[] = [];
  for (const p of raw) {
    const l = dd[dd.length - 1];
    if (!l || l.x !== p.x || l.y !== p.y) dd.push(p);
  }
  const simp: { x: number; y: number }[] = [];
  for (let k = 0; k < dd.length; k++) {
    if (k > 0 && k < dd.length - 1) {
      const a = dd[k - 1];
      const b = dd[k];
      const c = dd[k + 1];
      if ((a.x === b.x && b.x === c.x) || (a.y === b.y && b.y === c.y)) continue;
    }
    simp.push(dd[k]);
  }
  return simp.length >= 2 ? simp : null;
}

function routeOrthogonal(
  rects: RRect[],
  reqs: RouteReq[],
  prior?: Map<string, { srcRight: boolean; tgtRight: boolean }>,
  // Ids of residual-add ("+") nodes: their inputs are forced to enter
  // vertically (top & bottom) so the two summed wires converge like a real
  // junction instead of stacking on one side.
  addIds?: Set<string>,
): Map<string, RouteResult> {
  const result = new Map<string, RouteResult>();
  if (reqs.length === 0) return result;
  const M = ROUTE_MARGIN;

  // Shared Hanan grid lines: inflated node boundaries + EVERY candidate point
  // (all 4 sides of both endpoints of every edge) so any chosen side snaps to
  // an exact grid line.
  const xsRaw: number[] = [];
  const ysRaw: number[] = [];
  for (const r of rects) { xsRaw.push(r.x - M, r.x + r.w + M); ysRaw.push(r.y - M, r.y + r.h + M); }
  for (const q of reqs) {
    for (const c of q.srcCands) { xsRaw.push(c.x); ysRaw.push(c.y); }
    for (const c of q.tgtCands) { xsRaw.push(c.x); ysRaw.push(c.y); }
  }
  const xs = uniqSorted(xsRaw, 6);
  const ys = uniqSorted(ysRaw, 6);

  // Occupancy: how many committed edges already attach to a given node side.
  // Used to spread fan-in/out (e.g. residual-add) across sides instead of
  // stacking everything on one. Keyed `${nodeId}|in|side` / `${nodeId}|out|side`.
  const occ = new Map<string, number>();
  const occKey = (id: string, dir: "in" | "out", side: HandleSideName) => `${id}|${dir}|${side}`;

  const pathCost = (pts: { x: number; y: number }[]): number => {
    let len = 0;
    for (let i = 1; i < pts.length; i++) len += Math.abs(pts[i].x - pts[i - 1].x) + Math.abs(pts[i].y - pts[i - 1].y);
    const turns = Math.max(0, pts.length - 2);
    return len + ROUTE_TURN_PENALTY * turns;
  };

  // Stable edge order (already deterministic since reqs preserves edges order).
  for (const q of reqs) {
    const inflated: number[][] = [];
    for (const r of rects) {
      if (r.id === q.source || r.id === q.target) continue;
      inflated.push([r.x - M, r.x + r.w + M, r.y - M, r.y + r.h + M]);
    }
    const pr = prior?.get(q.id);
    const priorSrc: HandleSideName | null = pr ? (pr.srcRight ? "right" : "left") : null;
    const priorTgt: HandleSideName | null = pr ? (pr.tgtRight ? "right" : "left") : null;

    // Force "+" (residual_add) inputs to enter top/bottom so the two summed
    // wires converge vertically like a junction. Output of a "+" stays free
    // (alignment sends it toward its successor).
    const tgtCands = addIds?.has(q.target)
      ? q.tgtCands.filter((c) => c.side === "top" || c.side === "bottom")
      : q.tgtCands;
    const srcCands = q.srcCands;

    // Direction sanity: cull sides whose outward normal points away from the
    // other endpoint (prevents "exit the back to reach a node in front").
    let best: { cost: number; res: RouteResult; ss: HandleSideName; ts: HandleSideName } | null = null;
    for (const sc of srcCands) {
      for (const tc of tgtCands) {
        const dx = tc.x - sc.x;
        const dy = tc.y - sc.y;
        const sn = SIDE_NORMAL[sc.side];
        const tn = SIDE_NORMAL[tc.side];
        const dlen = Math.hypot(dx, dy) || 1;
        const srcFacing = (sn.x * dx + sn.y * dy) / dlen; // want >= ~0 (exit toward target)
        const tgtFacing = (tn.x * -dx + tn.y * -dy) / dlen; // want >= ~0 (face back toward source)
        // Prune hard back-facing unless it's the canonical prior side.
        const srcFacingOk = srcFacing >= -0.2 || sc.side === priorSrc;
        const tgtFacingOk = tgtFacing >= -0.2 || tc.side === priorTgt;
        if (!srcFacingOk || !tgtFacingOk) continue;

        const path = routeOne(sc.x, sc.y, tc.x, tc.y, xs, ys, inflated);
        if (!path) continue;

        let cost = pathCost(path);
        if (srcFacing < 0) cost += ROUTE_FACING_PENALTY * (-srcFacing);
        if (tgtFacing < 0) cost += ROUTE_FACING_PENALTY * (-tgtFacing);
        // Reward forward-facing alignment so the chosen side faces where the
        // wire comes from (top/bottom for vertical approaches, not just sides).
        cost -= ROUTE_ALIGNMENT_BONUS * Math.max(0, srcFacing);
        cost -= ROUTE_ALIGNMENT_BONUS * Math.max(0, tgtFacing);
        cost += ROUTE_OCCUPANCY_PENALTY * (occ.get(occKey(q.source, "out", sc.side)) ?? 0);
        cost += ROUTE_OCCUPANCY_PENALTY * (occ.get(occKey(q.target, "in", tc.side)) ?? 0);
        if (sc.side === priorSrc) cost -= ROUTE_SIDE_PRIOR_BONUS;
        if (tc.side === priorTgt) cost -= ROUTE_SIDE_PRIOR_BONUS;
        // Deterministic tie-break by side order.
        cost += (SIDE_ORDER.indexOf(sc.side) + SIDE_ORDER.indexOf(tc.side)) * 1e-3;

        if (!best || cost < best.cost) {
          best = { cost, res: { pts: path, srcSide: sc.side, tgtSide: tc.side }, ss: sc.side, ts: tc.side };
        }
      }
    }
    if (best) {
      result.set(q.id, best.res);
      occ.set(occKey(q.source, "out", best.ss), (occ.get(occKey(q.source, "out", best.ss)) ?? 0) + 1);
      occ.set(occKey(q.target, "in", best.ts), (occ.get(occKey(q.target, "in", best.ts)) ?? 0) + 1);
    }
  }
  return result;
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

  // Obstacle-avoiding orthogonal routing on the FINAL node positions. This is
  // the authoritative edge geometry — it guarantees no connector crosses a
  // node body, in both the wrapped (snake) and unwrapped cases. ELK's own
  // bend points are only a fallback if the router fails on an edge.
  let routed = new Map<string, RouteResult>();
  if (horizontal) {
    const rectOf = new Map<string, RRect>();
    out.forEach((n) => {
      const { w, h } = size(n);
      rectOf.set(n.id, { id: n.id, x: n.position.x, y: n.position.y, w, h });
    });
    const rects: RRect[] = [...rectOf.values()];
    const addIds = new Set(out.filter((n) => n.type === "residual_add").map((n) => n.id));
    const reqs: RouteReq[] = [];
    const prior = new Map<string, { srcRight: boolean; tgtRight: boolean }>();
    for (const e of edges) {
      const sr = rectOf.get(e.source);
      const tr = rectOf.get(e.target);
      if (!sr || !tr) continue;
      reqs.push({
        id: e.id,
        source: e.source,
        target: e.target,
        srcCands: candidatesFor(sr),
        tgtCands: candidatesFor(tr),
      });
      prior.set(e.id, {
        srcRight: nodeSourceRight.get(e.source) ?? true,
        tgtRight: nodeTargetRight.get(e.target) ?? false,
      });
    }
    routed = routeOrthogonal(rects, reqs, prior, addIds);
  }

  const outEdges = edges.map((e) => {
    const r = routed.get(e.id);
    if (r && r.pts.length >= 2) {
      // Chosen side → handle id MUST match so React Flow resolves the edge
      // endpoint to the same handle the route used (else bendsFresh fails and
      // it falls back to a box-crossing bezier).
      return {
        ...e,
        sourceHandle: `s-${r.srcSide}`,
        targetHandle: `t-${r.tgtSide}`,
        data: { ...(e.data as any), elkBends: r.pts },
      };
    }
    // Fallback (router found no path): use ELK bends if their ports agree with
    // the canonical left/right prior; otherwise clean bezier. Either way emit
    // explicit handle ids matching the prior so RF doesn't pick an arbitrary
    // handle (8 handles exist now → undefined handle is ambiguous).
    const sRight = nodeSourceRight.get(e.source) ?? true;
    const tRight = nodeTargetRight.get(e.target) ?? false;
    const fallbackHandles = {
      sourceHandle: sRight ? "s-right" : "s-left",
      targetHandle: tRight ? "t-right" : "t-left",
    };
    if (!wrapped) {
      const rec = bendsById.get(e.id);
      if (rec && rec.srcRight === sRight && rec.tgtRight === tRight) {
        return { ...e, ...fallbackHandles, data: { ...(e.data as any), elkBends: rec.pts } };
      }
    }
    return { ...e, ...fallbackHandles, data: { ...(e.data as any), elkBends: undefined } };
  });

  return { nodes: out, edges: outEdges };
}
