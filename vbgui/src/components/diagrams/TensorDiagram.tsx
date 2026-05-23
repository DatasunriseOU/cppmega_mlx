/**
 * cppmega-mlx-w2t6: tiny SVG mini-framework for tensor-flow diagrams
 * inside HelpModal / ExplainModal.
 *
 * Each diagram is declarative JSX: a <Diagram> wrapper picks the
 * canvas size + dark-theme background, then primitives like
 * <Tensor>, <Op>, <Arrow>, <Residual>, <Group> position elements via
 * absolute x/y coords. The container automatically scales to its
 * container width while preserving the aspect ratio.
 *
 * Colour palette pinned to the rest of the canvas:
 *   - background: rgba(15, 23, 42, 0.6) (#0f172a-ish)
 *   - tensor box: fill #1e293b, stroke #22d3ee (cyan)
 *   - op box: fill #312e81, stroke #818cf8 (indigo)
 *   - arrow: #94a3b8 (slate-400)
 *   - residual: #facc15 (yellow), dashed
 *   - text: #f1f5f9 (slate-50)
 */

import { type ReactNode } from "react";


export const DIAG_THEME = {
  bg:        "#0e1014",   // Raschka/3B1B-style deep neutral
  bgPanel:   "rgba(20, 23, 28, 0.85)",
  border:    "rgba(255, 255, 255, 0.06)",
  tensorBg:  "#1e293b",
  tensorFg:  "#22d3ee",
  opBg:      "#312e81",
  opFg:      "#818cf8",
  arrow:     "#9aa0a6",
  residual:  "#f5b841",
  text:      "#e8e8e8",
  textMuted: "#9aa0a6",
  // Role-coded hues per the unified Raschka + 3B1B + Distill style:
  roleQ:     "#f5b841",  // amber — query
  roleK:     "#4ec9b0",  // cyan  — key
  roleV:     "#b48ead",  // violet — value
  roleAttn:  "#5aa9e6",  // blue — attention
  roleAttnL: "#d96c8e",  // pink low end of attn heatmap
  roleOut:   "#7bc47f",  // green — output
  masked:    "#3a3f47",
} as const;


export type CellRole = "q" | "k" | "v" | "attn" | "out" | "gate"
                       | "hidden" | "raw" | "mask";

const ROLE_TINT: Record<CellRole, string> = {
  q:      DIAG_THEME.roleQ,
  k:      DIAG_THEME.roleK,
  v:      DIAG_THEME.roleV,
  attn:   DIAG_THEME.roleAttn,
  out:    DIAG_THEME.roleOut,
  gate:   "#fbbf24",
  hidden: "#94a3b8",
  raw:    "#64748b",
  mask:   DIAG_THEME.masked,
};


/** Map a float (clipped to [-clip, +clip]) to a hex colour:
 *  blue (high) → muted (zero) → pink (low). Used by MatrixGrid for
 *  per-cell heatmap rendering. */
export function heatmapColor(
  value: number, clip: number = 2.0, role: CellRole = "raw",
): string {
  const v = Math.max(-clip, Math.min(clip, value));
  const t = (v + clip) / (2 * clip);     // [0,1]
  // Diverging palette: pink (0) → grey (0.5) → blue/role (1).
  const lo = { r: 0xd9, g: 0x6c, b: 0x8e };
  const mid = { r: 0x24, g: 0x29, b: 0x33 };
  const hiHex = ROLE_TINT[role] === ROLE_TINT.raw
    ? "#5aa9e6" : ROLE_TINT[role];
  const hi = {
    r: parseInt(hiHex.slice(1, 3), 16),
    g: parseInt(hiHex.slice(3, 5), 16),
    b: parseInt(hiHex.slice(5, 7), 16),
  };
  const lerp = (a: number, b: number, k: number): number =>
    Math.round(a + (b - a) * k);
  const c = t < 0.5
    ? { r: lerp(lo.r, mid.r, t * 2),
        g: lerp(lo.g, mid.g, t * 2),
        b: lerp(lo.b, mid.b, t * 2) }
    : { r: lerp(mid.r, hi.r, (t - 0.5) * 2),
        g: lerp(mid.g, hi.g, (t - 0.5) * 2),
        b: lerp(mid.b, hi.b, (t - 0.5) * 2) };
  return `rgb(${c.r}, ${c.g}, ${c.b})`;
}


export interface DiagramProps {
  width?: number;
  height?: number;
  /** Diagram caption (rendered above the SVG). */
  caption?: string;
  children: ReactNode;
}

/** Root SVG container. Scales to width via `width="100%"` on the
 *  parent <svg>; coordinates inside are in the [0, width] × [0, height]
 *  user-space frame. */
export function Diagram({
  width = 480, height = 220, caption, children,
}: DiagramProps): JSX.Element {
  return (
    <div data-testid="tensor-diagram" style={{
      background: DIAG_THEME.bg,
      border: `1px solid ${DIAG_THEME.border}`,
      borderRadius: 6,
      padding: 10,
    }}>
      {caption && (
        <div data-testid="tensor-diagram-caption" style={{
          color: DIAG_THEME.textMuted,
          fontSize: 11,
          marginBottom: 6,
          fontFamily: "ui-monospace, monospace",
        }}>
          {caption}
        </div>
      )}
      <svg viewBox={`0 0 ${width} ${height}`}
           width="100%" preserveAspectRatio="xMidYMid meet"
           style={{ display: "block" }}>
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="6"
                   refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 Z" fill={DIAG_THEME.arrow} />
          </marker>
          <marker id="arrowhead-residual" markerWidth="6" markerHeight="6"
                   refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 Z" fill={DIAG_THEME.residual} />
          </marker>
        </defs>
        {children}
      </svg>
    </div>
  );
}


export interface TensorProps {
  x: number;
  y: number;
  w?: number;
  h?: number;
  label: string;
  /** Shape annotation rendered below the label, e.g. "(B, S, H)". */
  shape?: string;
  testid?: string;
}

/** Rounded-rect tensor block. */
export function Tensor({
  x, y, w = 90, h = 44, label, shape, testid,
}: TensorProps): JSX.Element {
  return (
    <g data-testid={testid}>
      <rect x={x} y={y} width={w} height={h} rx={6}
             fill={DIAG_THEME.tensorBg}
             stroke={DIAG_THEME.tensorFg}
             strokeWidth={1.5} />
      <text x={x + w / 2} y={y + (shape ? h / 2 - 2 : h / 2 + 4)}
             textAnchor="middle"
             fill={DIAG_THEME.text}
             fontSize={12} fontWeight={600}
             fontFamily="system-ui, sans-serif">
        {label}
      </text>
      {shape && (
        <text x={x + w / 2} y={y + h / 2 + 12}
               textAnchor="middle"
               fill={DIAG_THEME.textMuted}
               fontSize={9}
               fontFamily="ui-monospace, monospace">
          {shape}
        </text>
      )}
    </g>
  );
}


export interface OpProps {
  x: number;
  y: number;
  w?: number;
  h?: number;
  label: string;
  testid?: string;
}

/** Operation node (matmul / softmax / silu / etc.). Hex-shaped. */
export function Op({
  x, y, w = 70, h = 30, label, testid,
}: OpProps): JSX.Element {
  return (
    <g data-testid={testid}>
      <rect x={x} y={y} width={w} height={h} rx={14}
             fill={DIAG_THEME.opBg}
             stroke={DIAG_THEME.opFg}
             strokeWidth={1.2} />
      <text x={x + w / 2} y={y + h / 2 + 4}
             textAnchor="middle"
             fill={DIAG_THEME.text}
             fontSize={11}
             fontFamily="system-ui, sans-serif">
        {label}
      </text>
    </g>
  );
}


export interface ArrowProps {
  x1: number; y1: number; x2: number; y2: number;
  /** Optional label rendered halfway along the arrow. */
  label?: string;
  /** Bend along the y axis at this fraction of x1→x2. */
  bendY?: number;
  testid?: string;
}

/** Straight or right-angle arrow between two anchors. */
export function Arrow({
  x1, y1, x2, y2, label, bendY, testid,
}: ArrowProps): JSX.Element {
  let path: string;
  let midX: number, midY: number;
  if (bendY !== undefined) {
    const mx = x1 + (x2 - x1) * (bendY ?? 0.5);
    path = `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`;
    midX = mx;
    midY = (y1 + y2) / 2;
  } else {
    path = `M ${x1} ${y1} L ${x2} ${y2}`;
    midX = (x1 + x2) / 2;
    midY = (y1 + y2) / 2;
  }
  return (
    <g data-testid={testid}>
      <path d={path} fill="none"
             stroke={DIAG_THEME.arrow} strokeWidth={1.5}
             markerEnd="url(#arrowhead)" />
      {label && (
        <text x={midX} y={midY - 5} textAnchor="middle"
               fill={DIAG_THEME.textMuted}
               fontSize={10}
               fontFamily="ui-monospace, monospace">
          {label}
        </text>
      )}
    </g>
  );
}


/** Dashed yellow residual line (skip connection). */
export function Residual({
  x1, y1, x2, y2, label, bendY = 0.5, testid,
}: ArrowProps): JSX.Element {
  const mx = x1 + (x2 - x1) * bendY;
  const path =
    `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`;
  return (
    <g data-testid={testid}>
      <path d={path} fill="none"
             stroke={DIAG_THEME.residual} strokeWidth={1.5}
             strokeDasharray="4 4"
             markerEnd="url(#arrowhead-residual)" />
      {label && (
        <text x={mx + 4} y={(y1 + y2) / 2}
               fill={DIAG_THEME.residual}
               fontSize={10}
               fontFamily="ui-monospace, monospace">
          {label}
        </text>
      )}
    </g>
  );
}


/* ===========================================================
   MatrixGrid + RoleTensor + dataflow worked-example primitives
   =========================================================== */


export interface MatrixGridProps {
  /** Top-left x in SVG user-space. */
  x: number;
  y: number;
  /** 2-D numeric values (rows × cols). 1-D arrays are treated as
   *  a single row. */
  values: number[][] | number[];
  /** Label rendered above the grid. */
  label?: string;
  /** Bold colour-coded role used for the highlight rail + shape
   *  badge tint. */
  role?: CellRole;
  /** Optional shape suffix rendered top-right (e.g. "[4,4]"). When
   *  omitted, computed from `values`. */
  shape?: string;
  /** Cell side length in user-space units. */
  cell?: number;
  /** Clip range for the heatmap colour ramp. Cells > +clip / < -clip
   *  saturate to the role hue / pink. */
  clip?: number;
  /** Pre-rounded display floats — when omitted, uses values directly. */
  display?: (string | number)[][] | (string | number)[];
  testid?: string;
}


export function MatrixGrid({
  x, y, values, label, role = "raw", shape, cell = 26, clip = 2,
  display, testid,
}: MatrixGridProps): JSX.Element {
  const rows: number[][] = Array.isArray(values[0])
    ? (values as number[][])
    : ([values] as number[][]);
  const dispRows: (string | number)[][] = display
    ? (Array.isArray((display as never[])[0])
        ? (display as (string | number)[][])
        : ([display] as (string | number)[][]))
    : rows.map((r) => r.map((v) => v.toFixed(2)));
  const nRows = rows.length;
  const nCols = rows[0]?.length ?? 0;
  const totalW = nCols * cell;
  const totalH = nRows * cell;
  const shapeText = shape ??
    (nRows === 1 ? `[${nCols}]` : `[${nRows},${nCols}]`);
  const tint = ROLE_TINT[role];

  return (
    <g data-testid={testid ?? (label ? `matrix-${label}` : "matrix-grid")}>
      {label && (
        <text x={x} y={y - 6}
              fill={tint}
              fontSize={11} fontWeight={700}
              fontFamily="system-ui, sans-serif">
          {label}
        </text>
      )}
      <text x={x + totalW} y={y - 6}
            textAnchor="end"
            fill={DIAG_THEME.textMuted}
            fontSize={9}
            fontFamily="ui-monospace, monospace">
        {shapeText}
      </text>
      {/* role highlight rail along the left edge */}
      <rect x={x - 3} y={y} width={3} height={totalH}
             fill={tint} opacity={0.85} rx={1} />
      {rows.map((row, ri) =>
        row.map((v, ci) => {
          const cx = x + ci * cell;
          const cy = y + ri * cell;
          const fill = role === "mask" ? DIAG_THEME.masked
                                       : heatmapColor(v, clip, role);
          return (
            <g key={`${ri}-${ci}`}>
              <rect x={cx} y={cy} width={cell} height={cell}
                     fill={fill}
                     stroke="rgba(255,255,255,0.06)" strokeWidth={0.6} />
              <text x={cx + cell / 2} y={cy + cell / 2 + 3}
                    textAnchor="middle"
                    fill={DIAG_THEME.text}
                    fontSize={Math.max(8, cell * 0.32)}
                    fontFamily="ui-monospace, monospace">
                {String(dispRows[ri]?.[ci] ?? v)}
              </text>
            </g>
          );
        })
      )}
    </g>
  );
}


export interface MathLinkProps {
  /** Topic key — used to render the gloss text inline. */
  topic: string;
  gloss: string;
  url: string;
  testid?: string;
}

/** A reference chip showing a math-foundation link with a short gloss.
 *  Rendered inside <Section label="Math foundations"> as a list. */
export function MathLink({
  topic, gloss, url, testid,
}: MathLinkProps): JSX.Element {
  return (
    <a href={url} target="_blank" rel="noopener noreferrer"
       data-testid={testid ?? `math-link-${topic}`}
       style={{
         display: "block",
         padding: "6px 10px",
         margin: "4px 0",
         background: "rgba(74, 174, 224, 0.08)",
         border: "1px solid rgba(74, 174, 224, 0.25)",
         borderRadius: 6,
         color: DIAG_THEME.text,
         textDecoration: "none",
         fontSize: 12,
         fontFamily: "system-ui, sans-serif",
       }}>
      <strong style={{ color: DIAG_THEME.roleK, marginRight: 6 }}>
        {topic.replace(/_/g, " ")}
      </strong>
      <span style={{ color: DIAG_THEME.textMuted }}>— {gloss}</span>
    </a>
  );
}


/** A logical group/box around a sub-region of the diagram (e.g.
 *  multi-head attention). Renders a faint outlined rectangle with a
 *  small caption in the top-left corner. */
export interface GroupProps {
  x: number; y: number; w: number; h: number;
  label?: string;
  testid?: string;
  children?: ReactNode;
}

export function Group({
  x, y, w, h, label, testid, children,
}: GroupProps): JSX.Element {
  return (
    <g data-testid={testid}>
      <rect x={x} y={y} width={w} height={h} rx={8}
             fill="none"
             stroke="rgba(148, 163, 184, 0.3)"
             strokeWidth={1}
             strokeDasharray="2 4" />
      {label && (
        <text x={x + 6} y={y - 4}
               fill={DIAG_THEME.textMuted}
               fontSize={9}
               fontFamily="ui-monospace, monospace">
          {label}
        </text>
      )}
      {children}
    </g>
  );
}
