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
  bg:        "rgba(15, 23, 42, 0.6)",
  border:    "rgba(255, 255, 255, 0.05)",
  tensorBg:  "#1e293b",
  tensorFg:  "#22d3ee",
  opBg:      "#312e81",
  opFg:      "#818cf8",
  arrow:     "#94a3b8",
  residual:  "#facc15",
  text:      "#f1f5f9",
  textMuted: "#94a3b8",
} as const;


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
