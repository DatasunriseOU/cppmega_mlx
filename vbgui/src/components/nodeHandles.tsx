import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";

// Shared 4-side handle set. Every node renders one source + one target handle
// per side (top/right/bottom/left) with stable ids "s-<side>" / "t-<side>".
// The auto-align router (src/lib/elk.ts) picks which side each edge attaches to
// and writes edge.sourceHandle / edge.targetHandle accordingly, so wires can be
// routed/spread across all four sides (e.g. residual-add fan-in arriving from
// top AND bottom) instead of crowding left/right.
//
// Handle ids MUST stay stable and the SET must stay constant (always 8) — that
// lets React Flow keep handle bounds measured without churn; the router only
// changes WHICH existing handle an edge references.

export type HandleSide = "top" | "right" | "bottom" | "left";

export const HANDLE_SIDES: { side: HandleSide; pos: Position }[] = [
  { side: "top", pos: Position.Top },
  { side: "right", pos: Position.Right },
  { side: "bottom", pos: Position.Bottom },
  { side: "left", pos: Position.Left },
];

export const sourceHandleId = (s: HandleSide): string => `s-${s}`;
export const targetHandleId = (s: HandleSide): string => `t-${s}`;

export function FourSideHandles({ style }: { style?: CSSProperties }): JSX.Element {
  return (
    <>
      {HANDLE_SIDES.map(({ side, pos }) => (
        <Handle key={`t-${side}`} id={`t-${side}`} type="target" position={pos} style={style} isConnectable />
      ))}
      {HANDLE_SIDES.map(({ side, pos }) => (
        <Handle key={`s-${side}`} id={`s-${side}`} type="source" position={pos} style={style} isConnectable />
      ))}
    </>
  );
}
