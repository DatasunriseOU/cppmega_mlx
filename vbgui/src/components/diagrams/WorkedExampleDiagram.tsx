/**
 * Renders a WorkedExample (concrete numerical Q/K/V/scores/probs/y
 * tensors) using the MatrixGrid primitive. Tensors flow left-to-right
 * in the order declared by the steps array; arrows carry the step
 * label.
 *
 * Layout strategy: each tensor becomes one MatrixGrid block. We
 * compute a column-major layout: tensors that are operands of step
 * N appear in column N-1 (or earlier); the step's `to` tensor lives
 * in column N. A simple greedy assignment is good enough for the
 * 4-6-tensor examples we ship; more complex flows can override via
 * the optional `positions` map.
 *
 * 3-D tensors (h×seq×d) are flattened to a single 2-D slice for the
 * head 0 only (with a small "(head 0/h)" label) — full head stacks
 * would blow the diagram width.
 */

import {
  Diagram, MatrixGrid, Arrow, DIAG_THEME,
} from "./TensorDiagram";
import type { WorkedExample, WorkedTensor } from "./worked_examples";


interface BlockLayout {
  name: string;
  col: number;
  row: number;
}


function layoutTensors(ex: WorkedExample): Map<string, BlockLayout> {
  // Per-step column assignment. Inputs to step N land in column N-1
  // (clamped at 0); the step's `to` tensor lands in column N.
  const colOf = new Map<string, number>();
  ex.steps.forEach((s, idx) => {
    s.from.forEach((src) => {
      const cur = colOf.get(src);
      if (cur === undefined || cur > idx) colOf.set(src, idx);
    });
    colOf.set(s.to, idx + 1);
  });
  // Tensors that never appear in any step (rare — usually inputs).
  ex.tensors.forEach((t, i) => {
    if (!colOf.has(t.name)) colOf.set(t.name, 0 + i * 0);
  });

  // Per-column row stacking.
  const rowByCol = new Map<number, number>();
  const out = new Map<string, BlockLayout>();
  ex.tensors.forEach((t) => {
    const col = colOf.get(t.name) ?? 0;
    const row = rowByCol.get(col) ?? 0;
    rowByCol.set(col, row + 1);
    out.set(t.name, { name: t.name, col, row });
  });
  return out;
}


function flatten3D(values: WorkedTensor["values"]): number[][] | number[] {
  if (Array.isArray(values) && Array.isArray(values[0]) &&
      Array.isArray((values as number[][][])[0][0])) {
    // 3-D: take head 0.
    return (values as number[][][])[0];
  }
  return values as number[][] | number[];
}


function tensorWidth(t: WorkedTensor, cell: number): number {
  const v = t.values;
  if (typeof v[0] === "number") return v.length * cell;
  if (Array.isArray(v[0]) && typeof (v[0] as number[])[0] === "number") {
    return (v[0] as number[]).length * cell;
  }
  // 3D — head 0.
  const head0 = (v as number[][][])[0];
  return head0[0].length * cell;
}


function tensorHeight(t: WorkedTensor, cell: number): number {
  const v = t.values;
  if (typeof v[0] === "number") return cell;
  if (Array.isArray(v[0]) && typeof (v[0] as number[])[0] === "number") {
    return (v as number[][]).length * cell;
  }
  return (v as number[][][])[0].length * cell;
}


export interface WorkedExampleDiagramProps {
  example: WorkedExample;
  /** Cell side length in user-space units. */
  cell?: number;
  /** Horizontal gap between columns. */
  colGap?: number;
  /** Vertical gap between rows within a column. */
  rowGap?: number;
}


export function WorkedExampleDiagram({
  example, cell = 22, colGap = 80, rowGap = 30,
}: WorkedExampleDiagramProps): JSX.Element {
  const layout = layoutTensors(example);

  // Per-col max width + per-col→row max height so columns size to fit.
  const maxCols = Math.max(0, ...Array.from(layout.values(), (l) => l.col));
  const colXs: number[] = [];
  let cursorX = 12;
  for (let c = 0; c <= maxCols; c++) {
    colXs.push(cursorX);
    let maxW = 60;
    example.tensors.forEach((t) => {
      const lp = layout.get(t.name);
      if (lp && lp.col === c) {
        const w = tensorWidth(t, cell);
        if (w > maxW) maxW = w;
      }
    });
    cursorX += maxW + colGap;
  }
  const totalWidth = cursorX;

  // Per-(col, row) y-coordinate.
  const tensorPositions = new Map<string, {x: number; y: number}>();
  for (let c = 0; c <= maxCols; c++) {
    let y = 30;
    example.tensors.forEach((t) => {
      const lp = layout.get(t.name);
      if (lp && lp.col === c) {
        tensorPositions.set(t.name, { x: colXs[c], y });
        y += tensorHeight(t, cell) + rowGap;
      }
    });
  }
  const maxY = Math.max(
    50,
    ...Array.from(tensorPositions.values(), (p) => p.y + cell * 4),
  );

  return (
    <Diagram width={totalWidth + 12} height={maxY}
             caption={example.caption}>
      {example.tensors.map((t) => {
        const p = tensorPositions.get(t.name);
        if (!p) return null;
        return (
          <MatrixGrid key={t.name}
                       x={p.x} y={p.y}
                       label={t.name}
                       role={t.role}
                       shape={`[${t.shape.join(",")}]`}
                       values={flatten3D(t.values)}
                       cell={cell} />
        );
      })}
      {example.steps.map((s, i) => {
        const src = tensorPositions.get(s.from[s.from.length - 1]);
        const tgt = tensorPositions.get(s.to);
        if (!src || !tgt) return null;
        const srcT = example.tensors.find((t) => t.name === s.from[s.from.length - 1])!;
        const tgtT = example.tensors.find((t) => t.name === s.to)!;
        const x1 = src.x + tensorWidth(srcT, cell) + 2;
        const y1 = src.y + tensorHeight(srcT, cell) / 2;
        const x2 = tgt.x - 4;
        const y2 = tgt.y + tensorHeight(tgtT, cell) / 2;
        return (
          <Arrow key={`step-${i}`}
                  x1={x1} y1={y1} x2={x2} y2={y2}
                  label={s.label} />
        );
      })}
      <text x={totalWidth - 10} y={maxY - 6}
            textAnchor="end"
            fill={DIAG_THEME.textMuted}
            fontSize={9}
            fontFamily="ui-monospace, monospace">
        cell colour = value (pink ← 0 → blue/role)
      </text>
    </Diagram>
  );
}
