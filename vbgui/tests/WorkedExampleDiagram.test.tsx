/**
 * w2t6 follow-up: render concrete numerical worked examples + math
 * foundations links inside HelpModal — Raschka-style.
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  WORKED_EXAMPLES, MATH_FOUNDATIONS, TOPIC_FOUNDATIONS,
  TOPIC_WORKED_EXAMPLES, WorkedExampleDiagram, MatrixGrid,
} from "@/components/diagrams";
import { HelpIcon } from "@/components/HelpIcon";


describe("w2t6 follow-up: worked examples + math foundations", () => {
  it("ships >= 6 worked examples", () => {
    expect(Object.keys(WORKED_EXAMPLES).length).toBeGreaterThanOrEqual(6);
  });

  it("ships >= 12 math-foundation concepts", () => {
    expect(Object.keys(MATH_FOUNDATIONS).length).toBeGreaterThanOrEqual(12);
  });

  it("every math foundation has gloss + URL + key_insight", () => {
    for (const [k, m] of Object.entries(MATH_FOUNDATIONS)) {
      expect(m.gloss).toBeTruthy();
      expect(m.urls.length).toBeGreaterThanOrEqual(1);
      expect(m.urls[0].url).toMatch(/^https?:\/\//);
      expect(m.key_insight).toBeTruthy();
      expect(k).toMatch(/^[a-z_]+$/);
    }
  });

  it("attention worked example has Q/K/V/scores/probs/y tensors", () => {
    const ex = WORKED_EXAMPLES.attention;
    const names = ex.tensors.map((t) => t.name);
    expect(names).toContain("Q");
    expect(names).toContain("K");
    expect(names).toContain("V");
    expect(names).toContain("probs");
    expect(names).toContain("y");
  });

  it("rmsnorm worked example asserts ‖y‖_RMS = 1", () => {
    const ex = WORKED_EXAMPLES.rmsnorm;
    const yCheck = ex.tensors.find((t) => t.name === "‖y‖_RMS")!;
    expect(yCheck.values).toEqual([1.0]);
  });

  it("WorkedExampleDiagram renders cells with values", () => {
    const { container } = render(
      <WorkedExampleDiagram example={WORKED_EXAMPLES.dot_product} />,
    );
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    // Each tensor → MatrixGrid → 1+ <rect> + a <text>. The shipped
    // dot_product example has 4 tensors (a, b, aᵢ·bᵢ, y) with 3+3+3+1
    // = 10 cells + label/shape text. Sanity-bound at ≥ 10 rects.
    const rects = container.querySelectorAll("rect");
    expect(rects.length).toBeGreaterThanOrEqual(10);
    // The dot product value 11 must appear in the output cell.
    expect(svg!.textContent ?? "").toContain("11.00");
  });

  it("MatrixGrid handles 1-D input via single-row promotion", () => {
    const { container } = render(
      <svg viewBox="0 0 200 60" width={200} height={60}>
        <MatrixGrid x={10} y={10} values={[1, 2, 3]}
                     label="v" role="q" />
      </svg>,
    );
    const rects = container.querySelectorAll("rect");
    // 1 role rail + 3 cell rects.
    expect(rects.length).toBeGreaterThanOrEqual(3);
  });

  it("attention topic surfaces both diagram + worked + foundations",
    () => {
      render(<HelpIcon topic="brick_attention" />);
      fireEvent.click(screen.getByTestId("help-icon-brick_attention"));
      expect(screen.getByTestId("help-modal-diagram")).toBeDefined();
      expect(screen.getByTestId("help-modal-worked-example"))
        .toBeDefined();
      expect(screen.getByTestId("help-modal-math-foundations"))
        .toBeDefined();
      // Math-foundation links land with proper testids.
      expect(screen.getByTestId("math-link-attention_mechanism"))
        .toBeDefined();
      expect(screen.getByTestId("math-link-dot_product"))
        .toBeDefined();
    });

  it("topic without a worked example skips that section", () => {
    render(<HelpIcon topic="dim_env_H" />);
    fireEvent.click(screen.getByTestId("help-icon-dim_env_H"));
    expect(screen.queryByTestId("help-modal-worked-example"))
      .toBeNull();
  });

  it("rmsnorm adapter help shows the worked example", () => {
    render(<HelpIcon topic="adapter_rmsnorm" />);
    fireEvent.click(screen.getByTestId("help-icon-adapter_rmsnorm"));
    expect(screen.getByTestId("help-modal-worked-example"))
      .toBeDefined();
    expect(screen.getByTestId("help-modal-math-foundations"))
      .toBeDefined();
    // RMSNorm should link the rms_norm foundation.
    expect(screen.getByTestId("math-link-rms_norm")).toBeDefined();
  });

  it("topic registry coverage: every diagram brick has either an "
    + "explicit foundation list or skips gracefully", () => {
      for (const k of Object.keys(TOPIC_WORKED_EXAMPLES)) {
        const exKey = TOPIC_WORKED_EXAMPLES[k];
        expect(WORKED_EXAMPLES[exKey]).toBeDefined();
      }
      for (const k of Object.keys(TOPIC_FOUNDATIONS)) {
        for (const m of TOPIC_FOUNDATIONS[k]) {
          expect(MATH_FOUNDATIONS[m]).toBeDefined();
        }
      }
    });
});
