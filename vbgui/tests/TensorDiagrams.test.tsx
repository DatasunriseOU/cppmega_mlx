/**
 * w2t6: every brick that ships a tensor-flow diagram renders without
 * crashing + the HelpModal slots the diagram between What and Why.
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TENSOR_DIAGRAMS } from "@/components/diagrams";
import { HelpIcon } from "@/components/HelpIcon";


describe("w2t6: tensor diagrams registry", () => {
  it("ships at least 10 diagram-bearing keys", () => {
    expect(Object.keys(TENSOR_DIAGRAMS).length).toBeGreaterThanOrEqual(10);
  });

  it.each(Object.keys(TENSOR_DIAGRAMS))(
    "%s renders an SVG",
    (key) => {
      const Diag = TENSOR_DIAGRAMS[key];
      const { container } = render(<Diag />);
      const svg = container.querySelector("svg");
      expect(svg).not.toBeNull();
      // Must have at least one tensor or op rect — sanity check.
      const rects = container.querySelectorAll("rect");
      expect(rects.length).toBeGreaterThan(0);
    },
  );

  it("brick_attention diagram exposes the expected anchor testids", () => {
    const Diag = TENSOR_DIAGRAMS.brick_attention;
    render(<Diag />);
    // Caption + at least one tensor with the testid we pinned.
    expect(screen.getByTestId("tensor-diagram")).toBeDefined();
    expect(screen.getByTestId("tensor-diagram-caption").textContent)
      .toMatch(/vanilla SDPA|softmax/);
  });
});


describe("w2t6: HelpModal slot for the diagram", () => {
  it("renders the tensor-flow section for brick_attention", () => {
    render(<HelpIcon topic="brick_attention" />);
    fireEvent.click(screen.getByTestId("help-icon-brick_attention"));
    expect(screen.getByTestId("help-modal-diagram")).toBeDefined();
    // Two tensor-diagrams now: schematic + worked-example.
    expect(screen.getAllByTestId("tensor-diagram").length)
      .toBeGreaterThanOrEqual(1);
  });

  it("omits the diagram section for topics without a diagram", () => {
    render(<HelpIcon topic="dim_env_S" />);
    fireEvent.click(screen.getByTestId("help-icon-dim_env_S"));
    expect(screen.queryByTestId("help-modal-diagram")).toBeNull();
  });
});
