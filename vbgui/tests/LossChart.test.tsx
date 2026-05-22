import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { LossChart } from "@/components/LossChart";

describe("V7-F-foundation LossChart", () => {
  it("renders single-series SVG with one path and N circles", () => {
    render(<LossChart losses={[2.5, 2.1, 1.8]} />);
    const svg = screen.getByTestId("chart-svg");
    expect(svg.tagName.toLowerCase()).toBe("svg");
    const line = screen.getByTestId("chart-line");
    const d = line.getAttribute("d") ?? "";
    expect(d.startsWith("M")).toBe(true);
    expect(d.split("L").length).toBe(3); // M + 2 L commands -> 3 points
    expect(screen.getByTestId("chart-point-0")).toBeDefined();
    expect(screen.getByTestId("chart-point-1")).toBeDefined();
    expect(screen.getByTestId("chart-point-2")).toBeDefined();
  });

  it("encodes loss values as data-loss-value attributes (independent of DOM order)", () => {
    render(<LossChart losses={[3.0, 1.0]} />);
    const p0 = screen.getByTestId("chart-point-0");
    const p1 = screen.getByTestId("chart-point-1");
    expect(p0.getAttribute("data-loss-value")).toBe("3.000000");
    expect(p1.getAttribute("data-loss-value")).toBe("1.000000");
  });

  it("renders y-axis min/max labels reflecting the data range", () => {
    render(<LossChart losses={[2.5, 1.0, 4.0]} />);
    expect(screen.getByTestId("chart-axis-y-label-min").textContent)
      .toBe("1.000");
    expect(screen.getByTestId("chart-axis-y-label-max").textContent)
      .toBe("4.000");
  });

  it("renders multi-series with legend (F53 scaling sweep case)", () => {
    render(
      <LossChart
        losses={[2.5, 2.0]}
        series={[
          { label: "H64", values: [3.0, 2.7] },
          { label: "H128", values: [2.4, 1.9] },
        ]}
      />,
    );
    expect(screen.getByTestId("chart-line")).toBeDefined();
    expect(screen.getByTestId("chart-line-H64")).toBeDefined();
    expect(screen.getByTestId("chart-line-H128")).toBeDefined();
    expect(screen.getByTestId("chart-legend")).toBeDefined();
    expect(screen.getByTestId("chart-legend-loss")).toBeDefined();
    expect(screen.getByTestId("chart-legend-H64")).toBeDefined();
    expect(screen.getByTestId("chart-legend-H128")).toBeDefined();
  });

  it("handles single-point series without NaN coords", () => {
    render(<LossChart losses={[1.5]} />);
    const line = screen.getByTestId("chart-line");
    const d = line.getAttribute("d") ?? "";
    expect(d).toMatch(/^M\d+(\.\d+)?,\d+(\.\d+)?$/);
    expect(d.includes("NaN")).toBe(false);
  });

  it("returns no <path d> when losses are empty (and no series given)", () => {
    render(<LossChart losses={[]} testidPrefix="empty" />);
    // axes still rendered but no series group emitted.
    expect(screen.queryByTestId("empty-line")).toBeNull();
    expect(screen.getByTestId("empty-svg")).toBeDefined();
  });
});
