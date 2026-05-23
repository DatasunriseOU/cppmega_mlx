import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DimensionsTab } from "@/components/sidebar/DimensionsTab";

const LOG = [
  { brick: "attn_0", param: "num_heads", value: 2,
    source: "auto" as const, reason: "H=128/head_dim=64 → 2" },
  { brick: "attn_0", param: "head_dim",  value: 64,
    source: "user" as const, reason: "provided in BrickSpec.params" },
  { brick: "mlp_0",  param: "intermediate_size", value: 512,
    source: "auto" as const, reason: "4 * H (128) = 512" },
  { brick: "mlp_0",  param: "activation", value: "glu",
    source: "auto" as const, reason: "GLU default" },
];

describe("DimensionsTab", () => {
  it("renders one row per entry", () => {
    render(<DimensionsTab log={LOG} />);
    expect(screen.getByTestId("dim-row-attn_0-num_heads")).toBeTruthy();
    expect(screen.getByTestId("dim-row-attn_0-head_dim")).toBeTruthy();
    expect(screen.getByTestId("dim-row-mlp_0-intermediate_size")).toBeTruthy();
    expect(screen.getByTestId("dim-row-mlp_0-activation")).toBeTruthy();
  });

  it("shows source badges (auto vs user)", () => {
    render(<DimensionsTab log={LOG} />);
    expect(screen.getByTestId("dim-source-attn_0-num_heads").textContent)
      .toBe("auto");
    expect(screen.getByTestId("dim-source-attn_0-head_dim").textContent)
      .toBe("user");
  });

  it("source filter restricts to auto-only rows", () => {
    render(<DimensionsTab log={LOG} />);
    fireEvent.change(screen.getByTestId("dimensions-filter-source"),
                     { target: { value: "auto" } });
    expect(screen.queryByTestId("dim-row-attn_0-head_dim")).toBeNull();
    expect(screen.getByTestId("dim-row-attn_0-num_heads")).toBeTruthy();
  });

  it("brick filter substring-matches name", () => {
    render(<DimensionsTab log={LOG} />);
    fireEvent.change(screen.getByTestId("dimensions-filter-brick"),
                     { target: { value: "mlp" } });
    expect(screen.queryByTestId("dim-row-attn_0-num_heads")).toBeNull();
    expect(screen.getByTestId("dim-row-mlp_0-activation")).toBeTruthy();
  });

  it("row click fires onHighlight with brick id", () => {
    const onHighlight = vi.fn();
    render(<DimensionsTab log={LOG} onHighlight={onHighlight} />);
    fireEvent.click(screen.getByTestId("dim-row-attn_0-num_heads"));
    expect(onHighlight).toHaveBeenCalledWith("attn_0");
  });

  it("empty log shows placeholder", () => {
    render(<DimensionsTab log={[]} />);
    expect(screen.getByText(/No matching entries/)).toBeTruthy();
  });

  // G19: Apply button on auto rows
  it("G19: Apply button visible only on auto-source rows", () => {
    const log = [
      { brick: "attn", param: "head_dim", value: 64,
        source: "auto" as const, reason: "inferred from H" },
      { brick: "mlp", param: "intermediate_size", value: 256,
        source: "user" as const, reason: "user set" },
    ];
    render(<DimensionsTab log={log} onApply={() => {}} />);
    expect(screen.getByTestId("dim-row-attn-head_dim-apply")).toBeTruthy();
    expect(screen.queryByTestId(
      "dim-row-mlp-intermediate_size-apply")).toBeNull();
  });

  it("G19: clicking Apply fires onApply(entry)", () => {
    const onApply = vi.fn();
    const entry = { brick: "attn", param: "head_dim", value: 64,
                    source: "auto" as const, reason: "inferred from H" };
    render(<DimensionsTab log={[entry]} onApply={onApply} />);
    fireEvent.click(screen.getByTestId("dim-row-attn-head_dim-apply"));
    expect(onApply).toHaveBeenCalledWith(entry);
  });

  it("G19: Apply button absent when onApply prop missing", () => {
    render(<DimensionsTab log={[
      { brick: "attn", param: "head_dim", value: 64,
        source: "auto", reason: "x" },
    ]} />);
    expect(screen.queryByTestId(
      "dim-row-attn-head_dim-apply")).toBeNull();
  });

  it("V7-M36: flow trace renders an ordered list of every inference step", () => {
    render(<DimensionsTab log={[
      { brick: "attn", param: "num_heads", value: 4,
        source: "auto", reason: "H/head_dim" },
      { brick: "attn", param: "head_dim", value: 64,
        source: "user", reason: "spec override" },
      { brick: "mlp", param: "hidden_size", value: 128,
        source: "auto", reason: "from dim_env.H" },
    ]} />);
    const list = screen.getByTestId("dimensions-flow-trace-list");
    expect(list).toBeDefined();
    expect(screen.getByTestId("flow-step-0").getAttribute("data-brick"))
      .toBe("attn");
    expect(screen.getByTestId("flow-step-0").getAttribute("data-source"))
      .toBe("auto");
    expect(screen.getByTestId("flow-step-0").textContent)
      .toContain("attn.num_heads");
    expect(screen.getByTestId("flow-step-2").getAttribute("data-brick"))
      .toBe("mlp");
  });

  it("V7-M36: flow trace summary shows the step count", () => {
    render(<DimensionsTab log={[
      { brick: "a", param: "p", value: 1, source: "auto", reason: "r" },
      { brick: "b", param: "p", value: 2, source: "user", reason: "r" },
    ]} />);
    const trace = screen.getByTestId("dimensions-flow-trace");
    expect(trace.querySelector("summary")!.textContent)
      .toContain("2 steps");
  });
});
