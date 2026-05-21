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
});
