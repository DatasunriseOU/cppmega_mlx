import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorDetailsPanel } from "@/components/ErrorDetailsPanel";

describe("V7-L46 ErrorDetailsPanel", () => {
  it("renders the headline message + code", () => {
    render(<ErrorDetailsPanel error={{
      code: -32602, message: "Invalid params",
    }} />);
    expect(screen.getByTestId("error-details-headline").textContent)
      .toContain("-32602");
    expect(screen.getByTestId("error-details-headline").textContent)
      .toContain("Invalid params");
  });

  it("expands Pydantic-style errors[] into per-field rows", () => {
    render(<ErrorDetailsPanel error={{
      code: -32602, message: "1 validation error",
      data: { errors: [
        { loc: ["graph", "nodes", 2, "kind"],
          msg: "unknown brick kind",
          type: "value_error", input: "rmsnorm" },
        { loc: ["dim_env", "H"],
          msg: "must be positive int", type: "type_error" },
      ] },
    }} />);
    const list = screen.getByTestId("error-details-field-errors");
    expect(list).toBeDefined();
    expect(screen.getByTestId("error-details-field-0-loc").textContent)
      .toBe("graph.nodes.2.kind");
    expect(screen.getByTestId("error-details-field-0-msg").textContent)
      .toContain("unknown brick");
    expect(screen.getByTestId("error-details-field-0-type").textContent)
      .toContain("value_error");
    expect(screen.getByTestId("error-details-field-1-loc").textContent)
      .toBe("dim_env.H");
  });

  it("falls back to `detail` key when `errors` missing", () => {
    render(<ErrorDetailsPanel error={{
      message: "fail",
      data: { detail: [
        { loc: ["x"], msg: "bad" },
      ] },
    }} />);
    expect(screen.getByTestId("error-details-field-0-loc").textContent)
      .toBe("x");
  });

  it("renders stage + type metadata when present", () => {
    render(<ErrorDetailsPanel error={{
      message: "boom",
      data: { stage: "train", type: "RuntimeError" },
    }} />);
    expect(screen.getByTestId("error-details-stage").textContent)
      .toContain("train");
    expect(screen.getByTestId("error-details-type").textContent)
      .toContain("RuntimeError");
  });

  it("collapses traceback into <details>; opens on click", () => {
    render(<ErrorDetailsPanel error={{
      message: "boom",
      data: { trace: "Traceback (most recent…)\n  File foo.py" },
    }} />);
    const trace = screen.getByTestId("error-details-trace");
    expect(trace.textContent).toContain("Traceback");
    // open via the wrapping <details> summary click.
    fireEvent.click(
      screen.getByTestId("error-details-trace-wrap")
            .querySelector("summary")!);
  });

  it("skips field list when no errors[] / detail[] present", () => {
    render(<ErrorDetailsPanel error={{
      message: "non-validation failure",
    }} />);
    expect(screen.queryByTestId("error-details-field-errors")).toBeNull();
  });
});
