import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ScheduleEditor } from "@/components/ScheduleEditor";
import type { ScheduleSpecState } from "@/state/spec";

describe("ScheduleEditor", () => {
  it("renders the kind dropdown with 6 options", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           onChange={() => {}} />);
    const select = screen.getByTestId("schedule-kind-0") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual([
      "constant", "linear_warmup", "cosine", "wsd",
      "inv_sqrt", "polynomial",
    ]);
  });

  it("constant kind hides all conditional fields", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           value={{ kind: "constant" }}
                           onChange={() => {}} />);
    expect(screen.queryByTestId("schedule-warmup-0")).toBeNull();
    expect(screen.queryByTestId("schedule-total-0")).toBeNull();
    expect(screen.queryByTestId("schedule-sparkline")).toBeNull();
  });

  it("selecting cosine reveals total_steps + min_lr_ratio + sparkline", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <ScheduleEditor index={0} baseLr={1e-3} onChange={onChange} />,
    );
    fireEvent.change(screen.getByTestId("schedule-kind-0"),
                     { target: { value: "cosine" } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "cosine" }),
    );

    rerender(<ScheduleEditor index={0} baseLr={1e-3}
                             value={{ kind: "cosine", total_steps: 100 }}
                             onChange={onChange} />);
    expect(screen.getByTestId("schedule-warmup-0")).toBeTruthy();
    expect(screen.getByTestId("schedule-total-0")).toBeTruthy();
    expect(screen.getByTestId("schedule-min-ratio-0")).toBeTruthy();
    expect(screen.getByTestId("schedule-sparkline")).toBeTruthy();
  });

  it("selecting wsd reveals decay_steps", () => {
    const value: ScheduleSpecState = {
      kind: "wsd", warmup_steps: 10, decay_steps: 20, total_steps: 100,
    };
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           value={value} onChange={() => {}} />);
    expect(screen.getByTestId("schedule-decay-0")).toBeTruthy();
  });

  it("selecting polynomial reveals power", () => {
    const value: ScheduleSpecState = {
      kind: "polynomial", total_steps: 100, power: 3.0,
    };
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           value={value} onChange={() => {}} />);
    const input = screen.getByTestId("schedule-power-0") as HTMLInputElement;
    expect(input.value).toBe("3");
  });

  it("switching back to constant clears the schedule (emits undefined)", () => {
    const onChange = vi.fn();
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           value={{ kind: "cosine", total_steps: 100 }}
                           onChange={onChange} />);
    fireEvent.change(screen.getByTestId("schedule-kind-0"),
                     { target: { value: "constant" } });
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("sparkline renders a polyline of expected point count (50)", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                           value={{ kind: "cosine", total_steps: 100,
                                    warmup_steps: 10 }}
                           onChange={() => {}} />);
    const svg = screen.getByTestId("schedule-sparkline");
    const polyline = svg.querySelector("polyline");
    expect(polyline).not.toBeNull();
    const pts = polyline!.getAttribute("points")!.split(" ");
    expect(pts.length).toBe(50);
  });
});
