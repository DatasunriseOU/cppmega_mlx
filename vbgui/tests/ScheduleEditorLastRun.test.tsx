// V7-H45: ScheduleEditor renders lastRunScheduleKind echo pill.

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScheduleEditor } from "@/components/ScheduleEditor";

describe("V7-H45 ScheduleEditor last-run echo", () => {
  it("omits the pill when lastRunScheduleKind is null", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                            onChange={vi.fn()} />);
    expect(screen.queryByTestId("schedule-last-run-0")).toBeNull();
  });

  it("renders matched (green) echo when selected matches last run", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                            value={{ kind: "cosine",
                                     warmup_steps: 100,
                                     total_steps: 1000 }}
                            onChange={vi.fn()}
                            lastRunScheduleKind="cosine" />);
    const pill = screen.getByTestId("schedule-last-run-0");
    expect(pill.textContent).toContain("last run: cosine");
    expect(pill.textContent).not.toContain("≠");
  });

  it("renders mismatch (amber) when last run differs from selected", () => {
    render(<ScheduleEditor index={0} baseLr={1e-3}
                            value={{ kind: "cosine" }}
                            onChange={vi.fn()}
                            lastRunScheduleKind="constant" />);
    const pill = screen.getByTestId("schedule-last-run-0");
    expect(pill.textContent).toContain("last run: constant");
    expect(pill.textContent).toContain("≠ selected");
  });

  it("auto-clamps warmup_steps when total_steps is changed to be smaller", () => {
    const onChangeMock = vi.fn();
    render(
      <ScheduleEditor
        index={0}
        baseLr={1e-3}
        value={{ kind: "cosine", warmup_steps: 100, total_steps: 1000 }}
        onChange={onChangeMock}
      />
    );
    const totalInput = screen.getByTestId("schedule-total-0");
    const fireEvent = require("@testing-library/react").fireEvent;
    fireEvent.change(totalInput, { target: { value: "50" } });
    expect(onChangeMock).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "cosine",
        warmup_steps: 50,
        total_steps: 50,
      })
    );
  });
});
