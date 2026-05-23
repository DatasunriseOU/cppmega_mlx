import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RunHistoryPicker } from "@/components/RunHistoryPicker";

describe("V7-K7 RunHistoryPicker", () => {
  it("renders the latest sentinel + the history list", () => {
    render(<RunHistoryPicker history={["run-1", "run-2"]}
                              selected={null}
                              onSelect={() => {}} />);
    const sel = screen.getByTestId(
      "run-history-select") as HTMLSelectElement;
    const opts = Array.from(sel.options).map((o) => o.value);
    expect(opts).toEqual(["", "run-1", "run-2"]);
    expect(sel.value).toBe("");
  });

  it("count badge updates with history length", () => {
    render(<RunHistoryPicker history={["a", "b", "c"]}
                              selected={null}
                              onSelect={() => {}} />);
    expect(screen.getByTestId("run-history-count").textContent)
      .toContain("3 runs");
  });

  it("count badge uses singular 'run' for length 1", () => {
    render(<RunHistoryPicker history={["a"]}
                              selected={null}
                              onSelect={() => {}} />);
    expect(screen.getByTestId("run-history-count").textContent)
      .toContain("1 run in history");
  });

  it("selecting a real run forwards the id", () => {
    const onS = vi.fn();
    render(<RunHistoryPicker history={["run-1", "run-2"]}
                              selected={null}
                              onSelect={onS} />);
    fireEvent.change(screen.getByTestId("run-history-select"),
                     { target: { value: "run-2" } });
    expect(onS).toHaveBeenCalledWith("run-2");
  });

  it("selecting '(latest)' forwards null", () => {
    const onS = vi.fn();
    render(<RunHistoryPicker history={["run-1"]}
                              selected="run-1"
                              onSelect={onS} />);
    fireEvent.change(screen.getByTestId("run-history-select"),
                     { target: { value: "" } });
    expect(onS).toHaveBeenCalledWith(null);
  });
});
