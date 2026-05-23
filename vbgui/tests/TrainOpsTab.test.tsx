// UX#3 — TrainOpsTab consolidates K3-K8 train options + warm-start
// picker into a single sidebar tab. Asserts:
//   - the train-options-panel and run-history-picker are both mounted
//   - editing a train option fires onTrainOptionsChange
//   - changing warm-start fires onWarmStartSelect

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrainOpsTab } from "@/components/sidebar/TrainOpsTab";

describe("UX#3 TrainOpsTab", () => {
  const baseProps = {
    trainOptions: { val_every: 0 },
    history: ["run_a", "run_b"] as readonly string[],
    selectedWarmStart: null as string | null,
    graphEdges: [],
    rpc: null as never,
  };

  it("mounts TrainOptionsPanel + RunHistoryPicker under one tab", () => {
    render(
      <TrainOpsTab
        {...baseProps}
        onTrainOptionsChange={() => {}}
        onWarmStartSelect={() => {}}
      />,
    );
    expect(screen.getByTestId("train-options-panel")).toBeDefined();
    expect(screen.getByTestId("run-history-picker")).toBeDefined();
    expect(screen.getByTestId("train-ops-tab")).toBeDefined();
  });

  it("editing val_every fires onTrainOptionsChange with new payload", () => {
    const onTrainOptionsChange = vi.fn();
    render(
      <TrainOpsTab
        {...baseProps}
        onTrainOptionsChange={onTrainOptionsChange}
        onWarmStartSelect={() => {}}
      />,
    );
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-val_every"),
      { target: { value: "5" } });
    expect(onTrainOptionsChange).toHaveBeenCalled();
    const last = onTrainOptionsChange.mock.calls.at(-1)![0];
    expect(last.val_every).toBe(5);
  });

  it("selecting a warm-start run fires onWarmStartSelect", () => {
    const onWarmStartSelect = vi.fn();
    render(
      <TrainOpsTab
        {...baseProps}
        onTrainOptionsChange={() => {}}
        onWarmStartSelect={onWarmStartSelect}
      />,
    );
    fireEvent.change(screen.getByTestId("run-history-select"),
      { target: { value: "run_b" } });
    expect(onWarmStartSelect).toHaveBeenCalledWith("run_b");
  });
});
