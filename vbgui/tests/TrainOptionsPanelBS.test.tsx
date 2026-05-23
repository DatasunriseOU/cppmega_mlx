import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  TrainOptionsPanel,
} from "@/components/TrainOptionsPanel";

describe("V7-Q06.1 TrainOptionsPanel B/S controls", () => {
  it("exposes train-opt-B and train-opt-S inputs when open", () => {
    const onChange = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onChange} />);
    // Body is collapsed by default; expand it.
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    expect(screen.getByTestId("train-opt-B")).toBeDefined();
    expect(screen.getByTestId("train-opt-S")).toBeDefined();
  });

  it("calls onChange with parsed train_B int value", () => {
    const onChange = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onChange} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-B"),
                     { target: { value: "4" } });
    expect(onChange).toHaveBeenCalledWith({ train_B: 4 });
  });

  it("calls onChange with parsed train_S int value", () => {
    const onChange = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onChange} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-S"),
                     { target: { value: "128" } });
    expect(onChange).toHaveBeenCalledWith({ train_S: 128 });
  });

  it("clears the field back to undefined when emptied", () => {
    const onChange = vi.fn();
    render(<TrainOptionsPanel value={{ train_B: 4 }} onChange={onChange} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-B"),
                     { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith({ train_B: undefined });
  });
});
