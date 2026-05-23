import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrainOptionsPanel } from "@/components/TrainOptionsPanel";

describe("V7-K-block TrainOptionsPanel", () => {
  it("starts collapsed; toggle exposes the body", () => {
    render(<TrainOptionsPanel value={{}} onChange={() => {}} />);
    expect(screen.queryByTestId("train-options-body")).toBeNull();
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    expect(screen.getByTestId("train-options-body")).toBeDefined();
  });

  it("val_every input forwards parsed int (K3)", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-val_every"),
                     { target: { value: "50" } });
    expect(onC).toHaveBeenLastCalledWith({ val_every: 50 });
  });

  it("grad_clip_max_norm float (K4)", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(
      screen.getByTestId("train-opt-grad_clip_max_norm"),
      { target: { value: "0.5" } });
    expect(onC).toHaveBeenLastCalledWith({ grad_clip_max_norm: 0.5 });
  });

  it("loss_scaler init_scale + growth_interval (K5)", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(
      screen.getByTestId("train-opt-loss_scaler_init_scale"),
      { target: { value: "32768" } });
    expect(onC).toHaveBeenLastCalledWith({
      loss_scaler_init_scale: 32768,
    });
  });

  it("fake_ranks slider (K6) surfaces a numeric value", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{ fake_ranks: 1 }} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-fake_ranks"),
                     { target: { value: "8" } });
    expect(onC).toHaveBeenLastCalledWith({ fake_ranks: 8 });
  });

  it("abort_token override (K8)", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{}} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-abort_token"),
                     { target: { value: "my-cancel-handle" } });
    expect(onC).toHaveBeenLastCalledWith({
      abort_token: "my-cancel-handle",
    });
  });

  it("clearing an input forwards undefined (revert to default)", () => {
    const onC = vi.fn();
    render(<TrainOptionsPanel value={{ val_every: 50 }} onChange={onC} />);
    fireEvent.click(screen.getByTestId("train-options-toggle"));
    fireEvent.change(screen.getByTestId("train-opt-val_every"),
                     { target: { value: "" } });
    expect(onC).toHaveBeenLastCalledWith({ val_every: undefined });
  });
});
