import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrainExtrasOverlay } from "@/components/TrainExtrasOverlay";
import type { TrainExtras } from "@/components/TrainExtrasOverlay";

describe("V7-Q06.3 TrainExtrasOverlay styled panels", () => {
  it("hides plasticity/mtp/ifim/mhc panels when keys absent", () => {
    const extras: TrainExtras = { losses: [1.0, 0.9] };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.queryByTestId("extras-panel-plasticity")).toBeNull();
    expect(screen.queryByTestId("extras-panel-mtp")).toBeNull();
    expect(screen.queryByTestId("extras-panel-ifim")).toBeNull();
    expect(screen.queryByTestId("extras-panel-mhc")).toBeNull();
  });

  it("renders plasticity panel with FIRE/DASH/ReDo metrics", () => {
    const extras: TrainExtras = {
      plasticity: {
        fire_fired_at_step: 3,
        fire_keys_modified: ["a", "b"],
        dash_last_keys_count: 5,
        redo_last_recycled: 7,
      },
    };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.getByTestId("extras-panel-plasticity")).toBeDefined();
    expect(screen.getByTestId("extras-plasticity-fire-step").textContent)
      .toBe("3");
    expect(screen.getByTestId("extras-plasticity-fire-keys").textContent)
      .toContain("2 keys");
    expect(screen.getByTestId("extras-plasticity-dash-keys-count").textContent)
      .toBe("5");
    expect(screen.getByTestId("extras-plasticity-redo-recycled").textContent)
      .toBe("7");
  });

  it("renders MTP panel with k + beta + head_losses", () => {
    const extras: TrainExtras = {
      mtp: { k: 2, beta: 0.5, head_losses: [0.1234, 0.0567] },
    };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.getByTestId("extras-panel-mtp")).toBeDefined();
    expect(screen.getByTestId("extras-mtp-k").textContent).toBe("2");
    expect(screen.getByTestId("extras-mtp-beta").textContent).toContain("0.5");
    expect(screen.getByTestId("extras-mtp-head-losses").textContent)
      .toContain("0.1234");
    expect(screen.getByTestId("extras-mtp-head-losses").textContent)
      .toContain("0.0567");
  });

  it("renders IFIM panel with instr_loss + token_count + lambda", () => {
    const extras: TrainExtras = {
      ifim: { instr_loss: 1.2345, instr_token_count: 8, lambda: 0.1 },
    };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.getByTestId("extras-panel-ifim")).toBeDefined();
    expect(screen.getByTestId("extras-ifim-instr-loss").textContent)
      .toBe("1.2345");
    expect(screen.getByTestId("extras-ifim-token-count").textContent)
      .toBe("8");
    expect(screen.getByTestId("extras-ifim-lambda").textContent)
      .toBe("0.1000");
  });

  it("renders MHC panel with consistency_loss + heads_correlated", () => {
    const extras: TrainExtras = {
      mhc: { consistency_loss: 0.4567, heads_correlated: 0.89, lambda: 0.05 },
    };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.getByTestId("extras-panel-mhc")).toBeDefined();
    expect(screen.getByTestId("extras-mhc-consistency-loss").textContent)
      .toBe("0.4567");
    expect(screen.getByTestId("extras-mhc-heads-correlated").textContent)
      .toBe("0.8900");
    expect(screen.getByTestId("extras-mhc-lambda").textContent)
      .toBe("0.0500");
  });

  it("hides plasticity panel when all values are nullish", () => {
    const extras: TrainExtras = {
      plasticity: {
        fire_fired_at_step: undefined,
        dash_last_keys_count: undefined,
      },
    };
    render(<TrainExtrasOverlay extras={extras} />);
    expect(screen.queryByTestId("extras-panel-plasticity")).toBeNull();
  });
});
