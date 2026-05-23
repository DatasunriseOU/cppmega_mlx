// V7-H34/H35/H36: LiveTrainPanel last-pill renders per-step
// grad_norms summary, expert_load mini-bar, and mem_mb.

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveTrainPanel, type LiveTrainEvent } from "@/components/LiveTrainPanel";

const EV: LiveTrainEvent = {
  step: 5, loss: 1.2, lr: 1e-3, overflow: false,
  mem_mb: 234.5,
  grad_norms: { "layers.0": 0.7, "layers.1": 0.42, "layers.2": 0.91 },
  expert_load: [0.05, 0.35, 0.6, 0.0],
  ts: 1700000000.0,
};

describe("V7-H34/35/36 LiveTrainPanel per-step pill", () => {
  it("renders mem_mb (H35)", () => {
    render(<LiveTrainPanel events={[EV]} trainInFlight={true} />);
    expect(screen.getByTestId("live-train-panel-last-mem").textContent)
      .toContain("234.5MB");
  });

  it("renders grad-norm summary (H34) with brick count + max", () => {
    render(<LiveTrainPanel events={[EV]} trainInFlight={true} />);
    const el = screen.getByTestId("live-train-panel-last-grad-norms");
    expect(el.textContent).toContain("3b");
    expect(el.textContent).toContain("0.910");
  });

  it("renders expert-load mini-bar (H36) with one cell per expert", () => {
    render(<LiveTrainPanel events={[EV]} trainInFlight={true} />);
    expect(screen.getByTestId("live-train-panel-last-expert-load"))
      .toBeTruthy();
    for (let i = 0; i < 4; i++) {
      expect(screen.getByTestId(`live-train-panel-expert-${i}`))
        .toBeTruthy();
    }
  });

  it("omits expert-load bar when event has empty / null list", () => {
    const ev = { ...EV, expert_load: null };
    render(<LiveTrainPanel events={[ev]} trainInFlight={true} />);
    expect(screen.queryByTestId("live-train-panel-last-expert-load"))
      .toBeNull();
  });

  it("omits grad-norm summary when event has empty dict", () => {
    const ev = { ...EV, grad_norms: {} };
    render(<LiveTrainPanel events={[ev]} trainInFlight={true} />);
    expect(screen.queryByTestId("live-train-panel-last-grad-norms"))
      .toBeNull();
  });
});
