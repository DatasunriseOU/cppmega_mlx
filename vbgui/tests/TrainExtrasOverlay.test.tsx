import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrainExtrasOverlay } from "@/components/TrainExtrasOverlay";

describe("V7-M-block TrainExtrasOverlay", () => {
  it("renders nothing for empty extras", () => {
    render(<TrainExtrasOverlay extras={{}} />);
    const overlay = screen.getByTestId("train-extras-overlay");
    expect(overlay).toBeDefined();
    expect(screen.queryByTestId("extras-loss-chart-svg")).toBeNull();
    expect(screen.queryByTestId("extras-lr-chart-svg")).toBeNull();
  });

  it("M21 + M23: losses + smoothed + val overlay on one chart", () => {
    render(<TrainExtrasOverlay extras={{
      losses: [3.0, 2.5, 2.1],
      losses_smoothed: [3.0, 2.7, 2.4],
      val_losses: [2.9, 2.4],
    }} />);
    expect(screen.getByTestId("extras-loss-chart-line")).toBeDefined();
    expect(screen.getByTestId("extras-loss-chart-line-smoothed"))
      .toBeDefined();
    expect(screen.getByTestId("extras-loss-chart-line-val"))
      .toBeDefined();
  });

  it("M22: LR trajectory renders its own chart", () => {
    render(<TrainExtrasOverlay extras={{
      lr_trajectory: [0.0001, 0.0003, 0.0005],
    }} />);
    expect(screen.getByTestId("extras-lr-chart-svg")).toBeDefined();
    expect(screen.getByTestId("extras-lr-chart-line-lr")).toBeDefined();
  });

  it("M24: perplexity + bpb badges", () => {
    render(<TrainExtrasOverlay extras={{
      perplexity: 12.345, bits_per_byte: 1.234,
    }} />);
    expect(screen.getByTestId("extras-badge-perplexity-value")
      .textContent).toBe("12.345");
    expect(screen.getByTestId("extras-badge-bpb-value")
      .textContent).toBe("1.234");
  });

  it("M25: master + actual dtype badges", () => {
    render(<TrainExtrasOverlay extras={{
      master_dtype: "fp16", dtype_actual: "fp32",
    }} />);
    expect(screen.getByTestId("extras-badge-master_dtype-value")
      .textContent).toBe("fp16");
    expect(screen.getByTestId("extras-badge-dtype_actual-value")
      .textContent).toBe("fp32");
  });

  it("M26: fp8_active badge only when true", () => {
    const { rerender } = render(
      <TrainExtrasOverlay extras={{ fp8_active: false }} />);
    expect(screen.queryByTestId("extras-badge-fp8_active")).toBeNull();
    rerender(<TrainExtrasOverlay extras={{ fp8_active: true }} />);
    expect(screen.getByTestId("extras-badge-fp8_active-value")
      .textContent).toBe("ON");
  });

  it("M27: sharding panel with applied + per-rank bytes", () => {
    render(<TrainExtrasOverlay extras={{
      sharding_applied: true, per_rank_param_bytes: 12345678,
    }} />);
    expect(screen.getByTestId("extras-sharding-applied").textContent)
      .toContain("yes");
    expect(screen.getByTestId("extras-sharding-per-rank").textContent)
      .toContain("12,345,678");
  });

  it("M28: FIM badge with ratio percent", () => {
    render(<TrainExtrasOverlay extras={{
      fim_active: true, fim_ratio: 0.5,
    }} />);
    expect(screen.getByTestId("extras-badge-fim_active-value")
      .textContent).toBe("50.0%");
  });

  it("M29: side-channels observed renders a list of channels", () => {
    render(<TrainExtrasOverlay extras={{
      side_channels_observed: ["doc_ids", "token_ids"],
    }} />);
    expect(screen.getByTestId("extras-side-channel-doc_ids")).toBeDefined();
    expect(screen.getByTestId("extras-side-channel-token_ids")).toBeDefined();
  });

  it("M30: per-brick grad-norm bar for each brick", () => {
    render(<TrainExtrasOverlay extras={{
      per_brick_grad_norms: { "attn_0": 0.5, "mlp_0": 1.2 },
    }} />);
    const a = screen.getByTestId("extras-grad-norm-attn_0");
    const m = screen.getByTestId("extras-grad-norm-mlp_0");
    expect(a.getAttribute("data-grad-norm")).toBe("0.500000");
    expect(m.getAttribute("data-grad-norm")).toBe("1.200000");
  });

  it("M31: brick_kinds pill row", () => {
    render(<TrainExtrasOverlay extras={{
      model_summary: { brick_kinds: ["attention", "mlp", "moe"] },
    }} />);
    expect(screen.getByTestId("extras-brick-kind-attention")).toBeDefined();
    expect(screen.getByTestId("extras-brick-kind-mlp")).toBeDefined();
    expect(screen.getByTestId("extras-brick-kind-moe")).toBeDefined();
  });

  it("M31: brick_kinds accepts comma-string form too", () => {
    render(<TrainExtrasOverlay extras={{
      model_summary: { brick_kinds: "attention, mlp, moe" },
    }} />);
    expect(screen.getByTestId("extras-brick-kind-attention")).toBeDefined();
  });

  it("M32: MoE dashboard with 8 routing keys + per-expert bars", () => {
    render(<TrainExtrasOverlay extras={{
      routing_entropy: 1.23,
      load_balance_loss: 0.0045,
      per_expert_load: [10, 20, 5, 15],
      dropped_token_ratio: 0.07,
      rerouted_token_ratio: 0.01,
      overflow_ratio: 0.02,
      capacity_per_expert: 32,
      capacity_factor: 1.25,
      num_experts: 4,
    }} />);
    expect(screen.getByTestId("extras-moe-routing_entropy-value")
      .textContent).toBe("1.230");
    expect(screen.getByTestId("extras-moe-load_balance_loss-value")
      .textContent).toBe("0.0045");
    expect(screen.getByTestId("extras-moe-dropped_token_ratio-value")
      .textContent).toBe("7.00%");
    expect(screen.getByTestId("extras-moe-num_experts-value")
      .textContent).toBe("4");
    for (const i of [0, 1, 2, 3]) {
      expect(screen.getByTestId(`extras-moe-expert-${i}`)
        .getAttribute("data-load")).not.toBeNull();
    }
  });

  it("M33: grad-clip activity panel", () => {
    render(<TrainExtrasOverlay extras={{
      max_grad_norm_seen: 2.3456, num_clips: 7,
    }} />);
    expect(screen.getByTestId("extras-grad-clip-max").textContent)
      .toContain("2.3456");
    expect(screen.getByTestId("extras-grad-clip-count").textContent)
      .toContain("7");
  });

  it("M34: optimizer_kind badge", () => {
    render(<TrainExtrasOverlay extras={{
      optimizer_kind: "muon",
    }} />);
    expect(screen.getByTestId("extras-badge-optimizer_kind-value")
      .textContent).toBe("muon");
  });

  it("M35: gradient_reduce_ms badge", () => {
    render(<TrainExtrasOverlay extras={{
      gradient_reduce_ms: 12.7,
    }} />);
    expect(screen.getByTestId("extras-badge-gradient_reduce_ms-value")
      .textContent).toBe("12.7");
  });
});
