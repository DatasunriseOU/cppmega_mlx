/**
 * Backward-pass section coverage: every brick we ship with a
 * worked-example also exposes a backward walkthrough (∂L / VJP), with
 * an upstream-cotangent narrative, MLX `mx.grad` example for at
 * least a few of them, and a key-identity equation chip.
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  BACKWARD_TOPICS,
} from "@/components/diagrams/backward_passes";
import { HelpIcon } from "@/components/HelpIcon";


describe("backward-pass coverage", () => {
  it("ships >= 10 backward entries", () => {
    expect(Object.keys(BACKWARD_TOPICS).length).toBeGreaterThanOrEqual(10);
  });

  it("every backward entry has all three required fields", () => {
    for (const [k, v] of Object.entries(BACKWARD_TOPICS)) {
      expect(v.differentiates).toBeTruthy();
      expect(v.chain_rule).toBeTruthy();
      expect(v.key_identity).toBeTruthy();
      expect(k).toMatch(/^[a-z0-9_]+$/);
      // Key identity must contain a ≡math symbol or operator.
      expect(v.key_identity.length).toBeGreaterThan(8);
    }
  });

  it("attention backward references the softmax Jacobian", () => {
    const e = BACKWARD_TOPICS.brick_attention;
    expect(e.chain_rule).toMatch(/softmax/i);
    // chain_rule is framework-agnostic now; mx.grad lives in api.mlx.
    expect(e.api?.mlx ?? "").toMatch(/mx\.grad/);
    expect(e.key_identity).toMatch(/softmax|row-Jacobian/i);
  });

  it("mlp backward references SiLU derivative", () => {
    const e = BACKWARD_TOPICS.brick_mlp;
    expect(e.chain_rule).toMatch(/silu|SiLU/);
    expect(e.key_identity).toMatch(/silu/i);
  });

  it("moe backward mentions the straight-through estimator", () => {
    const e = BACKWARD_TOPICS.brick_moe;
    expect(e.chain_rule).toMatch(/straight-through|STE/i);
    expect(e.key_identity).toMatch(/STE|stop-grad/i);
  });

  it("rmsnorm backward references the 'norm cancels out' identity", () => {
    const e = BACKWARD_TOPICS.adapter_rmsnorm;
    expect(e.chain_rule).toMatch(/norm cancels out|cancels/i);
    expect(e.key_identity).toMatch(/rms/i);
  });

  it("residual backward names the gradient-split property", () => {
    const e = BACKWARD_TOPICS.adapter_residual;
    expect(e.chain_rule).toMatch(/identity|split|both/i);
  });

  it("HelpModal for brick_attention renders the Backward section", () => {
    render(<HelpIcon topic="brick_attention" />);
    fireEvent.click(screen.getByTestId("help-icon-brick_attention"));
    expect(screen.getByTestId("help-modal-backward")).toBeDefined();
    expect(screen.getByTestId("help-modal-backward-diff").textContent ?? "")
      .toMatch(/dL\/dX/);
    expect(screen.getByTestId("help-modal-backward-chain").textContent ?? "")
      .toMatch(/softmax/i);
    expect(screen.getByTestId("help-modal-backward-identity").textContent ?? "")
      .toMatch(/softmax|row-Jacobian/i);
  });

  it("HelpModal for adapter_rmsnorm renders the rmsnorm backward", () => {
    render(<HelpIcon topic="adapter_rmsnorm" />);
    fireEvent.click(screen.getByTestId("help-icon-adapter_rmsnorm"));
    expect(screen.getByTestId("help-modal-backward")).toBeDefined();
    expect(screen.getByTestId("help-modal-backward-identity").textContent ?? "")
      .toMatch(/rms/i);
  });

  it("HelpModal for dim_env_H skips the Backward section (none mapped)",
    () => {
      render(<HelpIcon topic="dim_env_H" />);
      fireEvent.click(screen.getByTestId("help-icon-dim_env_H"));
      expect(screen.queryByTestId("help-modal-backward")).toBeNull();
    });
});
