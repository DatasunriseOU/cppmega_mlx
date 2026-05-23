/**
 * Backward-pass API tab strip — framework-agnostic snippets across
 * MLX (Apple Silicon), PyTorch (CUDA/CPU), JAX (TPU/GPU/CPU). Default
 * tab is MLX; click a tab to swap the snippet. Tests pin the contract
 * so a CUDA deploy still sees PyTorch grad calls, not `mx.grad`.
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BACKWARD_TOPICS } from "@/components/diagrams";
import { HelpIcon } from "@/components/HelpIcon";


describe("backward-pass framework tabs", () => {
  it("every backward entry ships api snippets for the 3 frameworks",
    () => {
      for (const [k, v] of Object.entries(BACKWARD_TOPICS)) {
        expect(v.api).toBeTruthy();
        expect(v.api!.mlx).toBeTruthy();
        expect(v.api!.pytorch).toBeTruthy();
        expect(v.api!.jax).toBeTruthy();
        // The chain_rule must NOT mention any framework name —
        // it must be framework-agnostic; framework-specific calls
        // live in api.* exclusively.
        expect(v.chain_rule).not.toMatch(/mx\.grad/);
        expect(v.chain_rule).not.toMatch(/\.backward\(\)/);
        expect(v.chain_rule).not.toMatch(/jax\.grad/);
        expect(k).toBeTruthy();
      }
    });

  it("attention api references mx.grad / loss.backward / jax.grad",
    () => {
      const a = BACKWARD_TOPICS.brick_attention.api!;
      expect(a.mlx).toMatch(/mx\.grad/);
      expect(a.pytorch).toMatch(/loss\.backward|torch\.autograd/);
      expect(a.jax).toMatch(/jax\.grad/);
    });

  it("attention HelpModal shows tab strip with all three frameworks",
    () => {
      render(<HelpIcon topic="brick_attention" />);
      fireEvent.click(screen.getByTestId("help-icon-brick_attention"));
      expect(screen.getByTestId("help-modal-backward-api")).toBeDefined();
      expect(screen.getByTestId("help-modal-backward-api-tab-mlx"))
        .toBeDefined();
      expect(screen.getByTestId("help-modal-backward-api-tab-pytorch"))
        .toBeDefined();
      expect(screen.getByTestId("help-modal-backward-api-tab-jax"))
        .toBeDefined();
      // Default tab is MLX — its snippet renders.
      expect(screen.getByTestId(
        "help-modal-backward-api-snippet-mlx").textContent)
        .toMatch(/mx\.grad/);
    });

  it("click PyTorch tab swaps snippet to torch backward call", () => {
    render(<HelpIcon topic="brick_attention" />);
    fireEvent.click(screen.getByTestId("help-icon-brick_attention"));
    fireEvent.click(screen.getByTestId(
      "help-modal-backward-api-tab-pytorch"));
    expect(screen.getByTestId(
      "help-modal-backward-api-snippet-pytorch").textContent)
      .toMatch(/loss\.backward|torch\.autograd/);
    // MLX snippet should NOT be visible now.
    expect(screen.queryByTestId(
      "help-modal-backward-api-snippet-mlx")).toBeNull();
  });

  it("click JAX tab swaps snippet to jax.grad call", () => {
    render(<HelpIcon topic="brick_mlp" />);
    fireEvent.click(screen.getByTestId("help-icon-brick_mlp"));
    fireEvent.click(screen.getByTestId(
      "help-modal-backward-api-tab-jax"));
    expect(screen.getByTestId(
      "help-modal-backward-api-snippet-jax").textContent)
      .toMatch(/jax\.grad/);
  });
});
