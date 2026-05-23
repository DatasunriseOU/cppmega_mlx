import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HelpIcon, HELP_TOPICS } from "@/components/HelpIcon";

describe("V7-Q11 brick_<kind> alias for adapter kinds", () => {
  it("brick_rmsnorm resolves to the same entry as adapter_rmsnorm", () => {
    expect(HELP_TOPICS["brick_rmsnorm"]).toBeDefined();
    expect(HELP_TOPICS["brick_rmsnorm"]).toBe(HELP_TOPICS["adapter_rmsnorm"]);
  });

  it("brick_residual / brick_merge_heads / etc. all alias adapters", () => {
    for (const k of ["residual", "merge_heads", "split_heads",
                     "transpose_bnsd", "linear_bridge"]) {
      expect(HELP_TOPICS[`brick_${k}`], `missing brick_${k}`).toBeDefined();
      expect(HELP_TOPICS[`brick_${k}`])
        .toBe(HELP_TOPICS[`adapter_${k}`]);
    }
  });

  it("clicking brick_rmsnorm ? opens a modal with the RMSNorm content", () => {
    render(<HelpIcon topic="brick_rmsnorm" />);
    fireEvent.click(screen.getByTestId("help-icon-brick_rmsnorm"));
    expect(screen.getByTestId("help-modal-brick_rmsnorm")).toBeDefined();
    // The modal title should reference RMSNorm (the adapter entry's
    // title) rather than the missing-topic fallback.
    expect(screen.getByTestId("help-modal-title").textContent)
      .toMatch(/RMSNorm/i);
    expect(screen.queryByTestId("help-modal-missing")).toBeNull();
  });
});
