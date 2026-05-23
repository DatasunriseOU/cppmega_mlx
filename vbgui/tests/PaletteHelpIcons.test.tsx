import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Palette } from "@/components/Palette";
import { BRICKS, ADAPTERS } from "@/lib/bricks";
import { HELP_TOPICS } from "@/components/HelpIcon";

describe("V7-Q09 Palette help icons", () => {
  it("renders a ? icon next to every brick tile", () => {
    render(<Palette />);
    for (const b of BRICKS) {
      const icon = screen.queryByTestId(`help-icon-brick_${b.kind}`);
      expect(icon, `missing help-icon for brick ${b.kind}`)
        .not.toBeNull();
    }
  });

  it("renders a ? icon next to every adapter tile", () => {
    render(<Palette />);
    for (const a of ADAPTERS) {
      const icon = screen.queryByTestId(`help-icon-adapter_${a.kind}`);
      expect(icon, `missing help-icon for adapter ${a.kind}`)
        .not.toBeNull();
    }
  });

  it("HELP_TOPICS carries an entry for every brick + adapter", () => {
    for (const b of BRICKS) {
      const key = `brick_${b.kind}`;
      expect(HELP_TOPICS[key], `HELP_TOPICS missing ${key}`).toBeDefined();
      const t = HELP_TOPICS[key]!;
      expect(t.title.length).toBeGreaterThan(0);
      expect(t.what.length).toBeGreaterThan(0);
      expect(t.why.length).toBeGreaterThan(0);
      // V7-Q09 schema: brick topics must surface inputs/outputs/norm
      // so the operator knows what to connect + where to drop a norm.
      expect(t.inputs, `${key} missing inputs`).toBeDefined();
      expect(t.outputs, `${key} missing outputs`).toBeDefined();
      expect(t.normalization, `${key} missing normalization`).toBeDefined();
    }
    for (const a of ADAPTERS) {
      const key = `adapter_${a.kind}`;
      expect(HELP_TOPICS[key], `HELP_TOPICS missing ${key}`).toBeDefined();
    }
  });

  it("clicking a brick ? opens the help modal with the brick's content", () => {
    render(<Palette />);
    const icon = screen.getByTestId("help-icon-brick_attention");
    fireEvent.click(icon);
    const modal = screen.getByTestId("help-modal-brick_attention");
    expect(modal).toBeDefined();
    expect(screen.getByTestId("help-modal-inputs").textContent)
      .toContain("(B, S, H)");
    expect(screen.getByTestId("help-modal-outputs").textContent)
      .toContain("(B, S, H)");
    expect(screen.getByTestId("help-modal-normalization").textContent)
      .toContain("RMSNorm");
  });

  it("clicking an adapter ? opens its help modal", () => {
    render(<Palette />);
    const icon = screen.getByTestId("help-icon-adapter_rmsnorm");
    fireEvent.click(icon);
    const modal = screen.getByTestId("help-modal-adapter_rmsnorm");
    expect(modal).toBeDefined();
  });
});
