import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Palette } from "@/components/Palette";

describe("V7-Q12 HelpModal renders via createPortal(document.body)", () => {
  it("opening a brick ? mounts the modal under document.body, "
     + "NOT inside the Palette aside", () => {
    const { container } = render(<Palette />);

    fireEvent.click(screen.getByTestId("help-icon-brick_attention"));

    const modal = screen.getByTestId("help-modal-brick_attention");
    const palette = screen.getByTestId("palette");

    // The portal target is document.body — the modal MUST NOT be
    // inside the palette (or anywhere in the render container).
    expect(palette.contains(modal)).toBe(false);
    expect(container.contains(modal)).toBe(false);

    // It must be somewhere under document.body though.
    expect(document.body.contains(modal)).toBe(true);
  });

  it("opening an adapter ? also portals out", () => {
    const { container } = render(<Palette />);
    fireEvent.click(screen.getByTestId("help-icon-adapter_rmsnorm"));
    const modal = screen.getByTestId("help-modal-adapter_rmsnorm");
    expect(container.contains(modal)).toBe(false);
    expect(document.body.contains(modal)).toBe(true);
  });
});
