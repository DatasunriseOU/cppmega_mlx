import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HelpIcon, HELP_TOPICS } from "@/components/HelpIcon";

describe("HelpIcon + HelpModal", () => {
  it("renders the ? button with the topic-derived testid", () => {
    render(<HelpIcon topic="dim_env_H" />);
    expect(screen.getByTestId("help-icon-dim_env_H")).toBeDefined();
  });

  it("clicking the icon opens the modal with topic title + sections", () => {
    render(<HelpIcon topic="dim_env_H" />);
    fireEvent.click(screen.getByTestId("help-icon-dim_env_H"));
    expect(screen.getByTestId("help-modal-dim_env_H")).toBeDefined();
    const title = screen.getByTestId("help-modal-title");
    expect(title.textContent).toContain("H");
    expect(screen.getByTestId("help-modal-what")).toBeDefined();
    expect(screen.getByTestId("help-modal-why")).toBeDefined();
  });

  it("renders 'missing' panel for an unknown topic", () => {
    render(<HelpIcon topic="this_topic_does_not_exist" />);
    fireEvent.click(
      screen.getByTestId("help-icon-this_topic_does_not_exist"));
    expect(screen.getByTestId("help-modal-missing")).toBeDefined();
  });

  it("symbolic_dim_mismatch topic explains the decoupled-Q convention", () => {
    expect(HELP_TOPICS.symbolic_dim_mismatch).toBeDefined();
    render(<HelpIcon topic="symbolic_dim_mismatch" />);
    fireEvent.click(screen.getByTestId("help-icon-symbolic_dim_mismatch"));
    const why = screen.getByTestId("help-modal-why");
    expect(why.textContent?.toLowerCase()).toContain("decouple");
  });

  it("close button hides the modal", () => {
    render(<HelpIcon topic="dim_env_nh" />);
    fireEvent.click(screen.getByTestId("help-icon-dim_env_nh"));
    expect(screen.getByTestId("help-modal-dim_env_nh")).toBeDefined();
    fireEvent.click(screen.getByTestId("help-modal-close"));
    expect(screen.queryByTestId("help-modal-dim_env_nh")).toBeNull();
  });

  // UX-fix: prior implementation set backdropFilter:blur(8px) on the
  // fixed inset:0 overlay. Chrome recomposed the GPU layer on every
  // mousemove that crossed the overlay → visible flicker over the
  // React Flow canvas underneath. Fix drops the blur and promotes
  // the backdrop to its own compositing layer via translateZ(0).
  it("backdrop has no backdropFilter:blur (anti-flicker fix)", () => {
    render(<HelpIcon topic="dim_env_H" />);
    fireEvent.click(screen.getByTestId("help-icon-dim_env_H"));
    const backdrop = screen.getByTestId("help-modal-backdrop") as
      HTMLElement;
    expect(backdrop.style.backdropFilter || "").toBe("");
    expect(backdrop.style.transform).toContain("translateZ");
  });
});
