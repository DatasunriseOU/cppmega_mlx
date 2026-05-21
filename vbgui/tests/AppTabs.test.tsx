import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AppTabs } from "@/components/AppTabs";

describe("AppTabs", () => {
  it("renders the three tabs", () => {
    render(<AppTabs active="canvas" onChange={() => {}} />);
    expect(screen.getByTestId("app-tab-canvas")).toBeTruthy();
    expect(screen.getByTestId("app-tab-tokenizer")).toBeTruthy();
    expect(screen.getByTestId("app-tab-data")).toBeTruthy();
  });

  it("marks the active tab via aria-selected", () => {
    render(<AppTabs active="tokenizer" onChange={() => {}} />);
    expect(screen.getByTestId("app-tab-tokenizer")
                 .getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("app-tab-canvas")
                 .getAttribute("aria-selected")).toBe("false");
  });

  it("fires onChange with the clicked tab key", () => {
    const onChange = vi.fn();
    render(<AppTabs active="canvas" onChange={onChange} />);
    fireEvent.click(screen.getByTestId("app-tab-data"));
    expect(onChange).toHaveBeenCalledWith("data");
  });
});
