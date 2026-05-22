import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BottomStrip } from "@/components/BottomStrip";
import { INITIAL_SPEC, type SpecState } from "@/state/spec";

const _make = (status: SpecState["backend_status"]): SpecState => ({
  ...INITIAL_SPEC, backend_status: status,
});

describe("BottomStrip / G08", () => {
  it("connected status: no reconnecting indicator", () => {
    render(<BottomStrip state={_make("connected")} onHelpToggle={() => {}} />);
    expect(screen.queryByTestId("bottom-strip-reconnecting")).toBeNull();
  });

  it("reconnecting status: bottom-strip-reconnecting indicator present", () => {
    render(<BottomStrip state={_make("reconnecting")}
                        onHelpToggle={() => {}} />);
    expect(screen.getByTestId("bottom-strip-reconnecting")).toBeTruthy();
  });

  it("disconnected status: no reconnecting indicator (different state)", () => {
    render(<BottomStrip state={_make("disconnected")}
                        onHelpToggle={() => {}} />);
    expect(screen.queryByTestId("bottom-strip-reconnecting")).toBeNull();
  });
});
