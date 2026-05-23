// V7-H48: BottomStrip renders backend build_id pill.

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BottomStrip } from "@/components/BottomStrip";
import { INITIAL_SPEC } from "@/state/spec";

describe("V7-H48 BottomStrip backend build-id", () => {
  it("renders build-id when supplied", () => {
    render(<BottomStrip state={INITIAL_SPEC}
                         backendBuildId="abc123.1700000000" />);
    expect(screen.getByTestId("backend-build-id").textContent)
      .toBe("build abc123.1700000000");
  });

  it("omits the pill when build_id is null", () => {
    render(<BottomStrip state={INITIAL_SPEC} backendBuildId={null} />);
    expect(screen.queryByTestId("backend-build-id")).toBeNull();
  });
});
