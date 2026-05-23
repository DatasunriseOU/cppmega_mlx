// UX#2 — DimensionsTab now hosts the DimEnvEditor on top when the
// parent supplies dimEnv + onDimEnvApply. The dim_env editor strip
// used to live above the canvas; this verifies the relocation.

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DimensionsTab } from "@/components/sidebar/DimensionsTab";

const BASE_DIM_ENV = { H: 128, nh: 2, head_dim: 64, B: 1, S: 64 };

describe("UX#2 DimensionsTab hosts DimEnvEditor", () => {
  it("renders the env-editor section when dimEnv + onDimEnvApply provided", () => {
    render(
      <DimensionsTab log={[]}
                     dimEnv={BASE_DIM_ENV}
                     onDimEnvApply={() => {}} />,
    );
    expect(screen.getByTestId("dimensions-tab-env-editor")).toBeDefined();
    expect(screen.getByTestId("dim-env-editor")).toBeDefined();
    expect(screen.getByTestId("dim-env-apply")).toBeDefined();
  });

  it("does NOT render the env-editor when dimEnv is not provided", () => {
    render(<DimensionsTab log={[]} />);
    expect(screen.queryByTestId("dimensions-tab-env-editor")).toBeNull();
    expect(screen.queryByTestId("dim-env-editor")).toBeNull();
  });

  it("clicking Apply on the embedded editor fires onDimEnvApply", () => {
    const onDimEnvApply = vi.fn();
    render(
      <DimensionsTab log={[]}
                     dimEnv={BASE_DIM_ENV}
                     onDimEnvApply={onDimEnvApply} />,
    );
    fireEvent.click(screen.getByTestId("dim-env-apply"));
    expect(onDimEnvApply).toHaveBeenCalled();
    expect(onDimEnvApply.mock.calls[0][0]).toMatchObject({ H: 128, nh: 2 });
  });
});
