import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DimEnvEditor } from "@/components/DimEnvEditor";

describe("V7-F56b/F53 DimEnvEditor", () => {
  it("populates inputs from the value prop", () => {
    render(<DimEnvEditor
      value={{ H: 128, nh: 2, head_dim: 64, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    expect((screen.getByTestId("dim-env-H") as HTMLInputElement).value).toBe("128");
    expect((screen.getByTestId("dim-env-nh") as HTMLInputElement).value).toBe("2");
    expect((screen.getByTestId("dim-env-head_dim") as HTMLInputElement).value).toBe("64");
  });

  it("Apply calls onApply with the parsed numeric draft", () => {
    const onApply = vi.fn();
    render(<DimEnvEditor
      value={{ H: 128, nh: 2, head_dim: 64, B: 1, S: 8 }}
      onApply={onApply}
    />);
    fireEvent.change(screen.getByTestId("dim-env-H"),
      { target: { value: "256" } });
    fireEvent.change(screen.getByTestId("dim-env-nh"),
      { target: { value: "4" } });
    fireEvent.click(screen.getByTestId("dim-env-apply"));
    expect(onApply).toHaveBeenCalledTimes(1);
    const next = onApply.mock.calls[0]?.[0];
    expect(next.H).toBe(256);
    expect(next.nh).toBe(4);
    expect(next.head_dim).toBe(64);
  });

  it("renders an inline mismatch indicator when nh*head_dim != H", () => {
    render(<DimEnvEditor
      value={{ H: 128, nh: 3, head_dim: 50, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    const warn = screen.getByTestId("dim-env-inline-mismatch");
    expect(warn).toBeDefined();
    expect(warn.textContent).toContain("150");
    expect(warn.textContent).toContain("128");
  });

  it("hides the mismatch indicator when nh*head_dim == H", () => {
    render(<DimEnvEditor
      value={{ H: 128, nh: 2, head_dim: 64, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    expect(screen.queryByTestId("dim-env-inline-mismatch")).toBeNull();
  });

  it("re-evaluates the mismatch live as the user types (before Apply)", () => {
    render(<DimEnvEditor
      value={{ H: 128, nh: 2, head_dim: 64, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    expect(screen.queryByTestId("dim-env-inline-mismatch")).toBeNull();
    fireEvent.change(screen.getByTestId("dim-env-nh"),
      { target: { value: "3" } });
    // 3 * 64 = 192 ≠ 128 → mismatch surfaces without Apply.
    const warn = screen.getByTestId("dim-env-inline-mismatch");
    expect(warn.textContent).toContain("192");
  });

  it("offers Snap H → fix that applies the consistent value", () => {
    const onApply = vi.fn();
    render(<DimEnvEditor
      value={{ H: 128, nh: 3, head_dim: 50, B: 1, S: 8 }}
      onApply={onApply}
    />);
    const snap = screen.getByTestId("dim-env-fix-set-H");
    expect(snap.textContent).toContain("150");
    fireEvent.click(snap);
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0]?.[0]).toMatchObject({ H: 150 });
  });

  it("offers Snap head_dim → fix only when H is divisible by nh", () => {
    // H=128 / nh=8 = 16 → head_dim snap available.
    render(<DimEnvEditor
      value={{ H: 128, nh: 8, head_dim: 32, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    const snap = screen.getByTestId("dim-env-fix-set-head_dim");
    expect(snap.textContent).toContain("16");
  });

  it("hides Snap head_dim when H is not divisible by nh", () => {
    // H=128 / nh=3 = 42.66 → head_dim snap NOT available; only H snap.
    render(<DimEnvEditor
      value={{ H: 128, nh: 3, head_dim: 50, B: 1, S: 8 }}
      onApply={() => {}}
    />);
    expect(screen.queryByTestId("dim-env-fix-set-head_dim")).toBeNull();
    expect(screen.getByTestId("dim-env-fix-set-H")).toBeDefined();
  });
});
