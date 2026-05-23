import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SweepPanel } from "@/components/SweepPanel";

describe("V7-F53 SweepPanel", () => {
  it("renders the run button + no chart before any sweep", () => {
    render(<SweepPanel runner={async () => [1, 2]} />);
    expect(screen.getByTestId("scaling-sweep-run")).toBeDefined();
    expect(screen.queryByTestId("sweep-chart-svg")).toBeNull();
  });

  it("runs all H sizes sequentially and renders one series per H", async () => {
    const calls: number[] = [];
    const runner = vi.fn(async (H: number) => {
      calls.push(H);
      return [3.0 - H * 0.001, 2.5 - H * 0.001];
    });
    render(<SweepPanel runner={runner} hSizes={[64, 128]} />);
    fireEvent.click(screen.getByTestId("scaling-sweep-run"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-chart-svg")).toBeDefined();
    });
    expect(calls).toEqual([64, 128]);
    expect(screen.getByTestId("sweep-chart-line-H64")).toBeDefined();
    expect(screen.getByTestId("sweep-chart-line-H128")).toBeDefined();
    // Two points per H series.
    expect(screen.getByTestId("sweep-chart-point-H64-0")).toBeDefined();
    expect(screen.getByTestId("sweep-chart-point-H64-1")).toBeDefined();
    expect(screen.getByTestId("sweep-chart-point-H128-0")).toBeDefined();
  });

  it("disables Run button while a sweep is in flight", async () => {
    let resolveFirst: ((v: number[]) => void) | null = null;
    const runner = vi.fn((H: number) => new Promise<number[]>((res) => {
      if (H === 64) { resolveFirst = res; }
      else res([1.0]);
    }));
    render(<SweepPanel runner={runner} hSizes={[64, 128]} />);
    fireEvent.click(screen.getByTestId("scaling-sweep-run"));
    await waitFor(() => {
      expect((screen.getByTestId("scaling-sweep-run") as HTMLButtonElement).disabled)
        .toBe(true);
    });
    expect(screen.getByTestId("sweep-progress")).toBeDefined();
    resolveFirst!([1.0]);
    await waitFor(() => {
      expect((screen.getByTestId("scaling-sweep-run") as HTMLButtonElement).disabled)
        .toBe(false);
    });
  });

  it("surfaces a sweep-error when runner rejects, stops the sweep", async () => {
    const runner = vi.fn(async (H: number) => {
      if (H === 128) throw new Error("oom on H=128");
      return [1.0];
    });
    render(<SweepPanel runner={runner} hSizes={[64, 128, 256]} />);
    fireEvent.click(screen.getByTestId("scaling-sweep-run"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-error").textContent)
        .toContain("oom on H=128");
    });
    // H=64 series rendered, H=256 never reached.
    expect(screen.getByTestId("sweep-chart-line-H64")).toBeDefined();
    expect(screen.queryByTestId("sweep-chart-line-H256")).toBeNull();
  });
});
