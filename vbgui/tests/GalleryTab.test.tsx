import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GalleryTab } from "@/components/GalleryTab";
import type { GalleryCache } from "@/hooks/useGalleryCache";

const PRESETS = ["alpha", "beta", "gamma"] as const;

function buildCache(): GalleryCache {
  return {
    alpha: { preset: "alpha", bricks: 4, params_M: 10.5, mem_MB: 800,
             last_loss: 2.1, last_step_ms: 45, run_at: 1_000 },
    beta:  { preset: "beta",  bricks: 8, params_M: 25.0, mem_MB: 1600,
             last_loss: 1.7, last_step_ms: 90, run_at: 2_000 },
    // gamma intentionally missing — sorts to bottom for numeric cols.
  };
}

describe("V7-F58 GalleryTab", () => {
  it("renders one row per preset with cell testids", () => {
    render(<GalleryTab presets={PRESETS} cache={buildCache()} />);
    expect(screen.getByTestId("gallery-row-alpha")).toBeDefined();
    expect(screen.getByTestId("gallery-row-beta")).toBeDefined();
    expect(screen.getByTestId("gallery-row-gamma")).toBeDefined();
    expect(screen.getByTestId("gallery-cell-alpha-params_M").textContent)
      .toBe("10.500");
    expect(screen.getByTestId("gallery-cell-gamma-params_M").textContent)
      .toBe("—");
  });

  it("default sort is preset name ascending", () => {
    render(<GalleryTab presets={PRESETS} cache={buildCache()} />);
    const rows = screen.getAllByTestId(/^gallery-row-/);
    expect(rows.map((r) => r.getAttribute("data-testid"))).toEqual([
      "gallery-row-alpha", "gallery-row-beta", "gallery-row-gamma",
    ]);
    const presetHeader = screen.getByTestId("gallery-sort-preset");
    expect(presetHeader.getAttribute("aria-sort")).toBe("ascending");
  });

  it("clicking preset header toggles to descending", () => {
    render(<GalleryTab presets={PRESETS} cache={buildCache()} />);
    const presetHeader = screen.getByTestId("gallery-sort-preset");
    fireEvent.click(presetHeader);
    expect(presetHeader.getAttribute("aria-sort")).toBe("descending");
    const rows = screen.getAllByTestId(/^gallery-row-/);
    expect(rows.map((r) => r.getAttribute("data-testid"))).toEqual([
      "gallery-row-gamma", "gallery-row-beta", "gallery-row-alpha",
    ]);
  });

  it("sorts numeric column with missing values landing last", () => {
    render(<GalleryTab presets={PRESETS} cache={buildCache()} />);
    const paramsHeader = screen.getByTestId("gallery-sort-params_M");
    fireEvent.click(paramsHeader); // asc
    expect(paramsHeader.getAttribute("aria-sort")).toBe("ascending");
    let rows = screen.getAllByTestId(/^gallery-row-/);
    expect(rows.map((r) => r.getAttribute("data-testid"))).toEqual([
      "gallery-row-alpha", "gallery-row-beta", "gallery-row-gamma",
    ]);
    fireEvent.click(paramsHeader); // desc
    expect(paramsHeader.getAttribute("aria-sort")).toBe("descending");
    rows = screen.getAllByTestId(/^gallery-row-/);
    // beta(25) > alpha(10.5) > gamma(missing).
    expect(rows.map((r) => r.getAttribute("data-testid"))).toEqual([
      "gallery-row-beta", "gallery-row-alpha", "gallery-row-gamma",
    ]);
  });

  it("Refresh fires onRefresh with the row's preset name", () => {
    const onRefresh = vi.fn();
    render(<GalleryTab presets={PRESETS} cache={buildCache()}
                       onRefresh={onRefresh} />);
    fireEvent.click(screen.getByTestId("gallery-refresh-beta"));
    expect(onRefresh).toHaveBeenCalledWith("beta");
  });

  it("Refresh shows ellipsis label while preset is refreshing", () => {
    render(<GalleryTab presets={PRESETS} cache={buildCache()}
                       onRefresh={() => {}}
                       refreshing={new Set(["alpha"])} />);
    expect(screen.getByTestId("gallery-refresh-alpha").textContent).toBe("…");
    expect(screen.getByTestId("gallery-refresh-beta").textContent)
      .toBe("Refresh");
  });
});
