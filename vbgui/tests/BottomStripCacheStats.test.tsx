// V7-I07: BottomStrip surfaces /cache/stats hit_rate as a live chip.

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BottomStrip } from "@/components/BottomStrip";
import { INITIAL_SPEC, type SpecState } from "@/state/spec";
import type { CacheStats } from "@/hooks/useCacheStats";

const _state = (): SpecState => ({
  ...INITIAL_SPEC, backend_status: "connected",
});

describe("BottomStrip cache-stats chip / V7-I07", () => {
  it("renders '—' placeholder when cacheStats is null", () => {
    render(<BottomStrip state={_state()} cacheStats={null} />);
    const chip = screen.getByTestId("cache-stats-hit-rate");
    expect(chip.textContent).toBe("—");
    expect(chip.getAttribute("data-hit-rate")).toBe("");
  });

  it("renders hit_rate as percentage when cacheStats is present", () => {
    const stats: CacheStats = {
      size: 12, capacity: 50,
      hits: 80, misses: 20, evictions: 3,
      hit_rate: 0.8,
    };
    render(<BottomStrip state={_state()} cacheStats={stats} />);
    const chip = screen.getByTestId("cache-stats-hit-rate");
    expect(chip.textContent).toBe("80.0%");
    expect(chip.getAttribute("data-hit-rate")).toBe("0.8");
    expect(chip.getAttribute("data-hits")).toBe("80");
    expect(chip.getAttribute("data-misses")).toBe("20");
    expect(chip.getAttribute("data-evictions")).toBe("3");
    expect(chip.getAttribute("data-size")).toBe("12");
    expect(chip.getAttribute("data-capacity")).toBe("50");

    const sizeBadge = screen.getByTestId("cache-stats-size");
    expect(sizeBadge.textContent?.trim()).toBe("12/50");
  });

  it("renders 0.0% on a cold cache (zero hits + zero misses)", () => {
    const stats: CacheStats = {
      size: 0, capacity: 50,
      hits: 0, misses: 0, evictions: 0,
      hit_rate: 0.0,
    };
    render(<BottomStrip state={_state()} cacheStats={stats} />);
    expect(screen.getByTestId("cache-stats-hit-rate").textContent)
      .toBe("0.0%");
  });

  it("color-codes high hit-rate green via the dot's background", () => {
    const high: CacheStats = {
      size: 5, capacity: 10, hits: 80, misses: 20,
      evictions: 0, hit_rate: 0.8,
    };
    render(<BottomStrip state={_state()} cacheStats={high} />);
    const dot = screen.getByTestId("cache-stats").querySelector("span");
    expect(dot?.getAttribute("style"))
      .toMatch(/rgb\(16,\s*185,\s*129\)|#10b981/i);
  });

  it("color-codes mid hit-rate amber", () => {
    const mid: CacheStats = {
      size: 5, capacity: 10, hits: 50, misses: 50,
      evictions: 0, hit_rate: 0.5,
    };
    render(<BottomStrip state={_state()} cacheStats={mid} />);
    const dot = screen.getByTestId("cache-stats").querySelector("span");
    expect(dot?.getAttribute("style"))
      .toMatch(/rgb\(217,\s*119,\s*6\)|#d97706/i);
  });

  it("color-codes low hit-rate red", () => {
    const low: CacheStats = {
      size: 5, capacity: 10, hits: 10, misses: 90,
      evictions: 0, hit_rate: 0.1,
    };
    render(<BottomStrip state={_state()} cacheStats={low} />);
    const dot = screen.getByTestId("cache-stats").querySelector("span");
    expect(dot?.getAttribute("style"))
      .toMatch(/rgb\(220,\s*38,\s*38\)|#dc2626/i);
  });
});
