// V7-I07: useCacheStats hook polls /cache/stats and surfaces the
// latest snapshot to consumers.

import { describe, it, expect, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useCacheStats, formatHitRate } from "@/hooks/useCacheStats";

function _mockFetchOk(stats: Record<string, number>) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(stats),
  }) as unknown as typeof fetch;
}

describe("useCacheStats / V7-I07", () => {
  it("returns null until the first successful fetch resolves", async () => {
    const stats = { size: 4, capacity: 50, hits: 30, misses: 10,
                    evictions: 2, hit_rate: 0.75 };
    const fetchImpl = _mockFetchOk(stats);
    const { result } = renderHook(() => useCacheStats({
      baseUrl: "http://localhost:9999",
      intervalMs: 100,
      fetchImpl,
    }));
    expect(result.current).toBeNull();
    await waitFor(() => {
      expect(result.current).not.toBeNull();
    }, { timeout: 1000 });
    expect(result.current?.hits).toBe(30);
    expect(result.current?.hit_rate).toBeCloseTo(0.75);
  });

  it("targets the configured baseUrl + /cache/stats path", async () => {
    const stats = { size: 0, capacity: 4, hits: 0, misses: 0,
                    evictions: 0, hit_rate: 0.0 };
    const fetchImpl = _mockFetchOk(stats);
    renderHook(() => useCacheStats({
      baseUrl: "http://example.test:1234",
      intervalMs: 100,
      fetchImpl,
    }));
    await waitFor(() => {
      expect(fetchImpl).toHaveBeenCalled();
    });
    const url = (fetchImpl as unknown as ReturnType<typeof vi.fn>)
      .mock.calls[0][0];
    expect(url).toBe("http://example.test:1234/cache/stats");
  });

  it("keeps previous stats when a fetch fails (does not flap to null)",
     async () => {
    const okStats = { size: 1, capacity: 50, hits: 5, misses: 5,
                      evictions: 0, hit_rate: 0.5 };
    let call = 0;
    const fetchImpl = vi.fn().mockImplementation(() => {
      call += 1;
      if (call === 1) {
        return Promise.resolve({
          ok: true, status: 200,
          json: () => Promise.resolve(okStats),
        });
      }
      return Promise.reject(new Error("net down"));
    }) as unknown as typeof fetch;
    const { result } = renderHook(() => useCacheStats({
      baseUrl: "http://localhost:9999",
      intervalMs: 50,
      fetchImpl,
    }));
    await waitFor(() => expect(result.current).not.toBeNull());
    const first = result.current;
    // give the polling loop a chance to attempt the second fetch
    await new Promise((r) => setTimeout(r, 200));
    expect(result.current).toEqual(first);
    expect(call).toBeGreaterThanOrEqual(2);
  });
});

describe("formatHitRate / V7-I07", () => {
  it.each([
    [0.0, "0.0%"],
    [0.5, "50.0%"],
    [0.755, "75.5%"],
    [1.0, "100.0%"],
  ])("formats %p as %p", (input, expected) => {
    expect(formatHitRate(input)).toBe(expected);
  });

  it("returns '—' for NaN", () => {
    expect(formatHitRate(Number.NaN)).toBe("—");
  });
});
