import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useVerifyAfter } from "@/hooks/useVerifyAfter";

beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => { vi.useRealTimers(); });

describe("useVerifyAfter", () => {
  it("invokes runner once after debounce window", async () => {
    const runner = vi.fn(async () => {});
    const { result } = renderHook(() =>
      useVerifyAfter({ tick: 0 }, runner, { debounceMs: 100 }));
    act(() => result.current.schedule());
    expect(runner).not.toHaveBeenCalled();
    await act(async () => { vi.advanceTimersByTime(150); });
    expect(runner).toHaveBeenCalledTimes(1);
  });

  it("collapses multiple rapid schedules into one call", async () => {
    const runner = vi.fn(async () => {});
    const { result } = renderHook(() =>
      useVerifyAfter({ x: 0 }, runner, { debounceMs: 100 }));
    act(() => {
      result.current.schedule();
      result.current.schedule();
      result.current.schedule();
    });
    await act(async () => { vi.advanceTimersByTime(120); });
    expect(runner).toHaveBeenCalledTimes(1);
  });

  it("cancel prevents a pending invocation", async () => {
    const runner = vi.fn(async () => {});
    const { result } = renderHook(() =>
      useVerifyAfter({ x: 0 }, runner, { debounceMs: 100 }));
    act(() => result.current.schedule());
    act(() => result.current.cancel());
    await act(async () => { vi.advanceTimersByTime(200); });
    expect(runner).not.toHaveBeenCalled();
  });

  it("does not schedule when enabled=false", async () => {
    const runner = vi.fn(async () => {});
    const { result } = renderHook(() =>
      useVerifyAfter({ x: 0 }, runner, { debounceMs: 50, enabled: false }));
    act(() => result.current.schedule());
    await act(async () => { vi.advanceTimersByTime(200); });
    expect(runner).not.toHaveBeenCalled();
  });

  it("passes the latest payload into the runner", async () => {
    const runner = vi.fn(async () => {});
    let payload = { tick: 1 };
    const { result, rerender } = renderHook(
      ({ p }: { p: typeof payload }) =>
        useVerifyAfter(p, runner, { debounceMs: 50 }),
      { initialProps: { p: payload } },
    );
    act(() => result.current.schedule());
    payload = { tick: 99 };
    rerender({ p: payload });
    await act(async () => { vi.advanceTimersByTime(80); });
    expect(runner).toHaveBeenCalledWith({ tick: 99 });
  });
});
