import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useHistory } from "@/hooks/useHistory";

describe("V7-H03 useHistory", () => {
  it("starts with no undo / no redo", () => {
    const { result } = renderHook(() => useHistory<number>());
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
    expect(result.current.size).toBe(0);
  });

  it("push grows the size and enables undo after >=2 entries", () => {
    const { result } = renderHook(() => useHistory<number>());
    act(() => { result.current.push(1); });
    expect(result.current.canUndo).toBe(false);
    act(() => { result.current.push(2); });
    expect(result.current.canUndo).toBe(true);
    expect(result.current.size).toBe(2);
  });

  it("undo returns the previous snapshot and enables redo", () => {
    const { result } = renderHook(() => useHistory<number>());
    act(() => { result.current.push(10); });
    act(() => { result.current.push(20); });
    act(() => { result.current.push(30); });
    let prev: number | null = null;
    act(() => { prev = result.current.undo(); });
    expect(prev).toBe(20);
    expect(result.current.canRedo).toBe(true);
  });

  it("redo replays the snapshot popped by undo", () => {
    const { result } = renderHook(() => useHistory<number>());
    act(() => { result.current.push(1); });
    act(() => { result.current.push(2); });
    act(() => { result.current.push(3); });
    act(() => { result.current.undo(); });    // pop 3, prev=2
    let nxt: number | null = null;
    act(() => { nxt = result.current.redo(); });
    expect(nxt).toBe(3);
  });

  it("new push after undo clears redo stack (linear semantics)", () => {
    const { result } = renderHook(() => useHistory<number>());
    act(() => { result.current.push(1); });
    act(() => { result.current.push(2); });
    act(() => { result.current.undo(); });
    expect(result.current.canRedo).toBe(true);
    act(() => { result.current.push(99); });
    expect(result.current.canRedo).toBe(false);
  });

  it("capacity bounds the past stack", () => {
    const { result } = renderHook(() => useHistory<number>(3));
    act(() => { result.current.push(1); });
    act(() => { result.current.push(2); });
    act(() => { result.current.push(3); });
    act(() => { result.current.push(4); });
    expect(result.current.size).toBe(3);  // 1 was dropped
  });

  it("clear resets both stacks", () => {
    const { result } = renderHook(() => useHistory<number>());
    act(() => { result.current.push(1); });
    act(() => { result.current.push(2); });
    act(() => { result.current.clear(); });
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
    expect(result.current.size).toBe(0);
  });
});
