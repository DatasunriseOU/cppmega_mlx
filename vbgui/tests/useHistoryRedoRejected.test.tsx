// V7-H43: useHistory redo surfaces rejected flag for snapshots flagged
// via markRejected, so App.tsx can warn the user that redo lands on a
// known-bad spec.

import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useHistory } from "@/hooks/useHistory";

interface Snap { v: number; }

describe("V7-H43 useHistory rejected-aware redo", () => {
  it("undo+redo round-trip when no snapshots marked rejected", () => {
    const { result } = renderHook(() => useHistory<Snap>());
    act(() => {
      result.current.push({ v: 1 });
      result.current.push({ v: 2 });
    });
    let redoRes: { snapshot: Snap; rejected: boolean } | null = null;
    act(() => {
      result.current.undo();
      redoRes = result.current.redo();
    });
    expect(redoRes).not.toBeNull();
    expect(redoRes!.snapshot.v).toBe(2);
    expect(redoRes!.rejected).toBe(false);
  });

  it("markRejected stamps current snapshot; redo into it flags rejected",
     () => {
    const { result } = renderHook(() => useHistory<Snap>());
    act(() => {
      result.current.push({ v: 1 });
      result.current.push({ v: 2 });
      // Verify failed on v=2 — mark it rejected.
      result.current.markRejected();
      // User undoes back to v=1.
      result.current.undo();
    });
    let r: { snapshot: Snap; rejected: boolean } | null = null;
    act(() => { r = result.current.redo(); });
    expect(r).not.toBeNull();
    expect(r!.snapshot.v).toBe(2);
    expect(r!.rejected).toBe(true);
  });

  it("markRejected on empty history is a no-op (doesn't throw)", () => {
    const { result } = renderHook(() => useHistory<Snap>());
    expect(() => act(() => { result.current.markRejected(); }))
      .not.toThrow();
  });

  it("clear wipes both past + future + rejected flags", () => {
    const { result } = renderHook(() => useHistory<Snap>());
    act(() => {
      result.current.push({ v: 1 });
      result.current.push({ v: 2 });
      result.current.markRejected();
      result.current.undo();
      result.current.clear();
    });
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });
});
