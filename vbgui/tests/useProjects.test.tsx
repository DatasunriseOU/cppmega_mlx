import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useProjects } from "@/hooks/useProjects";

describe("V7-H02 useProjects", () => {
  beforeEach(() => {
    const storage =
      typeof window === "undefined" ? undefined : window.localStorage;
    try { storage?.clear(); } catch { /* ignore */ }
    try {
      storage?.removeItem("vbgui_projects_v1");
      storage?.removeItem("vbgui_active_project_v1");
    } catch { /* ignore */ }
  });

  it("starts empty + no active id", () => {
    const { result } = renderHook(() => useProjects<{ v: number }>());
    expect(result.current.projects).toEqual([]);
    expect(result.current.activeId).toBeNull();
    expect(result.current.active).toBeNull();
  });

  it("create adds a project + flips active", () => {
    const { result } = renderHook(() => useProjects<{ v: number }>());
    let id = "";
    act(() => { id = result.current.create("alpha", { v: 1 }); });
    expect(result.current.projects).toHaveLength(1);
    expect(result.current.activeId).toBe(id);
    expect(result.current.active?.name).toBe("alpha");
  });

  it("rename mutates name + preserves order/payload", () => {
    const { result } = renderHook(() => useProjects<{ v: number }>());
    let id = "";
    act(() => { id = result.current.create("alpha", { v: 1 }); });
    act(() => { result.current.rename(id, "beta"); });
    expect(result.current.active?.name).toBe("beta");
    expect(result.current.active?.payload).toEqual({ v: 1 });
  });

  it("remove drops project + clears active when it was the active one",
    () => {
      const { result } = renderHook(() => useProjects<{ v: number }>());
      let id = "";
      act(() => { id = result.current.create("alpha", { v: 1 }); });
      act(() => { result.current.remove(id); });
      expect(result.current.projects).toEqual([]);
      expect(result.current.activeId).toBeNull();
    });

  it("setActive swaps active without touching others", () => {
    const { result } = renderHook(() => useProjects<{ v: number }>());
    let a = "", b = "";
    act(() => {
      a = result.current.create("alpha", { v: 1 });
      b = result.current.create("beta", { v: 2 });
    });
    act(() => { result.current.setActive(a); });
    expect(result.current.activeId).toBe(a);
    expect(result.current.projects.map((p) => p.id)).toEqual([a, b]);
    expect(result.current.projects).toHaveLength(2);
  });

  it("updateActive writes payload of active project only", () => {
    const { result } = renderHook(() => useProjects<{ v: number }>());
    let a = "", b = "";
    act(() => {
      a = result.current.create("alpha", { v: 1 });
      b = result.current.create("beta", { v: 2 });
    });
    // active is b after both creates.
    act(() => { result.current.updateActive({ v: 99 }); });
    const ba = result.current.projects.find((p) => p.id === b);
    const aa = result.current.projects.find((p) => p.id === a);
    expect(ba?.payload).toEqual({ v: 99 });
    expect(aa?.payload).toEqual({ v: 1 });
  });

  it("hydrates from localStorage on next mount", async () => {
    const { result, unmount } = renderHook(
      () => useProjects<{ v: number }>());
    act(() => { result.current.create("alpha", { v: 1 }); });
    await waitFor(() => {
      expect(window.localStorage.getItem("vbgui_projects_v1"))
        .not.toBeNull();
    });
    unmount();
    // Fresh mount — should see the persisted project.
    const { result: r2 } = renderHook(
      () => useProjects<{ v: number }>());
    expect(r2.current.projects).toHaveLength(1);
    expect(r2.current.active?.name).toBe("alpha");
  });
});
