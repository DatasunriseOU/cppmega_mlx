// V7-H02: project-registry hook with localStorage persistence.
//
// Manages a list of named projects + the active id. Each entry stores
// an opaque payload (the App composes the spec+nodes+edges blob the
// rest of the workflow already serialises for Save/Load).
//
// The full ProjectTabs UI + per-project undo stack (H03 composition)
// land as follow-ups; this hook is the contract everything else binds
// to.

import { useCallback, useEffect, useState } from "react";

export interface Project<T> {
  id: string;
  name: string;
  payload: T;
}

export interface UseProjectsAPI<T> {
  projects: Project<T>[];
  activeId: string | null;
  active: Project<T> | null;
  create: (name: string, payload: T) => string;
  rename: (id: string, name: string) => void;
  remove: (id: string) => void;
  setActive: (id: string) => void;
  updateActive: (payload: T) => void;
}

const LS_PROJECTS = "vbgui_projects_v1";
const LS_ACTIVE = "vbgui_active_project_v1";

function _rid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `p_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function _readLS<T>(key: string, fallback: T): T {
  try {
    const storage = _storage();
    if (storage == null) return fallback;
    const raw = storage.getItem(key);
    if (raw == null) return fallback;
    return JSON.parse(raw) as T;
  } catch { return fallback; }
}

function _writeLS(key: string, value: unknown): void {
  const storage = _storage();
  if (storage == null) return;
  try { storage.setItem(key, JSON.stringify(value)); }
  catch { /* quota errors silently ignored */ }
}

function _storage(): Storage | null {
  const browserStorage =
    typeof window === "undefined" ? undefined : window.localStorage;
  const globalStorage =
    typeof localStorage === "undefined" ? undefined : localStorage;
  for (const candidate of [browserStorage, globalStorage]) {
    if (
      candidate != null &&
      typeof candidate.getItem === "function" &&
      typeof candidate.setItem === "function"
    ) {
      return candidate;
    }
  }
  return null;
}

export function useProjects<T>(): UseProjectsAPI<T> {
  const [projects, setProjects] = useState<Project<T>[]>(
    () => _readLS<Project<T>[]>(LS_PROJECTS, []));
  const [activeId, setActiveId] = useState<string | null>(
    () => _readLS<string | null>(LS_ACTIVE, null));

  useEffect(() => { _writeLS(LS_PROJECTS, projects); },
            [projects]);
  useEffect(() => { _writeLS(LS_ACTIVE, activeId); }, [activeId]);

  const create = useCallback((name: string, payload: T): string => {
    const id = _rid();
    setProjects((prev) => [...prev, { id, name, payload }]);
    setActiveId(id);
    return id;
  }, []);

  const rename = useCallback((id: string, name: string) => {
    setProjects((prev) => prev.map(
      (p) => (p.id === id ? { ...p, name } : p)));
  }, []);

  const remove = useCallback((id: string) => {
    setProjects((prev) => prev.filter((p) => p.id !== id));
    setActiveId((cur) => (cur === id ? null : cur));
  }, []);

  const setActive = useCallback((id: string) => setActiveId(id), []);

  const updateActive = useCallback((payload: T) => {
    setProjects((prev) => prev.map(
      (p) => (p.id === activeId ? { ...p, payload } : p)));
  }, [activeId]);

  const active = projects.find((p) => p.id === activeId) ?? null;
  return { projects, activeId, active, create, rename, remove,
           setActive, updateActive };
}
