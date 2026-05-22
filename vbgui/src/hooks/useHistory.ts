// V7-H03: bounded undo/redo stack for spec mutations.
//
// Generic over the snapshot shape. The caller (App) is responsible for
// pushing a snapshot every time a user-visible mutation lands and for
// applying the snapshot returned by undo/redo back into its reducers.
//
// History capacity defaults to 50 entries. Pushing past the cap drops
// the oldest snapshot. Redo stack is cleared on every push (the
// classic linear-undo semantics — once the user branches off, the
// forward history is gone).

import { useCallback, useRef, useState } from "react";

export interface HistoryAPI<T> {
  push: (snapshot: T) => void;
  undo: () => T | null;
  redo: () => T | null;
  canUndo: boolean;
  canRedo: boolean;
  size: number;
  clear: () => void;
}

export function useHistory<T>(capacity = 50): HistoryAPI<T> {
  const past = useRef<T[]>([]);
  const future = useRef<T[]>([]);
  // setVersion bumps to force re-renders when canUndo/canRedo change.
  const [, setVersion] = useState(0);

  const push = useCallback((snapshot: T) => {
    past.current.push(snapshot);
    if (past.current.length > capacity) past.current.shift();
    future.current = [];
    setVersion((v) => v + 1);
  }, [capacity]);

  const undo = useCallback((): T | null => {
    const top = past.current.pop();
    if (top === undefined) return null;
    future.current.push(top);
    setVersion((v) => v + 1);
    // Caller should reset to the PREVIOUS snapshot — the one before
    // the popped state. If we want classic undo (restore previous),
    // peek the now-top after pop.
    const prev = past.current[past.current.length - 1];
    return prev ?? null;
  }, []);

  const redo = useCallback((): T | null => {
    const next = future.current.pop();
    if (next === undefined) return null;
    past.current.push(next);
    setVersion((v) => v + 1);
    return next;
  }, []);

  const clear = useCallback(() => {
    past.current = [];
    future.current = [];
    setVersion((v) => v + 1);
  }, []);

  return {
    push,
    undo,
    redo,
    canUndo: past.current.length > 1,  // need at least 2 entries to undo
    canRedo: future.current.length > 0,
    size: past.current.length,
    clear,
  };
}
