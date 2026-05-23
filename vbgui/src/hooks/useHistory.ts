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
  // V7-H43: redo can land on a snapshot the backend later rejected
  // (e.g. user broke verify, undid, kept editing, redid back to the
  // broken state). Return value gets a `rejected` flag so the host
  // can toast + offer to skip. `snapshot` is still returned even
  // when rejected so the caller decides whether to apply.
  redo: () => { snapshot: T; rejected: boolean } | null;
  // V7-H43: tag the current top-of-past as rejected (e.g. verify
  // failed for this snapshot). Subsequent undo-then-redo onto this
  // snapshot surfaces the flag.
  markRejected: () => void;
  canUndo: boolean;
  canRedo: boolean;
  size: number;
  clear: () => void;
}

export function useHistory<T>(capacity = 50): HistoryAPI<T> {
  // V7-H43: parallel arrays for snapshots + verdicts so the API
  // surface stays {snapshot} without forcing every existing
  // consumer to deal with wrapper objects. rejected[i] mirrors
  // past[i]'s known-bad status.
  const past = useRef<T[]>([]);
  const rejected = useRef<boolean[]>([]);
  const future = useRef<T[]>([]);
  const futureRejected = useRef<boolean[]>([]);
  // setVersion bumps to force re-renders when canUndo/canRedo change.
  const [, setVersion] = useState(0);

  const push = useCallback((snapshot: T) => {
    past.current.push(snapshot);
    rejected.current.push(false);
    if (past.current.length > capacity) {
      past.current.shift();
      rejected.current.shift();
    }
    future.current = [];
    futureRejected.current = [];
    setVersion((v) => v + 1);
  }, [capacity]);

  const undo = useCallback((): T | null => {
    const top = past.current.pop();
    const topVerdict = rejected.current.pop() ?? false;
    if (top === undefined) return null;
    future.current.push(top);
    futureRejected.current.push(topVerdict);
    setVersion((v) => v + 1);
    const prev = past.current[past.current.length - 1];
    return prev ?? null;
  }, []);

  const redo = useCallback(():
    { snapshot: T; rejected: boolean } | null => {
    const next = future.current.pop();
    const nextVerdict = futureRejected.current.pop() ?? false;
    if (next === undefined) return null;
    past.current.push(next);
    rejected.current.push(nextVerdict);
    setVersion((v) => v + 1);
    return { snapshot: next, rejected: nextVerdict };
  }, []);

  const markRejected = useCallback(() => {
    if (rejected.current.length === 0) return;
    rejected.current[rejected.current.length - 1] = true;
    setVersion((v) => v + 1);
  }, []);

  const clear = useCallback(() => {
    past.current = [];
    rejected.current = [];
    future.current = [];
    futureRejected.current = [];
    setVersion((v) => v + 1);
  }, []);

  return {
    push,
    undo,
    redo,
    markRejected,
    canUndo: past.current.length > 1,
    canRedo: future.current.length > 0,
    size: past.current.length,
    clear,
  };
}
