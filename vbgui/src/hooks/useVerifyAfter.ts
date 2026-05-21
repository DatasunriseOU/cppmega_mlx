// Debounced verify trigger. The App calls verify(spec) at most once per
// debounceMs after any mutation. Cancels in-flight requests when a newer
// mutation lands.

import { useCallback, useEffect, useRef } from "react";

export interface UseVerifyAfterOptions {
  debounceMs?: number;
  enabled?: boolean;
}

export function useVerifyAfter<T>(
  payload: T,
  runner: (payload: T) => Promise<void>,
  opts: UseVerifyAfterOptions = {},
): { schedule: () => void; cancel: () => void } {
  const debounceMs = opts.debounceMs ?? 150;
  const enabled = opts.enabled ?? true;
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const payloadRef = useRef<T>(payload);
  const runnerRef = useRef(runner);

  useEffect(() => { payloadRef.current = payload; }, [payload]);
  useEffect(() => { runnerRef.current = runner; }, [runner]);

  const cancel = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = undefined;
  }, []);

  const schedule = useCallback(() => {
    if (!enabled) return;
    cancel();
    timerRef.current = setTimeout(() => {
      const snap = payloadRef.current;
      void runnerRef.current(snap).catch(() => undefined);
    }, debounceMs);
  }, [cancel, debounceMs, enabled]);

  useEffect(() => () => cancel(), [cancel]);
  return { schedule, cancel };
}
