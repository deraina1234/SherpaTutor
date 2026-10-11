import { useCallback, useEffect, useRef } from "react";
import type { InMsg } from "./socket";

// Gap between revealing one round and the next item in the queue.
const ROUND_GAP_MS = 600;

// Messages go in as they arrive and come out in order. After a round is shown,
// the next message waits ROUND_GAP_MS. Decision and hint add no extra wait, so
// they appear right after the last round has had its turn.
export function useRevealQueue(apply: (m: InMsg) => void) {
  const queue = useRef<InMsg[]>([]);
  const timer = useRef<number | null>(null);
  const applyRef = useRef(apply);
  const step = useRef<() => void>(() => {});

  useEffect(() => {
    applyRef.current = apply;
  }, [apply]);

  useEffect(() => {
    step.current = () => {
      timer.current = null;
      const m = queue.current.shift();
      if (!m) return;
      applyRef.current(m);
      const gap = m.type === "round" ? ROUND_GAP_MS : 0;
      if (gap > 0 || queue.current.length > 0) {
        timer.current = window.setTimeout(() => step.current(), gap);
      }
    };
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, []);

  const push = useCallback((m: InMsg) => {
    queue.current.push(m);
    if (timer.current === null) step.current();
  }, []);

  const clear = useCallback(() => {
    queue.current = [];
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  return { push, clear };
}