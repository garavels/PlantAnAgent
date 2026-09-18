import { useEffect, useMemo, useState } from "react";
import type { V2DemoFrame } from "../types/v2";
import { AGENT_ACTION_DWELL_MS, deriveAgentActions } from "./agentActions";

/** One clock drives both the equipment tour and permission to advance playback. */
export function useAgentWalkthrough(frame: V2DemoFrame | null, runId: string | undefined, moving: boolean) {
  const [visible, setVisible] = useState(false);
  const actions = useMemo(() => frame ? deriveAgentActions(frame) : [], [frame]);
  const [progress, setProgress] = useState({ frame, runId, index: 0, complete: false });
  const changed = progress.frame !== frame || progress.runId !== runId;
  // Reset on every displayed boundary, including a seek away and back before
  // the first dwell finishes. A previously completed tour must not resurface.
  if (changed) setProgress({ frame, runId, index: 0, complete: false });
  const index = changed ? 0 : Math.min(progress.index, Math.max(0, actions.length - 1));
  const ready = !visible || !actions.length || (!changed && progress.complete);

  useEffect(() => {
    if (!visible || !moving || ready) return;
    const timer = window.setTimeout(() => {
      setProgress({ frame, runId, index: Math.min(index + 1, actions.length - 1), complete: index >= actions.length - 1 });
    }, AGENT_ACTION_DWELL_MS);
    return () => window.clearTimeout(timer);
  }, [visible, moving, ready, frame, runId, index, actions.length]);

  return {
    visible,
    toggle: () => {
      setVisible((value) => !value);
      setProgress({ frame, runId, index: 0, complete: false });
    },
    actions,
    index,
    action: actions[index] ?? null,
    ready,
    select: (next: number) => {
      if (Number.isInteger(next) && next >= 0 && next < actions.length) {
        setProgress({ frame, runId, index: next, complete: false });
      }
    },
  };
}
