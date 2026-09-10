import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchV2Demo, resetV2Demo, stepV2Demo } from '../lib/v2-api'
import type { V2DemoSnapshot } from '../types/v2'

type Pending = 'refresh' | 'step' | 'reset' | null

/** Manual fixture controls. Mounting reads state; no effect advances physics. */
export function useV2Demo() {
  const [snapshot, setSnapshot] = useState<V2DemoSnapshot | null>(null)
  const [pending, setPending] = useState<Pending>('refresh')
  const [error, setError] = useState<string | null>(null)
  const [stale, setStale] = useState(false)
  const latest = useRef<V2DemoSnapshot | null>(null)
  const busy = useRef(false)
  const staleRef = useRef(false)
  const generation = useRef(0)
  const abort = useRef<AbortController | null>(null)

  const perform = useCallback(async (
    kind: Exclude<Pending, null>,
    operation: (signal: AbortSignal) => Promise<V2DemoSnapshot>,
  ) => {
    if (busy.current) return
    busy.current = true
    const id = ++generation.current
    const controller = new AbortController()
    abort.current = controller
    setPending(kind)
    setError(null)
    try {
      const next = await operation(controller.signal)
      if (id !== generation.current) return
      latest.current = next
      setSnapshot(next)
      staleRef.current = false
      setStale(false)
    } catch (reason) {
      if (id !== generation.current) return
      // A failed write can have committed remotely. Read before another write;
      // never retry a step automatically or optimistically move the clock.
      staleRef.current = true
      setStale(true)
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (id === generation.current) {
        busy.current = false
        abort.current = null
        setPending(null)
      }
    }
  }, [])

  const refresh = useCallback(() => perform('refresh', fetchV2Demo), [perform])

  const advance = useCallback((decisions: number) => {
    const current = latest.current
    if (!current || staleRef.current || current.status !== 'ready' || !Number.isInteger(decisions) || decisions < 1 || decisions > 4) return
    const remaining = current.frame.clock.decision_count - current.frame.clock.decision_index
    if (decisions > remaining) return
    return perform('step', (signal) => stepV2Demo(current, decisions, signal))
  }, [perform])

  const reset = useCallback((controller: V2DemoSnapshot['controller']['id']) => {
    if (staleRef.current) return
    return perform('reset', (signal) => resetV2Demo(controller, signal))
  }, [perform])

  const cancelPending = useCallback(() => {
    ++generation.current
    abort.current?.abort()
    busy.current = false
  }, [])

  useEffect(() => {
    void refresh()
    return cancelPending
  }, [refresh, cancelPending])

  return { snapshot, pending, error, stale, refresh, advance, reset }
}
