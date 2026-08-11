/** Drives the plant: holds the observed state, the action, and the clock.
 *
 * The browser owns the clock for this first iteration -- every tick asks the
 * backend to advance N timesteps. When a controller or agent takes over the
 * pacing, only this hook changes.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchState, resetPlant, stepPlant } from '../lib/api'
import type { PlantAction, PlantConfig, PlantState } from '../types/plant'

/** Wall-clock interval between advances. */
export const TICK_MS = 200
/** Timesteps applied per tick, per speed setting. */
export const SPEEDS = [1, 4, 16] as const
/** ~24 simulated hours of history at the default 60 s timestep. */
const MAX_FRAMES = 1440

export type LinkStatus = 'connecting' | 'online' | 'down'

const DEFAULT_ACTION: PlantAction = {
  battery_share: 0.4,
  electrolyser_share: 0.6,
  battery_support_kw: 600,
}

export interface PlantSession {
  state: PlantState | null
  frames: PlantState[]
  config: PlantConfig | null
  action: PlantAction
  playing: boolean
  speed: number
  link: LinkStatus
  error: string | null
  setAction: (action: PlantAction) => void
  setSpeed: (speed: number) => void
  togglePlay: () => void
  stepOnce: () => void
  reset: () => void
  reconnect: () => void
}

export function usePlantSession(): PlantSession {
  const [state, setState] = useState<PlantState | null>(null)
  const [frames, setFrames] = useState<PlantState[]>([])
  const [config, setConfig] = useState<PlantConfig | null>(null)
  const [action, setAction] = useState<PlantAction>(DEFAULT_ACTION)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<number>(SPEEDS[1])
  const [link, setLink] = useState<LinkStatus>('connecting')
  const [error, setError] = useState<string | null>(null)

  // the tick closure reads these refs, so changing the action mid-run does not
  // tear down and rebuild the interval
  const actionRef = useRef(action)
  const speedRef = useRef(speed)
  const busyRef = useRef(false)
  actionRef.current = action
  speedRef.current = speed

  const absorb = useCallback((next: PlantState[], replace = false) => {
    if (next.length === 0) return
    setState(next[next.length - 1])
    setFrames((previous) => (replace ? next : [...previous, ...next]).slice(-MAX_FRAMES))
  }, [])

  const fail = useCallback((reason: unknown) => {
    setPlaying(false)
    setLink('down')
    setError(reason instanceof Error ? reason.message : String(reason))
  }, [])

  const connect = useCallback(async () => {
    setLink('connecting')
    try {
      const payload = await fetchState()
      setConfig(payload.config ?? null)
      absorb(payload.frames.length > 0 ? payload.frames : [payload.state], true)
      setLink('online')
      setError(null)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, fail])

  useEffect(() => {
    void connect()
  }, [connect])

  // --- the clock --------------------------------------------------------
  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(async () => {
      if (busyRef.current) return // a slow tick skips rather than stacks up
      busyRef.current = true
      try {
        const payload = await stepPlant(actionRef.current, speedRef.current)
        absorb(payload.frames)
      } catch (reason) {
        fail(reason)
      } finally {
        busyRef.current = false
      }
    }, TICK_MS)
    return () => window.clearInterval(timer)
  }, [playing, absorb, fail])

  const stepOnce = useCallback(async () => {
    try {
      absorb((await stepPlant(actionRef.current, 1)).frames)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, fail])

  const reset = useCallback(async () => {
    setPlaying(false)
    try {
      const payload = await resetPlant()
      absorb([payload.state], true)
      setLink('online')
      setError(null)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, fail])

  return {
    state,
    frames,
    config,
    action,
    playing,
    speed,
    link,
    error,
    setAction,
    setSpeed,
    togglePlay: () => setPlaying((previous) => !previous),
    stepOnce: () => void stepOnce(),
    reset: () => void reset(),
    reconnect: () => void connect(),
  }
}
