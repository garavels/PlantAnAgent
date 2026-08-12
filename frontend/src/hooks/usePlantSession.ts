/** Drives the plant: holds the observed state, the action, and the clock.
 *
 * The browser owns the clock for this first iteration -- every tick asks the
 * backend to advance N timesteps. When a controller or agent takes over the
 * pacing, only this hook changes.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  clearFault as clearFaultRequest,
  clearFaults as clearFaultsRequest,
  fetchFaults,
  fetchScenarios,
  fetchState,
  injectFault as injectFaultRequest,
  loadScenario as loadScenarioRequest,
  resetPlant,
  stepPlant,
} from '../lib/api'
import type {
  FaultKind,
  PlantAction,
  PlantConfig,
  PlantEvent,
  PlantFault,
  PlantFrames,
  PlantState,
  ScenarioDef,
} from '../types/plant'

/** Wall-clock interval between advances. */
export const TICK_MS = 200
/** Timesteps applied per tick, per speed setting. */
export const SPEEDS = [1, 4, 16] as const
/** ~24 simulated hours of history at the default 60 s timestep. */
const MAX_FRAMES = 1440
/** Events kept client-side; the backend already trims its own stream. */
const MAX_EVENTS = 200

export type LinkStatus = 'connecting' | 'online' | 'down'

const DEFAULT_ACTION: PlantAction = {
  battery_share: 0.25,
  electrolyser_share: 0.45,
  dac_share: 0.2,
  methanator_share: 0.1,
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
  events: PlantEvent[]
  faults: PlantFault[]
  faultKinds: FaultKind[]
  scenarios: ScenarioDef[]
  scenario: string | null
  setAction: (action: PlantAction) => void
  setSpeed: (speed: number) => void
  togglePlay: () => void
  stepOnce: () => void
  reset: () => void
  reconnect: () => void
  injectFault: (type: string, severity?: number) => void
  clearFault: (faultId: string) => void
  clearFaults: () => void
  loadScenario: (scenario: string) => void
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
  const [events, setEvents] = useState<PlantEvent[]>([])
  const [faults, setFaults] = useState<PlantFault[]>([])
  const [faultKinds, setFaultKinds] = useState<FaultKind[]>([])
  const [scenarios, setScenarios] = useState<ScenarioDef[]>([])
  const [scenario, setScenario] = useState<string | null>(null)

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

  /** Conditions ride along with every response, so keep them in one place. */
  const absorbConditions = useCallback((payload: PlantFrames) => {
    if (payload.events) setEvents(payload.events.slice(-MAX_EVENTS))
    if (payload.faults) setFaults(payload.faults)
    if (payload.scenario !== undefined) setScenario(payload.scenario)
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
      absorbConditions(payload)
      const [scenarioList, faultList] = await Promise.all([fetchScenarios(), fetchFaults()])
      setScenarios(scenarioList.scenarios)
      setFaultKinds(faultList.catalogue)
      setLink('online')
      setError(null)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, absorbConditions, fail])

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
        absorbConditions(payload)
      } catch (reason) {
        fail(reason)
      } finally {
        busyRef.current = false
      }
    }, TICK_MS)
    return () => window.clearInterval(timer)
  }, [playing, absorb, absorbConditions, fail])

  const stepOnce = useCallback(async () => {
    try {
      const payload = await stepPlant(actionRef.current, 1)
      absorb(payload.frames)
      absorbConditions(payload)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, absorbConditions, fail])

  const reset = useCallback(async () => {
    setPlaying(false)
    try {
      const payload = await resetPlant()
      absorb([payload.state], true)
      absorbConditions(payload)
      setLink('online')
      setError(null)
    } catch (reason) {
      fail(reason)
    }
  }, [absorb, absorbConditions, fail])

  // --- abnormal operation ------------------------------------------------
  /** Faults only change conditions; the state arrives with the next step. */
  const mutateFaults = useCallback(
    async (call: Promise<PlantFrames>) => {
      try {
        absorbConditions(await call)
        setError(null)
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason))
      }
    },
    [absorbConditions],
  )

  const loadScenario = useCallback(
    async (id: string) => {
      setPlaying(false)
      try {
        const payload = await loadScenarioRequest(id)
        absorb([payload.state], true)
        absorbConditions(payload)
        setLink('online')
        setError(null)
      } catch (reason) {
        fail(reason)
      }
    },
    [absorb, absorbConditions, fail],
  )

  return {
    state,
    frames,
    config,
    action,
    playing,
    speed,
    link,
    error,
    events,
    faults,
    faultKinds,
    scenarios,
    scenario,
    setAction,
    setSpeed,
    togglePlay: () => setPlaying((previous) => !previous),
    stepOnce: () => void stepOnce(),
    reset: () => void reset(),
    reconnect: () => void connect(),
    injectFault: (type, severity) => void mutateFaults(injectFaultRequest(type, severity)),
    clearFault: (faultId) => void mutateFaults(clearFaultRequest(faultId)),
    clearFaults: () => void mutateFaults(clearFaultsRequest()),
    loadScenario: (id) => void loadScenario(id),
  }
}
