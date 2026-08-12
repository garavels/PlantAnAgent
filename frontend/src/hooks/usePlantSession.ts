/** Subscribes to the plant. Since V5 it no longer drives it.
 *
 * The backend owns the clock, the controller and the action guard, so this hook
 * is an adapter: it opens the SSE stream, keeps the last frames, events, faults
 * and control status, and turns operator gestures into requests. Play, speed and
 * mode are *reported* by the backend, never assumed here -- what the UI shows is
 * what the plant is actually doing.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  clearFault as clearFaultRequest,
  clearFaults as clearFaultsRequest,
  fetchFaults,
  fetchScenarios,
  fetchState,
  injectFault as injectFaultRequest,
  investigatePlant,
  loadScenario as loadScenarioRequest,
  pause as pauseRequest,
  play as playRequest,
  resetPlant,
  setAgentEnabled as setAgentEnabledRequest,
  setControlMode as setControlModeRequest,
  setManualAction,
  setSpeed as setSpeedRequest,
  STREAM_URL,
  stepPlant,
} from '../lib/api'
import type {
  AgentSnapshot,
  ControlMode,
  ControlStatus,
  FaultKind,
  PlantAction,
  PlantConfig,
  PlantEvent,
  PlantFault,
  PlantFrames,
  PlantState,
  ScenarioDef,
} from '../types/plant'

/** Timesteps per backend tick, per speed setting. */
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
  /** The manual setpoint as edited here; the backend validates it on apply. */
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
  control: ControlStatus | null
  /** The supervisory agent: status, latest incident, recent trace. */
  agent: AgentSnapshot | null
  setAction: (action: PlantAction) => void
  setSpeed: (speed: number) => void
  setMode: (mode: ControlMode) => void
  togglePlay: () => void
  stepOnce: () => void
  reset: () => void
  reconnect: () => void
  injectFault: (type: string, severity?: number) => void
  clearFault: (faultId: string) => void
  clearFaults: () => void
  loadScenario: (scenario: string) => void
  /** Ask for one supervisor run now, outside the trigger rules. */
  investigate: () => void
  setAgentEnabled: (enabled: boolean) => void
}

export function usePlantSession(): PlantSession {
  const [state, setState] = useState<PlantState | null>(null)
  const [frames, setFrames] = useState<PlantState[]>([])
  const [config, setConfig] = useState<PlantConfig | null>(null)
  const [action, setLocalAction] = useState<PlantAction>(DEFAULT_ACTION)
  const [link, setLink] = useState<LinkStatus>('connecting')
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<PlantEvent[]>([])
  const [faults, setFaults] = useState<PlantFault[]>([])
  const [faultKinds, setFaultKinds] = useState<FaultKind[]>([])
  const [scenarios, setScenarios] = useState<ScenarioDef[]>([])
  const [scenario, setScenario] = useState<string | null>(null)
  const [control, setControl] = useState<ControlStatus | null>(null)
  const [agent, setAgent] = useState<AgentSnapshot | null>(null)
  const [generation, setGeneration] = useState(0)

  // payloads arrive twice for whoever asked for them (once as the response,
  // once over the stream), so frames are keyed on the timestep they belong to
  const lastStepRef = useRef(-1)
  const editedActionRef = useRef(false)

  const absorb = useCallback((payload: PlantFrames) => {
    if (payload.config) setConfig(payload.config)
    if (payload.events) setEvents(payload.events.slice(-MAX_EVENTS))
    if (payload.faults) setFaults(payload.faults)
    if (payload.scenario !== undefined) setScenario(payload.scenario)
    if (payload.agent) setAgent(payload.agent)
    if (payload.control) {
      setControl(payload.control)
      // adopt the backend's setpoint until the operator touches a slider
      if (!editedActionRef.current) setLocalAction(payload.control.manual_action)
    }

    const replace = payload.replace ?? false
    const incoming = payload.frames.length > 0 ? payload.frames : [payload.state]
    const fresh = replace ? incoming : incoming.filter((frame) => frame.step > lastStepRef.current)
    if (replace) lastStepRef.current = -1
    if (fresh.length === 0) {
      if (replace) setState(payload.state)
      return
    }
    lastStepRef.current = fresh[fresh.length - 1].step
    setState(fresh[fresh.length - 1])
    setFrames((previous) => (replace ? fresh : [...previous, ...fresh]).slice(-MAX_FRAMES))
  }, [])

  const report = useCallback((reason: unknown) => {
    setError(reason instanceof Error ? reason.message : String(reason))
  }, [])

  /** Requests answer with the same payload the stream carries. */
  const ask = useCallback(
    async (call: Promise<PlantFrames>) => {
      try {
        absorb(await call)
        setError(null)
      } catch (reason) {
        report(reason)
      }
    },
    [absorb, report],
  )

  // --- the stream --------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    setLink('connecting')

    // catalogues are static, so they are fetched rather than streamed
    void (async () => {
      try {
        const [scenarioList, faultList] = await Promise.all([fetchScenarios(), fetchFaults()])
        if (cancelled) return
        setScenarios(scenarioList.scenarios)
        setFaultKinds(faultList.catalogue)
      } catch (reason) {
        if (!cancelled) report(reason)
      }
    })()

    const source = new EventSource(STREAM_URL)
    source.onopen = () => {
      setLink('online')
      setError(null)
    }
    source.onmessage = (event: MessageEvent<string>) => {
      absorb(JSON.parse(event.data) as PlantFrames)
      setLink('online')
    }
    // EventSource reconnects by itself; say so and keep the last frames on screen
    source.onerror = () => {
      setLink('down')
      setError('Lost the plant feed — reconnecting')
    }

    // if the stream cannot be established at all, at least draw the plant once
    void fetchState()
      .then((payload) => {
        if (!cancelled) absorb(payload)
      })
      .catch((reason: unknown) => {
        if (!cancelled) report(reason)
      })

    return () => {
      cancelled = true
      source.close()
    }
  }, [absorb, report, generation])

  // --- operator gestures --------------------------------------------------
  const setAction = useCallback(
    (next: PlantAction) => {
      editedActionRef.current = true
      setLocalAction(next)
      void ask(setManualAction(next))
    },
    [ask],
  )

  const togglePlay = useCallback(() => {
    void ask(control?.playing ? pauseRequest() : playRequest())
  }, [ask, control?.playing])

  return {
    state,
    frames,
    config,
    action,
    playing: control?.playing ?? false,
    speed: control?.speed ?? SPEEDS[1],
    link,
    error,
    events,
    faults,
    faultKinds,
    scenarios,
    scenario,
    control,
    agent,
    setAction,
    setSpeed: (speed) => void ask(setSpeedRequest(speed)),
    setMode: (mode) => void ask(setControlModeRequest(mode)),
    togglePlay,
    stepOnce: () => void ask(stepPlant(1)),
    reset: () => void ask(resetPlant()),
    reconnect: () => setGeneration((previous) => previous + 1),
    injectFault: (type, severity) => void ask(injectFaultRequest(type, severity)),
    clearFault: (faultId) => void ask(clearFaultRequest(faultId)),
    clearFaults: () => void ask(clearFaultsRequest()),
    loadScenario: (id) => void ask(loadScenarioRequest(id)),
    investigate: () => void ask(investigatePlant()),
    setAgentEnabled: (enabled) => void ask(setAgentEnabledRequest(enabled)),
  }
}
