/** The only place that knows how to reach the plant.
 *
 * Since V5 the backend owns the plant: it runs the clock, decides the action in
 * AUTO and validates it in MANUAL. So these are requests, not commands -- the UI
 * asks the backend to play, to inject a fault, to change the envelope, and then
 * watches `/api/stream` for what actually happened.
 *
 * Everything returned here is *telemetry*: what the plant reports, which a
 * sensor fault is allowed to distort. The UI never sees the simulator's truth.
 */

import type {
  AgentSnapshot,
  AgentTraceResponse,
  Constraints,
  ControlMode,
  FaultCatalogue,
  PlantAction,
  PlantFrames,
  ScenarioCatalogue,
  ScenarioDef,
  ScenarioDraft,
  ScenarioSaved,
} from '../types/plant'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!response.ok) throw new Error(await reason(response, init?.method ?? 'GET', path))
  return (await response.json()) as T
}

/** The backend's own explanation of a refusal, since validation errors are read by operators. */
async function reason(response: Response, method: string, path: string): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string' && body.detail) return body.detail
  } catch {
    // no JSON body; fall back to the status line
  }
  return `${method} ${path} -> ${response.status}`
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

/** Live plant updates, one payload per backend tick. */
export const STREAM_URL = `${BASE}/stream`

/** Current state plus recent history, so a fresh page load can draw its charts. */
export function fetchState(history = 480): Promise<PlantFrames> {
  return request<PlantFrames>(`/state?history=${history}`)
}

// --- simulation controls ---------------------------------------------------
export function play(): Promise<PlantFrames> {
  return post<PlantFrames>('/simulation/play')
}

export function pause(): Promise<PlantFrames> {
  return post<PlantFrames>('/simulation/pause')
}

/** Timesteps applied per backend tick. */
export function setSpeed(speed: number): Promise<PlantFrames> {
  return post<PlantFrames>('/simulation/speed', { speed })
}

/** Advance by hand while the clock is paused. */
export function stepPlant(repeat = 1): Promise<PlantFrames> {
  return post<PlantFrames>('/simulation/step', { repeat })
}

export function resetPlant(): Promise<PlantFrames> {
  return post<PlantFrames>('/simulation/reset')
}

// --- control ---------------------------------------------------------------
export function setControlMode(mode: ControlMode): Promise<PlantFrames> {
  return post<PlantFrames>('/controller/config', { mode })
}

/** The manual setpoint. The backend still validates it before applying it. */
export function setManualAction(action: PlantAction): Promise<PlantFrames> {
  return post<PlantFrames>('/controller/config', { action })
}

export function setConstraints(constraints: Partial<Constraints>): Promise<PlantFrames> {
  return post<PlantFrames>('/controller/config', { constraints })
}

/** Which controller drives in AUTO. RL is refused unless a checkpoint is loaded. */
export function setController(controller: string): Promise<PlantFrames> {
  return post<PlantFrames>('/controller/config', { controller })
}

// --- abnormal operation ----------------------------------------------------
export function fetchScenarios(): Promise<ScenarioCatalogue> {
  return request<ScenarioCatalogue>('/scenarios')
}

/** Reset the plant into a predefined, repeatable situation. */
export function loadScenario(scenario: string): Promise<PlantFrames> {
  return post<PlantFrames>('/scenarios', { scenario })
}

/** Natural language -> a validated scenario draft. Nothing is saved or run yet. */
export function generateScenario(prompt: string): Promise<ScenarioDraft> {
  return post<ScenarioDraft>('/scenarios/generate', { prompt })
}

/** Keep a reviewed draft for this session; the backend validates it again. */
export function saveScenario(scenario: ScenarioDef): Promise<ScenarioSaved> {
  return post<ScenarioSaved>('/scenarios/save', { scenario: draft(scenario) })
}

/** A saved scenario travels in seconds; the generator's format speaks hours. */
function draft(scenario: ScenarioDef): Record<string, unknown> {
  return {
    name: scenario.name,
    description: scenario.description,
    start_hour: scenario.start_hour,
    initial: scenario.initial,
    faults: scenario.faults.map((fault) => ({
      type: fault.type,
      severity: fault.severity,
      start_hour: (fault.start_s ?? 0) / 3600,
      duration_hours: fault.duration_s === null ? null : fault.duration_s / 3600,
      channel: fault.channel,
    })),
    notes: [],
  }
}

export function fetchFaults(): Promise<FaultCatalogue> {
  return request<FaultCatalogue>('/faults')
}

export function injectFault(type: string, severity?: number): Promise<PlantFrames> {
  return post<PlantFrames>('/faults', severity === undefined ? { type } : { type, severity })
}

export function clearFaults(): Promise<PlantFrames> {
  return request<PlantFrames>('/faults', { method: 'DELETE' })
}

export function clearFault(faultId: string): Promise<PlantFrames> {
  return request<PlantFrames>(`/faults/${faultId}`, { method: 'DELETE' })
}

// --- the supervisory agent -------------------------------------------------
/** Agent status, latest incident and recent trace. Also arrives on the stream. */
export function fetchAgent(trace = 60): Promise<AgentSnapshot> {
  return request<AgentSnapshot>(`/agent?trace=${trace}`)
}

/** The full trace, or just one run of it. */
export function fetchAgentTrace(run?: number, limit = 120): Promise<AgentTraceResponse> {
  const query = run === undefined ? `limit=${limit}` : `limit=${limit}&run=${run}`
  return request<AgentTraceResponse>(`/agent/trace?${query}`)
}

/** "Investigate current plant": one supervisor run, now, whatever the triggers say. */
export function investigatePlant(): Promise<PlantFrames> {
  return post<PlantFrames>('/agent/investigate')
}

/** Off leaves V5 behaviour exactly: controller, envelope, guard, nothing above. */
export function setAgentEnabled(enabled: boolean): Promise<PlantFrames> {
  return post<PlantFrames>('/agent/config', { enabled })
}
