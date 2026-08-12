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
  Constraints,
  ControlMode,
  FaultCatalogue,
  PlantAction,
  PlantFrames,
  ScenarioCatalogue,
} from '../types/plant'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!response.ok) throw new Error(`${init?.method ?? 'GET'} ${path} -> ${response.status}`)
  return (await response.json()) as T
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

// --- abnormal operation ----------------------------------------------------
export function fetchScenarios(): Promise<ScenarioCatalogue> {
  return request<ScenarioCatalogue>('/scenarios')
}

/** Reset the plant into a predefined, repeatable situation. */
export function loadScenario(scenario: string): Promise<PlantFrames> {
  return post<PlantFrames>('/scenarios', { scenario })
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
