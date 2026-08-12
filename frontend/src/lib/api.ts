/** The only place that knows how to reach the plant.
 *
 * Swap these calls for a websocket, an agent endpoint or an in-browser WASM
 * simulator and nothing else in the UI has to change.
 *
 * Everything returned here is *telemetry*: what the plant reports, which a
 * sensor fault is allowed to distort. The UI never sees the simulator's truth.
 */

import type { FaultCatalogue, PlantAction, PlantFrames, ScenarioCatalogue } from '../types/plant'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!response.ok) throw new Error(`${init?.method ?? 'GET'} ${path} -> ${response.status}`)
  return (await response.json()) as T
}

/** Current state plus recent history, so a fresh page load can draw its charts. */
export function fetchState(history = 480): Promise<PlantFrames> {
  return request<PlantFrames>(`/state?history=${history}`)
}

/** Advance the plant `repeat` timesteps under one action. */
export function stepPlant(action: PlantAction, repeat = 1): Promise<PlantFrames> {
  return request<PlantFrames>('/step', {
    method: 'POST',
    body: JSON.stringify({ action, repeat }),
  })
}

export function resetPlant(): Promise<PlantFrames> {
  return request<PlantFrames>('/reset', { method: 'POST' })
}

// --- abnormal operation ----------------------------------------------------
export function fetchScenarios(): Promise<ScenarioCatalogue> {
  return request<ScenarioCatalogue>('/scenarios')
}

/** Reset the plant into a predefined, repeatable situation. */
export function loadScenario(scenario: string): Promise<PlantFrames> {
  return request<PlantFrames>('/scenarios', {
    method: 'POST',
    body: JSON.stringify({ scenario }),
  })
}

export function fetchFaults(): Promise<FaultCatalogue> {
  return request<FaultCatalogue>('/faults')
}

export function injectFault(type: string, severity?: number): Promise<PlantFrames> {
  return request<PlantFrames>('/faults', {
    method: 'POST',
    body: JSON.stringify(severity === undefined ? { type } : { type, severity }),
  })
}

export function clearFaults(): Promise<PlantFrames> {
  return request<PlantFrames>('/faults', { method: 'DELETE' })
}

export function clearFault(faultId: string): Promise<PlantFrames> {
  return request<PlantFrames>(`/faults/${faultId}`, { method: 'DELETE' })
}
