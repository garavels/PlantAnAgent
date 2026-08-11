/** The only place that knows how to reach the plant.
 *
 * Swap these four calls for a websocket, an agent endpoint or an in-browser
 * WASM simulator and nothing else in the UI has to change.
 */

import type { PlantAction, PlantFrames } from '../types/plant'

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
