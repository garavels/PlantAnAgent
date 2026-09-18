import type { V2DemoSnapshot } from '../types/v2'
import type { SiteActivationRequest } from '../workspace/siteTypes'

const BASE = '/api/v2/demo'

/** Only public presentation data crosses this boundary. */
async function request(path: string, init: RequestInit = {}, base = BASE): Promise<V2DemoSnapshot> {
  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...init.headers },
    cache: 'no-store',
  })
  if (!response.ok) {
    let detail = `The V2 service returned ${response.status}.`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
      else if (body.detail) detail = JSON.stringify(body.detail)
    } catch {
      // A proxy can return a non-JSON error; the HTTP status remains useful.
    }
    throw new Error(detail)
  }
  const snapshot = (await response.json()) as V2DemoSnapshot
  if (
    snapshot.schema !== 'plant-v2-demo-v1' ||
    !['interactive_fixture', 'site_forecast'].includes(snapshot.mode) ||
    typeof snapshot.run_id !== 'string' ||
    !Number.isInteger(snapshot.revision) ||
    !snapshot.frame ||
    !Array.isArray(snapshot.history)
  ) {
    throw new Error('The V2 service returned an unsupported snapshot. Refresh after checking the service version.')
  }
  return snapshot
}

export function activateSiteWorkspace(config: SiteActivationRequest, signal?: AbortSignal): Promise<V2DemoSnapshot> {
  return request('/activate', { method: 'POST', body: JSON.stringify(config), signal }, '/api/sites')
}

export function deactivateSiteWorkspace(signal?: AbortSignal): Promise<V2DemoSnapshot> {
  return request('/deactivate', { method: 'POST', signal }, '/api/sites')
}

export function fetchV2Demo(signal?: AbortSignal): Promise<V2DemoSnapshot> {
  return request('', { signal })
}

export function resetV2Demo(controller: V2DemoSnapshot['controller']['id'], signal?: AbortSignal): Promise<V2DemoSnapshot> {
  return request('/reset', { method: 'POST', body: JSON.stringify({ controller }), signal })
}

export function stepV2Demo(snapshot: V2DemoSnapshot, decisions: number, signal?: AbortSignal): Promise<V2DemoSnapshot> {
  return request('/step', {
    method: 'POST',
    body: JSON.stringify({ expected_run_id: snapshot.run_id, expected_revision: snapshot.revision, decisions }),
    signal,
  })
}
