import type { SiteExample, SiteForecast, SiteRunJob, SiteRunRequest } from "./siteTypes";

const BASE = "/api/sites";

export class SiteApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SiteApiError";
    this.status = status;
  }
}

export const STUDY_UNAVAILABLE_MESSAGE = "This study is no longer available. The server may have restarted or the result may have expired. Run the simulation again to create a new study.";

export function isUnavailableStudy(error: unknown) {
  return error instanceof SiteApiError && error.status === 404;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `The location service returned ${response.status}. Try again or choose a saved example.`;
    try {
      const body = await response.json() as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((item: { msg?: string }) => item.msg ?? "Invalid input").join(". ");
    } catch { /* Preserve the HTTP error when a proxy returns HTML. */ }
    throw new SiteApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export const siteApi = {
  examples: (signal?: AbortSignal) => request<SiteExample[]>("/examples", { signal }),
  example: (id: string, signal?: AbortSignal) => request<SiteForecast>(`/examples/${encodeURIComponent(id)}`, { signal }),
  savedForecast: (id: string, signal?: AbortSignal) => request<SiteForecast>(`/forecasts/${encodeURIComponent(id)}/download`, { signal }),
  forecast: (site: { latitude: number; longitude: number; name?: string; refresh?: boolean }, signal?: AbortSignal) =>
    request<SiteForecast>("/forecast", { method: "POST", body: JSON.stringify(site), signal }),
  run: (config: SiteRunRequest, signal?: AbortSignal) =>
    request<SiteRunJob>("/runs", { method: "POST", body: JSON.stringify(config), signal }),
  job: (id: string, signal?: AbortSignal) => request<SiteRunJob>(`/runs/${encodeURIComponent(id)}`, { signal }),
  download: (id: string) => `${BASE}/forecasts/${encodeURIComponent(id)}/download`,
};

export const EUROPE_BOUNDS = { west: -25, east: 45, south: 34, north: 72 };

/** Matches the bundled Natural Earth equirectangular map, including its aspect ratio. */
export function projectSite(latitude: number, longitude: number) {
  return { x: ((longitude + 25) / 70) * 720, y: ((72 - latitude) / 38) * 608 };
}

export function unprojectSite(x: number, y: number) {
  return {
    latitude: Math.round(Math.max(34, Math.min(72, 72 - (y / 608) * 38)) * 10000) / 10000,
    longitude: Math.round(Math.max(-25, Math.min(45, -25 + (x / 720) * 70)) * 10000) / 10000,
  };
}

export function isEuropeanSite(latitude: number, longitude: number) {
  return Number.isFinite(latitude) && Number.isFinite(longitude) && latitude >= 34 && latitude <= 72 && longitude >= -25 && longitude <= 45;
}

/** Keep the planner's additional lookahead out of the ten-day study display. */
export function studyDays(hours: SiteForecast["hours"]) {
  const studyHours = hours.slice(0, 240);
  return Array.from({ length: Math.ceil(studyHours.length / 24) }, (_, index) => studyHours.slice(index * 24, (index + 1) * 24));
}
