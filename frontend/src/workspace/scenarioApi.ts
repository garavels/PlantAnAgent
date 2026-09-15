import type { V2DemoSnapshot } from "../types/v2";
import type { ScenarioCapabilities, ScenarioController, ScenarioGeneration, ScenarioPreview, ScenarioSpec } from "./scenarioTypes";

const BASE = "/api/scenario-studio";

export class ScenarioApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ScenarioApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = `The scenario service returned ${response.status}. Your draft is still available; try again.`;
    try {
      const body = await response.json() as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((issue: { loc?: (string | number)[]; msg?: string }) => {
        const field = issue.loc?.filter((part) => part !== "body" && part !== "scenario").join(" / ");
        return `${field ? `${field}: ` : ""}${issue.msg ?? "Invalid input"}`;
      }).join(". ");
    } catch { /* A gateway can return HTML. Preserve the actionable status. */ }
    throw new ScenarioApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const scenarioApi = {
  capabilities: (signal?: AbortSignal) => request<ScenarioCapabilities>("/capabilities", { signal }),
  generate: (prompt: string, signal?: AbortSignal) => request<ScenarioGeneration>("/generate", { method: "POST", body: JSON.stringify({ prompt }), signal }),
  preview: (scenario: ScenarioSpec, signal?: AbortSignal) => request<ScenarioPreview>("/preview", { method: "POST", body: JSON.stringify({ scenario }), signal }),
};

export async function activateScenarioWorkspace(previewId: string, controller: ScenarioController, signal?: AbortSignal): Promise<V2DemoSnapshot> {
  const snapshot = await request<V2DemoSnapshot>("/activate", {
    method: "POST", body: JSON.stringify({ preview_id: previewId, controller }), signal,
  });
  if (snapshot.schema !== "plant-v2-demo-v1" || snapshot.mode !== "site_forecast" || typeof snapshot.run_id !== "string" || !Number.isInteger(snapshot.revision) || !snapshot.frame || !Array.isArray(snapshot.history)) {
    throw new Error("The scenario service returned an unsupported simulation. Refresh after checking the service version.");
  }
  return snapshot;
}
