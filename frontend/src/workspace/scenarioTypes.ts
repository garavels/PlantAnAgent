import type { SiteLocation, SiteSource } from "./siteTypes";

export type ScenarioController = "MPC" | "H";
export type SavedScenarioSite = "seville" | "trieste" | "stockholm";
export interface ScenarioSite {
  name: string;
  latitude: number;
  longitude: number;
  source: "saved" | "live";
  example_id: SavedScenarioSite | null;
}
export interface ScenarioSunlight {
  start_hour: number;
  end_hour: number;
  scale: number;
  forecast_known: boolean;
}
export interface ScenarioFault {
  type: string;
  severity: number;
  start_hour: number;
  duration_hours: number | null;
  channel: string | null;
}
export interface ScenarioDelivery { due_hour: number; kg: number }
export interface ScenarioSpec {
  version: "plant-scenario-v1";
  name: string;
  description: string;
  duration_days: number;
  seed: number;
  site: ScenarioSite;
  solar_kw: number;
  battery_kwh: number;
  initial: { battery_soc: number; h2_kg: number; co2_kg: number; ch4_kg: number };
  sunlight: ScenarioSunlight[];
  faults: ScenarioFault[];
  deliveries: ScenarioDelivery[];
  economics: { delivered_service_value_eur_per_kg: number; unmet_delivery_penalty_eur_per_kg: number };
  notes: string[];
  unsupported: string[];
}
export interface ScenarioExample { title: string; prompt: string; scenario: ScenarioSpec }
export interface ScenarioCapabilities {
  llm_available: boolean;
  model: string;
  defaults: ScenarioSpec;
  examples: ScenarioExample[];
  faults: { type: string; label: string; default_severity: number }[];
  sensor_channels: string[];
}
export interface ScenarioGeneration { scenario: ScenarioSpec; warnings: string[] }
export interface ScenarioPreview extends ScenarioGeneration {
  preview_id: string;
  forecast: { id: string; site: SiteLocation; start_time: string; source: SiteSource };
  summary: { duration_hours: number; start_time: string; end_time: string; total_delivery_kg: number; delivery_count: number };
}
