export interface SiteLocation {
  name: string;
  latitude: number;
  longitude: number;
  elevation_m: number | null;
  timezone: string;
}

export interface SiteSource {
  provider: string;
  url: string;
  retrieved_at: string;
  forecast_issued_at: string | null;
  model: string;
  kind: "live_forecast" | "cached_forecast";
  note: string;
}

export interface SiteCloudCover {
  cloud_cover_pct?: number | null;
  cloud_cover_low_pct?: number | null;
  cloud_cover_mid_pct?: number | null;
  cloud_cover_high_pct?: number | null;
}

export interface SiteForecast {
  id: string;
  site: SiteLocation;
  source: SiteSource;
  start_time: string;
  hours: (SiteCloudCover & {
    time: string;
    shortwave_w_m2: number;
    temperature_c: number;
    humidity_pct: number;
    wind_m_s: number;
  })[];
  assumptions: string[];
}

export interface SiteExample {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  available: boolean;
}

export interface SiteRunRequest {
  forecast_id: string;
  solar_kw: number;
  battery_kwh: number;
  controller: "MPC" | "H";
  inject_fault: boolean;
  solar_scale: 0.7 | 1;
  compare_baseline: boolean;
}

export type SiteActivationRequest = Omit<SiteRunRequest, "compare_baseline">;

export interface SiteRunResult {
  schema: "site-simulation-v1";
  forecast_id: string;
  site: SiteLocation;
  source: SiteSource;
  start_time: string;
  end_time: string;
  horizon_hours: number;
  controller: {
    id: string;
    label: string;
    uses_forecast: boolean;
    planning_horizon_hours: number | null;
    replanning_interval_hours?: number | null;
    solve_count?: number;
    fallback_count: number;
  };
  parameters: {
    solar_kw: number;
    battery_kwh: number;
    solar_scale: number;
    inject_fault: boolean;
  };
  metrics: {
    methane_produced_kg: number;
    solar_energy_kwh: number;
    curtailed_energy_kwh: number;
    curtailment_pct: number;
    delivery_requested_kg: number;
    delivery_delivered_kg: number;
    delivery_service_pct: number;
    battery_final_soc: number;
    battery_min_soc: number;
    utilisation: { electrolyser: number; dac: number; methanator: number };
    limiting_subsystem: string;
  };
  history: {
    time: string;
    elapsed_hours: number;
    battery_soc: number;
    solar_kw: number;
    forecast_solar_kw: number;
    methane_kg_h: number;
    methane_total_kg: number;
    curtailed_kw: number;
    electrolyser_kw: number;
    dac_kw: number;
    methanator_kw: number;
    limiting_subsystem: string;
    fault_active: boolean;
  }[];
  fault: {
    enabled: boolean;
    type: string;
    start_hours: number;
    duration_hours: number;
    note: string;
  };
  assumptions: string[];
  runtime_seconds: number;
  baseline?: SiteRunResult | null;
}

export interface SiteRunJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed";
  progress: number;
  result: SiteRunResult | null;
  error: string | null;
}
