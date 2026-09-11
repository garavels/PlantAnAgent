import type { PlantConfig, PlantState } from "./plant";

/** Operator-safe presentation contract. Never contains future truth or an audit manifest. */
export interface V2DemoSnapshot {
  schema: "plant-v2-demo-v1";
  run_id: string;
  revision: number;
  mode: "interactive_fixture";
  status: "ready" | "complete" | "failed";
  error: string | null;
  controller: { id: "H" | "MPC" | "E-beta" | "PPO"; label: string; uses_forecast: boolean; policy_id?: string; checkpoint?: Record<string, string | number> };
  available_controllers: {
    id: "H" | "MPC" | "E-beta";
    label: string;
    available: boolean;
    reason: string | null;
  }[];
  environment: {
    id: string;
    observation_count: number;
    action_count: number;
    decision_minutes: number;
    physics_minutes: number;
    horizon_hours: number;
    fixture_seed: number;
    interface_digest: string;
    task_digest: string;
  };
  config: PlantConfig;
  /** Last completely projected boundary; a failed interval can have partial physics. */
  frame: V2DemoFrame;
  /** Every retained public decision boundary, including reset. */
  history: V2DemoHistoryPoint[];
  events: PlantEvent[];
}

export interface V2DemoFrame {
  /** Absent in historical strict daily-return recordings. */
  operating_policy?: "inventory-targets-v1";
  state: PlantState;
  weather: WeatherView;
  commitment: CommitmentView | null;
  commitment_profile: Record<string, unknown> | null;
  metrics: Record<string, number | boolean | Record<string, number>>;
  assumptions: Record<string, unknown>;
  clock: {
    elapsed_hours: number;
    decision_index: number;
    decision_count: number;
    cycle_index: number;
    cycle_count: number;
    hours_to_cycle_end: number;
  };
  delivery: {
    next_due_hours: number | null;
    next_requested_kg: number;
    delivered_kg: number;
    unmet_kg: number;
    requested_kg: number;
    commitments: {
      due_hours: number;
      requested_kg: number;
      delivered_kg: number | null;
      unmet_kg: number | null;
      status: "pending" | "met" | "shortfall";
    }[];
  };
  forecasts: {
    issued_at_hours: number;
    price_current_eur_mwh: number;
    prices: { offset_hours: number; mean: number; p10: number; p90: number }[];
    /** Solar provides aggregate mean/p10 bands only; no upper quantile is published. */
    solar: {
      offset_hours: number;
      end_offset_hours: number;
      mean_kw: number;
      p10_kw: number;
    }[];
  };
  resources: {
    id: string;
    label: string;
    unit: "kWh" | "kg";
    current: number;
    required: number;
    margin: number;
  }[];
  economics: {
    net_value_eur: number;
    /** Signed contributions; their sum equals net_value_eur. */
    entries: { id: string; label: string; value_eur: number }[];
    assumption_note: string;
  };
  decision: {
    operation?: OperatingOutlook | null;
    at_hours: number;
    controller_id: string;
    computation_ms: number;
    notes: string[];
    targets: {
      id: string;
      label: string;
      requested_kw: number;
      applied_mean_kw: number;
    }[];
    guard_adjustments: number;
    execution_digest: string;
    command_digest: string;
    input_forecasts: V2DemoFrame["forecasts"];
    input_state: PlantState;
    minutes: {
      offset_minutes: number;
      solar_kw: number;
      electrolyser_kw: number;
      dac_kw: number;
      methanator_kw: number;
      battery_charge_kw: number;
      battery_support_kw: number;
      adjustments: string[];
    }[];
  } | null;
  /** Only an actual retained MPC plan. H does not invent a future schedule. */
  plan: {
    issued_at_hours: number;
    points: {
      offset_hours: number;
      electrolyser_kw: number;
      dac_kw: number;
      methanator_kw: number;
      battery_charge_kw: number;
      battery_kwh: number;
      h2_kg: number;
      co2_kg: number;
      ch4_kg: number;
      delivered_kg: number;
    }[];
  } | null;
  cycle_settlements: {
    cycle_index: number;
    feasible: boolean;
    targets: Record<string, number>;
    actual: Record<string, number>;
    deficits: Record<string, number>;
    tolerances: Record<string, number>;
  }[];
}

export interface OperatingOutlook {
  policy_id: string;
  mode: "scheduled" | "recovery" | "fallback";
  forecast_statistic: "issued_mean";
  restoration_policy: "operating_target";
  issued_at_hours: number;
  horizon_hours: number;
  projected_shortfall_kg: number | null;
  fallback_controller: string | null;
  solver_failure: { reason: string; exception_type: string; stages: Record<string, unknown>[] } | null;
  resources: { id: string; label: string; unit: string; reference: number; current: number; projected: number | null; projected_deficit: number | null }[];
  deliveries: { due_hours: number; requested_kg: number; predicted_kg: number; shortfall_kg: number }[];
}

export interface V2DemoHistoryPoint {
  elapsed_hours: number;
  solar_kw: number;
  clear_sky_kw: number;
  h2_kg: number;
  co2_kg: number;
  battery_soc: number;
  ch4_kg: number;
  delivered_kg: number;
  net_value_eur: number;
}

export interface WeatherView {
  sample_end_hours: number;
  status: "observed" | "night" | "unattributed";
  clear_sky_kw: number;
  available_kw: number;
  resource_factor: number | null;
  attenuation: number | null;
  curtailed_kw: number;
  spatial_basis: string;
}

export interface CommitmentView {
  mode: string;
  startup_remaining_s: number;
  cooldown_remaining_s: number;
  min_up_remaining_s: number;
  min_down_remaining_s: number;
  ramp_up_headroom_kw: number;
  ramp_down_headroom_kw: number;
  cold_starts: number;
  hot_starts: number;
  forced_stops: number;
  aborted_starts: number;
  startup_energy_kwh: number;
  standby_energy_kwh: number;
  low_load_time_s: number;
  mode_time_s: Record<string, number>;
}

export interface PlantEvent {
  id: string;
  decision_index: number;
  at_hours: number;
  kind: string;
  title: string;
  detail: string;
  unit: string | null;
}

export interface PublicRecording {
  schema: "plant-public-recording-v1";
  snapshot: V2DemoSnapshot;
  frames: V2DemoFrame[];
  sha256: string;
  evidence_level: string;
}

export interface RecordingSummary {
  id: string;
  controller?: string;
  hours?: number;
  status: string;
  source: string;
  sha256?: string;
  available: boolean;
  delivered_kg?: number;
  unmet_kg?: number;
  net_value_eur?: number;
  failed_cycles?: number;
  fallback_decisions?: number;
  recovery_decisions?: number;
}
