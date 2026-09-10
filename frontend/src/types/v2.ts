import type { PlantConfig, PlantState } from './plant'

/** Operator-safe presentation contract. Never contains future truth or an audit manifest. */
export interface V2DemoSnapshot {
  schema: 'plant-v2-demo-v1'
  run_id: string
  revision: number
  mode: 'interactive_fixture'
  status: 'ready' | 'complete' | 'failed'
  error: string | null
  controller: { id: 'H' | 'E-beta'; label: string; uses_forecast: boolean }
  available_controllers: { id: 'H' | 'E-beta'; label: string; available: boolean; reason: string | null }[]
  environment: {
    id: string
    observation_count: number
    action_count: number
    decision_minutes: number
    physics_minutes: number
    horizon_hours: number
    fixture_seed: number
    interface_digest: string
    task_digest: string
  }
  config: PlantConfig
  /** Last completely projected boundary; a failed interval can have partial physics. */
  frame: V2DemoFrame
  /** Every retained public decision boundary, including reset. */
  history: V2DemoHistoryPoint[]
}

export interface V2DemoFrame {
  state: PlantState
  clock: {
    elapsed_hours: number
    decision_index: number
    decision_count: number
    cycle_index: number
    cycle_count: number
    hours_to_cycle_end: number
  }
  delivery: {
    next_due_hours: number | null
    next_requested_kg: number
    delivered_kg: number
    unmet_kg: number
    requested_kg: number
    commitments: {
      due_hours: number
      requested_kg: number
      delivered_kg: number | null
      unmet_kg: number | null
      status: 'pending' | 'met' | 'shortfall'
    }[]
  }
  forecasts: {
    issued_at_hours: number
    price_current_eur_mwh: number
    prices: { offset_hours: number; mean: number; p10: number; p90: number }[]
    /** Solar provides aggregate mean/p10 bands only; no upper quantile is published. */
    solar: { offset_hours: number; end_offset_hours: number; mean_kw: number; p10_kw: number }[]
  }
  resources: {
    id: string
    label: string
    unit: 'kWh' | 'kg'
    current: number
    required: number
    margin: number
  }[]
  economics: {
    net_value_eur: number
    /** Signed contributions; their sum equals net_value_eur. */
    entries: { id: string; label: string; value_eur: number }[]
    assumption_note: string
  }
  decision: {
    at_hours: number
    controller_id: string
    computation_ms: number
    notes: string[]
    targets: { id: string; label: string; requested_kw: number; applied_mean_kw: number }[]
    guard_adjustments: number
    execution_digest: string
    command_digest: string
  } | null
  /** Only an actual retained MPC plan. H does not invent a future schedule. */
  plan: {
    issued_at_hours: number
    points: {
      offset_hours: number
      electrolyser_kw: number
      dac_kw: number
      methanator_kw: number
      battery_charge_kw: number
    }[]
  } | null
  cycle_settlements: { cycle_index: number; feasible: boolean }[]
}

export interface V2DemoHistoryPoint {
  elapsed_hours: number
  solar_kw: number
  battery_soc: number
  ch4_kg: number
  delivered_kg: number
  net_value_eur: number
}
