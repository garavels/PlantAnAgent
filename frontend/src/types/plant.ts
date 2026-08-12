/** Mirror of `plant_sim.state` / `plant_sim.config`. Keep in sync by hand for now. */

export interface PlantAction {
  battery_share: number
  electrolyser_share: number
  dac_share: number
  methanator_share: number
  battery_support_kw: number
}

export type PlantStatus = 'normal' | 'degraded' | 'critical' | 'offline'
export type UnitStatus = 'nominal' | 'degraded' | 'offline'
export type EventLevel = 'info' | 'warning' | 'critical'

export interface ComponentHealth {
  status: UnitStatus
  degradation: number
  offline: boolean
}

/** One thing the simulator noticed, in its own words. */
export interface PlantEvent {
  seq: number
  time_s: number
  clock: string
  level: EventLevel
  code: string
  message: string
}

/** A fault or disturbance the plant has been given. */
export interface PlantFault {
  id: string
  type: string
  label: string
  target: string
  mode: 'degradation' | 'outage' | 'sensor'
  severity: number
  channel: string | null
  start_s: number
  end_s: number | null
  duration_s: number | null
  active: boolean
  cleared: boolean
}

/** An injectable fault type, as advertised by the simulator. */
export interface FaultKind {
  type: string
  label: string
  target: string
  mode: 'degradation' | 'outage' | 'sensor'
  description: string
  default_severity: number
  default_channel: string | null
}

export interface ScenarioDef {
  id: string
  name: string
  description: string
  start_hour: number
  initial: { battery_soc: number | null; h2_kg: number | null; co2_kg: number | null; ch4_kg: number | null }
  faults: {
    type: string
    severity: number
    start_s: number | null
    duration_s: number | null
    channel: string | null
  }[]
}

export type ControlMode = 'auto' | 'manual'

export interface ControllerInfo {
  id: string
  label: string
  description: string
}

/** Mirror of `controllers.Constraints`: the envelope the backend enforces. */
export interface Constraints {
  electrolyser_max_kw: number | null
  dac_max_kw: number | null
  methanator_max_kw: number | null
  battery_support_max_kw: number | null
  battery_min_soc: number
}

/** How the plant is being driven right now, straight from the backend. */
export interface ControlStatus {
  mode: ControlMode
  controller: ControllerInfo
  controllers: ControllerInfo[]
  constraints: Constraints
  manual_action: PlantAction
  applied_action: PlantAction
  /** Why the last action looked the way it did. */
  notes: string[]
  /** What the backend had to clamp before applying it. */
  adjustments: string[]
  playing: boolean
  speed: number
  tick_ms: number
}

export interface PlantState {
  step: number
  time_s: number

  solar_available_kw: number
  solar_potential_kw: number
  solar_output_factor: number
  solar_to_battery_kw: number
  solar_to_electrolyser_kw: number
  solar_to_dac_kw: number
  solar_to_methanator_kw: number
  solar_curtailed_kw: number

  battery_soc: number
  battery_energy_kwh: number
  battery_power_kw: number
  battery_to_electrolyser_kw: number
  battery_capacity_kwh: number

  electrolyser_power_kw: number
  electrolyser_load_frac: number
  electrolyser_online: boolean
  h2_production_kg_h: number
  electrolyser_efficiency: number
  electrolyser_temperature_c: number

  h2_level_kg: number
  h2_capacity_kg: number
  h2_fill: number

  dac_power_kw: number
  dac_load_frac: number
  dac_online: boolean
  co2_capture_kg_h: number

  co2_level_kg: number
  co2_capacity_kg: number
  co2_fill: number

  methanator_power_kw: number
  methanator_load_frac: number
  methanator_online: boolean
  ch4_production_kg_h: number

  ch4_level_kg: number
  ch4_capacity_kg: number
  ch4_fill: number

  energy_produced_kwh: number
  energy_curtailed_kwh: number
  h2_produced_total_kg: number
  co2_captured_total_kg: number
  ch4_produced_total_kg: number

  plant_status: PlantStatus
  health: Record<string, ComponentHealth>

  action: PlantAction

  // derived, added by PlantState.to_dict()
  time_h: number
  time_of_day_h: number
  day: number
  clock: string
}

export interface PlantConfig {
  dt_s: number
  dt_h: number
  steps_per_day: number
  electrolyser_min_load_kw: number
  solar: { peak_kw: number; sunrise_h: number; sunset_h: number }
  battery: {
    capacity_kwh: number
    max_charge_kw: number
    max_discharge_kw: number
    charge_efficiency: number
    discharge_efficiency: number
    min_soc: number
    initial_soc: number
  }
  electrolyser: { rated_kw: number; min_load_frac: number; specific_energy_kwh_per_kg: number }
  h2_tank: { capacity_kg: number; initial_kg: number }
  dac: { rated_kw: number; specific_energy_kwh_per_kg: number }
  co2_tank: { capacity_kg: number; initial_kg: number }
  methanator: {
    rated_kw: number
    specific_energy_kwh_per_kg: number
    h2_kg_per_kg_ch4: number
    co2_kg_per_kg_ch4: number
  }
  ch4_tank: { capacity_kg: number; initial_kg: number }
}

/** One backend update: new frames plus everything that came with them. */
export interface PlantFrames {
  state: PlantState
  frames: PlantState[]
  /** True when the frames are a new history, not a continuation of it. */
  replace?: boolean
  config?: PlantConfig
  events?: PlantEvent[]
  faults?: PlantFault[]
  scenario?: string | null
  control?: ControlStatus
}

export interface ScenarioCatalogue {
  scenarios: ScenarioDef[]
  scenario: string | null
}

export interface FaultCatalogue {
  catalogue: FaultKind[]
  faults: PlantFault[]
}
