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
  /** False for a controller the backend cannot select yet, e.g. RL with no checkpoint. */
  available?: boolean
  /** `ready` | `unavailable` | `failed`. */
  state?: string
  /** Why it is unavailable, shown next to the disabled option. */
  reason?: string
  /** V8: `unrated` | `good` | `suspect` | `poor`, from the controller health window. */
  grade?: string
}

/** V8: how well the driver is driving. Counted, never learned. */
export interface ControllerHealth {
  controller: string
  grade: 'unrated' | 'good' | 'suspect' | 'poor'
  steps: number
  summary: string
  concerns: string[]
  counts: Record<string, number>
  rates: Record<string, number>
}

/** V8: one logged handover, with the reason that justified it. */
export interface ControllerSwitch {
  time_s: number
  clock: string
  previous: string
  current: string
  /** e.g. `RL -> HEURISTIC`. */
  summary: string
  reason: string
  by: 'operator' | 'agent' | 'system'
  previous_grade: string
}

/** V8: the control hierarchy above the timestep -- who drives, how, how well. */
export interface ControlHierarchy {
  active_id: string
  preferred_id: string
  operating_state: 'normal' | 'degraded' | 'safe'
  health: ControllerHealth
  last_switch: ControllerSwitch | null
  switches: ControllerSwitch[]
  switch_count: number
  cooldown_s: number
  /** Envelope fields the supervisor currently owns. */
  agent_envelope: string[]
}

/** The RL checkpoint slot: whether a policy can drive, and why not when it cannot. */
export interface RLStatus {
  state: 'ready' | 'unavailable' | 'failed'
  available: boolean
  reason: string
  checkpoint: string | null
  metadata: {
    algorithm: string
    total_timesteps: number
    trained: boolean
    spec_version: number
  } | null
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
  /** V7: the RL checkpoint slot. Unavailable until a checkpoint is loaded. */
  rl?: RLStatus
  constraints: Constraints
  /** V8: the controller manager's view of itself. */
  hierarchy?: ControlHierarchy
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

// --- V6: the supervisory agent ---------------------------------------------
/** Mirror of `agent.trace.EntryKind`: the operational steps of one run. */
export type AgentEntryKind =
  | 'trigger'
  | 'observation'
  /** V10: what HEPA says may happen, as opposed to what the plant is doing. */
  | 'prediction'
  | 'tool'
  | 'diagnosis'
  | 'decision'
  | 'action'
  | 'result'
  | 'error'

export type AgentStatus = 'monitoring' | 'investigating' | 'attention' | 'disabled'
export type Confidence = 'low' | 'medium' | 'high'

/** One line of the supervisor's shift log. Never chain-of-thought. */
export interface AgentTraceEntry {
  seq: number
  run: number
  clock: string
  time_s: number
  kind: AgentEntryKind
  message: string
  detail: Record<string, unknown>
}

/** One situation the supervisor picked up, and what it made of it. */
export interface AgentIncident {
  id: string
  run: number
  opened_clock: string
  opened_time_s: number
  trigger: string
  trigger_detail: string
  key: string
  category: string
  component: string | null
  confidence: Confidence
  summary: string
  /** Envelope changes actually applied, already validated by the backend. */
  actions: string[]
  human_intervention_required: boolean
  resolved: boolean
  closed_clock: string | null
}

// --- V10: predicted event risk ---------------------------------------------
/** Mirror of `agent.risk`: how urgent a predicted event is, and where it is going. */
export type RiskLevel = 'low' | 'elevated' | 'high' | 'critical'
export type RiskTrend = 'stable' | 'rising' | 'falling'

/** Whether a forecast can be served at all. Only `active` carries numbers. */
export type RiskStatus = 'active' | 'unavailable' | 'error'

/** One event at one horizon: `P(event within Δt)`, what it means, how it moved. */
export interface EventRisk {
  event: string
  horizon: string
  probability: number
  level: RiskLevel
  trend: RiskTrend
  previous: number | null
  delta: number | null
}

/** Mirror of `RiskMonitor.snapshot()`. Empty of numbers unless status is active. */
export interface RiskSnapshot {
  status: RiskStatus
  available: boolean
  reason: string | null
  /** `checkpoint`, or `fixture:<name>` when the numbers are hand-written test data. */
  source: string
  updated_clock: string | null
  updated_time_s: number | null
  forecasts: number
  events: string[]
  horizons: string[]
  risks: EventRisk[]
  /** The risks that justify waking the supervisor. */
  attention: EventRisk[]
  refresh_s: number
  thresholds: Record<string, { elevated: number; high: number; critical: number }>
}

/** Mirror of `SupervisorAgent.snapshot()`. */
export interface AgentSnapshot {
  enabled: boolean
  status: AgentStatus
  busy: boolean
  /** Which supervisor answered: the model, or the rule book behind it. */
  engine: string
  runs: number
  last_run_clock: string | null
  last_error: string | null
  cooldown_s: number
  min_gap_s: number
  human_intervention_required: boolean
  incident: AgentIncident | null
  incidents: AgentIncident[]
  trace: AgentTraceEntry[]
  /** V10: what may happen, next to what happened. */
  risk: RiskSnapshot
}

export interface AgentTraceResponse {
  trace: AgentTraceEntry[]
  incidents: AgentIncident[]
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
  /** The supervisory agent, as of this payload. */
  agent?: AgentSnapshot
}

export interface ScenarioCatalogue {
  scenarios: ScenarioDef[]
  scenario: string | null
}

export interface FaultCatalogue {
  catalogue: FaultKind[]
  faults: PlantFault[]
}
