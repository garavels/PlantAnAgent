/** Mirror of `plant_sim.state` / `plant_sim.config`. Keep in sync by hand for now. */

export interface PlantAction {
  battery_share: number
  electrolyser_share: number
  battery_support_kw: number
}

export interface PlantState {
  step: number
  time_s: number

  solar_available_kw: number
  solar_to_battery_kw: number
  solar_to_electrolyser_kw: number
  solar_curtailed_kw: number

  battery_soc: number
  battery_energy_kwh: number
  battery_power_kw: number
  battery_to_electrolyser_kw: number

  electrolyser_power_kw: number
  electrolyser_load_frac: number
  electrolyser_online: boolean
  h2_production_kg_h: number

  h2_level_kg: number
  h2_capacity_kg: number
  h2_fill: number

  energy_produced_kwh: number
  energy_curtailed_kwh: number
  h2_produced_total_kg: number

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
}

export interface PlantFrames {
  state: PlantState
  frames: PlantState[]
  config?: PlantConfig
}
