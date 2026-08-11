/** The state adapter: PlantState in, visual descriptors out.
 *
 * This module is the ONLY place in the 3D view that knows simulator field
 * names. Every scene component receives these plain, normalised descriptors
 * and adds no physics of its own -- the same contract the single-line diagram
 * already honours.
 *
 * `intensity` values are clamped 0..1 against nameplate ratings from
 * `PlantConfig`, so meshes can treat them as simple multipliers for glow,
 * level and packet speed.
 */

import { clamp } from '../../lib/format'
import type { PlantConfig, PlantState } from '../../types/plant'

export interface FlowVisual {
  /** Current throughput, in the carrier's own unit (kW or kg/h). */
  value: number
  /** Throughput relative to its nameplate reference, 0..1. */
  intensity: number
  /** True when flow is above the noise floor. */
  live: boolean
}

export interface PlantVisuals {
  sun: {
    /** 0 at night, 1 at solar noon -- mirrors the simulator's half-sine. */
    elevation: number
    /** Smoothed daylight factor used for sky, lamps and exposure. */
    dayFactor: number
    /** Sun azimuth progress across the day, 0 at sunrise, 1 at sunset. */
    phase: number
  }
  solar: {
    kw: number
    /** vs peak array power, 0..1 */
    frac: number
    daylight: boolean
  }
  battery: {
    soc: number
    energyKwh: number
    capacityKwh: number
    /** signed: + charging, - discharging */
    powerKw: number
    charging: boolean
    discharging: boolean
  }
  electrolyser: {
    online: boolean
    powerKw: number
    ratedKw: number
    loadFrac: number
    h2kgH: number
  }
  tank: {
    kg: number
    capacityKg: number
    fill: number
    kgH: number
  }
  flows: {
    solarToHub: FlowVisual
    hubToBattery: FlowVisual
    hubToStack: FlowVisual
    batteryToStack: FlowVisual
    stackToTank: FlowVisual
    curtail: FlowVisual
  }
}

const FLOW_EPSILON = 0.5

function flow(value: number, reference: number): FlowVisual {
  return { value, intensity: clamp(value / Math.max(reference, 1)), live: value > FLOW_EPSILON }
}

export function deriveVisuals(state: PlantState, config: PlantConfig | null): PlantVisuals {
  const peakKw = config?.solar.peak_kw ?? 5000
  const ratedKw = config?.electrolyser.rated_kw ?? 3000
  const maxChargeKw = config?.battery.max_charge_kw ?? 2500
  const maxDischargeKw = config?.battery.max_discharge_kw ?? 2500
  const ratedKgH = ratedKw / (config?.electrolyser.specific_energy_kwh_per_kg ?? 52)

  const sunrise = config?.solar.sunrise_h ?? 6
  const sunset = config?.solar.sunset_h ?? 19.5
  const phase = clamp((state.time_of_day_h - sunrise) / (sunset - sunrise))
  // The simulator's solar curve is a half-sine between sunrise and sunset;
  // the twin's sun rides the same curve so light and generation never disagree.
  const elevation = Math.sin(Math.PI * phase)
  const dayFactor = clamp(state.solar_available_kw / (0.22 * peakKw))

  return {
    sun: { elevation, dayFactor, phase },
    solar: {
      kw: state.solar_available_kw,
      frac: clamp(state.solar_available_kw / peakKw),
      daylight: state.solar_available_kw > FLOW_EPSILON,
    },
    battery: {
      soc: clamp(state.battery_soc),
      energyKwh: state.battery_energy_kwh,
      capacityKwh: config?.battery.capacity_kwh ?? 0,
      powerKw: state.battery_power_kw,
      charging: state.battery_power_kw > FLOW_EPSILON,
      discharging: state.battery_power_kw < -FLOW_EPSILON,
    },
    electrolyser: {
      online: state.electrolyser_online,
      powerKw: state.electrolyser_power_kw,
      ratedKw,
      loadFrac: clamp(state.electrolyser_load_frac),
      h2kgH: state.h2_production_kg_h,
    },
    tank: {
      kg: state.h2_level_kg,
      capacityKg: state.h2_capacity_kg,
      fill: clamp(state.h2_fill),
      kgH: state.h2_production_kg_h,
    },
    flows: {
      solarToHub: flow(state.solar_available_kw, peakKw),
      hubToBattery: flow(state.solar_to_battery_kw, maxChargeKw),
      hubToStack: flow(state.solar_to_electrolyser_kw, ratedKw),
      batteryToStack: flow(state.battery_to_electrolyser_kw, maxDischargeKw),
      stackToTank: flow(state.h2_production_kg_h, ratedKgH),
      curtail: flow(state.solar_curtailed_kw, peakKw),
    },
  }
}

/** Frame-rate independent easing toward a target; the only smoothing the twin
 * applies. Visual lag is a presentation concern, so it lives here and in the
 * components -- never in the simulator. */
export const damp = (current: number, target: number, delta: number, lambda = 5) =>
  current + (target - current) * (1 - Math.exp(-lambda * delta))
