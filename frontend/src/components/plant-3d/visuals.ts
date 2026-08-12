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
  dac: {
    online: boolean
    powerKw: number
    ratedKw: number
    loadFrac: number
    co2kgH: number
  }
  methanator: {
    online: boolean
    powerKw: number
    ratedKw: number
    loadFrac: number
    ch4kgH: number
    /** True when the reactor wants to run but a feedstock is empty. */
    starved: boolean
  }
  h2Tank: TankVisual
  co2Tank: TankVisual
  ch4Tank: TankVisual
  flows: {
    solarToHub: FlowVisual
    hubToBattery: FlowVisual
    hubToStack: FlowVisual
    batteryToStack: FlowVisual
    stackToTank: FlowVisual
    hubToDac: FlowVisual
    hubToMethanator: FlowVisual
    dacToCo2: FlowVisual
    h2ToMethanator: FlowVisual
    co2ToMethanator: FlowVisual
    methanatorToCh4: FlowVisual
    curtail: FlowVisual
  }
}

export interface TankVisual {
  kg: number
  capacityKg: number
  fill: number
  /** Signed net flow, kg/h: positive fills, negative drains. */
  netKgH: number
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
  const dacRatedKw = config?.dac.rated_kw ?? 1200
  const dacRatedKgH = dacRatedKw / (config?.dac.specific_energy_kwh_per_kg ?? 2)
  const metRatedKw = config?.methanator.rated_kw ?? 180
  const metRatedKgH = metRatedKw / (config?.methanator.specific_energy_kwh_per_kg ?? 1.2)
  // feedstock consumption follows the reactor recipe from the plant config
  const h2PerCh4 = config?.methanator.h2_kg_per_kg_ch4 ?? 0.5
  const co2PerCh4 = config?.methanator.co2_kg_per_kg_ch4 ?? 2.75
  const h2ToMetKgH = state.ch4_production_kg_h * h2PerCh4
  const co2ToMetKgH = state.ch4_production_kg_h * co2PerCh4

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
    dac: {
      online: state.dac_online,
      powerKw: state.dac_power_kw,
      ratedKw: dacRatedKw,
      loadFrac: clamp(state.dac_load_frac),
      co2kgH: state.co2_capture_kg_h,
    },
    methanator: {
      online: state.methanator_online,
      powerKw: state.methanator_power_kw,
      ratedKw: metRatedKw,
      loadFrac: clamp(state.methanator_load_frac),
      ch4kgH: state.ch4_production_kg_h,
      starved:
        !state.methanator_online && (state.h2_level_kg < 0.05 || state.co2_level_kg < 0.05),
    },
    h2Tank: {
      kg: state.h2_level_kg,
      capacityKg: state.h2_capacity_kg,
      fill: clamp(state.h2_fill),
      netKgH: state.h2_production_kg_h - h2ToMetKgH,
    },
    co2Tank: {
      kg: state.co2_level_kg,
      capacityKg: state.co2_capacity_kg,
      fill: clamp(state.co2_fill),
      netKgH: state.co2_capture_kg_h - co2ToMetKgH,
    },
    ch4Tank: {
      kg: state.ch4_level_kg,
      capacityKg: state.ch4_capacity_kg,
      fill: clamp(state.ch4_fill),
      netKgH: state.ch4_production_kg_h,
    },
    flows: {
      solarToHub: flow(state.solar_available_kw, peakKw),
      hubToBattery: flow(state.solar_to_battery_kw, maxChargeKw),
      hubToStack: flow(state.solar_to_electrolyser_kw, ratedKw),
      batteryToStack: flow(state.battery_to_electrolyser_kw, maxDischargeKw),
      stackToTank: flow(state.h2_production_kg_h, ratedKgH),
      hubToDac: flow(state.solar_to_dac_kw, dacRatedKw),
      hubToMethanator: flow(state.solar_to_methanator_kw, metRatedKw),
      dacToCo2: flow(state.co2_capture_kg_h, dacRatedKgH),
      h2ToMethanator: flow(h2ToMetKgH, ratedKgH),
      co2ToMethanator: flow(co2ToMetKgH, dacRatedKgH),
      methanatorToCh4: flow(state.ch4_production_kg_h, metRatedKgH),
      curtail: flow(state.solar_curtailed_kw, peakKw),
    },
  }
}

/** Frame-rate independent easing toward a target; the only smoothing the twin
 * applies. Visual lag is a presentation concern, so it lives here and in the
 * components -- never in the simulator. */
export const damp = (current: number, target: number, delta: number, lambda = 5) =>
  current + (target - current) * (1 - Math.exp(-lambda * delta))
