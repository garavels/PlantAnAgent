import type { ScenarioCapabilities, ScenarioDelivery, ScenarioExample, ScenarioSite, ScenarioSpec, SavedScenarioSite } from "./scenarioTypes";

export const SAVED_SCENARIO_SITES: Record<SavedScenarioSite, ScenarioSite> = {
  trieste: { name: "Trieste, Italy", latitude: 45.6495, longitude: 13.7768, source: "saved", example_id: "trieste" },
  seville: { name: "Seville, Spain", latitude: 37.3891, longitude: -5.9845, source: "saved", example_id: "seville" },
  stockholm: { name: "Stockholm, Sweden", latitude: 59.3293, longitude: 18.0686, source: "saved", example_id: "stockholm" },
};

export function recurringDeliveries(days: number, firstHour: number, interval: number, kg: number): ScenarioDelivery[] {
  const horizon = days * 24;
  if (![horizon, firstHour, interval, kg].every(Number.isFinite) || firstHour <= 0 || interval <= 0 || kg <= 0 || firstHour > horizon) return [];
  const count = Math.floor((horizon - firstHour + 1e-9) / interval) + 1;
  if (count > 80) return [];
  return Array.from({ length: count }, (_, i) => ({ due_hour: Math.round((firstHour + interval * i) * 100) / 100, kg }));
}

export const DEFAULT_SCENARIO: ScenarioSpec = {
  version: "plant-scenario-v1", name: "A week in Trieste", description: "A week of renewable methane production with regular deliveries.",
  duration_days: 7, seed: 42, site: SAVED_SCENARIO_SITES.trieste, solar_kw: 5000, battery_kwh: 8000,
  initial: { battery_soc: 0.5, h2_kg: 300, co2_kg: 1500, ch4_kg: 150 },
  sunlight: [], faults: [], deliveries: recurringDeliveries(7, 6, 6, 150),
  economics: { delivered_service_value_eur_per_kg: 4, unmet_delivery_penalty_eur_per_kg: 20 }, notes: [], unsupported: [],
};

const cloudyWeek: ScenarioSpec = {
  ...structuredClone(DEFAULT_SCENARIO), name: "A cloudy week in Trieste",
  description: "Low starting reserves, two cloudy days and a six-hour CO₂ capture outage test daily deliveries.",
  initial: { battery_soc: 0.3, h2_kg: 200, co2_kg: 3600, ch4_kg: 150 },
  sunlight: [{ start_hour: 24, end_hour: 72, scale: 0.3, forecast_known: true }],
  faults: [{ type: "dac_outage", severity: 1, start_hour: 84, duration_hours: 6, channel: null }],
  deliveries: recurringDeliveries(7, 19, 24, 150),
  notes: ["Days are 24-hour periods from the saved weather's start. Evening deliveries occur 19 hours into each period.", "Hydrogen starts at 10% of its 2,000 kg tank; CO₂ starts at 60% of its 6,000 kg tank."],
};

export const SCENARIO_EXAMPLES: ScenarioExample[] = [
  { title: "Cloudy week", prompt: "Simulate one week in Trieste. Start with battery at 30%, hydrogen at 10% and CO₂ at 60%. Give days two and three 70% less sunlight than forecast. Shut down CO₂ capture for six hours on day four. Deliver 150 kg of methane each evening.", scenario: cloudyWeek },
  { title: "Equipment outage", prompt: "Run a week in Seville with the usual inventories. Trip the methanator at hour 54 for twelve hours. Deliver 150 kg of methane every six hours.", scenario: { ...structuredClone(DEFAULT_SCENARIO), name: "Methanator outage in Seville", site: SAVED_SCENARIO_SITES.seville, description: "A twelve-hour reactor outage interrupts scheduled methane deliveries.", faults: [{ type: "methanator_outage", severity: 1, start_hour: 54, duration_hours: 12, channel: null }] } },
  { title: "Delivery pressure", prompt: "Run a week in Stockholm. Require 200 kg of methane every six hours. Value methane deliveries at €6 per kg and missed deliveries at €30 per kg.", scenario: { ...structuredClone(DEFAULT_SCENARIO), name: "Delivery pressure in Stockholm", site: SAVED_SCENARIO_SITES.stockholm, description: "More frequent demand tests production and inventory reserves.", deliveries: recurringDeliveries(7, 6, 6, 200), economics: { delivered_service_value_eur_per_kg: 6, unmet_delivery_penalty_eur_per_kg: 30 } } },
];

export const FALLBACK_FAULTS: ScenarioCapabilities["faults"] = [
  { type: "dac_outage", label: "CO₂ capture outage", default_severity: 1 },
  { type: "methanator_outage", label: "Methanator outage", default_severity: 1 },
  { type: "electrolyser_degradation", label: "Electrolyser degradation", default_severity: 0.4 },
  { type: "dac_degradation", label: "CO₂ capture degradation", default_severity: 0.4 },
  { type: "battery_degradation", label: "Battery capacity fade", default_severity: 0.3 },
  { type: "solar_drop", label: "Solar generation drop", default_severity: 0.6 },
  { type: "sensor_drift", label: "Sensor drift", default_severity: 0.17 },
];

export function faultLabel(type: string) {
  return FALLBACK_FAULTS.find((fault) => fault.type === type)?.label ?? type.replaceAll("_", " ");
}

export function scenarioOffset(hours: number) {
  if (!Number.isFinite(hours)) return "Enter a time";
  const minutes = Math.round(hours * 60);
  return `Day ${Math.floor(minutes / 1440) + 1}, ${String(Math.floor(minutes / 60) % 24).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
}

/** Changing the horizon keeps events in the run, preserving exact times where possible. */
export function resizeScenario(scenario: ScenarioSpec, durationDays: number): ScenarioSpec {
  const horizon = durationDays * 24;
  return {
    ...scenario, duration_days: durationDays,
    sunlight: scenario.sunlight.filter((period) => period.start_hour < horizon).map((period) => ({ ...period, end_hour: Math.min(period.end_hour, horizon) })),
    faults: scenario.faults.filter((fault) => fault.start_hour < horizon).map((fault) => ({ ...fault, duration_hours: fault.duration_hours === null ? null : Math.min(fault.duration_hours, horizon - fault.start_hour) })),
    deliveries: scenario.deliveries.filter((delivery) => delivery.due_hour <= horizon),
  };
}

export function scenarioIssues(scenario: ScenarioSpec): string[] {
  const issues: string[] = [];
  const horizon = scenario.duration_days * 24;
  if (!Number.isInteger(scenario.duration_days) || scenario.duration_days < 1 || scenario.duration_days > 10) issues.push("Choose a duration between 1 and 10 whole days.");
  if (scenario.unsupported.length) issues.push("Resolve the unsupported requests before checking this scenario.");
  const periods = [...scenario.sunlight].sort((a, b) => a.start_hour - b.start_hour);
  if (periods.some((period) => !Number.isInteger(period.start_hour) || !Number.isInteger(period.end_hour) || period.start_hour < 0 || period.end_hour > horizon || period.end_hour <= period.start_hour)) issues.push("Sunlight periods must use whole hours and end after they start, within the scenario.");
  if (periods.some((period, i) => i > 0 && period.start_hour < periods[i - 1].end_hour)) issues.push("Sunlight periods overlap. Choose separate time ranges.");
  if (periods.some((period) => !Number.isFinite(period.scale) || period.scale < 0 || period.scale > 1)) issues.push("Sunlight remaining must be between 0% and 100% of the forecast.");
  const quarterHour = (value: number) => Number.isFinite(value) && Number.isInteger(value * 4);
  if (scenario.faults.some((fault) => !quarterHour(fault.start_hour) || fault.start_hour < 0 || fault.start_hour >= horizon || (fault.duration_hours !== null && (!quarterHour(fault.duration_hours) || fault.duration_hours <= 0 || fault.start_hour + fault.duration_hours > horizon)))) issues.push("Faults must start within the scenario and use 15-minute intervals. Their duration cannot extend past its end.");
  if (scenario.deliveries.some((delivery) => !quarterHour(delivery.due_hour) || delivery.due_hour <= 0 || delivery.due_hour > horizon || !Number.isFinite(delivery.kg) || delivery.kg <= 0)) issues.push("Deliveries need a positive quantity and a deadline within the scenario, in 15-minute intervals.");
  if (new Set(scenario.deliveries.map((delivery) => delivery.due_hour)).size !== scenario.deliveries.length) issues.push("Each delivery needs its own deadline. Combine quantities due at the same hour.");
  if (scenario.economics.unmet_delivery_penalty_eur_per_kg <= scenario.economics.delivered_service_value_eur_per_kg) issues.push("Missed delivery cost must exceed the delivery value.");
  if (scenario.sunlight.length > 16 || scenario.faults.length > 12 || scenario.deliveries.length > 80) issues.push("Use at most 16 sunlight periods, 12 faults and 80 deliveries.");
  return issues;
}

export function scenarioDays(scenario: ScenarioSpec) {
  const days = Number.isInteger(scenario.duration_days) ? Math.max(1, Math.min(10, scenario.duration_days)) : 7;
  return Array.from({ length: days }, (_, day) => {
    const start = day * 24;
    const end = start + 24;
    const sunlight = scenario.sunlight.filter((period) => period.start_hour < end && period.end_hour > start);
    const faults = scenario.faults.filter((fault) => fault.start_hour < end && fault.start_hour + (fault.duration_hours ?? days * 24) > start);
    // A deadline at midnight belongs to the period that just finished.
    const deliveries = scenario.deliveries.filter((delivery) => delivery.due_hour > start && delivery.due_hour <= end);
    const reducedHours = sunlight.reduce((sum, period) => sum + (Math.min(period.end_hour, end) - Math.max(period.start_hour, start)) * (1 - period.scale), 0);
    return { day: day + 1, start, end, sunlight, faults, deliveries, sunlightScale: Math.max(0, Math.min(1, 1 - reducedHours / 24)) };
  });
}
