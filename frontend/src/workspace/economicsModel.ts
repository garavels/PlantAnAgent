import type { V2DemoFrame } from "../types/v2";

export type EconomicPeriod = "run" | "day" | "interval";

export const ECONOMIC_LEDGER_IDS = {
  delivery: "delivered_service_value",
  penalty: "unmet_delivery_penalty",
  solar: "solar_opportunity_cost",
  batteryWear: "battery_throughput_reserve",
  electrolyserWear: "electrolyser_life_reserve",
  methaneInventory: "methane_endowment_liability_change",
  batteryInventory: "battery_endowment_liability_change",
} as const;

const ledgerLabels: Record<string, string> = {
  [ECONOMIC_LEDGER_IDS.delivery]: "Delivered methane value",
  [ECONOMIC_LEDGER_IDS.penalty]: "Missed-delivery penalties",
  [ECONOMIC_LEDGER_IDS.solar]: "Solar opportunity cost",
  [ECONOMIC_LEDGER_IDS.batteryWear]: "Battery wear reserve",
  [ECONOMIC_LEDGER_IDS.electrolyserWear]: "Electrolyser wear reserve",
  [ECONOMIC_LEDGER_IDS.methaneInventory]: "Gas inventory adjustment",
  [ECONOMIC_LEDGER_IDS.batteryInventory]: "Battery inventory adjustment",
};

/** Public runs retain one frame per 15-minute decision, including reset. */
export function periodStartIndex(
  index: number,
  period: EconomicPeriod,
): number {
  if (period === "run") return 0;
  return Math.max(0, index - (period === "day" ? 96 : 1));
}

export function numericMetric(frame: V2DemoFrame, key: string): number | null {
  const value = frame.metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Always compare physical stock with reset, independently of the selected reporting period. */
export function inventoryChangeSinceOpening(
  frame: V2DemoFrame,
  resource: V2DemoFrame["resources"][number],
  openingFrame?: V2DemoFrame | null,
): number | null {
  if (!Number.isFinite(resource.current)) return null;
  const reset = openingFrame?.clock.decision_index === 0 && openingFrame.clock.elapsed_hours === 0
    ? openingFrame
    : frame.clock.decision_index === 0 && frame.clock.elapsed_hours === 0 ? frame : null;
  const retained = reset?.resources.find((item) => item.id === resource.id && item.unit === resource.unit)?.current;
  if (typeof retained === "number" && Number.isFinite(retained)) return resource.current - retained;
  // Historical interactive recordings also retain reset stock in cumulative metrics.
  const keys: Record<string, string> = {
    battery: "initial_battery_energy_kwh", h2: "initial_h2_kg",
    co2: "initial_co2_kg", ch4: "initial_ch4_kg",
  };
  const opening = keys[resource.id] ? numericMetric(frame, keys[resource.id]) : null;
  return opening === null ? null : resource.current - opening;
}

export function ledgerValue(frame: V2DemoFrame, id: string): number | null {
  const value = frame.economics.entries.find(
    (entry) => entry.id === id,
  )?.value_eur;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricDelta(
  frame: V2DemoFrame,
  baseline: V2DemoFrame,
  key: string,
): number | null {
  const current = numericMetric(frame, key);
  const previous = numericMetric(baseline, key);
  return current === null || previous === null ? null : current - previous;
}

function hoursDelta(
  frame: V2DemoFrame,
  baseline: V2DemoFrame,
  key: string,
): number | null {
  const seconds = metricDelta(frame, baseline, key);
  return seconds === null ? null : seconds / 3600;
}

/** Only an absent contribution at reset is a known zero. */
function ledgerBoundaryValue(frame: V2DemoFrame, id: string): number | null {
  const value = ledgerValue(frame, id);
  if (value !== null) return value;
  return frame.clock.decision_index === 0 &&
    !frame.economics.entries.some((entry) => entry.id === id)
    ? 0
    : null;
}

function ledgerDelta(
  frame: V2DemoFrame,
  baseline: V2DemoFrame,
  id: string,
): number | null {
  const current = ledgerBoundaryValue(frame, id);
  const previous = ledgerBoundaryValue(baseline, id);
  return current === null || previous === null ? null : current - previous;
}

export function economicSummary(frame: V2DemoFrame, baseline: V2DemoFrame) {
  const settled = frame.delivery.commitments.filter(
    (commitment) =>
      commitment.due_hours > baseline.clock.elapsed_hours &&
      commitment.due_hours <= frame.clock.elapsed_hours &&
      commitment.status !== "pending" &&
      commitment.delivered_kg !== null &&
      commitment.unmet_kg !== null,
  );
  const requestedKg = settled.reduce(
    (sum, commitment) => sum + commitment.requested_kg,
    0,
  );
  const deliveredKg =
    frame.delivery.delivered_kg - baseline.delivery.delivered_kg;
  const solarKwh = metricDelta(frame, baseline, "solar_kwh");
  const curtailedKwh = metricDelta(frame, baseline, "curtailed_kwh");
  const ids = new Set([
    ...Object.values(ECONOMIC_LEDGER_IDS),
    ...frame.economics.entries.map((entry) => entry.id),
    ...baseline.economics.entries.map((entry) => entry.id),
  ]);
  const entries = [...ids].flatMap((id) => {
    const value = ledgerDelta(frame, baseline, id);
    return value === null
      ? []
      : [
          {
            id,
            label:
              ledgerLabels[id] ??
              frame.economics.entries.find((entry) => entry.id === id)?.label ??
              baseline.economics.entries.find((entry) => entry.id === id)
                ?.label ??
              id,
            value,
          },
        ];
  });
  const methaneAdjustment = ledgerDelta(
    frame,
    baseline,
    ECONOMIC_LEDGER_IDS.methaneInventory,
  );
  const batteryAdjustment = ledgerDelta(
    frame,
    baseline,
    ECONOMIC_LEDGER_IDS.batteryInventory,
  );

  return {
    netValue: frame.economics.net_value_eur - baseline.economics.net_value_eur,
    entries,
    deliveredKg,
    requestedKg,
    unmetKg: frame.delivery.unmet_kg - baseline.delivery.unmet_kg,
    fillRate: requestedKg > 0 ? deliveredKg / requestedKg : null,
    settledCount: settled.length,
    metCount: settled.filter((commitment) => commitment.status === "met")
      .length,
    producedKg: metricDelta(frame, baseline, "ch4_produced_kg"),
    acceptedSolarKwh:
      solarKwh === null || curtailedKwh === null
        ? null
        : Math.max(0, solarKwh - curtailedKwh),
    batteryThroughputKwh: metricDelta(
      frame,
      baseline,
      "battery_throughput_kwh",
    ),
    operatingHours: hoursDelta(
      frame,
      baseline,
      "electrolyser_operating_time_s",
    ),
    lowLoadHours: hoursDelta(frame, baseline, "electrolyser_low_load_time_s"),
    coldStarts: metricDelta(frame, baseline, "electrolyser_cold_starts"),
    hotStarts: metricDelta(frame, baseline, "electrolyser_hot_starts"),
    forcedStops: metricDelta(frame, baseline, "electrolyser_forced_stops"),
    controlledStops: metricDelta(
      frame,
      baseline,
      "electrolyser_controlled_stops",
    ),
    nonproductiveKwh: metricDelta(
      frame,
      baseline,
      "electrolyser_nonproductive_energy_kwh",
    ),
    inventoryAdjustment:
      methaneAdjustment === null || batteryAdjustment === null
        ? null
        : methaneAdjustment + batteryAdjustment,
  };
}
