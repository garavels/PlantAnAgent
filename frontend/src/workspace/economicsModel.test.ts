/// <reference types="node" />
import assert from "node:assert/strict";
import { test } from "node:test";
import type { V2DemoFrame } from "../types/v2";
import {
  economicSummary,
  ledgerValue,
  numericMetric,
  periodStartIndex,
} from "./economicsModel.ts";

type SummaryFrame = Pick<
  V2DemoFrame,
  "clock" | "delivery" | "economics" | "metrics"
>;

function frame(overrides: Partial<SummaryFrame> = {}): V2DemoFrame {
  return {
    clock: {
      elapsed_hours: 0,
      decision_index: 0,
      decision_count: 672,
      cycle_index: 0,
      cycle_count: 7,
      hours_to_cycle_end: 24,
    },
    delivery: {
      next_due_hours: 6,
      next_requested_kg: 240,
      delivered_kg: 0,
      unmet_kg: 0,
      requested_kg: 0,
      commitments: [],
    },
    economics: {
      net_value_eur: 0,
      entries: [],
      assumption_note: "Illustrative",
    },
    metrics: {},
    ...overrides,
  } as V2DemoFrame;
}

function at(hours: number, overrides: Partial<SummaryFrame> = {}): V2DemoFrame {
  return frame({
    ...overrides,
    clock: {
      ...frame().clock,
      elapsed_hours: hours,
      decision_index: hours * 4,
    },
  });
}

function ledger(values: Record<string, number>): V2DemoFrame["economics"] {
  return {
    net_value_eur: Object.values(values).reduce((sum, value) => sum + value, 0),
    entries: Object.entries(values).map(([id, value_eur]) => ({
      id,
      value_eur,
      label: id,
    })),
    assumption_note: "Illustrative",
  };
}

test("reset and pre-delivery periods have no settled-delivery denominator", () => {
  const reset = frame();
  const early = at(2, {
    delivery: {
      ...reset.delivery,
      requested_kg: 6720,
      commitments: [
        {
          due_hours: 6,
          requested_kg: 240,
          delivered_kg: null,
          unmet_kg: null,
          status: "pending",
        },
      ],
    },
  });
  for (const current of [reset, early]) {
    const summary = economicSummary(current, reset);
    assert.equal(summary.requestedKg, 0);
    assert.equal(summary.fillRate, null);
    assert.equal(summary.settledCount, 0);
    assert.equal(summary.metCount, 0);
  }
});

test("delivery periods exclude their starting boundary and include settlements at their ending boundary", () => {
  const baseline = at(6, {
    delivery: { ...frame().delivery, delivered_kg: 200, unmet_kg: 40 },
  });
  const current = at(18, {
    delivery: {
      ...frame().delivery,
      delivered_kg: 590,
      unmet_kg: 130,
      requested_kg: 6720,
      commitments: [
        {
          due_hours: 6,
          requested_kg: 240,
          delivered_kg: 200,
          unmet_kg: 40,
          status: "shortfall",
        },
        {
          due_hours: 12,
          requested_kg: 240,
          delivered_kg: 240,
          unmet_kg: 0,
          status: "met",
        },
        {
          due_hours: 18,
          requested_kg: 240,
          delivered_kg: 150,
          unmet_kg: 90,
          status: "shortfall",
        },
        {
          due_hours: 18,
          requested_kg: 999,
          delivered_kg: null,
          unmet_kg: null,
          status: "pending",
        },
        {
          due_hours: 24,
          requested_kg: 240,
          delivered_kg: 240,
          unmet_kg: 0,
          status: "met",
        },
      ],
    },
  });
  const summary = economicSummary(current, baseline);
  assert.equal(summary.deliveredKg, 390);
  assert.equal(summary.unmetKg, 90);
  assert.equal(summary.requestedKg, 480);
  assert.equal(summary.fillRate, 390 / 480);
  assert.equal(summary.settledCount, 2);
  assert.equal(summary.metCount, 1);
});

test("all reported activity and ledger flows use cumulative differences", () => {
  const baseline = at(4, {
    economics: ledger({
      delivered_service_value: 800,
      solar_opportunity_cost: -100,
      methane_endowment_liability_change: -40,
      battery_endowment_liability_change: -5,
    }),
    metrics: {
      ch4_produced_kg: 10,
      solar_kwh: 200,
      curtailed_kwh: 50,
      battery_throughput_kwh: 1000,
      electrolyser_operating_time_s: 3600,
      electrolyser_low_load_time_s: 900,
      electrolyser_cold_starts: 1,
      electrolyser_hot_starts: 2,
      electrolyser_forced_stops: 3,
      electrolyser_controlled_stops: 4,
      electrolyser_nonproductive_energy_kwh: 5,
    },
  });
  const current = at(8, {
    economics: ledger({
      delivered_service_value: 1600,
      solar_opportunity_cost: -170,
      methane_endowment_liability_change: -90,
      battery_endowment_liability_change: -15,
    }),
    metrics: {
      ch4_produced_kg: 30,
      solar_kwh: 500,
      curtailed_kwh: 90,
      battery_throughput_kwh: 1300,
      electrolyser_operating_time_s: 9000,
      electrolyser_low_load_time_s: 2700,
      electrolyser_cold_starts: 2,
      electrolyser_hot_starts: 4,
      electrolyser_forced_stops: 4,
      electrolyser_controlled_stops: 5,
      electrolyser_nonproductive_energy_kwh: 9,
    },
  });
  const summary = economicSummary(current, baseline);
  assert.equal(summary.netValue, 670);
  assert.equal(
    summary.entries.reduce((sum, entry) => sum + entry.value, 0),
    670,
  );
  assert.equal(summary.inventoryAdjustment, -60);
  assert.equal(summary.producedKg, 20);
  assert.equal(summary.acceptedSolarKwh, 260);
  assert.equal(summary.batteryThroughputKwh, 300);
  assert.equal(summary.operatingHours, 1.5);
  assert.equal(summary.lowLoadHours, 0.5);
  assert.equal(summary.coldStarts, 1);
  assert.equal(summary.hotStarts, 2);
  assert.equal(summary.forcedStops, 1);
  assert.equal(summary.controlledStops, 1);
  assert.equal(summary.nonproductiveKwh, 4);
});

test("replenishment reverses inventory charges independently of battery throughput", () => {
  const depleted = at(4, {
    economics: ledger({
      methane_endowment_liability_change: -120,
      battery_endowment_liability_change: -30,
    }),
    metrics: { battery_throughput_kwh: 100 },
  });
  const restored = at(8, {
    economics: ledger({
      methane_endowment_liability_change: -20,
      battery_endowment_liability_change: 0,
    }),
    metrics: { battery_throughput_kwh: 500 },
  });
  const summary = economicSummary(restored, depleted);
  assert.equal(summary.inventoryAdjustment, 130);
  assert.equal(summary.batteryThroughputKwh, 400);
});

test("missing, boolean, object and nonfinite numeric metrics remain unavailable", () => {
  const invalid = at(2, {
    metrics: {
      ch4_produced_kg: Number.NaN,
      solar_kwh: 100,
      battery_throughput_kwh: Number.POSITIVE_INFINITY,
      electrolyser_cold_starts: true,
      electrolyser_hot_starts: { count: 1 },
      electrolyser_low_load_time_s: 0,
    },
    economics: ledger({ invalid: Number.NaN }),
  });
  assert.equal(numericMetric(invalid, "missing"), null);
  assert.equal(numericMetric(invalid, "ch4_produced_kg"), null);
  assert.equal(numericMetric(invalid, "electrolyser_cold_starts"), null);
  assert.equal(numericMetric(invalid, "electrolyser_hot_starts"), null);
  assert.equal(ledgerValue(invalid, "missing"), null);
  assert.equal(ledgerValue(invalid, "invalid"), null);
  const summary = economicSummary(invalid, frame());
  for (const key of [
    "producedKg",
    "acceptedSolarKwh",
    "batteryThroughputKwh",
    "operatingHours",
    "lowLoadHours",
    "coldStarts",
    "hotStarts",
    "forcedStops",
    "controlledStops",
    "nonproductiveKwh",
  ] as const) {
    assert.equal(summary[key], null, key);
  }
});

test("24-hour and interval windows clip to the reset frame", () => {
  assert.equal(periodStartIndex(0, "day"), 0);
  assert.equal(periodStartIndex(48, "day"), 0);
  assert.equal(periodStartIndex(96, "day"), 0);
  assert.equal(periodStartIndex(120, "day"), 24);
  assert.equal(periodStartIndex(120, "run"), 0);
  assert.equal(periodStartIndex(0, "interval"), 0);
  assert.equal(periodStartIndex(120, "interval"), 119);
});

test("absent reset contributions are known zero, while unavailable nonreset contributions are omitted", () => {
  const reset = frame();
  const resetSummary = economicSummary(reset, reset);
  assert.equal(resetSummary.entries.length, 7);
  assert.ok(resetSummary.entries.every((entry) => entry.value === 0));
  assert.equal(resetSummary.inventoryAdjustment, 0);

  const baseline = at(4, {
    economics: {
      ...ledger({
        delivered_service_value: 400,
        solar_opportunity_cost: -100,
        battery_throughput_reserve: -5,
        methane_endowment_liability_change: -40,
        battery_endowment_liability_change: -5,
        invalid_at_opening: Number.POSITIVE_INFINITY,
      }),
      net_value_eur: 250,
    },
  });
  const current = at(8, {
    economics: {
      ...ledger({
        delivered_service_value: 400,
        solar_opportunity_cost: Number.NaN,
        methane_endowment_liability_change: -30,
        battery_endowment_liability_change: Number.NEGATIVE_INFINITY,
        invalid_at_opening: 20,
        absent_at_opening: 40,
      }),
      net_value_eur: 350,
    },
  });
  const summary = economicSummary(current, baseline);
  assert.deepEqual(
    summary.entries.map(({ id, value }) => ({ id, value })),
    [
      { id: "delivered_service_value", value: 0 },
      { id: "methane_endowment_liability_change", value: 10 },
    ],
  );
  assert.equal(summary.netValue, 100);
  assert.equal(summary.inventoryAdjustment, null);
  assert.equal(
    economicSummary(baseline, reset).entries.find(
      (entry) => entry.id === "delivered_service_value",
    )?.value,
    400,
  );
  assert.deepEqual(economicSummary(at(2), reset).entries, []);
  assert.equal(economicSummary(at(2), reset).inventoryAdjustment, null);
  assert.ok(
    !economicSummary(
      baseline,
      frame({ economics: ledger({ delivered_service_value: Number.NaN }) }),
    ).entries.some((entry) => entry.id === "delivered_service_value"),
  );
});
