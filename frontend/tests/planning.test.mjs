import assert from "node:assert/strict";
import { test } from "node:test";
import { latestPlanDelivery, plannedInventoryAtFrame, retainedPlanTiming } from "../src/workspace/planningModel.ts";

function plan(issuedAt = 0, count = 96, intervalMinutes = 15) {
  return {
    issued_at_hours: issuedAt,
    points: Array.from({ length: count }, (_, index) => ({
      offset_hours: index * intervalMinutes / 60,
      battery_kwh: 1000 - index * 10,
      h2_kg: index,
      co2_kg: index * 2,
      ch4_kg: index * 3,
    })),
  };
}

function at(retainedPlan, elapsedHours) {
  return { plan: retainedPlan, clock: { elapsed_hours: elapsedHours } };
}

test("the first completed interval compares to its end inventory", () => {
  const retained = plan(3);
  assert.equal(plannedInventoryAtFrame(at(retained, 3.25), 15), retained.points[0]);
  assert.equal(plannedInventoryAtFrame(at(retained, 3), 15), null);
});

test("a retained site plan compares the matching interval instead of its first row", () => {
  const retained = plan(6);
  const timing = retainedPlanTiming(retained, 15);
  assert.equal(plannedInventoryAtFrame(at(retained, 8.5), 15), retained.points[9]);
  assert.equal(plannedInventoryAtFrame(at(retained, 9), 15), retained.points[11]);
  assert.deepEqual(timing.domain, [6, 30]);
  assert.equal(timing.durationHours, 24);
});

test("unmatched boundaries and frames outside the plan have no prediction", () => {
  const retained = plan(6, 4);
  for (const elapsed of [5.75, 6, 6.125, 7.25, Number.NaN]) {
    assert.equal(plannedInventoryAtFrame(at(retained, elapsed), 15), null);
  }
  assert.equal(plannedInventoryAtFrame(at(retained, 7), 15), retained.points[3]);
  assert.equal(plannedInventoryAtFrame(at(null, 7), 15), null);
  assert.equal(retainedPlanTiming(null, 15), null);
  assert.equal(retainedPlanTiming(plan(6, 0), 15), null);
});

test("the plot domain ends with a shortened mission plan", () => {
  const retained = plan(167, 4);
  assert.deepEqual(retainedPlanTiming(retained, 15), {
    intervalHours: 0.25,
    startHours: 167,
    endHours: 168,
    durationHours: 1,
    domain: [167, 168],
  });
});

test("timing follows the configured decision duration and each retained offset", () => {
  const retained = plan(2, 4, 10);
  retained.points = [retained.points[1], retained.points[3]];
  const timing = retainedPlanTiming(retained, 10);
  assert.equal(timing.intervalHours, 1 / 6);
  assert.equal(timing.startHours, 2 + 1 / 6);
  assert.equal(timing.endHours, 2 + 4 / 6);
  assert.equal(plannedInventoryAtFrame(at(retained, 2 + 2 / 6), 10), retained.points[0]);
  assert.equal(plannedInventoryAtFrame(at(retained, 2 + 3 / 6), 10), null);
  assert.equal(plannedInventoryAtFrame(at(retained, 2 + 4 / 6), 10), retained.points[1]);
});

test("invalid decision durations cannot create a prediction", () => {
  const retained = plan();
  for (const duration of [0, -15, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(retainedPlanTiming(retained, duration), null);
    assert.equal(plannedInventoryAtFrame(at(retained, 0.25), duration), null);
  }
});

function deliveryFrame() {
  return {
    plan: plan(0),
    clock: { elapsed_hours: 0.25 },
    delivery: { next_due_hours: 6, next_requested_kg: 240 },
    // Current methane and the total shortfall must not become the next prediction.
    state: { ch4_level_kg: 900 },
    decision: {
      controller_id: "MPC",
      at_hours: 0,
      operation: {
        policy_id: "operational-economic-mpc-v1",
        mode: "recovery",
        forecast_statistic: "issued_mean",
        restoration_policy: "operating_target",
        issued_at_hours: 0,
        horizon_hours: 24,
        projected_shortfall_kg: 70,
        fallback_controller: null,
        solver_failure: null,
        resources: [],
        deliveries: [
          { due_hours: 6, requested_kg: 240, predicted_kg: 190, shortfall_kg: 50 },
          { due_hours: 12, requested_kg: 240, predicted_kg: 220, shortfall_kg: 20 },
          { due_hours: 18, requested_kg: 240, predicted_kg: 240, shortfall_kg: 0 },
        ],
      },
    },
  };
}

test("next-delivery prediction uses its individual retained projection", () => {
  const frame = deliveryFrame();
  assert.deepEqual(latestPlanDelivery(frame), {
    delivery: frame.decision.operation.deliveries[0],
    issuedAtHours: 0,
  });
  frame.clock.elapsed_hours = 6.25;
  frame.delivery.next_due_hours = 12;
  assert.equal(latestPlanDelivery(frame).delivery, frame.decision.operation.deliveries[1]);
  frame.clock.elapsed_hours = 12.25;
  frame.delivery.next_due_hours = 18;
  assert.equal(latestPlanDelivery(frame).delivery.shortfall_kg, 0);
});

test("a cached site plan keeps its original issue time as execution advances", () => {
  const frame = deliveryFrame();
  frame.clock.elapsed_hours = 2.75;
  frame.decision.at_hours = 2.5;
  frame.decision.operation.policy_id = "site-forecast-operating-mpc-v1";
  frame.decision.operation.mode = "scheduled";
  frame.decision.operation.executing_at_hours = 2.5;
  assert.deepEqual(latestPlanDelivery(frame), {
    delivery: frame.decision.operation.deliveries[0],
    issuedAtHours: 0,
  });
});

test("delivery matching tolerates floating-point timestamps and quantities", () => {
  const frame = deliveryFrame();
  frame.decision.operation.issued_at_hours += 1e-10;
  frame.decision.operation.deliveries[0].due_hours += 1e-10;
  frame.decision.operation.deliveries[0].requested_kg += 1e-8;
  assert.equal(latestPlanDelivery(frame).delivery, frame.decision.operation.deliveries[0]);
});

test("missing, unsupported, fallback, and completed plans have no delivery prediction", () => {
  const cases = [
    (frame) => { frame.plan = null; },
    (frame) => { frame.plan.points = []; },
    (frame) => { frame.decision = null; },
    (frame) => { delete frame.decision.operation; },
    (frame) => { frame.decision.operation = null; },
    (frame) => { frame.decision.controller_id = "H"; },
    (frame) => { frame.decision.controller_id = "PPO"; },
    (frame) => { frame.decision.operation.mode = "fallback"; },
    (frame) => { frame.decision.operation.fallback_controller = "H"; },
    (frame) => { frame.decision.operation.solver_failure = { reason: "unavailable" }; },
    (frame) => { frame.decision.operation.forecast_statistic = "unknown"; },
    (frame) => { frame.decision.operation.restoration_policy = "unknown"; },
    (frame) => { frame.delivery.next_due_hours = null; },
    (frame) => { frame.clock.elapsed_hours = 6; },
    (frame) => { frame.clock.elapsed_hours = 24; },
    (frame) => { frame.decision.operation.deliveries = []; },
  ];
  for (const change of cases) {
    const frame = deliveryFrame();
    change(frame);
    assert.equal(latestPlanDelivery(frame), null, change.toString());
  }
});

test("mismatched or ambiguous retained delivery evidence is unavailable", () => {
  const cases = [
    (frame) => { frame.decision.operation.issued_at_hours = 0.25; },
    (frame) => { frame.delivery.next_due_hours = 7; },
    (frame) => { frame.delivery.next_requested_kg = 300; },
    (frame) => { frame.decision.operation.horizon_hours = 3; },
    (frame) => { frame.plan.issued_at_hours = frame.decision.operation.issued_at_hours = 1; },
    (frame) => { frame.decision.operation.deliveries.push({ ...frame.decision.operation.deliveries[0] }); },
    (frame) => { frame.decision.operation.deliveries[0].shortfall_kg = 70; },
    (frame) => { frame.decision.operation.deliveries[0].predicted_kg = 250; },
  ];
  for (const change of cases) {
    const frame = deliveryFrame();
    change(frame);
    assert.equal(latestPlanDelivery(frame), null, change.toString());
  }
});

test("nonfinite or negative inputs cannot create a delivery prediction", () => {
  const setters = [
    (frame, value) => { frame.clock.elapsed_hours = value; },
    (frame, value) => { frame.delivery.next_due_hours = value; },
    (frame, value) => { frame.delivery.next_requested_kg = value; },
    (frame, value) => { frame.plan.issued_at_hours = value; },
    (frame, value) => { frame.decision.operation.issued_at_hours = value; },
    (frame, value) => { frame.decision.operation.horizon_hours = value; },
    (frame, value) => { frame.decision.operation.deliveries[0].due_hours = value; },
    (frame, value) => { frame.decision.operation.deliveries[0].requested_kg = value; },
    (frame, value) => { frame.decision.operation.deliveries[0].predicted_kg = value; },
    (frame, value) => { frame.decision.operation.deliveries[0].shortfall_kg = value; },
  ];
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    for (const set of setters) {
      const frame = deliveryFrame();
      set(frame, value);
      assert.equal(latestPlanDelivery(frame), null, `${set.toString()}: ${value}`);
    }
  }
});
