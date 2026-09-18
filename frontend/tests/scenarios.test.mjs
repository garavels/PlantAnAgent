import assert from "node:assert/strict";
import { test } from "node:test";
import { activateScenarioWorkspace, scenarioApi, ScenarioApiError } from "../src/workspace/scenarioApi.ts";
import { DEFAULT_SCENARIO, SCENARIO_EXAMPLES, recurringDeliveries, resizeScenario, scenarioDays, scenarioIssues, scenarioOffset } from "../src/workspace/scenarioModel.ts";

test("prepared scenarios work without generation and keep all deadlines within their horizon", () => {
  for (const scenario of [DEFAULT_SCENARIO, ...SCENARIO_EXAMPLES.map((example) => example.scenario)]) {
    assert.deepEqual(scenarioIssues(scenario), []);
    assert.equal(scenario.site.source, "saved");
    assert.ok(scenario.site.example_id);
    assert.ok(scenario.deliveries.every((delivery) => delivery.due_hour <= scenario.duration_days * 24));
  }
});

test("recurring deliveries include a final deadline at the run boundary, without exceeding the schedule limit", () => {
  assert.deepEqual(recurringDeliveries(1, 6, 6, 150), [6, 12, 18, 24].map((due_hour) => ({ due_hour, kg: 150 })));
  assert.equal(recurringDeliveries(7, 18, 24, 200).length, 7);
  assert.deepEqual(recurringDeliveries(10, 0.25, 0.25, 150), []);
  for (const args of [[1, 25, 6, 150], [1, 0, 6, 150], [1, 6, 0, 150], [1, 6, 6, 0], [NaN, 6, 6, 150]]) {
    assert.deepEqual(recurringDeliveries(...args), []);
  }
});

test("shortening a scenario removes future events and trims ongoing intervals without mutating the original", () => {
  const draft = structuredClone(DEFAULT_SCENARIO);
  draft.sunlight = [{ start_hour: 12, end_hour: 48, scale: 0.3, forecast_known: true }, { start_hour: 72, end_hour: 96, scale: 0.5, forecast_known: false }];
  draft.faults = [{ type: "dac_outage", severity: 1, start_hour: 18, duration_hours: 12, channel: null }, { type: "dac_outage", severity: 1, start_hour: 24, duration_hours: null, channel: null }];
  const shortened = resizeScenario(draft, 1);
  assert.deepEqual(shortened.sunlight, [{ start_hour: 12, end_hour: 24, scale: 0.3, forecast_known: true }]);
  assert.equal(shortened.faults.length, 1);
  assert.equal(shortened.faults[0].duration_hours, 6);
  assert.equal(shortened.deliveries.at(-1).due_hour, 24);
  assert.deepEqual(scenarioIssues(shortened), []);
  assert.equal(draft.sunlight[0].end_hour, 48);
  assert.equal(draft.faults.length, 2);
});

test("timeline assigns midnight deliveries to the completed day and keeps boundaries precise", () => {
  const draft = resizeScenario(structuredClone(DEFAULT_SCENARIO), 2);
  draft.sunlight = [{ start_hour: 24, end_hour: 48, scale: 0.3, forecast_known: true }];
  draft.faults = [{ type: "dac_outage", severity: 1, start_hour: 20, duration_hours: 8, channel: null }];
  const days = scenarioDays(draft);
  assert.equal(days.length, 2);
  assert.deepEqual(days.map((day) => day.deliveries.length), [4, 4]);
  assert.equal(days[0].deliveries.at(-1).due_hour, 24);
  assert.equal(days[0].sunlightScale, 1);
  assert.ok(Math.abs(days[1].sunlightScale - 0.3) < 1e-9);
  assert.deepEqual(days.map((day) => day.faults.length), [1, 1]);
  assert.equal(scenarioOffset(24.25), "Day 2, 00:15");
  assert.equal(scenarioOffset(23.75), "Day 1, 23:45");
});

test("preview validation identifies overlapping sunlight, duplicate deadlines and unsupported requests", () => {
  const draft = structuredClone(DEFAULT_SCENARIO);
  draft.sunlight = [{ start_hour: 0, end_hour: 25, scale: 0.5, forecast_known: true }, { start_hour: 24, end_hour: 48, scale: 0.5, forecast_known: true }];
  draft.deliveries.push({ due_hour: 6, kg: 10 });
  draft.unsupported = ["Freeze the electrolyser"];
  draft.economics.unmet_delivery_penalty_eur_per_kg = 1;
  const issues = scenarioIssues(draft);
  assert.ok(issues.some((issue) => issue.includes("overlap")));
  assert.ok(issues.some((issue) => issue.includes("own deadline")));
  assert.ok(issues.some((issue) => issue.includes("unsupported")));
  assert.ok(issues.some((issue) => issue.includes("exceed")));
});

test("generation and preview preserve the exact operator input and cancellation signal", async () => {
  const original = globalThis.fetch;
  const calls = [];
  const abort = new AbortController();
  try {
    globalThis.fetch = async (url, init) => {
      calls.push({ url, init });
      return new Response(JSON.stringify({ scenario: DEFAULT_SCENARIO, warnings: [] }));
    };
    await scenarioApi.generate("Make day two cloudy", abort.signal);
    await scenarioApi.preview(DEFAULT_SCENARIO, abort.signal);
    assert.deepEqual(calls.map((call) => call.url), ["/api/scenario-studio/generate", "/api/scenario-studio/preview"]);
    assert.deepEqual(JSON.parse(calls[0].init.body), { prompt: "Make day two cloudy" });
    assert.deepEqual(JSON.parse(calls[1].init.body), { scenario: DEFAULT_SCENARIO });
    assert.ok(calls.every((call) => call.init.signal === abort.signal && call.init.method === "POST"));
  } finally { globalThis.fetch = original; }
});

test("activation sends only the reviewed preview identity and controller, and validates the active twin response", async () => {
  const original = globalThis.fetch;
  const snapshot = { schema: "plant-v2-demo-v1", mode: "site_forecast", run_id: "scenario-run", revision: 0, frame: {}, history: [] };
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(url, "/api/scenario-studio/activate");
      assert.deepEqual(JSON.parse(init.body), { preview_id: "reviewed-preview", controller: "H" });
      return new Response(JSON.stringify(snapshot));
    };
    assert.deepEqual(await activateScenarioWorkspace("reviewed-preview", "H"), snapshot);
    globalThis.fetch = async () => new Response(JSON.stringify({ ...snapshot, mode: "interactive_fixture" }));
    await assert.rejects(activateScenarioWorkspace("reviewed-preview", "H"), /unsupported simulation/);
  } finally { globalThis.fetch = original; }
});

test("validation failures identify the edited field and expired previews retain their status", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: [{ loc: ["body", "scenario", "sunlight", 0, "end_hour"], msg: "Must be later than start" }] }), { status: 422 });
    await assert.rejects(scenarioApi.preview(DEFAULT_SCENARIO), /sunlight \/ 0 \/ end_hour: Must be later than start/);
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Preview expired. Check the scenario again." }), { status: 404 });
    await assert.rejects(activateScenarioWorkspace("expired", "MPC"), (error) => error instanceof ScenarioApiError && error.status === 404 && error.message.includes("Preview expired"));
    globalThis.fetch = async () => new Response("Bad gateway", { status: 502 });
    await assert.rejects(scenarioApi.capabilities(), /502/);
  } finally { globalThis.fetch = original; }
});
