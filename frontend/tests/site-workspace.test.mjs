import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { activateSiteWorkspace, deactivateSiteWorkspace, fetchV2Demo, stepV2Demo } from "../src/lib/v2-api.ts";
import { checkRecording } from "../src/workspace/recording.ts";
import { siteTime, time } from "../src/workspace/format.ts";

const forecast = JSON.parse(readFileSync(new URL("../../backend/src/plant_backend/data/site_forecasts/trieste.json", import.meta.url), "utf8"));
const parameters = { solar_kw: 5000, battery_kwh: 8000, solar_scale: 0.7, inject_fault: true };
function recording(count = 961) {
  const frames = Array.from({ length: count }, (_, i) => {
    const hour = forecast.hours[Math.floor(Math.max(0, i * 900 - 60) / 3600)];
    return {
      clock: { decision_index: i, decision_count: 960, elapsed_hours: i / 4 },
      weather: { reference_kind: "forecast_pv", site_sample: { ...hour, interval_start: hour.time, time: new Date(Date.parse(forecast.start_time) + i * 900_000).toISOString() } },
      forecasts: { forecast_id: forecast.id },
    };
  });
  return { schema: "plant-site-recording-v1", frames, snapshot: {
    schema: "plant-v2-demo-v1", mode: "site_forecast", run_id: "site-run", revision: count - 1,
    site_forecast: forecast, site_parameters: parameters, controller: { id: "H" },
    environment: { id: "plant-site-forecast-v1", fixture_seed: null, horizon_hours: 240 },
    frame: frames.at(-1), history: [],
  } };
}

test("site playback admits ten days while preserving strict time and forecast identity", () => {
  const record = recording();
  assert.equal(checkRecording(record), record);
  for (const mutate of [
    r => { r.frames[5].clock.elapsed_hours = 1; },
    r => { r.frames[4].weather.site_sample.interval_start = r.snapshot.site_forecast.hours[1].time; },
    r => { r.frames[3].weather.site_sample.time = "2026-09-15T06:00:00Z"; },
    r => { r.frames[2].forecasts.forecast_id = "other"; },
    r => { r.snapshot.environment.fixture_seed = 9000034; },
    r => { r.schema = "plant-public-recording-v1"; },
  ]) {
    const copy = structuredClone(record); mutate(copy);
    assert.throws(() => checkRecording(copy));
  }
  assert.equal(checkRecording(recording(1)).frames.length, 1);
});

test("all site clocks use the downloaded UTC origin, including midnight and historical boundaries", () => {
  assert.equal(siteTime("2026-09-14T23:00:00Z", 0), "14 Sept, 23:00 UTC");
  assert.equal(siteTime("2026-09-14T23:00:00Z", 1.25), "15 Sept, 00:15 UTC");
  assert.equal(siteTime("2026-09-14T23:00:00Z", 240), "24 Sept, 23:00 UTC");
  assert.equal(time(0), "D1 06:00");
});

test("site recordings preserve zero versus missing clouds and reject shifted or altered weather", () => {
  const record = structuredClone(recording(1));
  record.snapshot.site_forecast.hours[0].cloud_cover_pct = 0;
  record.frames[0].weather.site_sample.cloud_cover_pct = 0;
  record.frames[0].weather.site_sample.conditions_time = new Date(Date.parse(record.snapshot.site_forecast.hours[0].time) + 3_600_000).toISOString();
  assert.equal(checkRecording(record), record);
  record.frames[0].weather.site_sample.cloud_cover_pct = null;
  assert.throws(() => checkRecording(record), /retained site forecast/);
  record.frames[0].weather.site_sample.cloud_cover_pct = 0;
  record.frames[0].weather.site_sample.conditions_time = record.snapshot.site_forecast.hours[0].time;
  assert.throws(() => checkRecording(record), /retained site forecast/);
});

test("site activation, refresh and step use the same live snapshot and preserve optimistic concurrency", async () => {
  const original = globalThis.fetch;
  const snapshot = recording(1).snapshot;
  const calls = [];
  const abort = new AbortController();
  try {
    globalThis.fetch = async (url, init) => {
      calls.push({ url, init });
      return new Response(JSON.stringify(snapshot), { status: 200 });
    };
    const config = { forecast_id: forecast.id, controller: "H", ...parameters };
    assert.deepEqual(await activateSiteWorkspace(config, abort.signal), snapshot);
    assert.deepEqual(await fetchV2Demo(), snapshot);
    await stepV2Demo(snapshot, 4);
    await deactivateSiteWorkspace();
    assert.deepEqual(calls.map(c => c.url), ["/api/sites/activate", "/api/v2/demo", "/api/v2/demo/step", "/api/sites/deactivate"]);
    assert.deepEqual(JSON.parse(calls[0].init.body), config);
    assert.equal(calls[0].init.signal, abort.signal);
    assert.deepEqual(JSON.parse(calls[2].init.body), { expected_run_id: "site-run", expected_revision: 0, decisions: 4 });
    globalThis.fetch = async () => new Response(JSON.stringify({ ...snapshot, mode: "recorded_research" }));
    await assert.rejects(fetchV2Demo(), /unsupported snapshot/);
  } finally { globalThis.fetch = original; }
});
