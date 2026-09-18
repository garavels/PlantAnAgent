import assert from "node:assert/strict";
import { test } from "node:test";
import { isEuropeanSite, isUnavailableStudy, projectSite, unprojectSite, siteApi, SiteApiError, studyDays } from "../src/workspace/siteApi.ts";

test("expired jobs retain their HTTP status so polling stops while transient failures remain retryable", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Study no longer available" }), { status: 404 });
    await assert.rejects(siteApi.job("expired-study"), (error) => {
      assert.ok(error instanceof SiteApiError);
      assert.equal(error.status, 404);
      assert.equal(isUnavailableStudy(error), true);
      return true;
    });
    globalThis.fetch = async () => new Response("Temporary gateway failure", { status: 502 });
    await assert.rejects(siteApi.job("still-running"), (error) => {
      assert.ok(error instanceof SiteApiError);
      assert.equal(error.status, 502);
      assert.equal(isUnavailableStudy(error), false);
      return true;
    });
    assert.equal(isUnavailableStudy(new TypeError("Failed to fetch")), false);
  } finally { globalThis.fetch = original; }
});

test("the study displays ten 24-hour intervals while preserving extra planner lookahead", () => {
  const hours = Array.from({ length: 264 }, (_, i) => ({ time: new Date(Date.UTC(2026, 8, 14, 23 + i)).toISOString(), shortwave_w_m2: i }));
  const days = studyDays(hours);
  assert.equal(days.length, 10);
  assert.ok(days.every((day) => day.length === 24));
  assert.equal(days[0][0].time, "2026-09-14T23:00:00.000Z");
  assert.equal(days.at(-1).at(-1).time, "2026-09-24T22:00:00.000Z");
  assert.equal(hours.length, 264);
});

test("map coordinates place the region corners and round-trip European sites", () => {
  assert.deepEqual(projectSite(72, -25), { x: 0, y: 0 });
  assert.deepEqual(projectSite(34, 45), { x: 720, y: 608 });
  for (const [latitude, longitude] of [[37.3891, -5.9845], [45.6495, 13.7768], [59.3293, 18.0686]]) {
    const point = projectSite(latitude, longitude);
    assert.deepEqual(unprojectSite(point.x, point.y), { latitude, longitude });
  }
});

test("map clicks clamp to supported bounds and coordinate validation rejects non-finite values", () => {
  assert.deepEqual(unprojectSite(-10, -10), { latitude: 72, longitude: -25 });
  assert.deepEqual(unprojectSite(1000, 1000), { latitude: 34, longitude: 45 });
  for (const [latitude, longitude] of [[NaN, 10], [45, Infinity], [33.9, 10], [72.1, 10], [45, -25.1], [45, 45.1]]) assert.equal(isEuropeanSite(latitude, longitude), false);
  assert.equal(isEuropeanSite(34, -25), true);
  assert.equal(isEuropeanSite(72, 45), true);
});

test("forecast requests preserve selected coordinates and cancellation signals", async () => {
  const original = globalThis.fetch;
  const abort = new AbortController();
  const response = { id: "forecast-test" };
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(url, "/api/sites/forecast");
      assert.equal(init.signal, abort.signal);
      assert.equal(init.method, "POST");
      assert.deepEqual(JSON.parse(init.body), { latitude: 45.6495, longitude: 13.7768, refresh: true });
      return new Response(JSON.stringify(response), { headers: { "content-type": "application/json" } });
    };
    assert.deepEqual(await siteApi.forecast({ latitude: 45.6495, longitude: 13.7768, refresh: true }, abort.signal), response);
  } finally { globalThis.fetch = original; }
});

test("service errors expose a usable reason and non-JSON failures retain HTTP status", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Live weather unavailable; choose a saved example." }), { status: 503 });
    await assert.rejects(siteApi.examples(), /Live weather unavailable; choose a saved example/);
    globalThis.fetch = async () => new Response("Gateway unavailable", { status: 502 });
    await assert.rejects(siteApi.examples(), /502/);
  } finally { globalThis.fetch = original; }
});
