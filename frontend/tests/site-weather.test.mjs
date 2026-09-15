import assert from "node:assert/strict";
import { test } from "node:test";
import { cloudReadout, missionDays, scenarioForecastPower, utcWeatherTime, weatherReadout } from "../src/workspace/siteWeatherModel.ts";

test("cloud readouts distinguish clear skies, overcast and unavailable samples", () => {
  const clear = cloudReadout({ cloud_cover_pct: 0, cloud_cover_low_pct: 0, cloud_cover_mid_pct: null, cloud_cover_high_pct: 10 });
  assert.equal(clear.available, true);
  assert.equal(clear.total, "0%");
  assert.match(clear.caption, /Forecast cloud cover 0%/);
  assert.deepEqual(clear.layers.map(layer => layer.value), ["0%", "Unavailable", "10%"]);
  assert.equal(cloudReadout({ cloud_cover_pct: 100 }).total, "100%");
  for (const value of [undefined, null, NaN, Infinity, -1, 101, "70"]) {
    const unavailable = cloudReadout({ cloud_cover_pct: value });
    assert.equal(unavailable.available, false);
    assert.equal(unavailable.total, "Unavailable");
    assert.match(unavailable.caption, /Cloud data unavailable/);
  }
});

test("site runs describe the downloaded PV reference and reduced-sun scenario without clear-sky claims", () => {
  const weather = { reference_kind: "forecast_pv", available_kw: 700, clear_sky_kw: 1000, status: "unattributed", attenuation: null };
  assert.deepEqual(weatherReadout(weather), { heading: "30% below forecast PV", detail: "700 kW available / 1,000 kW forecast PV" });
  assert.equal(weatherReadout({ ...weather, available_kw: 1000 }).heading, "Solar follows the downloaded forecast");
  assert.doesNotMatch(JSON.stringify(weatherReadout(weather)), /clear-sky|cloud/);
});

test("zero forecast PV is handled without division by zero or a fabricated cloud attribution", () => {
  const readout = weatherReadout({ reference_kind: "forecast_pv", available_kw: 0, clear_sky_kw: 0, status: "night", attenuation: null });
  assert.equal(readout.heading, "No solar generation in the forecast");
  assert.doesNotMatch(JSON.stringify(readout), /NaN|Infinity|cloud/);
});

test("existing synthetic weather keeps its clear-sky attribution", () => {
  assert.equal(weatherReadout({ available_kw: 750, clear_sky_kw: 1000, status: "observed", attenuation: 0.25 }).heading, "25% below clear-sky potential");
  assert.equal(weatherReadout({ available_kw: 0, clear_sky_kw: 0, status: "night", attenuation: null }).detail, "0 kW available · cloud conditions unobserved");
});

test("mission day selectors follow the actual horizon and dates remain UTC", () => {
  assert.equal(missionDays(168), 7);
  assert.equal(missionDays(240), 10);
  assert.equal(missionDays(25), 2);
  assert.equal(utcWeatherTime("2026-09-14T23:00:00Z"), "14 Sept 2026, 23:00 UTC");
});

test("authored sunlight changes the forecast reference only when known and preserves hour boundaries", () => {
  const periods = [{ start_hour: 0, end_hour: 1, scale: 0.3, forecast_known: true }, { start_hour: 1, end_hour: 2, scale: 0.2, forecast_known: false }];
  assert.equal(scenarioForecastPower(1000, 0, periods), 300);
  assert.equal(scenarioForecastPower(1000, 1, periods), 300);
  assert.equal(scenarioForecastPower(1000, 1.25, periods), 1000);
  assert.deepEqual(weatherReadout({ reference_kind: "forecast_pv", available_kw: 300, clear_sky_kw: 1000, scenario_name: "Cloudy week", scenario_forecast_scale: 0.3, status: "observed", attenuation: null }), {
    heading: "Solar follows the scenario forecast", detail: "300 kW available / 300 kW forecast PV",
  });
});
