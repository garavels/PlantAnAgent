import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";
import { siteSunPosition } from "../src/components/plant-3d/siteSun.ts";

// Exercise the actual pure adapter with its existing extensionless TS imports.
function compile(file) {
  const source = readFileSync(file, "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
  }).outputText.replace(/from ['"]([^'"]+)['"]/g, (_match, name) => {
    assert.ok(name.startsWith("."));
    return `from "${compile(new URL(`${name}.ts`, file))}"`;
  });
  return `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`;
}
const { deriveVisuals } = await import(compile(new URL("../src/components/plant-3d/visuals.ts", import.meta.url)));
const DEG = 180 / Math.PI;
const stockholm = { name: "Stockholm", latitude: 59.3293, longitude: 18.0686, elevation_m: 24, timezone: "UTC" };

const state = {
  time_of_day_h: 0, plant_status: "normal", health: {},
  solar_available_kw: 850, solar_potential_kw: 1700,
  solar_to_battery_kw: 200, solar_to_electrolyser_kw: 400,
  solar_to_dac_kw: 150, solar_to_methanator_kw: 50, solar_curtailed_kw: 50,
  battery_soc: 0.5, battery_energy_kwh: 500, battery_capacity_kwh: 1000,
  battery_power_kw: 200, battery_to_electrolyser_kw: 0,
  electrolyser_online: true, electrolyser_power_kw: 400, electrolyser_load_frac: 0.4,
  electrolyser_temperature_c: 60, electrolyser_efficiency: 1, h2_production_kg_h: 8,
  dac_online: true, dac_power_kw: 150, dac_load_frac: 0.15, co2_capture_kg_h: 75,
  methanator_online: true, methanator_power_kw: 50, methanator_load_frac: 0.5, ch4_production_kg_h: 40,
  h2_level_kg: 50, h2_capacity_kg: 100, h2_fill: 0.5,
  co2_level_kg: 50, co2_capacity_kg: 100, co2_fill: 0.5,
  ch4_level_kg: 50, ch4_capacity_kg: 100, ch4_fill: 0.5,
};
const config = {
  solar: { peak_kw: 5000, sunrise_h: 6, sunset_h: 18 },
  battery: { capacity_kwh: 1000, max_charge_kw: 500, max_discharge_kw: 500 },
  electrolyser: { rated_kw: 1000, specific_energy_kwh_per_kg: 50 },
  dac: { rated_kw: 1000, specific_energy_kwh_per_kg: 2 },
  methanator: { rated_kw: 100, specific_energy_kwh_per_kg: 1.2, h2_kg_per_kg_ch4: 0.5, co2_kg_per_kg_ch4: 2.75 },
};
const weather = (time) => ({
  reference_kind: "forecast_pv", status: "observed", clear_sky_kw: 1700,
  available_kw: 850, attenuation: 0.5, sample_end_hours: 0,
  site_sample: { time, interval_start: time, shortwave_w_m2: 400, forecast_solar_kw: 1700 },
});

test("sun geometry follows explicit UTC offsets and never the browser timezone", () => {
  assert.deepEqual(
    siteSunPosition(45, 13, "2026-06-21T12:00:00Z"),
    siteSunPosition(45, 13, "2026-06-21T14:00:00+02:00"),
  );
  assert.equal(siteSunPosition(45, 13, "2026-06-21T12:00:00"), null);
  assert.equal(siteSunPosition(45, 13, "invalid"), null);
  assert.equal(siteSunPosition(NaN, 13, "2026-06-21T12:00:00Z"), null);
});

test("latitude, season and longitude set daylight instead of fixed 06:00 dawn", () => {
  const summer = siteSunPosition(stockholm.latitude, stockholm.longitude, "2026-06-21T11:00:00Z");
  const winter = siteSunPosition(stockholm.latitude, stockholm.longitude, "2026-12-21T11:00:00Z");
  assert.ok(summer.altitude * DEG > 50);
  assert.ok(winter.altitude * DEG > 5 && winter.altitude * DEG < 10);
  assert.ok(siteSunPosition(stockholm.latitude, stockholm.longitude, "2026-06-21T04:00:00Z").altitude > 0);
  assert.ok(siteSunPosition(stockholm.latitude, stockholm.longitude, "2026-12-21T04:00:00Z").altitude < 0);
  const east = siteSunPosition(45, 30, "2026-03-20T12:00:00Z");
  const west = siteSunPosition(45, -15, "2026-03-20T12:00:00Z");
  assert.ok(east.direction[0] < 0); // afternoon sun is west of the observer
  assert.ok(west.direction[0] > 0); // morning sun is east of the observer
});

test("equinoctial noon and leap-year polar daylight produce finite normalized directions", () => {
  assert.ok(siteSunPosition(0, 0, "2026-03-20T12:00:00Z").altitude * DEG > 87);
  for (const date of ["2024-02-29T12:00:00Z", "2024-06-21T00:00:00Z", "2024-12-21T00:00:00Z"]) {
    const result = siteSunPosition(71, 25, date);
    assert.ok(Math.abs(Math.hypot(...result.direction) - 1) < 1e-12);
    assert.ok(Number.isFinite(result.azimuth));
  }
  assert.ok(siteSunPosition(71, 25, "2024-06-21T00:00:00Z").altitude > 0);
  assert.ok(siteSunPosition(71, 25, "2024-12-21T00:00:00Z").altitude < 0);
});

test("scene uses the recorded site timestamp and preserves actual solar power and flow", () => {
  const visuals = deriveVisuals(state, config, weather("2026-06-21T04:00:00Z"), stockholm);
  assert.equal(visuals.sun.siteMode, true);
  assert.equal(visuals.sun.discVisible, true); // PlantState hour0 must not override the site sample
  assert.ok(visuals.sun.elevation > 0);
  assert.equal(visuals.solar.kw, 850);
  assert.equal(visuals.solar.potentialKw, 1700);
  assert.equal(visuals.solar.referenceKind, "forecast_pv");
  assert.equal(visuals.solar.frac, 850 / 5000);
  assert.equal(visuals.sun.directFactor, 850 / 5000);
  assert.equal(visuals.flows.solarToHub.value, 850);
  assert.equal(visuals.flows.hubToStack.value, 400);
});

test("night, zero radiation and missing timestamps never produce a synthetic daytime sun", () => {
  const night = deriveVisuals({ ...state, time_of_day_h: 12, solar_available_kw: 0 }, config, weather("2026-12-21T00:00:00Z"), stockholm);
  assert.equal(night.sun.elevation, 0);
  assert.equal(night.sun.discVisible, false);
  assert.equal(night.sun.dayFactor, 0);
  assert.equal(night.sun.directFactor, 0);
  const darkDay = deriveVisuals({ ...state, solar_available_kw: 0 }, config, weather("2026-06-21T11:00:00Z"), stockholm);
  assert.ok(darkDay.sun.elevation > 0);
  assert.equal(darkDay.sun.discVisible, false);
  const unknown = deriveVisuals(state, config, weather(undefined), stockholm);
  assert.equal(unknown.sun.discVisible, false);
  assert.equal(unknown.sun.direction, undefined);
});

test("ordinary synthetic runs keep their configured solar curve", () => {
  const visuals = deriveVisuals({ ...state, time_of_day_h: 12 }, config);
  assert.equal(visuals.sun.elevation, 1);
  assert.equal(visuals.sun.phase, 0.5);
  assert.equal(visuals.sun.siteMode, undefined);
  assert.equal(visuals.solar.referenceKind, "clear_sky");
});
