import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Render the real inspector and decision component without a browser dependency.
const require = createRequire(import.meta.url);
const modules = new Map();
function compile(file) {
  if (modules.has(file.href)) return modules.get(file.href);
  const source = readFileSync(file, "utf8");
  let compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  compiled = compiled.replace(/^import "[^"\n]+\.css";\n/gm, "");
  compiled = compiled.replace(/from "([^"]+)"/g, (_match, specifier) => {
    if (!specifier.startsWith(".")) return `from "${pathToFileURL(require.resolve(specifier))}"`;
    const dependency = [".ts", ".tsx"].map((extension) => new URL(`${specifier}${extension}`, file)).find(existsSync);
    assert.ok(dependency, `Cannot resolve ${specifier} from ${file}`);
    return `from "${compile(dependency)}"`;
  });
  const url = `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`;
  modules.set(file.href, url);
  return url;
}
const { EquipmentInspector } = await import(compile(new URL("../src/workspace/EquipmentInspector.tsx", import.meta.url)));
const { EquipmentDecision } = await import(compile(new URL("../src/workspace/EquipmentDecision.tsx", import.meta.url)));

const config = {
  dt_s: 60,
  solar: { peak_kw: 1000 },
  battery: { min_soc: 0.1, max_charge_kw: 150, max_discharge_kw: 200 },
  electrolyser: { rated_kw: 500, min_load_frac: 0.2 },
  dac: { rated_kw: 200 },
  // Deliberately different from the default recipe: the inspector must use configuration.
  methanator: { rated_kw: 100, h2_kg_per_kg_ch4: 0.75, co2_kg_per_kg_ch4: 3 },
};

function makeFrame() {
  return {
    state: {
      solar_available_kw: 420, solar_potential_kw: 600, solar_curtailed_kw: 10,
      solar_to_electrolyser_kw: 300, solar_to_dac_kw: 80, solar_to_methanator_kw: 20, solar_to_battery_kw: 10,
      battery_soc: 0.3, battery_energy_kwh: 300, battery_capacity_kwh: 1000,
      battery_power_kw: -30, battery_to_electrolyser_kw: 40,
      electrolyser_online: true, electrolyser_power_kw: 340, electrolyser_load_frac: 0.68,
      electrolyser_temperature_c: 72, electrolyser_efficiency: 0.92, h2_production_kg_h: 12,
      dac_online: true, dac_power_kw: 80, dac_load_frac: 0.4, co2_capture_kg_h: 80,
      methanator_online: true, methanator_power_kw: 20, methanator_load_frac: 0.2, ch4_production_kg_h: 20,
      h2_level_kg: 100, h2_capacity_kg: 500, co2_level_kg: 800, co2_capacity_kg: 1000,
      ch4_level_kg: 600, ch4_capacity_kg: 2000, health: {},
    },
    weather: { status: "observed", attenuation: 0.3 },
    commitment: {
      mode: "running", startup_remaining_s: 0, min_up_remaining_s: 0, min_down_remaining_s: 0, cooldown_remaining_s: 0,
      ramp_up_headroom_kw: 60, ramp_down_headroom_kw: 40, cold_starts: 2, hot_starts: 3,
      forced_stops: 1, aborted_starts: 4, startup_energy_kwh: 5.5,
    },
    commitment_profile: null,
    metrics: { battery_throughput_kwh: 125 },
    clock: { elapsed_hours: 2.25, hours_to_cycle_end: 21.75 },
    delivery: { next_due_hours: 4, next_requested_kg: 1000 },
    resources: [
      { id: "h2", unit: "kg", required: 120, margin: -20 },
      { id: "co2", unit: "kg", required: 700, margin: 100 },
      { id: "battery", unit: "kWh", required: 500, margin: -200 },
      { id: "ch4", unit: "kg", required: 100, margin: 500 },
    ],
    decision: {
      at_hours: 2, controller_id: "MPC",
      targets: [
        { id: "electrolyser", label: "Electrolyser", requested_kw: 400, applied_mean_kw: 320 },
        { id: "dac", label: "Direct air capture", requested_kw: 100, applied_mean_kw: 70 },
        { id: "methanator", label: "Methanator", requested_kw: 30, applied_mean_kw: 25 },
        { id: "battery", label: "Battery", requested_kw: 40, applied_mean_kw: 80 },
      ],
      minutes: [
        { battery_support_kw: 20, adjustments: [] },
        { battery_support_kw: 40, adjustments: [] },
      ],
      input_state: { h2_level_kg: 105, co2_level_kg: 795, ch4_level_kg: 850 },
    },
  };
}

const render = (unit, frame = makeFrame(), plantConfig = config) => renderToStaticMarkup(createElement(EquipmentInspector, { unit, frame, config: plantConfig }));
const renderDecision = (unit, frame) => renderToStaticMarkup(createElement(EquipmentDecision, { unit, frame }));

test("site controller commands use the retained weather boundary calendar", () => {
  const frame = makeFrame();
  frame.weather = { ...frame.weather, reference_kind: "forecast_pv", site_sample: { time: "2026-09-15T01:15:00Z" } };
  const markup = renderDecision("electrolyser", frame);
  assert.match(markup, /15 Sept, 01:00 UTC/);
  assert.match(markup, /15 Sept, 01:15 UTC/);
  assert.doesNotMatch(markup, /D1 08:/);
});

test("site inspector sample, delivery and retained interval share the UTC plant clock", () => {
  const frame = makeFrame();
  frame.weather = { ...frame.weather, reference_kind: "forecast_pv" };
  frame.clock.time_utc = "2026-09-15T01:15:00Z";
  const markup = render("ch4Tank", frame);
  assert.match(markup, /15 Sept, 01:15 UTC/);
  assert.match(markup, /15 Sept, 01:00 UTC/);
  assert.match(markup, /15 Sept, 03:00 UTC/);
  assert.doesNotMatch(markup, /D1 /);
});

test("site solar inspector reports forecast PV and scenario reduction without cloud or clear-sky claims", () => {
  const frame = makeFrame();
  frame.weather = { ...frame.weather, reference_kind: "forecast_pv", clear_sky_kw: 1000, available_kw: 700 };
  frame.clock.time_utc = "2026-09-15T01:15:00Z";
  const markup = render("solar", frame);
  assert.match(markup, /Forecast PV reference/);
  assert.match(markup, /Below forecast PV/);
  assert.match(markup, /30%/);
  assert.doesNotMatch(markup, /clear-sky|cloud|D1 /i);
});
const text = (markup) => markup.replace(/<[^>]*>/g, "").replaceAll("&amp;", "&").replaceAll("&#x27;", "'");
const readings = (markup) => Object.fromEntries([...markup.matchAll(/<dt>(.*?)<\/dt><dd>(.*?)<\/dd>/gs)].map(([, label, value]) => [text(label), text(value)]));
const disclosures = (markup) => [...markup.matchAll(/<details\b([^>]*)>([\s\S]*?)<\/details>/g)].map(([, attributes, body]) => ({ attributes, body }));
const status = (markup) => text(markup.match(/<p class="w-equipment-status[^"]*">(.*?)<\/p>/s)[1]);

test("process readings distinguish the current sample from the completed interval mean", () => {
  const markup = render("electrolyser");
  const rows = readings(markup);
  assert.equal(rows["Actual power"], "340 kW");
  assert.equal(rows["Requested power"], "400 kW");
  assert.equal(rows["Applied mean"], "320 kW");
  assert.match(text(markup), /D1 08:15 · 1-minute sample/);
  assert.match(text(markup), /Previous 15-minute interval · MPCD1 08:00 → D1 08:15/);
  const frame = makeFrame();
  frame.clock.elapsed_hours = 2.5;
  const longer = text(render("electrolyser", frame, { ...config, dt_s: 120 }));
  assert.match(longer, /2-minute sample/);
  assert.match(longer, /Previous 30-minute interval/);
});

test("only active commitment timers are prominent and cumulative diagnostics start collapsed", () => {
  const idleTimers = render("electrolyser");
  assert.doesNotMatch(text(idleTimers), /Startup remaining|Must stay on for|Restart allowed in|Cooldown remaining/);
  const frame = makeFrame();
  frame.commitment.startup_remaining_s = 180;
  frame.commitment.min_up_remaining_s = 420;
  const markup = render("electrolyser", frame);
  assert.equal(readings(markup)["Startup remaining"], "0h 03m");
  assert.equal(readings(markup)["Must stay on for"], "0h 07m");
  assert.doesNotMatch(text(markup), /Restart allowed in|Cooldown remaining/);
  const details = disclosures(markup).find(({ body }) => body.startsWith("<summary>Details</summary>"));
  assert.ok(details);
  assert.doesNotMatch(details.attributes, /\bopen\b/);
  assert.equal(readings(details.body)["Cold / hot starts"], "2 / 3");
  assert.equal(readings(details.body)["Cumulative startup energy"], "5.5 kWh");
  assert.doesNotMatch(markup.replace(details.body, ""), /Cold \/ hot starts|Cumulative startup energy/);
});

test("battery reserve uses the physical minimum and charging distinguishes ceiling, mean and support", () => {
  const frame = makeFrame();
  const markup = render("battery", frame);
  const rows = readings(markup);
  assert.equal(status(markup), "Charging and discharging");
  assert.equal(rows["Reserve above physical minimum"], "200 kWh");
  assert.equal(rows["Net discharge"], "30 kW");
  assert.equal(rows["Charge ceiling"], "40 kW");
  assert.equal(rows["Charging mean"], "80 kW");
  assert.equal(rows["Automatic support mean"], "30 kW");
  const explanation = disclosures(markup).find(({ body }) => body.includes("About battery charging"));
  assert.doesNotMatch(explanation.attributes, /\bopen\b/);
  assert.match(text(explanation.body), /excludes recovered solar spill, so charging can exceed it/);
  assert.match(text(explanation.body), /support to the electrolyser is dispatched automatically/);
  frame.state.solar_to_battery_kw = 0;
  assert.equal(status(render("battery", frame)), "Discharging");
  frame.state.battery_energy_kwh = 90;
  frame.state.battery_soc = 0.09;
  const depleted = readings(render("battery", frame));
  assert.equal(depleted["Reserve above physical minimum"], "0 kWh");
  assert.equal(depleted.Reserve, "At physical minimum");
});

test("gas balances and methanator consumption use the configured recipe", () => {
  const h2 = render("h2Tank");
  const co2 = render("co2Tank");
  assert.equal(status(h2), "Draining");
  assert.equal(readings(h2)["From electrolyser"], "12 kg/h");
  assert.equal(readings(h2)["To methanator"], "15 kg/h");
  assert.equal(readings(h2)["Net outflow"], "3 kg/h");
  assert.equal(readings(h2)["Inventory position"], "20 kg below");
  assert.equal(status(co2), "Filling");
  assert.equal(readings(co2)["From air capture"], "80 kg/h");
  assert.equal(readings(co2)["To methanator"], "60 kg/h");
  assert.equal(readings(co2)["Net inflow"], "20 kg/h");
  const process = readings(render("methanator"));
  assert.equal(process["Hydrogen consumption"], "15 kg/h");
  assert.equal(process["CO₂ consumption"], "60 kg/h");
  assert.equal(readings(render("dac"))["CO₂ storage space"], "200 kg");
});

test("inventory references follow the frame's operating policy", () => {
  const frame = makeFrame();
  assert.equal(readings(render("h2Tank", frame))["Daily return minimum"], "120 kg");
  assert.match(text(render("h2Tank", frame)), /Return due in 21h 45m/);
  frame.operating_policy = "inventory-targets-v1";
  const rows = readings(render("h2Tank", frame));
  assert.equal(rows["Opening reference"], "120 kg");
  assert.equal(rows["Daily return minimum"], undefined);
  assert.doesNotMatch(text(render("h2Tank", frame)), /Return due in/);
});

test("ramp limits surface only when running and more restrictive than the physical load range", () => {
  const frame = makeFrame();
  let rows = readings(render("electrolyser", frame));
  assert.equal(rows["Maximum rise next minute"], "60 kW");
  assert.equal(rows["Maximum fall next minute"], "40 kW");
  frame.commitment.mode = "starting";
  rows = readings(render("electrolyser", frame));
  assert.equal(rows["Maximum rise next minute"], undefined);
  assert.equal(rows["Maximum fall next minute"], undefined);
  frame.commitment.mode = "running";
  frame.commitment.ramp_up_headroom_kw = 160;
  frame.commitment.ramp_down_headroom_kw = 240;
  const unrestricted = render("electrolyser", frame);
  assert.doesNotMatch(text(unrestricted), /Active limits|Maximum rise next minute|Maximum fall next minute/);
});

test("full output storage and empty methanator feedstocks expose the measured limit", () => {
  const frame = makeFrame();
  for (const [tank, process, level, capacity, label] of [
    ["h2Tank", "electrolyser", "h2_level_kg", "h2_capacity_kg", "Hydrogen storage"],
    ["co2Tank", "dac", "co2_level_kg", "co2_capacity_kg", "CO₂ storage"],
    ["ch4Tank", "methanator", "ch4_level_kg", "ch4_capacity_kg", "Methane storage"],
  ]) {
    frame.state[level] = frame.state[capacity];
    for (const unit of [tank, process]) assert.equal(readings(render(unit, frame))[label], "Full");
  }
  frame.state.h2_level_kg = 0;
  frame.state.co2_level_kg = 0;
  const rows = readings(render("methanator", frame));
  assert.equal(rows["Hydrogen storage"], "Empty");
  assert.equal(rows["CO₂ storage"], "Empty");
});

test("methane delivery readiness uses only current stock and bounds coverage and the gap", () => {
  const frame = makeFrame();
  const markup = render("ch4Tank", frame);
  assert.match(text(markup), /D1 10:00In 1h 45m/);
  assert.equal(readings(markup)["Delivery quantity"], "1,000 kg");
  assert.equal(readings(markup)["Covered by current stock"], "600 kg");
  assert.equal(readings(markup)["Still to produce"], "400 kg");
  assert.match(markup, /aria-label="600 kg of 1,000 kg covered by current stock"><span style="width:60%"/);
  for (const [stock, covered, gap] of [[1500, "1,000 kg", "0 kg"], [-20, "0 kg", "1,000 kg"]]) {
    frame.state.ch4_level_kg = stock;
    const rows = readings(render("ch4Tank", frame));
    assert.equal(rows["Covered by current stock"], covered);
    assert.equal(rows["Still to produce"], gap);
  }
  frame.delivery.next_due_hours = null;
  const complete = render("ch4Tank", frame);
  assert.match(text(complete), /No deliveries remaining in this mission/);
  assert.equal(readings(complete)["Still to produce"], undefined);
  assert.equal(readings(complete)["Covered by current stock"], undefined);
});

test("methane production inflow is separate from interval inventory change that includes deliveries", () => {
  const markup = render("ch4Tank");
  assert.equal(status(markup), "Receiving methane");
  assert.equal(readings(markup)["From methanator"], "20 kg/h");
  const details = disclosures(markup).find(({ body }) => body.startsWith("<summary>Details</summary>"));
  assert.equal(readings(details.body)["Inventory change"], "-250 kg");
  assert.match(text(details.body), /Inventory change includes production and scheduled deliveries/);
  assert.doesNotMatch(status(markup), /Filling|Draining/);
});

test("solar and storage show their own context without duplicate controller command sections", () => {
  for (const unit of ["solar", "h2Tank", "co2Tank", "ch4Tank"]) {
    const frame = makeFrame();
    assert.equal(renderDecision(unit, frame), "");
    assert.doesNotMatch(render(unit, frame), /Controller command|Dispatch comparison|Requested power|Applied mean/);
    frame.decision = null;
    assert.equal(renderDecision(unit, frame), "");
  }
  const solar = readings(render("solar"));
  assert.equal(solar["Being used"], "410 kW");
  assert.equal(solar.Electrolyser, "300 kW");
  assert.equal(solar["Below clear-sky potential"], "30%");
  const hub = render("hub");
  assert.equal(readings(hub).Electrolyser, "340 kW");
  assert.equal(readings(hub)["Battery support to electrolyser"], "40 kW");
  assert.match(hub, /<th scope="row">Battery charging<\/th><td>40<\/td><td>80<\/td>/);
});

test("initial frames and missing targets have an explicit empty command state", () => {
  const frame = makeFrame();
  frame.decision = null;
  for (const unit of ["battery", "electrolyser", "dac", "methanator", "hub"]) {
    const markup = render(unit, frame);
    assert.match(text(markup), /Initial state/);
    assert.match(text(markup), /No controller command has executed yet/);
    assert.doesNotMatch(markup, /NaN|Infinity|Previous 15-minute interval/);
  }
  const missing = makeFrame();
  missing.decision.targets = [];
  missing.decision.minutes = [];
  const markup = render("battery", missing);
  assert.match(text(markup), /No power target was recorded for this interval/);
  assert.equal(readings(markup)["Automatic support mean"], undefined);
});

test("fault status uses simulator unit identifiers and distinguishes an idle component", () => {
  const frame = makeFrame();
  frame.state.dac_online = false;
  frame.state.health.dac = { status: "nominal", offline: false, degradation: 0 };
  assert.equal(status(render("dac", frame)), "Idle");
  frame.state.health.dac = { status: "offline", offline: true, degradation: 0 };
  assert.equal(status(render("dac", frame)), "Offline · fault");
  for (const [unit, key] of [["h2Tank", "h2_tank"], ["co2Tank", "co2_tank"], ["ch4Tank", "ch4_tank"]]) {
    frame.state.health[key] = { status: "offline", offline: true, degradation: 0 };
    assert.equal(status(render(unit, frame)), "Offline · fault");
  }
  frame.state.health.electrolyser = { status: "degraded", offline: false, degradation: 0.4 };
  assert.equal(status(render("electrolyser", frame)), "Running · degraded");
});

test("recorded execution adjustments are deduplicated and scoped to the affected component", () => {
  const frame = makeFrame();
  const adjustment = "Battery at its minimum SOC: battery support dropped";
  frame.decision.minutes[0].adjustments = [adjustment, "DAC is offline: power dropped"];
  frame.decision.minutes[1].adjustments = [adjustment];
  const electrolyser = text(render("electrolyser", frame));
  assert.equal(electrolyser.split(adjustment).length - 1, 1);
  assert.doesNotMatch(electrolyser, /DAC is offline/);
  const dac = text(render("dac", frame));
  assert.match(dac, /DAC is offline: power dropped/);
  assert.doesNotMatch(dac, /Battery at its minimum/);
  assert.doesNotMatch(text(render("methanator", frame)), /Execution adjustments/);
});
