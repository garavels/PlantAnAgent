import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const modules = new Map();
function compile(file) {
  if (modules.has(file.href)) return modules.get(file.href);
  let source = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
  }).outputText.replace(/^import "[^"\n]+\.css";\n/gm, "");
  source = source.replace(/from "([^"]+)"/g, (_match, specifier) => {
    if (!specifier.startsWith(".")) return `from "${pathToFileURL(require.resolve(specifier))}"`;
    const dependency = [".ts", ".tsx"].map((extension) => new URL(`${specifier}${extension}`, file)).find(existsSync);
    assert.ok(dependency);
    return `from "${compile(dependency)}"`;
  });
  const url = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  modules.set(file.href, url);
  return url;
}
const { checkRecording, featuredRecording, missionDescription } = await import(compile(new URL("../src/workspace/recording.ts", import.meta.url)));
const { ResearchMissionSummary } = await import(compile(new URL("../src/workspace/ResearchPlayback.tsx", import.meta.url)));
const { EquipmentDecision } = await import(compile(new URL("../src/workspace/EquipmentDecision.tsx", import.meta.url)));
const { inventoryChangeSinceOpening } = await import(compile(new URL("../src/workspace/economicsModel.ts", import.meta.url)));
const recording = JSON.parse(readFileSync(new URL("../../demo/recordings/cd1ff361-33ac-5f84-9c94-8363ffdbe12f.json", import.meta.url), "utf8"));

test("the actual verified PPO artifact is admitted as a complete research playback", () => {
  assert.equal(checkRecording(recording), recording);
  assert.equal(recording.frames.length, 673);
  assert.equal(recording.frames[0].clock.elapsed_hours, 0);
  assert.equal(recording.frames.at(-1).clock.elapsed_hours, 168);
  assert.match(missionDescription(recording.snapshot, recording), /25 kg CH₄\/h · 150 kg every 6 hours/);
});

test("research admission rejects mismatched identity, timeline and unsupported success claims", () => {
  const mutations = [
    (copy) => { copy.snapshot.mode = "interactive_fixture"; },
    (copy) => { copy.provenance.scenario_seed = 9930001; },
    (copy) => { copy.provenance.evaluation.passed = 16; },
    (copy) => { copy.provenance.electricity.fixed_costs_included = true; },
    (copy) => { copy.provenance.checkpoint_sha256 = "not a checksum"; },
    (copy) => { copy.frames[40].clock.elapsed_hours = 42; },
    (copy) => { copy.frames.at(-1).delivery.commitments[0].status = "shortfall"; },
    (copy) => { copy.frames.at(-1).cycle_settlements[0].feasible = false; },
  ];
  for (const mutate of mutations) {
    const copy = structuredClone(recording);
    mutate(copy);
    assert.throws(() => checkRecording(copy));
  }
});

test("legacy fixture admission remains separate and strict", () => {
  const legacy = structuredClone(recording);
  legacy.schema = "plant-public-recording-v1";
  legacy.snapshot.mode = "interactive_fixture";
  delete legacy.provenance;
  assert.throws(() => checkRecording(legacy), /public fixture/);
  legacy.snapshot.environment.fixture_seed = 9000034;
  assert.equal(checkRecording(legacy), legacy);
  legacy.snapshot.mode = "recorded_research";
  assert.throws(() => checkRecording(legacy), /public fixture/);
});

test("only an available explicitly featured research recording takes precedence", () => {
  const old = { id: "old", available: true, source: "saved", controller: "MPC" };
  const unavailable = { id: "missing", available: false, featured: true, mode: "recorded_research" };
  const featured = { id: "ppo", available: true, featured: true, mode: "recorded_research" };
  assert.equal(featuredRecording([old, unavailable, featured]), featured);
  assert.equal(featuredRecording([old, unavailable]), undefined);
});

test("the selected success is distinguished from broader held-out reliability and excluded costs", () => {
  const markup = renderToStaticMarkup(createElement(ResearchMissionSummary, { provenance: recording.provenance }));
  for (const text of ["Scenario 9930000", "28/28", "7/7", "14/16 full missions passed", "25 kg CH₄/h", "€0/MWh marginal electricity charge", "fixed costs excluded"]) {
    assert.ok(markup.includes(text), text);
  }
  assert.doesNotMatch(markup, /16\/16|commercial profit|live control/i);
});

test("actual unrecorded guard notes remain unknown in the equipment inspector", () => {
  const frame = recording.frames[1];
  assert.equal(frame.decision.guard_adjustments, null);
  const markup = renderToStaticMarkup(createElement(EquipmentDecision, { frame, unit: "electrolyser" }));
  assert.match(markup, /Detailed guard adjustments were not recorded/);
  assert.doesNotMatch(markup, /No adjustment|NaN|Infinity/);
});

test("research inventory changes use actual retained reset stock for all four resources", () => {
  const reset = recording.frames[0];
  const final = recording.frames.at(-1);
  assert.equal(final.metrics.initial_battery_energy_kwh, undefined);
  for (const resource of final.resources) {
    const initial = reset.resources.find((item) => item.id === resource.id);
    const change = inventoryChangeSinceOpening(final, resource, reset);
    assert.equal(change, resource.current - initial.current, resource.id);
    assert.ok(change > 0, resource.id);
    assert.equal(inventoryChangeSinceOpening(reset, initial, reset), 0, resource.id);
    // A selected period's opening cannot stand in for the run's physical reset.
    assert.equal(inventoryChangeSinceOpening(final, resource, recording.frames[576]), null, resource.id);
  }
});

test("historical inventory metrics remain supported and missing values stay unknown", () => {
  const legacy = { clock: { decision_index: 32, elapsed_hours: 8 }, metrics: { initial_h2_kg: 10 } };
  const h2 = { id: "h2", unit: "kg", current: 8 };
  assert.equal(inventoryChangeSinceOpening(legacy, h2), -2);
  assert.equal(inventoryChangeSinceOpening(legacy, { id: "co2", unit: "kg", current: 20 }), null);
  assert.equal(inventoryChangeSinceOpening(legacy, { ...h2, current: Number.NaN }), null);
});
