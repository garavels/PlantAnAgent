import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { AGENT_ACTION_DWELL_MS, deriveAgentActions } from "../src/workspace/agentActions.ts";

const recording = JSON.parse(readFileSync(new URL("../../demo/recordings/cd1ff361-33ac-5f84-9c94-8363ffdbe12f.json", import.meta.url), "utf8"));

function frame(targets, minutes = []) {
  return { decision: { targets, minutes } };
}

const target = (id, requested_kw, applied_mean_kw) => ({ id, label: id, requested_kw, applied_mean_kw });

test("the recorded PPO week tours only its four issued targets in physical order", () => {
  assert.deepEqual(deriveAgentActions(recording.frames[0]), []);
  assert.equal(AGENT_ACTION_DWELL_MS, 2600);
  for (const recorded of recording.frames.slice(1)) {
    const actions = deriveAgentActions(recorded);
    assert.deepEqual(actions.map((action) => action.unit), ["electrolyser", "battery", "dac", "methanator"]);
    for (const action of actions) {
      const source = recorded.decision.targets.find((item) => item.id === action.unit);
      assert.equal(action.requestedKw, source.requested_kw);
      assert.equal(action.appliedKw, source.applied_mean_kw);
      assert.match(action.detail, /Detailed guard adjustments were not recorded/);
      assert.doesNotMatch(action.explanation, /because|forecast|decided to|shut|off|NaN|Infinity/i);
    }
  }
});

test("zero and tiny commands preserve the distinction between requested and applied power", () => {
  const [zero, tiny] = deriveAgentActions(frame([
    target("electrolyser", 0, 12),
    target("dac", 0.000042, 0),
  ]));
  assert.match(zero.explanation, /requested 0 kW/);
  assert.match(zero.explanation, /averaged 12 kW/);
  assert.doesNotMatch(zero.title + zero.explanation, /off|shut|stop/i);
  assert.match(tiny.explanation, /requested 0\.000042 kW/);
  assert.match(tiny.explanation, /averaged 0 kW/);
  assert.equal(zero.detail, undefined);
});

test("battery charging above its ceiling is separate from automatic electrolyser support", () => {
  const [battery] = deriveAgentActions(frame([target("battery", 40, 80)], [
    { battery_support_kw: 20, adjustments: [] },
    { battery_support_kw: 40, adjustments: [] },
  ]));
  assert.equal(battery.requestLabel, "Charge ceiling");
  assert.equal(battery.appliedLabel, "Charging mean");
  assert.match(battery.explanation, /40 kW direct-charge ceiling/);
  assert.match(battery.explanation, /charging averaged 80 kW/);
  assert.match(battery.detail, /recovers unused solar power/);
  assert.match(battery.detail, /Automatic battery support.*averaged 30 kW/);
  assert.doesNotMatch(battery.detail, /guard|clamp|intervention/i);

  const [missing] = deriveAgentActions(frame([target("battery", 0, 0)]));
  assert.match(missing.detail, /dispatched automatically/);
  assert.doesNotMatch(missing.detail, /support.*averaged 0 kW/);
});

test("execution notes are deduplicated and attributed only to the affected equipment", () => {
  const actions = deriveAgentActions(frame([
    target("electrolyser", 100, 60), target("battery", 0, 0),
    target("dac", 10, 0), target("methanator", 5, 5),
  ], [
    { battery_support_kw: 0, adjustments: ["DAC is offline: allocation dropped", "Battery is offline: battery support dropped", "Unknown system note"] },
    { battery_support_kw: 0, adjustments: ["DAC is offline: allocation dropped"] },
  ]));
  const details = Object.fromEntries(actions.map((action) => [action.unit, action.detail ?? ""]));
  assert.match(details.electrolyser, /Battery is offline: battery support dropped/);
  assert.match(details.battery, /Battery is offline: battery support dropped/);
  assert.doesNotMatch(details.electrolyser + details.battery, /DAC/);
  assert.equal(details.dac, "Recorded adjustments: DAC is offline: allocation dropped");
  assert.equal(details.methanator, "");
  assert.doesNotMatch(Object.values(details).join(" "), /Unknown system note/);
});

test("empty, unsupported, nonfinite and negative targets never invent tour stops", () => {
  assert.deepEqual(deriveAgentActions(frame([])), []);
  assert.deepEqual(deriveAgentActions(frame([
    target("solar", 100, 100), target("electrolyser", NaN, 100),
    target("battery", 10, Infinity), target("dac", -1, 0), target("methanator", 0, -1),
  ])), []);
  assert.deepEqual(deriveAgentActions(frame([target("methanator", 5, 4)])).map((action) => action.unit), ["methanator"]);
});
