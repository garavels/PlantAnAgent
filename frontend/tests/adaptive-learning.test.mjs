import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { compareSummary, mean, parseResearchResponse } from "../src/workspace/adaptiveLearning.ts";

const artifact = { id: "closed-record", bytes: 123, sha256: "a".repeat(64) };
const seeds = Array.from({ length: 8 }, (_, i) => 9700300 + i);
const policies = ["A-scalar-gamma0995", "B-scalar-gamma1000", "A-mpc-residual-scalar", "B-mpc-residual-pi"];
const actorIds = [...policies, "H", "OperatingMPC", "initial-mpc-residual", "M24", "E-beta"];
function response() {
  return {
    status: "available",
    evidence: {
      schema_version: "plant-public-adaptive-learning-v1",
      source: { git_head: "a".repeat(40), git_tree: "b".repeat(40) },
      provenance: { exported_from_closed_evidence: true, independent_verification_passed: true, development_only: true, artifacts: [artifact] },
      task: { hours: 168, decisions: 672, deliveries: 28, requested_methane_kg: 6720, development_seeds: [...seeds] },
      policies: policies.map((id) => ({
        id, label: id, campaign_id: "009", algorithm: "scalar PPO", optimizer_seed: 101,
        training_status: "completed", evaluation_status: "evaluated", operational_status: "experimental",
        transitions: 16384, minibatch_updates: 640, reward_gamma: 0.995, gae_lambda: 0.95, reward_scale: 0.001,
        checkpoint: artifact, training_source: artifact, limitations: ["Development only"],
        rollouts: [1, 2].map((index) => ({ index, transitions: index * 8192, reward_sum: -10, completed_episodes: 8, complete_cycles: 80, training_mean_reward: null, explained_variance: 0, value_loss: 0.1, approx_kl: null, clip_fraction: 0.02, mean_episode_cost: null, multipliers: null })),
      })),
      actors: actorIds.map((id) => ({ id, label: id, kind: policies.includes(id) ? "ppo" : "mpc", evidence_origin: "reused_batch008", cases: seeds.map((seed) => {
        const stopped = ["M24", "E-beta"].includes(id);
        return {
          seed, status: stopped ? "stopped" : "completed", decisions: stopped ? 0 : 672, failures: stopped ? ["No feasible plan"] : [],
          net_operating_value_eur: stopped ? null : -100, delivered_kg: stopped ? null : 6000, battery_throughput_kwh: stopped ? null : 1000,
          hard_safety_passed: stopped ? null : true, prefix_hard_safety_passed: true, all_daily_returns_passed: stopped ? null : false,
          mpc_modes: stopped ? null : { scheduled: 0, recovery: 670, fallback: 2 },
          deliveries: stopped ? [] : Array.from({ length: 28 }, (_, i) => ({ ordinal: i + 1, delivered_kg: 200, requested_kg: 240 })),
          daily_resources: stopped ? [] : Array.from({ length: 7 }, (_, i) => ({ cycle: i + 1, deficits: { battery: 1, h2: 2, co2: 3, ch4: 4 } })),
        };
      }) })),
      comparisons: [{ candidate: policies[0], comparator: "OperatingMPC", evidence_batch: "009", cases: seeds.map((seed, index) => ({ seed, available: true, candidate_failures: [], comparator_failures: [], delta_net_operating_value_eur: [-8, -4, -2, -1, 1, 2, 4, 8][index], delta_delivered_kg: 0, delta_battery_throughput_kwh: 0, delivery_regressions: index < 4 ? [{ ordinal: 1, difference_kg: -5 }] : [], resource_regressions: index < 2 ? [{ cycle: 1, resource: "battery", difference: 1 }] : [] })) }],
      history: [{ id: "009", label: "Batch 009", status: "completed", summary: "Two policies evaluated", policy_ids: policies, source: artifact, limitations: [] }],
      registered_study: { status: "pending", families: 3, optimizer_seeds: 5, policies: 15, protected_seeds_opened: false },
      limitations: ["No held-out evaluation"], sha256: "c".repeat(64),
    },
  };
}

test("closed development record preserves sparse diagnostics and stopped cases", () => {
  const data = parseResearchResponse(response());
  assert.equal(data.policies[0].rollouts.length, 2);
  assert.equal(data.policies[0].rollouts[0].training_mean_reward, null);
  assert.equal(data.actors.find((actor) => actor.id === "E-beta").cases[0].hard_safety_passed, null);
  assert.equal(data.actors.find((actor) => actor.id === "OperatingMPC").cases[0].mpc_modes.recovery, 670);
});

test("pending evidence is distinct from malformed or failed availability", () => {
  assert.equal(parseResearchResponse({ status: "pending", evidence: null }), null);
  for (const bad of [null, {}, { status: "available", evidence: null }, { status: "pending", evidence: {} }, { status: "error", evidence: null }]) assert.throws(() => parseResearchResponse(bad));
});

test("unsafe, unverified or incomplete values cannot reach the interface", () => {
  const mutations = [
    (e) => { e.policies[0].operational_status = "approved"; },
    (e) => { e.provenance.independent_verification_passed = false; },
    (e) => { e.task.development_seeds.reverse(); },
    (e) => { e.actors[0].cases.pop(); },
    (e) => { e.actors[0].cases[0].net_operating_value_eur = Infinity; },
    (e) => { e.actors[0].cases[0].daily_resources[0].deficits.ch4 = undefined; },
    (e) => { e.actors[0].cases[0].mpc_modes.recovery = 671; },
    (e) => { e.actors[0].cases[0].deliveries.pop(); },
    (e) => { e.actors[8].cases[0].hard_safety_passed = true; },
    (e) => { e.actors[8].cases[0].net_operating_value_eur = 0; },
    (e) => { e.policies[0].rollouts[1].transitions = 65536; },
    (e) => { e.policies[0].rollouts[0].multipliers = [1, 2]; },
    (e) => { e.comparisons[0].cases[0].delivery_regressions[0].difference_kg = 1; },
    (e) => { e.comparisons[0].cases[0].resource_regressions[0].resource = "unknown"; },
    (e) => { e.registered_study.protected_seeds_opened = true; },
    (e) => { e.sha256 = "corrupt"; },
  ];
  mutations.forEach((mutate) => { const value = response(); mutate(value.evidence); assert.throws(() => parseResearchResponse(value)); });
});

test("comparison retains economic losses and every recorded regression", () => {
  const comparison = parseResearchResponse(response()).comparisons[0];
  assert.deepEqual(compareSummary(comparison), { available: true, meanValue: 0, worstValue: -8, wins: 4, deliveryRegressions: 4, resourceRegressions: 2 });
});

test("one incomplete case disables the complete-panel summary", () => {
  const comparison = parseResearchResponse(response()).comparisons[0];
  comparison.cases[7] = { ...comparison.cases[7], available: false, delta_net_operating_value_eur: null, delta_delivered_kg: null, delta_battery_throughput_kwh: null, delivery_regressions: [], resource_regressions: [] };
  assert.deepEqual(compareSummary(comparison), { available: false, meanValue: null, worstValue: null, wins: null, deliveryRegressions: null, resourceRegressions: null });
  assert.equal(compareSummary(undefined).meanValue, null);
  assert.equal(mean([-100, null]), null);
  assert.equal(mean([]), null);
  assert.equal(mean([-100, -50]), -75);
});


test("published export and frontend contract agree on negative tails and stopped controllers", () => {
  const evidence = JSON.parse(readFileSync(new URL("../../backend/src/plant_backend/data/adaptive_learning.json", import.meta.url), "utf8"));
  const data = parseResearchResponse({ status: "available", evidence });
  assert.deepEqual(data.policies.map((policy) => policy.rollouts.length), [8, 8, 2, 2]);
  assert.equal(data.actors.length, 9);
  assert.equal(data.history.length, 10);
  const summary = compareSummary(data.comparisons.find((pair) => pair.candidate === "A-scalar-gamma0995" && pair.comparator === "OperatingMPC"));
  assert.ok(Math.abs(summary.meanValue - 143.86637927732176) < 1e-7);
  assert.ok(Math.abs(summary.worstValue + 717.3340987535366) < 1e-7);
  assert.equal(summary.deliveryRegressions, 23);
  assert.equal(summary.resourceRegressions, 91);
  for (const id of ["M24", "E-beta"]) {
    const actor = data.actors.find((row) => row.id === id);
    assert.ok(actor.cases.every((row) => row.status === "stopped" && row.net_operating_value_eur === null && row.hard_safety_passed === null));
  }
  const mpc = data.actors.find((row) => row.id === "OperatingMPC");
  assert.equal(mpc.cases.reduce((sum, row) => sum + row.mpc_modes.recovery, 0), 5327);
  assert.equal(mpc.cases.reduce((sum, row) => sum + row.mpc_modes.fallback, 0), 49);
});
