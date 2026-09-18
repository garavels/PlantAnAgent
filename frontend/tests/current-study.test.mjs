import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/workspace/currentStudy.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText;
const { currentStudy } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
const resultRoot = new URL("../../docs/research/results/feasible25_offgrid_20260915/", import.meta.url);
const readJson = (path, root = resultRoot) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const ppo = readJson("ppo-101-held-out/summary.json");
const references = [readJson("coverage-held-out-a/summary.json"), readJson("coverage-held-out-b/summary.json")];
const baselineRuns = references.flatMap((panel) => panel.runs);
const benchmark = readJson("../../benchmarks/feasible25_offgrid_v1.json");
const training = readJson("coverage-ppo-v1/training-closure.json");
const endpoints = readJson("coverage-ppo-v1/endpoints-frozen.json");
const closure = readJson("experiment-closure.json");
const recording = readJson("../../demo/recordings/cd1ff361-33ac-5f84-9c94-8363ffdbe12f.json", import.meta.url);

test("current study identifies the completed PPO endpoint used by the demo", () => {
  const row = training.rows.find((item) => item.optimizer_seed === currentStudy.optimizerSeed);
  const endpoint = endpoints.endpoints.find((item) => item.optimizer_seed === currentStudy.optimizerSeed);
  assert.ok(row);
  assert.ok(endpoint);
  assert.deepEqual(currentStudy.training, { steps: row.num_timesteps, completedAt: training.completed_at });
  assert.equal(row.actor_parameters_changed, true);
  assert.equal(row.checkpoint_reopen_verified, true);
  assert.equal(currentStudy.checkpointSha256, row.final_checkpoint.sha256);
  assert.equal(currentStudy.checkpointSha256, endpoint.checkpoint.sha256);
  assert.equal(currentStudy.checkpointSha256, recording.provenance.checkpoint_sha256);
  assert.equal(currentStudy.optimizerSeed, recording.provenance.optimizer_seed);
  assert.equal(currentStudy.benchmarkSha256, benchmark.contract_sha256);
  assert.equal(currentStudy.benchmarkSha256, recording.provenance.benchmark_contract_sha256);
  for (const run of ppo.runs) {
    assert.equal(run.controller_configuration.optimizer_seed, currentStudy.optimizerSeed);
    assert.equal(run.controller_configuration.training_transitions, currentStudy.training.steps);
  }
});

test("the comparison covers the same held-out scenarios, mission and electricity accounting", () => {
  assert.equal(new Set(baselineRuns.map((run) => run.seed)).size, currentStudy.evaluation.scenarios);
  assert.equal(new Set(ppo.runs.map((run) => run.seed)).size, currentStudy.evaluation.scenarios);
  assert.deepEqual(baselineRuns.map((run) => run.seed), ppo.runs.map((run) => run.seed));
  assert.deepEqual(ppo.requested_seeds, benchmark.splits.held_out);
  assert.deepEqual(references.flatMap((panel) => panel.requested_seeds), ppo.requested_seeds);
  for (const panel of [ppo, ...references]) {
    assert.equal(panel.split, "held_out");
    assert.equal(panel.benchmark_contract_sha256, currentStudy.benchmarkSha256);
    assert.equal(panel.evaluation_freeze_sha256, closure.evaluation_freeze_sha256);
    assert.deepEqual(panel.runs.map((run) => run.seed), panel.requested_seeds);
  }
  for (const reference of references) assert.equal(reference.controller, "daily-coverage-offgrid-v1");
  assert.equal(ppo.controller, "coverage-ppo-101");
  assert.equal(currentStudy.evaluation.deliveryRateKgH, benchmark.delivery_rate_kg_h);
  assert.equal(currentStudy.evaluation.hours, benchmark.source_template.episode.horizon_h);
  for (const run of [...ppo.runs, ...baselineRuns]) {
    assert.equal(run.execution_complete, true);
    assert.equal(run.physical_metrics.hours, currentStudy.evaluation.hours);
    assert.equal(run.delivery_rate_kg_h, currentStudy.evaluation.deliveryRateKgH);
    assert.equal(run.electricity_model.mode, "dedicated_pv_no_export");
    assert.equal(run.electricity_model.marginal_solar_cost_eur_per_mwh, 0);
    assert.equal(run.electricity_model.solar_capital_recovery_eur_per_week, null);
    assert.equal(run.electricity_model.solar_fixed_om_eur_per_week, null);
    assert.equal(run.economic_metrics.components_eur.solar_opportunity_cost, 0);
    assert.equal(run.source_reconstruction.source_weather_preserved, true);
    assert.equal(run.source_reconstruction.source_initial_inventory_preserved, true);
  }
  for (const run of ppo.runs) {
    const reference = baselineRuns.find((item) => item.seed === run.seed);
    assert.equal(run.controller_configuration.policy_identity.prior.id, reference.controller);
    assert.deepEqual(run.delivery.schedule, reference.delivery.schedule);
    assert.deepEqual(run.electricity_model, reference.electricity_model);
    for (const key of ["initial_h2_kg", "initial_co2_kg", "initial_ch4_kg", "initial_battery_energy_kwh", "solar_kwh"]) {
      assert.equal(run.physical_metrics[key], reference.physical_metrics[key]);
    }
  }
});

test("pass counts count complete scenarios and agree with the retained results", () => {
  for (const [name, runs, controller] of [
    ["baseline", baselineRuns, "daily-coverage-offgrid-v1"],
    ["ppo", ppo.runs, "coverage-ppo-101"],
  ]) {
    assert.deepEqual(currentStudy.evaluation[name], {
      missionsPassed: runs.filter((run) => run.mission_passed).length,
      deliveriesPassed: runs.filter((run) => run.every_delivery_passed).length,
      safetyPassed: runs.filter((run) => run.hard_safety_passed).length,
    });
    const group = closure.groups.find((item) => item.split === "held_out" && item.controller === controller);
    assert.equal(group.cases, currentStudy.evaluation.scenarios);
    assert.equal(group.full_missions, currentStudy.evaluation[name].missionsPassed);
    assert.equal(group.safe_full_weeks, currentStudy.evaluation[name].safetyPassed);
  }
});

test("the remaining PPO mission failures are only first-day methane returns", () => {
  const failures = ppo.runs.filter((run) => !run.mission_passed);
  assert.equal(failures.length, currentStudy.evaluation.failure.scenarios);
  assert.equal(currentStudy.evaluation.failure.resource, "methane");
  for (const run of failures) {
    assert.equal(run.every_delivery_passed, true);
    assert.equal(run.hard_safety_passed, true);
    const failedReturns = run.cycle_settlements.filter((cycle) => !cycle.feasible);
    assert.equal(failedReturns.length, 1);
    assert.equal(failedReturns[0].cycle_index, currentStudy.evaluation.failure.day);
    assert.deepEqual(Object.keys(failedReturns[0].deficits).filter((resource) => failedReturns[0].deficits[resource] > failedReturns[0].tolerances[resource]), ["ch4"]);
  }
});
