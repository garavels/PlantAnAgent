/**
 * Public summary of the closed feasible25/off-grid study, not live training state.
 * Sources under docs/research/results/feasible25_offgrid_20260915/:
 * - coverage-ppo-v1/training-closure.json and endpoints-frozen.json
 * - ppo-101-held-out/summary.json
 * - coverage-held-out-a/summary.json and coverage-held-out-b/summary.json
 * Mission: docs/research/benchmarks/feasible25_offgrid_v1.json.
 * current-study.test.mjs checks this summary against those retained artifacts.
 * Pass counts are scenarios; completedAt is the three-endpoint training closure.
 */
export const currentStudy = {
  optimizerSeed: 101,
  checkpointSha256: "6c36cc24692a7d46528436f6f5fabcff1e1bed2862153a49eb5e1bd8cc49ba27",
  benchmarkSha256: "deae125d30f30cf63860aeaec76db566dffee18e6523de2fa508b26f57dafe7e",
  training: {
    steps: 65_536,
    completedAt: "2026-09-15T12:36:49.574411+00:00",
  },
  evaluation: {
    scenarios: 16,
    hours: 168,
    deliveryRateKgH: 25,
    baseline: { missionsPassed: 14, deliveriesPassed: 16, safetyPassed: 16 },
    ppo: { missionsPassed: 14, deliveriesPassed: 16, safetyPassed: 16 },
    failure: { scenarios: 2, day: 1, resource: "methane" },
  },
} as const;
