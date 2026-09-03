# Research programme

This directory is the auditable record for the Environment V2 research programme.
The programme's objective is to turn Plant An Agent from a convincing integration
prototype into a discriminative, reproducible benchmark for layered industrial
autonomy.

The governing question is not whether a particular learning algorithm can achieve
a high scalar reward. It is whether each intelligence layer produces measurable
operational value under realistic uncertainty, while retaining explicit safety and
fallback behavior.

## Files

- [`JOURNAL.md`](JOURNAL.md) is chronological and append-only in spirit. It records
  commands, observations, decisions, failures, and changes of direction.
- [`ROADMAP.md`](ROADMAP.md) defines hypotheses, experimental gates, benchmark arms,
  and the staged implementation programme.
- [`ENVIRONMENT_V1_AUDIT.md`](ENVIRONMENT_V1_AUDIT.md) records the reproducible
  saturation diagnosis that motivates Environment V2.
- [`STATE_OF_THE_ART.md`](STATE_OF_THE_ART.md) defines the external evidence bar,
  product wedge, benchmark comparators, and limits on current claims.
- [`ENVIRONMENT_V2_SPEC.md`](ENVIRONMENT_V2_SPEC.md) defines the commercial control
  problem, layer boundaries, time/forecast/commitment contracts, current 57-field
  interface and objective boundary, staged simulator changes, and acceptance gates.
- [`ENVIRONMENT_V2_ALPHA_RESULTS.md`](ENVIRONMENT_V2_ALPHA_RESULTS.md) records the
  first clean, paired delivery-only calibration and the decision it supports.
- [`ENVIRONMENT_V2_WEATHER_RESULTS.md`](ENVIRONMENT_V2_WEATHER_RESULTS.md) records
  the clean, paired weather-alpha calibration, independent replay audit, failure-tail
  analysis, and resulting Gate B decision.
- [`ELECTROLYSER_COMMITMENT_PROFILE.md`](ELECTROLYSER_COMMITMENT_PROFILE.md) freezes
  the evidence boundary, assumptions, state semantics, compatibility contract, and
  acceptance tests for the first technology-labelled equipment model now integrated
  into commitment-alpha V2.
- [`COMMITMENT_ALPHA_CALIBRATION.md`](COMMITMENT_ALPHA_CALIBRATION.md) records the
  clean, paired 3-seed development screen, selects 35 kg/h as the primary development
  target and 42 kg/h as a development stress point, and preserves the boundary that
  neither is formal or held-out evidence.
- [`CAUSAL_CONTROLLER_EXPERIMENT.md`](CAUSAL_CONTROLLER_EXPERIMENT.md) preregisters
  the transparent A0–A3 development comparison, its causal information boundary,
  paired estimands, failure handling, tuning boundary, and future prospective gates.
- [`CAUSAL_CONTROLLER_RESULTS.md`](CAUSAL_CONTROLLER_RESULTS.md) reports every
  preregistered development rollout, paired effect, treatment-expression check,
  inventory balance, negative controller finding, and the resulting trajectory-planning
  benchmark decision.
- [`PLANNING_BENCHMARK_PROTOCOL.md`](PLANNING_BENCHMARK_PROTOCOL.md) preregisters the
  privileged continuous-relaxation bound, exact-simulator best-known feasible planner,
  causal-MPC gate, proof obligations, information boundary, and failure rules.
- [`FEASIBLE_PLANNER_PROTOCOL.md`](FEASIBLE_PLANNER_PROTOCOL.md) freezes the exact
  future-information planner's action templates, deterministic search and polishing
  budgets, clone/deduplication contracts, task score, fresh-replay proof, and fail-closed
  evidence schema before any full-horizon planner result is opened.
- [`FEASIBLE_PLANNER_PREFREEZE_AUDIT.md`](FEASIBLE_PLANNER_PREFREEZE_AUDIT.md) maps
  every frozen pre-outcome obligation to executable evidence and records the final
  physical replay boundary fixtures required before the six development cells may run.
- [`FEASIBLE_PLANNER_REPAIR_001.md`](FEASIBLE_PLANNER_REPAIR_001.md) preserves the
  first registered-cell failure and freezes the narrow binary64 legacy-action repair
  before its implementation or any further protected execution.
- [`FEASIBLE_PLANNER_RESULTS.md`](FEASIBLE_PLANNER_RESULTS.md) reports the complete
  repaired six-cell development matrix, exact controllability bracket, separate
  endowment-restoring diagnostic, operating mechanism, limitations, and decision to
  build a causal service MPC.
- [`tools/`](tools/) retains the exact hash-bound one-shot drivers used to run the
  endowment diagnostic and construct its compact Git report from the protected external
  artifacts.

Small canonical benchmark outputs may be committed under [`results/`](results/) when
they make a reported conclusion directly auditable. Large generated datasets, model
checkpoints, exploratory outputs, and redundant reruns remain outside Git. Every reported
experiment must record enough provenance to reproduce it: commit, configuration, seeds,
environment contract version, dependency versions, and the location or digest of
generated artifacts.

## Research rules

1. Preserve Environment V1 as a reproducible baseline.
2. Change one scientifically meaningful difficulty class at a time.
3. Compare against the heuristic and at least one model-based or oracle baseline.
4. Separate training, validation, and held-out stress regimes.
5. Report distributions and tail risk, not a single best rollout.
6. Keep hard safety in the guard; do not hide violations in reward weights.
7. Use ablations to attribute value to RL, supervision, and HEPA separately.
8. Treat negative results as results and record them.
9. Do not claim commercial value without translating metrics into operational units.
10. Commit coherent, tested increments with the journal entry that explains them.
