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
- [`CAUSAL_SERVICE_MPC_PROTOCOL.md`](CAUSAL_SERVICE_MPC_PROTOCOL.md) preregisters the
  first deployable-information service MPC: its forecast projection, mixed-integer
  model, solver/failure contract, development thresholds, evidence payload, and
  pre-outcome implementation gate.
- [`CAUSAL_SERVICE_MPC_PREFREEZE_AUDIT.md`](CAUSAL_SERVICE_MPC_PREFREEZE_AUDIT.md)
  maps every implementation-gate obligation to executable evidence, reviews the
  source authority boundaries, and preserves the tiny-fixture/full-rollout boundary.
- [`CAUSAL_SERVICE_MPC_REPORT_FREEZE.md`](CAUSAL_SERVICE_MPC_REPORT_FREEZE.md)
  freezes schema-7 battery-throughput instrumentation, six-cell admission, exact
  planner cross-binding, no-subset aggregation, and executable interpretations of
  all seven development gates before any registered outcome is opened.
- [`results/v2_service_mpc_matrix_59e377c.md`](results/v2_service_mpc_matrix_59e377c.md)
  records the first registered six-cell result: a G1 engineering failure caused by
  numerical curtailment and solver-residual contracts, with no eligible performance
  aggregate and complete external artifact hashes.
- [`CAUSAL_SERVICE_MPC_REPLICATION_PROTOCOL.md`](CAUSAL_SERVICE_MPC_REPLICATION_PROTOCOL.md)
  freezes the narrow post-failure numerical and reporting repairs, discloses partial
  repair diagnostics, version-separates the new evidence, and preregisters the unchanged
  six-cell engineering replication before any repaired full rollout is opened.
- [`results/v2_service_mpc_replication_3796eb5.md`](results/v2_service_mpc_replication_3796eb5.md)
  records the complete execution-valid replication, its G1/G7 passes and G2--G6 failures,
  the 204-of-206 aborted-start diagnosis, and the decision to repair the supervisory
  actuation contract before objective tuning, added realism, or RL training.
- [`CAUSAL_SERVICE_MPC_ACTUATION_BRIDGE_PROTOCOL.md`](CAUSAL_SERVICE_MPC_ACTUATION_BRIDGE_PROTOCOL.md)
  freezes the expected-flow versus contingent-authority hypothesis, permitted bridge-only
  repair, executable obligations, single-cell mechanism probe, and decision rule before
  implementation. The versioned realization, fail-closed audit runner, and complete
  regression gate were completed before its one authorized outcome was opened.
- [`results/v2_service_mpc_actuation_bridge_probe_709e477.md`](results/v2_service_mpc_actuation_bridge_probe_709e477.md)
  records the probe's B1 canonicalization failure, the 223-decision startup-recovery prefix,
  the separately labelled target-fidelity failure, and the decision to replace the held
  share/additive-support seam with absolute setpoints and a deterministic inner dispatcher.
- [`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md`](CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md)
  freezes the new absolute command, reserve-aware one-minute allocation algorithm,
  feasibility-preserving solver selection, evidence graph, implementation gate, and first
  development-run decision rule before any new controller code or outcome is opened. Its
  solver-selection prerequisite is now implemented and adversarially tested under a new
  settings identity. Its typed absolute command and pure minute dispatcher are implemented
  behind a new opt-in API; the historical controller identity and strict behavior remain
  intact. The opt-in Environment V2 execution door now applies a fresh dispatch and the
  unchanged guard on every physical minute, returning a hash-chained requested/guarded/
  applied ledger. The separately versioned command-producing controller now emits only this
  absolute intent and requires that physical ledger before its next solve. A run-level deep
  validator now checks decision/event order, public and physical boundary chains, residuals,
  minute counts, target-deficit attribution, support authority, energy aggregates, and
  separate controller/solver/dispatcher latencies. A fixed one-shot runner now binds the
  exact cell, protocols, dependency lock, source revision, non-overwriting publication, and
  S1--S8 decision rule. All twelve outcome-free gates pass at clean implementation revision
  `a968a343`; exactly one registered mechanism run is admitted but remains unopened.
- [`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_CLARIFICATION_001.md`](CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_CLARIFICATION_001.md)
  preserves the original protocol while replacing its impossible byte-identical cross-version
  model/plan requirement with a frozen three-boundary structural-equivalence test. It binds
  exact historical and new identities, requires identical forecast projections, normalized
  models, and complete selected-plan quantities, and retains truthful version-separated
  provenance.
- [`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_ADMISSION.md`](CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_ADMISSION.md)
  maps every outcome-free implementation gate to executable evidence and freezes the
  one-shot runner's clean-source, failure-retention, publication, and S1--S8 contracts. It
  records the exact clean revision and verification evidence authorized for one execution.
- [`CAUSAL_SERVICE_MPC_SETPOINT_EXECUTION_REGISTRATION.md`](CAUSAL_SERVICE_MPC_SETPOINT_EXECUTION_REGISTRATION.md)
  prerecords the exact admitted source revision and external destination for the single
  rate-35/seed-0 mechanism execution before its outcome is opened.
- [`results/v2_service_mpc_setpoint_a968a34.md`](results/v2_service_mpc_setpoint_a968a34.md)
  records the first-and-only mechanism result: all S1--S8 gates pass and full delivery is
  restored, but A0 already reaches the same delivery ceiling with lower energy intensity
  and battery throughput. The setpoint architecture is accepted; superiority is not.
- [`COMMON_SETPOINT_AUTHORITY_PROTOCOL.md`](COMMON_SETPOINT_AUTHORITY_PROTOCOL.md)
  freezes phase 1 of controller parity: a controller-neutral four-target value, the exact
  shared allocation kernel, and a normalized RL adapter that cannot command additive battery
  support or inherit MPC provenance. The phase is now implemented outcome-free: MPC delegates
  to the shared kernel, historical Gym/checkpoint semantics remain unchanged, and the complete
  1,106-test repository gate passes.
- [`COMMON_SETPOINT_COMMAND_PROTOCOL.md`](COMMON_SETPOINT_COMMAND_PROTOCOL.md)
  freezes the separately versioned observation-bound command, common minute dispatcher,
  opt-in four-target Gym door, A0 authority adapter, MPC provenance adapter, and outcome-free
  equivalence gate. The complete door is now implemented outcome-free: normalized RL actions,
  transparent A0 targets, and MPC provenance all reach the same dispatcher, guard, and plant;
  the historical environment and checkpoint identity remain isolated.
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
