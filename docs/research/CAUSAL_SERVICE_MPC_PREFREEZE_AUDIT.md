# Causal service-MPC pre-freeze audit

Date: **2026-09-03**

Status: **all preregistered implementation-gate obligations are represented by
executable evidence; registered execution remains forbidden until the complete gate is
repeated from a clean committed revision. No registered 72-hour service-MPC cell has
been opened.**

## Purpose and evidence boundary

This document maps every item in `CAUSAL_SERVICE_MPC_PROTOCOL.md` section
“Implementation gate before full rollouts” to source and executable evidence. It is a
source audit, not a controller result. The audit reads from the frozen protocol into the
implementation and then from each implementation boundary back into an independently
asserted test. It does not infer correctness from solver status or from a favorable plant
outcome.

All optimization and exact-plant executions used before the freeze are deliberately tiny
fixtures. The longest exact-plant controller fixture is one hour (four decisions and 60
one-minute physics steps). The longest optimization fixture is ten 15-minute intervals.
The registered matrix remains exactly:

```text
delivery rates: 35.0 and 42.0 kg CH4/h
seeds:          0, 1, and 2
horizon:        72 h
decisions:      288 at 15-minute cadence
physics:        4,320 one-minute steps
order:          35/0, 35/1, 35/2, 42/0, 42/1, 42/2
```

Those six cells are already development-contaminated by the earlier transparent-controller
and feasible-planner work. They can answer the frozen engineering gate only. They are not
held-out, prospective, field, economic, commercial, or commission-performance evidence.

## Requirement-to-evidence matrix

| # | Frozen obligation | Maintained executable evidence | Audit conclusion |
|---:|---|---|---|
| 1 | Forecast reconstruction across darkness, boundaries, bands, shortened horizons, impossible issues, and exact energy | `test_projection_preserves_every_band_and_public_clear_sky_shape`, `test_projection_handles_sunrise_sunset_and_midnight_without_point_sampling`, `test_shortened_horizon_is_a_byte_identical_prefix_without_renormalization`, `test_projection_rejects_public_energy_above_clear_sky_support`, and `test_clear_sky_integral_is_additive_across_midnight` | `mpc_forecast.py` uses analytic clear-sky interval integrals, preserves every issued mean band within the frozen tolerance, and binds the trajectory to the complete public issuance. |
| 2 | Counterfactual causal isolation from private truth, seed, and future vintages | `test_private_truth_seed_and_future_vintages_cannot_change_same_public_request`, `test_identical_public_inputs_produce_identical_plan_action_and_evidence_identity`, `test_projection_identity_depends_only_on_declared_public_inputs`, and the foreign-issuance case in `test_model_rejects_inputs_outside_its_declared_validity_domain` | Two environments with different private truth, exogenous identity, forecast streams, and future issue identities produce byte-identical requests and semantic evidence when given the same detached public boundary. A projection from a different complete issue is rejected even when its used mean path is numerically identical. |
| 3 | Hand-solvable abundant, zero-resource, battery-only, delivery-order, storage, stoichiometry, and terminal-SOC behavior | the complete `test_rl_v2_mpc_physics_fixtures.py` suite, especially `test_zero_resource_and_no_usable_battery_has_exact_zero_service`, `test_abundant_solar_converts_exact_stoichiometric_endowment_before_delivery`, `test_full_methane_tank_blocks_same_interval_production_before_settlement`, `test_battery_endowment_improves_service_but_cannot_cross_terminal_floor`, and `test_battery_only_endowment_produces_hydrogen_and_closes_energy_balance` | Exact expected mass and energy balances are asserted independently of the objective. The fixtures distinguish pre-conversion tank headroom from pre-delivery settlement, isolate battery discharge at zero solar, and enforce the terminal battery floor. |
| 4 | Every initial commitment mode/timer, startup energy and delay, minimum up/down behavior, and no simultaneous battery charge/discharge | all six parameter rows in `test_public_commitment_state_is_mapped_to_conservative_fixed_prefixes`, `test_partial_initial_start_uses_a_full_stable_offer_conservatively`, `test_model_prohibits_late_starts_that_hide_terminal_obligations`, both rows of `test_cold_start_delay_energy_and_minimum_times_are_hand_solved`, and the battery-direction assertions in the physics and solver fixtures | OFF, COLD_STARTING, RUNNING, HOT_STANDBY, HOT_STARTING, and COOLDOWN public states map to explicit fixed prefixes. Real MILP solves exhibit the four-interval powered startup, two-interval remaining down lock, four running intervals, exact startup energy, and exclusive charge/support direction. |
| 5 | Fail closed for malformed/unsupported input, missing runtime, infeasibility, timeout/non-optimal status, non-finite or invalid results, and invalid first action | `test_model_rejects_inputs_outside_its_declared_validity_domain`, `test_missing_optional_runtime_raises_without_a_fallback`, `test_infeasible_primary_stage_retains_failed_solver_evidence`, `test_solver_exception_is_typed_and_retains_stage_evidence`, `test_status_zero_solution_must_pass_the_independent_residual_audit`, `test_time_limit_status_is_retained_without_a_solution_or_fallback`, `test_nonfinite_status_zero_vector_is_retained_as_validation_failure`, `test_secondary_failure_retains_successful_primary_stage`, `test_plan_extraction_failure_retains_both_successful_stages`, and `test_first_action_requiring_sanitizer_repair_fails_closed_with_plan_evidence` | No rejected solve or action emits an MPC request or hidden fallback. Ordered evidence retains every attempted successful prefix and the final rejected stage, including a status-zero result rejected by independent residual validation. |
| 6 | Deterministic model, plan, action, and semantic evidence identities across reconstruction and repeated solves | `test_model_has_a_reproducible_complete_sparse_identity`, `test_plan_binds_stages_steps_and_delivery_totals`, `test_identical_public_inputs_produce_identical_plan_action_and_evidence_identity`, and `test_first_decision_is_valid_and_fully_hash_bound` | Canonical model bytes and semantic solution identities are stable. Wall-clock latency remains serialized evidence but is excluded from plan and semantic-decision hashes; a changed solution digest changes the plan identity. |
| 7 | One-step and short-horizon comparison of planned quantities with exact guard/plant accounting | `test_one_interval_execution_closes_the_first_plan_against_public_truth`, `test_complete_short_episode_reconciles_every_plan_with_public_plant_truth`, and `test_next_compute_auto_reconciles_an_unobserved_pending_transition` | The controller closes each planned first interval only from the next detached public boundary. The exact evaluator separately binds guarded requests and per-physics-step application; residuals are diagnostics and are never rewritten as model feasibility. |
| 8 | Complete evaluator evidence, failed-solve retention, no fallback, source boundaries, pairing, immutable descriptors, and no incomplete aggregate | the complete `test_rl_v2_mpc_evaluation.py` suite; `test_paired_controller_evaluation_records_declared_information_and_settings`; `test_controller_failure_is_retained_and_pair_is_explicitly_unavailable`; `test_failure_in_any_planned_arm_invalidates_every_aggregate_contrast`; `test_pair_hash_mismatch_is_not_claimed_as_common_random_numbers`; and `test_executed_descriptor_or_copy_mismatch_fails_closed` | Successful runs bind every decision to its exact requested, guarded, and applied execution. Failed cells retain their valid prefix, report zero fallback, invalidate paired summaries, and cannot borrow a mismatched exogenous identity. Start/end worktree identities and descriptor integrity are checked independently by the generic causal evaluator. |
| 9 | Formatting, strict typing, compilation, dependency lock, focused and complete tests, and whitespace | the commands recorded under “Candidate freeze checks”; final counts must be reproduced after commit | This obligation is procedural and is not satisfied by unit-test logic. Any failure reopens the implementation and forbids the registered matrix. |
| 10 | Independent source audit of all preceding obligations | this matrix plus the module-boundary review below | Every protocol item maps to named maintained evidence. The audit identifies limits rather than promoting tiny fixtures into performance claims. |

## Module-boundary review

The source responsibilities remain deliberately separated:

- `mpc_forecast.py` accepts one detached public issue and static solar geometry. It has no
  environment, seed, private weather tape, future issue, or solver access.
- `mpc_schema.py` owns immutable settings, plan, stage, and failure contracts. A stage is
  successful only with status zero, finite optimization evidence, and a solution digest;
  a typed solve error with stage evidence must end in an unsuccessful stage.
- `mpc_model.py` converts only the current public observation, its issue-bound projection,
  static plant/safety data, public commitment profile, and settings into canonical sparse
  algebra. It does not import SciPy.
- `mpc_solver.py` is the sole optional SciPy/HiGHS boundary. It translates the immutable
  algebra, performs both lexicographic stages, recomputes objectives and residuals, and
  never substitutes a plan or action after failure.
- `mpc_execution.py` turns only the selected first interval into the existing five-field
  request, closes the subsequent public residual, and builds exact application/run
  evidence. Failure and run format identities are centralized constants.
- `mpc_controller.py` owns only the receding-horizon clock, public evidence ledger, and
  one pending reconciliation. Its immutable descriptor declares the success, failure,
  and run evidence contracts plus the no-fallback policy.
- `evaluation.py` owns the environment and privileged per-step `info`. It never returns
  that private execution record to the controller; it uses it only to audit the public
  residual and exact request-to-plant path.

The independent sparse residual audit is not a proof that the 15-minute approximation is
the one-minute plant. The deliberate differences—minute-scale ramping, hot transitions,
shortfall grace, nonlinear part-load efficiency, dispatch priority, and storage refusal—
remain measured model-to-plant residuals. They are candidates for attributed realism
only after the frozen development matrix diagnoses their materiality.

## Candidate freeze checks

Before this audit was committed, the complete seven-file MPC test surface passed **79
tests**. The seven production modules passed strict type checking with transitive legacy
modules treated as external; the two touched integration-test modules passed the same
strict body check with only the untyped external `pytest.mark.parametrize` decorator
diagnostic disabled. All 14 touched Python files passed formatting and whitespace checks.

The committed-source authorization run must repeat:

1. the complete MPC/controller/evaluator focused surface;
2. strict source and test-body type checks under the declared import boundary;
3. Python bytecode compilation;
4. `uv lock --check` without changing `uv.lock`;
5. deterministic formatter and Git whitespace checks; and
6. the complete repository test suite.

Passing those checks authorizes only the six development-contaminated service-MPC cells.
It does not authorize holdout opening, RL training, product claims, or commission claims.

## Execution discipline after freeze

1. Run the six cells in the frozen rate-major, seed-minor order and retain a failed cell
   without selective rerun for a stronger answer.
2. Each cell must contain A0, A2, and service MPC on one exact exogenous episode and must
   retain complete controller and application evidence.
3. Require clean, unchanged start/end worktree identities, the raw `uv.lock` digest, and
   the committed source revision for every output.
4. Attach prior F/U development values only after exact cell/manifest identity matching;
   never expose the privileged trajectory to the controller.
5. Publish every cell, balance, latency, residual, reliability, treatment-expression,
   and preregistered gate decision, including an unfavorable or invalid result.
6. Make no model or threshold change after the first cell. Preserve the full matrix before
   any versioned amendment.
