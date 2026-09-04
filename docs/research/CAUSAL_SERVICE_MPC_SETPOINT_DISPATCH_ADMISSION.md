# Absolute-setpoint MPC implementation admission audit

Status: **candidate implementation complete; final clean-revision verification pending**

Date: **2026-09-04**

This document audits whether the outcome-free implementation obligations in
`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md` and clarification 001 have executable
evidence. It does not contain, authorize by itself, or infer the result of the registered
72-hour mechanism cell.

## Frozen authority

| Item | Identity |
|---|---|
| Dispatch protocol | `21b12eaa245b21d20d9458b9e3d0b8d163c682ef85ad9d5579fde392647e8782` |
| Clarification 001 | `5aecfc8e87b87deeb8fb03a4f226854f1b6245ec762209325f88c9c08aaea349` |
| Dependency lock | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| One-shot contract | `421561b2ee75650024c11fdf55d806012275b44838a33f6b3714635fa689e304` |
| Registered cell | rate 35 kg/h, seed 0, 72 h, 288 decisions, 4,320 physical minutes |
| Manifest | `1e036f69c0514ad34a307b2d213fab3c781561ab6ad1efd199ed8313eca8dae4` |
| Exogenous episode | `2e0b7fd35b85c2a0b9d220c32151a00f96adbd7e24e610bb3b17740345903368` |

The executable runner is `plant_backend.rl.v2.mpc_setpoint_experiment`. Its public command
accepts only `--output`; the controller, settings, seed, episode, gates, lock, and protocol
identities cannot be selected from the command line. The destination must be a new absolute
JSON path outside the repository. Publication uses an atomic no-overwrite link, and the
runner requires the same complete, clean Git worktree identity before execution, before
publication, and after publication.

## Outcome-free implementation gates

| Gate | Executable evidence | Audit assessment |
|---|---|---|
| 1. Historical action compatibility | `test_historical_action_doors_do_not_emit_setpoint_fields`; `test_v1_and_v2_are_explicit_coexisting_contracts`; `test_v2_passes_the_gymnasium_contract_checker`; complete regression suite | Implemented; final clean run pending. |
| 2. Setpoint schema rejection | `test_setpoint_schema_rejects_invalid_or_unbound_commands`; `test_realization_rejects_identity_and_legacy_stage_substitution` | Implemented; negative, non-finite, over-envelope, policy, and identity substitutions are rejected. |
| 3. Dispatcher operating regimes | `test_surplus_dispatch_tracks_targets_without_duplicate_support`; `test_deficit_dispatch_uses_support_only_for_the_target_shortfall`; `test_unsupported_loads_share_scarcity_proportionally`; `test_reserve_bound_shortfall_is_explicit_and_never_overoffered`; `test_night_dispatch_can_reach_the_stack_target_from_reserve`; `test_derated_operator_envelope_is_respected_by_target_and_support`; `test_offline_stack_is_attributed_to_the_unchanged_guard`; `test_dispatch_grid_preserves_every_frozen_invariant` | Implemented across day/night, surplus/deficit, reserve, offline, derated, and invariant-grid fixtures. |
| 4. Four-minute 600 kW cold start | `test_four_fluctuating_minutes_hold_a_600_kw_cold_start_target` | Implemented with support equal only to the contemporaneous shortfall. |
| 5. No additive duplicate power | `test_surplus_dispatch_tracks_targets_without_duplicate_support`; `test_deficit_dispatch_uses_support_only_for_the_target_shortfall` | Implemented as an absolute target, not target plus support. |
| 6. Guard attribution | `test_guard_and_component_refusal_remain_separate_causes`; `test_offline_stack_is_attributed_to_the_unchanged_guard` | Implemented; guarded reduction remains distinct from physical refusal. |
| 7. Ramp/storage attribution | `test_ramp_and_storage_refusal_are_distinct_in_minute_evidence` | Implemented with separate commitment, headroom, and guarded-to-applied evidence. |
| 8. Determinism and future-truth isolation | `test_dispatch_is_byte_deterministic_and_rejects_a_changed_envelope`; `test_private_later_truth_cannot_change_command_or_same_minute_dispatch`; `test_identical_public_input_produces_identical_plan_command_and_identity`; `test_private_episode_identity_cannot_change_same_public_setpoint` | Implemented at dispatcher, controller, and environment boundaries. |
| 9. Three-boundary equivalence | `test_first_three_frozen_boundaries_have_exact_structural_equivalence` under clarification 001 | Implemented: all 24 identities reproduce; projections, normalized models, and complete plan semantics are exact. |
| 10. Solver canonicalization controls | `test_feasible_raw_co2_state_survives_an_invalid_scaled_bound_projection`; `test_feasible_gated_support_noise_still_selects_the_projected_vector`; `test_materially_invalid_raw_vector_remains_rejected`; `test_projection_that_breaks_a_stage_lock_selects_the_raw_vector`; `test_historical_settings_retain_the_strict_projection_failure_contract` | Implemented without loosening solver or physical tolerances. |
| 11. Substitution rejection | `test_minute_dispatch_evidence_rejects_policy_and_reachability_substitution`; `test_minute_and_interval_evidence_reject_substitution_and_chain_breaks`; `test_decision_evidence_rejects_identity_stage_and_closure_substitution`; `test_run_builder_rejects_public_chain_and_latency_substitution`; `test_outer_and_graph_digest_substitution_fail_closed`; `test_artifact_report_digest_rejects_substitution` | Implemented from command through published run artifact. |
| 12. Clean-revision quality gate | Focused tests, complete tests, Black, compilation, strict targeted MyPy, lock digest, `git diff --check`, and clean Git status | Pending the final committed-revision verification recorded below. |

## One-shot outcome contract

The runner computes S1--S8 directly from deep-validated typed run evidence and complete
episode metrics. S1 requires exactly 288 closed decisions, 576 successful stages, 4,320
ordered minute applications, exact identity-list cardinalities, and zero fallback. If S1
fails, S2--S8 are recorded as `not_evaluated`; partial physical outcomes cannot masquerade
as a complete mechanism result.

After S1, the gates remain independent:

- S2 checks pre-guard overshoot and simultaneous battery charge/support;
- S3 checks reachable target tracking and complete deficit decomposition;
- S4 checks productive operation, aborted starts, and forced stops;
- S5 checks reserve penalty, minimum and terminal SOC, invalid decisions, and shutdown;
- S6 checks digest, order, accounting, decomposition, and support authority;
- S7 reports the frozen delivery-direction diagnostic without upgrading it to superiority;
- S8 reports controller and inner-dispatch latency separately.

A controller, environment, reconciliation, or evidence-construction exception is retained
as a typed runner failure. If the evidence graph itself cannot be validated, raw controller
events, completed dispatcher timings, and any unreconciled interval execution are retained
under an explicitly unvalidated failure field. The runner never substitutes a fallback.

## Authorization rule

This candidate becomes authorized for **one** execution only after every gate in the table
above, including gate 12, passes from a committed clean revision. Authorization does not
permit a same-revision retry. This is a governance constraint rather than a claim of global
CLI enforcement: an operator could select another external directory that a stateless
process cannot discover. The first artifact's exact destination must therefore be recorded
at authorization, and that result must be retained whether it passes, fails during
execution, or completes with one or more failed S-gates.

No multi-seed matrix, PPO training, HEPA fitting, environment modification, or controller
tuning is authorized by this audit. The first result selects the next work package according
to the frozen protocol rather than being optimized after inspection.

## Final clean-revision verification

Pending. No 72-hour setpoint trajectory has been opened.
