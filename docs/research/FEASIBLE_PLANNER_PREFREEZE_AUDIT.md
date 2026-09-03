# Feasible planner pre-freeze audit

Date: **2026-09-03**

Status: **all version-1 pre-outcome obligations are represented by executable tests;
final authorization still requires these checks to pass again from the clean committed
execution revision. No registered 72-hour planner cell has been opened.**

## Purpose and evidence boundary

This audit maps every item in `FEASIBLE_PLANNER_PROTOCOL.md` section
“Required pre-outcome validation” to maintained evidence. It is a code-freeze gate, not
a planner result. Exact-plant tests use only explicitly noncanonical two-, five-, or
seven-decision episodes. Optimisation tests use hand-solvable noncanonical LP fixtures.
Worker tests replace episode sampling and the solver or reject the request before either
operation. No test invokes the registered worker on a valid rate/seed pair.

The development matrix remains exactly:

```text
delivery rates: 35.0 and 42.0 kg CH4/h
seeds:          0, 1, and 2
horizon:        72 h
decisions:      288 at 15-minute cadence
physics:        4,320 one-minute steps
```

These six cells are already development-contaminated and can answer only the engineering
questions P1--P3 in the planning protocol. They are not held-out, prospective, field,
financial, or commission evidence.

## Requirement-to-evidence matrix

| # | Frozen obligation | Maintained executable evidence | Audit conclusion |
|---:|---|---|---|
| 1 | Canonical eligibility and fault rejection | `test_early_admission_separates_registered_runs_from_small_fixtures`, `test_untouched_sample_episode_is_verified_by_default`, `test_default_canonical_check_rejects_a_mutated_sample_episode`, `test_faulted_episode_fails_closed_instead_of_ignoring_capacity_loss` | Registered and fixture scopes are disjoint; altered or faulted episodes fail closed. |
| 2 | Stable settings, library, action, source-lineage, state, outcome, and result hashes | `test_protocol_anchors_and_registered_payloads_are_exact`, `test_raw_canonical_hash_and_settings_revalidation_are_detached`, `test_frozen_action_library_and_a0_payloads_are_exact`, `test_exact_serializer_preserves_binary64_and_rejects_open_grammar`, `test_replay_core_and_runner_records_validate_and_detach` | Frozen identities reconstruct from exact canonical preimages and reject mutation. |
| 3 | Template order/formulas, deduplication, share closure, sustainable support, partial holds, sunrise routing, and projection binding | the complete `test_rl_v2_planner_actions.py` suite, especially the frozen reference-vector, sunrise, degraded-capacity, all-template, and partial-hold tests; plus `test_shortened_final_macro_and_counter_formula_are_exact` | Every action source and numerical routing boundary is executable and binary64-sensitive. |
| 4 | Clone equivalence, nested isolation, sibling independence, and deterministic evolution | `test_clone_allowlist_rejects_path_poisoning_and_mutable_closure`, `test_optimized_clone_matches_deepcopy_and_isolates_operational_state`, and the candidate ownership/cleanup tests | Search clones match ordinary deep copy while mutable operational state remains isolated. |
| 5 | Exhaustive agreement when the beam covers the tiny tree | `test_primary_terminal_stream_matches_exhaustive_one_macro_choice` and `test_two_macro_search_matches_an_independent_exhaustive_enumerator` | The generic beam agrees with independent exhaustive enumeration in its exact coverage regime. |
| 6 | Exact A0 generation/replay and survival through every beam cut | `test_exact_a0_port_beam_polish_and_fresh_replay_integrate_on_tiny_plant`, `test_a0_cursor_injection_and_coordinate_fixed_regions_use_exact_states`, and `test_real_nonterminal_beam_retains_a0_and_releases_every_candidate` | A0 has an independent replay, remains outside pruning, and is injected from the exact rolling cursor. |
| 7 | Delayed cold-start paths survive protected-mode selection | `test_a0_and_each_commitment_mode_survive_a_pressured_cut` plus `test_exact_replay_crosses_startup_settlement_sunrise_and_partial_macro` | Mode lanes preserve the non-myopic precursor and the exact plant fixture executes a real cold start. |
| 8 | State-signature sensitivity to all declared mutable state | the field-by-field signature tests in `test_rl_v2_planner_state.py` covering inventories, commitment state/timers, prior power, ledgers, metric histories, events, and throughput | No known decision-relevant mutable field is omitted from the deduplication identity. |
| 9 | Deterministic duplicate survivor and frontier order | `test_same_state_dedup_is_insertion_independent_and_releases_losers`, `test_exact_a0_prefix_wins_an_equal_rank_state_duplicate`, and `test_frontier_order_is_a0_then_modes_then_global_rank_fill` | Insertion order cannot choose the retained state; A0 and mode precedence are explicit. |
| 10 | Direct binary64 final ordering, exact service fractions, every field and tie rule | the final-score, service-cross-product, signed-zero, lineage, and comparison tests in `test_rl_v2_planner_schema.py` and `test_intermediate_rank_and_complete_trace_comparators_are_exact` | The ten-field comparator is exhaustive and does not collapse one-ULP distinctions. |
| 11 | Coordinate boundaries, one sweep, strict acceptance, fixed regions, and terminal discipline | `test_coordinate_polish_uses_two_origins_fixed_blocks_and_strict_acceptance`, `test_byte_identical_beam_origin_is_elided_and_equal_trials_retain_a0`, and the exact fixed-prefix/suffix runtime integration | Coarse/fine blocks follow the frozen schedule and cannot mutate the untouched trace region. |
| 12 | Exact terminal replay across startup, settlement, sunrise, and a partial final block | `test_exact_replay_crosses_startup_settlement_sunrise_and_partial_macro` | One seven-decision exact trajectory crosses dawn, accumulates cold-start exposure, settles at physics step 105, ends through a three-decision macro, and reproduces all terminal payloads. |
| 13 | Reject altered trace length/order/content, one-ULP state changes, wrong reset identity, and delivery-audit mutation | `test_exact_replay_throughput_includes_charge_then_discharge`, `test_runtime_admission_regenerates_identity_before_any_plant_step`, `test_runtime_admission_rejects_rehashed_manifest_and_live_reset_mutation`, `test_replay_and_core_reject_identity_action_binding_and_audit_tampering`, and the one-ULP signature/score tests | Omitted, appended, reordered, and changed actions plus identity/audit corruption all fail before promotion. |
| 14 | Throughput includes a charge-then-discharge sequence | `test_throughput_and_final_score_use_only_authoritative_sources` and `test_exact_replay_throughput_includes_charge_then_discharge` | Both isolated arithmetic and a real two-decision sunset plant trace include bus-side charge and support discharge, then replay bit-exactly. |
| 15 | Stable phase-specific failures for limits, memory, clone, generation, step, source, and replay | the failure-path tests across `test_rl_v2_planner_search.py`, `test_rl_v2_planner_runtime.py`, and `test_rl_v2_planner_evidence.py`, including edge ceiling, resource cleanup, timing phase, external watchdog, malformed child, and missing end provenance | Failures retain stable phase/code/counters and release owned plant environments; no best-so-far schedule is relabelled complete. |
| 16 | Detached immutable results and validating writer before file creation | `test_replay_core_and_runner_records_validate_and_detach`, `test_atomic_writer_reopens_and_revalidates_external_artifact`, `test_writer_rejects_repository_and_relative_destinations`, and writer-cleanup failure tests | Only a strictly reconstructed graph can be atomically written outside Git and accepted after reopening. |
| 17 | Privileged result rejected by causal evaluation | `test_privileged_planner_artifact_is_rejected_by_causal_controller_runner` | The causal runner rejects the artifact before environment reset or any plant decision. |

## Physical boundary fixture details

The seven-decision test starts at local 05:45 and ends at 07:30. Its first four-decision
macro uses the frozen full-stack template, spanning the zero-solar boundary and a real
one-hour cold-start sequence. Its terminal three-decision macro uses the charge template.
The only delivery settles at exact physics step 105. A fresh finalist replay must match
the evaluated request trace, source lineage, metrics, simulator truth, telemetry, ordered
delivery audit, throughput, score, and outcome fingerprints.

The independent two-decision throughput test starts at local 18:45. The charge template
first increases physical battery energy; the full-stack template then reduces it and
records positive battery discharge. The planner accumulator must exceed discharge alone
because it also includes the preceding bus-side charge. Omitted, appended, reordered,
and behaviorally changed variants are rejected. The original trace then reproduces the
terminal throughput as the same binary64 value on a fresh reset.

## Candidate freeze checks

On the complete candidate working content:

- the seven planner schema/action/state/search/runtime/evidence/upper-bound files pass
  **266 tests**;
- the complete repository passes **879 tests with 2 skips**;
- the sole warning is the existing Starlette `TestClient` deprecation;
- planner source formatting, strict targeted type checking, Python compilation,
  dependency-lock validation, and whitespace validation pass;
- all exact-plant outcomes are at most seven decisions and explicitly carry
  `noncanonical_unit_fixture` scope.

These counts must be reproduced after this audit and its tests are committed, with a
clean worktree, before invoking `make v2-planner`. A failing gate reopens the freeze and
forbids execution until corrected and recommitted.

## Execution discipline after freeze

1. Run one registered cell per fresh process and one external JSON artifact.
2. Require `evidence_ready=true`; retain and report any failure artifact without using a
   partial schedule or rerunning selectively for a stronger result.
3. Complete all six declared cells unless a frozen failure rule prevents continuation.
4. Verify each artifact digest and complete source/runtime boundary before aggregation.
5. Report `A0`, `F`, `U`, `F-A0`, `U-F`, bracket status, terminal inventories, battery
   throughput, curtailment, and commitment exposure for every cell.
6. Keep these results labelled inspected development evidence. Freeze new seeds,
   regimes, thresholds, and failure rules before any prospective comparison.
