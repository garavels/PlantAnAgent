# Causal service-MPC actuation-bridge protocol

Status: **frozen before implementation and before the bridge mechanism probe**

Date: **2026-09-04**

## Motivation and claim boundary

The execution-valid replication at source `3796eb5` passed its completeness and latency
gates but failed every controller-performance gate. Its compact report has internal digest
`5a5acf5f46ea0363bfee90925b1346d33b11ffc0766fb214ee55d8baaee2df13`.
Across the six MPC rollouts, 204 of 206 physical cold starts aborted. The result and
post-outcome diagnosis are recorded in
`results/v2_service_mpc_replication_3796eb5.md`.

This protocol is a **development mechanism test on an already inspected cell**. It is not
a new replication, a holdout result, controller selection evidence, or a commercial claim.
Its purpose is to decide whether the existing share-based supervisory interface can express
the first MPC target reliably enough to justify further work.

## Frozen causal hypothesis

The current optimizer and plant assign different meanings to `battery_support_kw`:

- the MILP variable is expected battery discharge in a 15-minute mean power balance;
- the plant action is a contingent top-up ceiling, from which only the instantaneous
  uncovered electrolyser demand is discharged; and
- first-action realization copies the expected flow into the authority ceiling.

`H_bridge`: when planned discharge is smaller than the electrolyser target, this mapping
removes the authority needed to cover one-minute solar variation. It causes commitment
shortfall and repeated startup aborts even when the model's 15-minute balance is valid and
the battery is above reserve.

The falsifiable prediction is that separating expected discharge from contingent authority,
without changing the plan, will materially reduce forced/aborted starts and create stable
productive intervals on the frozen rate-35/seed-0 development trajectory.

## Permitted implementation scope

Only the controller-to-plant realization boundary may change:

1. Preserve the four planned solar-route shares and their existing binary64 closure.
2. Retain `first_step.battery_support_kw` as the expected discharge used by the MILP.
3. Add a separately named, hash-bound contingent support-authority value to requested-action
   evidence.
4. For an active first-step electrolyser target, set authority to
   `min(active support cap, startup power + productive power)`. For an inactive target,
   authority is exactly zero so an off/stop request remains reachable.
5. Continue passing the resulting ordinary five-field `Action` through the unchanged safety
   guard. The guard remains authoritative for component limits, battery availability, and
   reserve SOC. The simulator continues to withdraw only support actually required.
6. Version the action-realization domain and descriptor so output cannot be confused with
   the `3796eb5` controller, while retaining the original Git result.

The following are forbidden in this work package:

- changes to the MILP variables, constraints, objectives, weights, horizon, solver, or
  feasibility tolerances;
- changes to forecast construction or information available to the controller;
- changes to plant physics, commitment timers, shortfall grace, component sizes, safety
  limits, reward, demand, or initial-state sampling;
- seed-specific behavior, future weather truth, fallback, or post-hoc action repair; and
- PPO training, HEPA changes, NLP/supervisor changes, or additional environment realism.

## Required executable evidence

Before a full development probe, tests must establish:

1. inactive plans retain zero support authority and an exactly unchanged low-level action;
2. active plans retain the original four shares and expected discharge while exposing the
   separately bounded authority;
3. a 600 kW startup target with a smaller expected support flow remains funded under a
   lower realized solar step, while actual battery discharge equals only the uncovered gap;
4. the reserve guard can still reduce or remove authority and that intervention is reported;
5. action, decision, request, guarded-action, interval-power, residual, and run digests
   cross-bind the new schema without accepting legacy-schema substitution;
6. identical public inputs remain deterministic and private truth cannot alter a request;
7. the original model and plan identities are unchanged for an identical public boundary;
   and
8. the focused MPC suite and complete repository regression pass from a clean revision.

No probe is authorized if any requirement fails.

## Frozen development mechanism probe

Exactly one 72-hour cell is permitted initially:

- delivery rate: `35.0 kg/h`;
- seed: `0`;
- decision interval: `900 s`;
- expected decisions: `288`;
- expected solver stages: `576`;
- controller: the versioned commitment-aware realization with otherwise default frozen MPC
  settings; and
- comparator: the immutable `3796eb5` MPC artifact, raw SHA-256
  `80f5d27af14d4d7633e194ed49307b31de96bd2c164485fcba51762d99333a13`.

The new probe must run once, in a fresh process, to a new absolute path outside Git. Source
revision, clean-worktree identity, dependency-lock digest, runtime, controller descriptor,
settings, episode manifest, exogenous episode identity, timestamps, decision ledger,
application ledger, and artifact digest must be retained. No source edit is permitted after
execution starts. If it fails, the partial failure evidence is the result; no fallback or
same-revision retry is allowed.

## Predeclared interpretation gates

| Gate | Criterion |
|---|---|
| B1 — execution | 288 successful decisions, 576 successful solver stages, every transition reconciled, zero fallback. |
| B2 — mechanism | At least one productive first interval; at most four forced stops and at most four aborted starts, versus 41/41 in the frozen comparator. |
| B3 — safety | Zero invalid/out-of-range action, zero shutdown step, no SOC below the active 15% reserve, terminal SOC at least 25%. |
| B4 — direction | Delivered methane is strictly greater than 647.7138003263635 kg, the frozen controller result. This is diagnostic direction only, not a superiority claim. |
| B5 — accounting | Planned support, authorized support, and applied support are all present and digest-bound; applied support never exceeds guarded authority apart from declared numerical tolerance. |
| B6 — latency | Decision p99 at most 2 s and maximum at most 5 s. |

These gates are reported independently. Passing B2 does not imply B4, and passing B4 does
not establish value against A0. The A0 result for this cell remains 2,520 kg with 12/12
commitments and is context, not the mechanism-probe acceptance threshold.

## Decision rule

- If B1, B2, B3, B5, and B6 pass, retain the bridge as a viable compatibility layer and
  preregister a broader development screen before making any controller claim.
- If B2 fails while B1 passes, stop patching share realization and design an explicit
  absolute-process-setpoint contract with a deterministic one-minute inner dispatcher.
- If B2 passes but B3 or B5 fails, reject the bridge because it closes commitment by hiding
  unplanned energy or weakening safety/accounting.
- If the mechanism gates pass but B4 fails, preserve the bridge and localize the next
  model/plant mismatch; do not tune objective weights from this one cell.

Only after the actuator decision is closed may the programme strengthen cyclic economics,
add the next realism feature, and define an RL benchmark against the resulting credible
model-based controller.
