# Absolute-process-setpoint and deterministic inner-dispatch protocol

Status: **frozen before implementation and before any setpoint-controller rollout**

Date: **2026-09-04**

## Why this work package exists

The execution-valid `3796eb5` service-MPC replication showed that the original held-share
action starved one-minute electrolyser commitments. The separately frozen `709e477`
actuation bridge then removed startup starvation over a 223-decision prefix, but exposed a
more fundamental defect: `battery_support_kw` is additive power, not a target-aware reserve.
The bridge applied 14,410.009 kWh of battery support against 28.500 kWh planned and
overshot positive electrolyser targets by 717.568 kW on average. Its run ultimately failed
B1 when post-solve bound canonicalization broke a raw-feasible CO2 balance.

The immutable result and its post-outcome claim boundary are recorded in
`results/v2_service_mpc_actuation_bridge_probe_709e477.md`. The bridge will not be rerun or
retrofitted with a new acceptance rule. This protocol begins a separately versioned control
architecture.

## Product-level architecture

The intended product sits above millisecond equipment controls and below the human/agent
supervisor:

```text
HEPA risk + operator/agent envelope        advisory / slow supervisory context
                    ↓
service MPC or future RL policy            one causal 15-minute operating intent
                    ↓
absolute process setpoints                 explicit kW targets, not indirect shares
                    ↓
deterministic inner dispatcher             one-minute contemporaneous bus allocation
                    ↓
independent safety guard                    limits, availability, reserve; always final
                    ↓
unchanged physical simulator / future PLC  exact component acceptance and dynamics
```

The natural-language supervisor does not allocate minute power. HEPA does not control the
plant. Either may later change a bounded operating envelope or risk mode. Model-based and
learned controllers must eventually issue the same typed intent through the same dispatcher
and guard so algorithm comparisons do not confound control authority.

This architecture is commercially preferable to a policy-specific action encoding. Plant
integrators already reason about unit setpoints, ramps, interlocks, and dispatch priorities.
A slow optimizer may be replaced without replacing the deterministic realization and safety
layers, while shadow-mode and hardware-in-the-loop evidence can compare requested, guarded,
and applied power directly.

## Frozen hypotheses

`H_setpoint`: an explicit total electrolyser-power target, translated against the current
one-minute solar bus and available reserve, can remove both the old startup undershoot and
the bridge's additive overshoot without changing physical commitment dynamics.

`H_dispatch`: a deterministic, outcome-independent allocation rule can make every deviation
from the supervisory target attributable to one of four causes: contemporaneous resource
shortfall, the independent guard, component/commitment refusal, or numerical tolerance.

`H_canonical`: when a raw solver vector passes the declared model audit but its optional
bound projection does not, retaining the raw feasible vector preserves mathematical
validity. The solver tolerance and physical balance constraints need not be loosened.

These hypotheses are falsifiable independently. Solver completion does not establish
dispatch correctness; dispatch correctness does not establish controller value.

## Work-package boundaries

Permitted changes are limited to:

1. a separately versioned absolute-setpoint command and controller descriptor;
2. a pure deterministic one-minute setpoint-to-`Action` dispatcher in the backend;
3. an opt-in Environment V2 setpoint execution door and its evidence/accounting path;
4. a new service-MPC realization that maps the unchanged first MILP interval to setpoints;
5. feasibility-preserving solver-output canonicalization, with the `709e477` failure bound
   and explicitly evidenced; and
6. focused tests, a one-shot development runner, documentation, and compact result report.

The following are forbidden in this work package:

- changes to the MILP decision variables, constraints, objective coefficients, planning
  horizon, public forecast, delivery schedule, or information set;
- looser solver feasibility, plan-balance, component, reserve, or accounting tolerances;
- changes to simulator physics, commitment state transitions, ramp limits, startup energy,
  part-load efficiency, equipment sizes, reward, or initial-state/weather sampling;
- seed-specific behavior, future weather truth, hidden fallback, or a second action after
  observing an interval outcome;
- tuning the service objective from the bridge result;
- PPO training, HEPA fitting, supervisor/NLP changes, or added environment realism; and
- changing the five-dimensional Gym action or historical share-action execution path.

The `709e477` bridge controller and artifact remain historical evidence. The new controller
must use a new identifier and cannot overwrite that semantic version.

## Absolute setpoint command

The immutable command schema is `service-mpc-process-setpoints-v1`. It binds:

- controller and selected-plan SHA-256 identities;
- the public forecast issue and first projected interval identity;
- decision index and intended 15-minute hold duration;
- total electrolyser input target in kW, equal to first-step startup plus productive power;
- DAC and methanator input targets in kW;
- a battery-charge target ceiling in kW;
- the MILP's expected battery-support flow in kW, retained as planned energy rather than
  actuator authority;
- all four original planned solar routes and planned curtailment for residual diagnosis;
- active process/support/reserve constraints; and
- inner-dispatch policy identifier
  `reserve-aware-proportional-unsupported-loads-v1`.

Every number must be finite and non-negative. Process setpoints cannot exceed the active
equipment envelope. Expected support cannot exceed the electrolyser target or support cap.
An inactive electrolyser target is exactly zero and remains an explicit reachable stop/off
request. The setpoint command contains no realized solar value and no future truth.

## Deterministic one-minute dispatcher

The inner dispatcher is a pure function of the held setpoint command, the current detached
plant boundary, active constraints/configuration, and the exact solar bus about to be
dispatched in that same physical minute. This contemporaneous bus value is already available
to the safety layer; exposing it to the actuator does not expand the 15-minute controller's
forecast information.

Let:

- `E`, `D`, `M`, and `C` be electrolyser, DAC, methanator, and battery-charge setpoints;
- `S` be current available solar kW; and
- `B` be battery support currently available above reserve, bounded by the support cap,
  discharge rating/efficiency, electrolyser envelope, and current health.

The exact allocation is:

1. Reserve `min(S, max(0, E - B))` solar kW for the electrolyser. This is the minimum solar
   required to keep `E` reachable with currently available support.
2. Allocate the remaining solar to unsupported process targets `D` and `M` with the common
   factor `min(1, remaining / (D + M))`; if `D + M` is zero, both allocations are zero.
   This proportional rule avoids a hidden DAC-versus-methanator ordering preference.
3. Allocate any remaining solar to the electrolyser up to `E`, reducing battery support.
4. Request exactly `min(B, E - electrolyser_solar)` battery support.
5. Only if the electrolyser needs no support, allocate remaining solar to battery charging
   up to `C`. Leave any residual solar unallocated for the existing spill/curtailment path.
6. Convert the four solar kW routes into a downward-closed unit simplex using `S`; at night
   all shares are zero. Emit one ordinary five-field `Action`.
7. Pass that action through the unchanged safety guard immediately before the simulator.

Required invariants, within the existing `1e-7` model-balance tolerance, are:

- requested electrolyser solar plus requested support never exceeds `E`;
- it equals `E` whenever `S + B` and the declared allocation rule make `E` reachable;
- applied support is never requested merely because `E` is positive;
- requested battery charging and support are never simultaneously positive;
- the four shares are finite, non-negative, and sum to at most one;
- DAC/methanator requested power never exceeds its setpoint;
- support never exceeds current reserve-aware availability; and
- the same inputs produce byte-identical commands and evidence.

The inner dispatcher is not an optimizer and contains no reward, forecast, delivery, or
seed logic. It may not repair a plan after observing future physics.

## Safety and authority boundary

Support availability will be extracted into one pure, shared safety calculation so the
dispatcher and guard do not maintain divergent reserve arithmetic. The guard remains
authoritative and reruns every minute against current telemetry and the exact bus. A
dispatcher prediction is never permission to bypass the guard.

Offline/derated equipment, depleted storage, exact commitment/ramp acceptance, and simulator
spill may still reduce applied power. Each cause must remain visible. The setpoint layer may
not classify a guard or component refusal as target achievement.

## Feasibility-preserving solver canonicalization

The existing order—raw audit, optional bound projection, projected audit—is retained. The
selection rule changes narrowly:

1. A raw vector that fails the current absolute-plus-scaled audit is still rejected.
2. Construct the same nearest-bound and near-integral projection as today.
3. Audit the complete projected vector against all model and lexicographic-lock rows.
4. If the projection passes, select it as before.
5. If the projection fails while the raw vector passed, retain the raw vector exactly.
6. Plan extraction and its existing physical-balance/schema validation still run on the
   selected vector. If the raw selection cannot form a valid plan, fail; do not sanitize it.

The settings/descriptor must version this selection rule. Stage evidence must state whether
the projected or raw-feasible vector was selected, the number and maximum magnitude of
proposed projections, and both audit summaries. No tolerance changes are permitted.

Required adversarial fixtures include:

- the 6,000 kg CO2-state case where a `3.495807901682279e-05 kg` raw-feasible state would
  be independently projected to zero and break a material equality;
- the earlier `3.935447649175992e-05 kW` gated support case where projection to zero is
  feasible and is required for a physically valid plan;
- a materially infeasible raw vector, which remains rejected; and
- a projection that preserves rows but changes the locked objective outside its existing
  audit tolerance, which remains rejected through the extra-row audit.

## Evidence schema

Each 15-minute setpoint application retains:

1. public observation, forecast, projection, model, solver-stage, plan, and setpoint digests;
2. all one-minute dispatcher input/output records in order;
3. contemporaneous solar, available support, requested solar routes, support, and unallocated
   solar for each minute;
4. setpoint target, pre-guard offered power, guarded offer, and physically applied power;
5. guard interventions, component refusal/rerouting, commitment state, and storage headroom;
6. integrated planned, target, requested, guarded, and applied kWh by channel;
7. positive and negative target-error integrals, maximum instantaneous overshoot/undershoot,
   and counts; and
8. controller and dispatcher latency distributions separately.

Digests bind setpoint -> minute dispatch -> ordinary request -> guarded action -> applied
power -> next public residual. A complete run must have exactly 288 setpoint decisions and
4,320 ordered minute records for the 72-hour cell. Compact reports may replace raw ledgers
only after deep validation and must retain ledger SHA-256 identities.

## Outcome-free implementation gate

No full setpoint-controller rollout is authorized until all of the following pass:

1. Historical Gym-vector and `step_action(Action)` behavior remains exactly compatible.
2. Setpoint schemas reject non-finite, negative, over-cap, mismatched-plan, and unknown-policy
   records.
3. Day, night, surplus, deficit, reserve-bound, offline, and derated dispatcher fixtures
   satisfy every frozen invariant and energy identity.
4. A 600 kW cold-start target remains exactly 600 kW across a fluctuating four-minute
   sequence when resource/support is sufficient, with support equal only to minute shortfall.
5. A running target with ample solar never receives the bridge's additive duplicate power.
6. The unchanged guard can reduce a request, and the target shortfall is attributed to that
   intervention rather than hidden.
7. Simulator refusal and ramp/storage limits are separately attributed.
8. Identical inputs are deterministic; changing private future truth cannot change the held
   setpoint or same-minute dispatch.
9. The first three frozen seed-0 observation/model/plan identities remain unchanged before
   the controller trajectories diverge.
10. Both canonicalization counterexamples and material-rejection controls pass.
11. Setpoint, minute, guard, application, residual, and run digest substitution tests fail
    closed.
12. Focused suites, complete repository regression, formatting, compilation, strict targeted
    types, lock identity, and Git whitespace pass from a committed clean revision.

Tiny and short-transition tests are allowed during implementation. No full 72-hour
controller outcome may be opened to satisfy this gate.

## First development mechanism run

After the implementation gate, one new external, non-overwriting 72-hour rate-35/seed-0
development run may be preregistered from the clean implementation revision. It remains a
previously inspected mechanism cell, not holdout or controller-superiority evidence. There
is no same-revision retry and no fallback.

The following gates will be bound into its executable runner:

| Gate | Criterion |
|---|---|
| S1 — execution | 288 successful decisions, 576 successful solver stages, 4,320 ordered minute dispatches, every transition reconciled, zero fallback. |
| S2 — no overshoot | Pre-guard electrolyser offer never exceeds its absolute target by more than `1e-7 kW`; zero simultaneous charge/support requests. |
| S3 — reachable tracking | Whenever the dispatcher's declared resource/support feasibility test is true, pre-guard offer equals target within `1e-7 kW`; every deficit is attributed. |
| S4 — mechanism | Productive operation is positive; at most four aborted starts and at most four forced stops. |
| S5 — safety | Zero reserve breach, invalid/out-of-range decision, or shutdown; terminal SOC at least 25%. |
| S6 — accounting | Zero digest, order, energy-conservation, target-decomposition, or authority violations; applied support never exceeds guarded target shortfall. |
| S7 — direction | Delivery strictly exceeds the frozen 647.7138003263635 kg controller result; diagnostic only. |
| S8 — latency | Controller p99 <= 2 s and max <= 5 s; dispatcher p99 <= 5 ms and max <= 25 ms on the recorded runtime. |

All gates are independent after S1. Planned-versus-applied support and target-versus-applied
power are reported even when accounting passes. No ratio is silently treated as acceptable.

## Decision rule

- If S2 or S3 fails, stop: the setpoint seam or dispatcher is wrong. Do not tune the MPC.
- If S6 fails, reject the implementation even when physical service improves.
- If S1 fails, preserve the typed failure and repair only the evidenced implementation
  defect under a separately recorded revision; do not report S2--S8 as complete outcomes.
- If S1--S6 and S8 pass but S7 fails, retain the control architecture and localize the
  remaining model/objective mismatch.
- If all gates pass, run a separately frozen multi-seed development matrix before changing
  the environment or making a controller claim.

Only after target realization is credible may the programme add cyclic economics,
degradation, thermal/methanator dynamics, or other realism. PPO then competes through the
same setpoint/dispatcher/guard interface and must close a measured residual gap left by
forecast-aware heuristic and MPC baselines. HEPA and the language supervisor are evaluated
later as predictive and supervisory layers, never credited for work performed by the inner
dispatcher.
