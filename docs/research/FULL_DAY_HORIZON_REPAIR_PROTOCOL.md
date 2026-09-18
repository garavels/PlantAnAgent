# Full-day service-MPC horizon repair: preregistered mechanism protocol

Status: **frozen after the robustness result and before implementation or any M24 action on
seeds 20--24**

Date: **2026-09-05**

## Decision question

Does extending only the causal service MPC planning horizon from one six-hour commitment period
to the complete public 24-hour window repair the registered full-day service failure without
changing the plant, episode, forecast product, controller objective weights, action authority,
safety guard, or economic task?

This is an outcome-informed engineering repair probe. The failed M6 trajectories are already
known and the five seeds are already development data. The new M24 outcomes have not been run.
The purpose is to isolate whether missing commitment lookahead is sufficient to recover
full-day service before adding product-form terminal constraints, changing the objective, or
training RL.

## Prior evidence and claim boundary

The immutable source result is `results/v2_economic_robustness_7b1b656.md`. Its registered
matrix completed all 20 cells and passed execution, identity, accounting, safety, and authority
gates. M6 was service-comparable to H in all ten six-hour cells and none of the ten 24-hour
cells. Every full-day M6 arm substantially completed its first two commitments, partially
completed its third, and delivered zero at its fourth while retaining battery energy and
convertible feedstock.

The bound reference identities are:

| Evidence | SHA-256 |
|---|---|
| Executed source revision | `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` |
| Robustness runner contract | `6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38` |
| Raw robustness report | `26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755` |
| Report-domain identity | `e0fe451861cda01f0a788adafed9bf2767f778d0141d7c76147b6eb0f3831f7f` |
| Deterministic evidence | `cff97bfc678f0b7ca47c516f3f7a80e39434aae83a6597576b1539eba5110b96` |
| Dependency lock | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |

This probe may establish that a specific causal horizon repair restores service on contaminated
development cells. It cannot establish controller superiority, RL value, statistical
generalization, calibrated economics, site transfer, commercial savings, or commission merit.

## Single admitted intervention

The new arm is labelled **M24**. It is the existing
`ServiceMPCSetpointControllerV1` with:

```text
task_horizon_h     = 24
planning_horizon_h = 24
interval_s         = 900
```

Every other `ServiceMPCSetpointSettingsV1` field remains byte-for-byte at its current default:

```text
terminal_battery_soc_target       = 0.25
delivery_lock_tolerance_kg        = 1e-6
energy_reconstruction_tolerance  = 1e-12
solver_presolve                   = true
solver_mip_rel_gap                = 1e-6
solver_mip_feasibility_tolerance = 1e-7
solver_time_limit_s               = 2
minimum_fill_weight               = 100
terminal_ch4_equivalent_weight    = 10
terminal_battery_weight           = 1
cold_start_weight                 = 0.10
curtailment_weight                = 0.01
battery_throughput_weight         = 0.01
```

The resulting M24 settings SHA-256 is
`67877c94438fc1bf1bd3ad564f17d860f77cea10b82f07bba0d52b6a6f6b1778`. The rejected M6
full-day settings SHA-256 remains
`28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f`.

No terminal weight, delivery weight, solver option, reserve, plant limit, tariff, economic
coefficient, command conversion, or fallback may change in this probe. M24 continues to consume
only `ControllerObservationV2`, the current issued-mean forecast, public commitment state,
static plant/constraint data, and its own prior public transition. It receives no simulator
truth, episode seed, future forecast vintage, old controller outcome, or privileged planner
schedule while deciding.

The hypothesis is deliberately narrow: with a 24-hour prefix, `infer_delivery_targets` can
represent the announced settlement plus its three fixed public recurrences. Explicit delivery
constraints should make finished methane valuable before the overnight window even though the
unchanged terminal tie-breaker still treats CH4 and convertible feedstock as equivalent.

## Frozen development matrix

Only the ten previously opened 24-hour cells are eligible:

| Axis | Frozen values |
|---|---|
| horizon | 24 h |
| delivery rate | 35 kg/h, 42 kg/h |
| seed | 20, 21, 22, 23, 24 |
| new controller | M24 only |

The canonical order is rate 35 seeds 20--24, then rate 42 seeds 20--24. This creates ten new
controller trajectories, 960 controller decisions, 1,920 solver stages, and 14,400 one-minute
applications. H and M6 are immutable reference arms loaded from the verified robustness
artifacts; they are not rerun.

All other task values remain those recorded in the reference cells:

- default `PlantConfig` and `Constraints`;
- `plant-env-v2-economic-wear-alpha` physical/economic execution;
- default `EconomicAssumptionsV1` and synthetic 24-hour tariff;
- default diagnostic `RewardWeightsV2`;
- start hour 06:00;
- 15-minute control and one-minute physics;
- four recurring six-hour methane settlements; and
- the exact seed-specific initial inventory, weather, forecast, tariff, plant, constraint, and
  checkpoint-contract identities already recorded for each cell.

The implementation may use non-protected identifiers at or above 904 for unit fixtures. It may
construct models and solve deliberately synthetic tiny fixtures. It may not reset a seed
20--24 environment through M24 until the protocol, runner, verifier, tests, and admission audit
are committed cleanly and a separate execution registration fixes the exact revision and new
external destination.

## Runner and artifact contract

The one-shot runner must:

1. accept the original `report.json` and a new absolute output directory outside Git;
2. reject a missing, extra, modified, incompletely verified, or wrong-identity reference
   directory before creating an environment;
3. require a clean, completely identified source revision, exact protocol bytes, and unchanged
   dependency lock before output creation and again after execution;
4. construct and reset one fresh M24 environment per cell and require its complete manifest,
   exogenous, economic-episode, assumptions, and task identities to match the corresponding
   immutable reference cell before requesting an action;
5. execute cells sequentially in canonical order with no controller-internal fallback;
6. retain every decision, two-stage solve, command/setpoint, minute dispatch, guard action,
   physical application, next boundary, residual, settlement, reward, economic component,
   failure, and latency record;
7. publish each cell immediately by exclusive creation and deeply verify it before proceeding;
8. retain a failed cell and continue, without retry, seed substitution, parameter change, or
   surviving-subset aggregate; and
9. publish one compact report that references raw, full, deterministic, and evidence-graph
   identities for all old and new artifacts.

Wall-clock timings remain in full evidence and are excluded from deterministic evidence.
Reference H/M6 arms must be cited by their existing byte and semantic hashes rather than copied
into the new cell files.

The independent verifier must reject missing, extra, reordered, renamed, duplicated, path-
escaping, truncated, byte-changed, identity-mismatched, or semantically inconsistent reference
or M24 artifacts. It must recompute every gate below from the full artifacts rather than trust
the compact report.

## Registered measurements

Every cell reports M24 and referenced H/M6 values for:

- total and per-settlement requested, delivered, unmet, and fill;
- commitments settled/met and worst-commitment fill;
- first-decision horizon, forecast, and all delivery-target records;
- CH4, H2, CO2, and battery states immediately after settlements two and three;
- accepted process/solar energy, curtailment, battery throughput/support, and terminal energy;
- electrolyser starts, stops, operating/exposure time, and nonproductive energy;
- corrections, plant refusals, target deficits, guard interventions, shutdown, forced stops,
  and out-of-range decisions;
- complete additive economic components and default/no-methane paired differences, labelled
  ineligible wherever H/M24 service is not comparable; and
- controller, solve-stage, dispatcher, cell, and whole-run latency outside deterministic
  performance identity.

Per-rate summaries retain all five seed values, missingness, mean, population standard
deviation, median, extrema, sign counts, and the exact seeds attaining extrema. No aggregate may
mix rates.

## Frozen gates

### F1 — source and reference custody

Pass only if the old 21-artifact robustness directory deeply verifies at the exact identities
above, the new source gate is clean and stable, the protocol and lock hashes match, all ten
declared reference cells appear once in order, and no undeclared source or destination exists.

### F2 — complete M24 execution and evidence

Pass only if all ten M24 arms reach normal termination with 96 decisions, 1,440 physical
minutes, four settlements, 192 successful solver stages, closed transitions, exact evidence
cardinality, and no fallback. Any failure fails the complete gate; no surviving subset replaces
the matrix.

### F3 — identity, accounting, safety, and authority

Pass only if every M24 reset identity equals its reference cell; reward/component/interval and
terminal-liability accounting closes; and every arm reports zero shutdown, out-of-range
decision, forced stop, chain/accounting violation, requested-target overshoot,
reachable-tracking violation, support-authority violation, or unattributed target deficit.
Fully attributed guard interventions and plant refusals remain reportable.

### F4 — horizon intervention fidelity

Pass only if every M24 descriptor has the exact settings digest above; its first decision spans
96 intervals; its first forecast projection contains the unchanged 24-hour public issue; and
its first delivery list contains exactly four ordered targets at boundaries 24, 48, 72, and 96,
with the first labelled announced and the remaining three labelled inferred. All non-horizon
settings and environment identities must equal M6/reference values.

### F5 — full-day service repair

Pass only if:

- M24 never delivers less or leaves more unmet product than M6 by more than `1e-6 kg` in any
  cell;
- M24 delivers a positive amount at the fourth settlement in every cell;
- at least four of five cells in each rate stratum deliver the complete request within
  `1e-6 kg` and meet all four commitments; and
- at least four of five cells in each rate stratum are service-comparable with H within
  `1e-6 kg` for both delivered and unmet methane.

A cell where M24 safely exceeds H service remains visible but cannot enter an equal-service
economic tie-break.

### F6 — mechanism magnitude and product-form adequacy

Pass only if, separately at each rate:

- mean M24 unmet methane is at least 90% lower than mean M6 unmet methane;
- at least four of five cells hold enough CH4 immediately after settlement three to cover the
  complete fourth request within `1e-6 kg`; and
- the realized fourth-settlement delivery agrees with the preceding public-boundary CH4
  drawdown and retained settlement evidence.

This gate tests the diagnosed mechanism. It does not require terminal feedstocks or battery
energy to move in a favorable direction.

### F7 — equal-service economic expression

Within only the predeclared H/M24 service-comparable cells, apply the existing R5 and R6 rules:
absolute default net-value separation above 1 EUR, at least one registered raw material driver,
and separation above 1 EUR after removing methane-endowment liability with a nonterminal driver.
Pass only if at least four of five cells in each rate stratum satisfy both rules. All five cells,
including any ineligible one, remain in the denominator and report.

Recompute the existing 0/0.5/1/2 methane, wear, and solar sensitivities and all leave-out views.
Every ranking flip is reported. F7 tests whether repairing service leaves a discriminative
development task; it is not a controller-selection or calibrated-economic gate.

### F8 — operational latency and interpretation

Report every latency. Pass the engineering latency check only if each M24 arm has controller
p99/max no greater than 2/5 seconds and dispatcher p99/max no greater than 5/25 milliseconds.
The report must repeat this claim boundary:

> Contaminated development mechanism only; no RL, controller-superiority, calibrated-economic,
> held-out, site, commercial, savings, commission, or generalization claim.

## Decision rule

- If F1--F3 fails, repair evidence, execution, accounting, or authority under a new revision;
  do not interpret service.
- If F4 fails, the experiment did not isolate the registered horizon intervention; do not
  interpret F5--F7.
- If F5 or F6 fails, horizon alone is insufficient. The next protocol must add explicit
  deliverable-CH4 adequacy across renewable outages and/or change the physical energy-supply
  architecture; objective-weight tuning is not an accepted repair.
- If F5/F6 passes but F7 fails, full-day service is repaired but the alpha economic task remains
  saturated or coefficient-dependent. Add calibrated operational pressure before RL.
- If F1--F7 and the latency check pass, the causal horizon repair is accepted on development
  evidence. This authorizes a separately frozen bounded PPO baseline and tariff-aware economic
  MPC implementation, plus design of a cyclic Environment V2 beta. It does not authorize a
  comparative or commercial claim.

## Explicit exclusions

This protocol does not:

- rerun, replace, or relabel H or M6;
- select a favorable horizon from a sweep;
- change objective weights or tune against opened seeds;
- train, resume, select, or evaluate an RL policy;
- add a tariff-aware, stochastic, robust, or product-form-terminal MPC;
- calibrate economic coefficients or claim cost savings;
- introduce historical weather, faults, sensor imperfections, degradation dynamics, grid
  exchange, water, compression, product quality, or multi-unit constraints;
- use new held-out scenarios; or
- evaluate HEPA, the language supervisor, operator workflow, HIL, or field behavior.

Those remain later, separately versioned steps. This probe answers one question only: whether
the already public 24-hour horizon is enough to convert the observed full-day failure from a
one-commitment controller into a service-credible benchmark.
