# Causal service-MPC development protocol

Date: **2026-09-03**

Status: **pre-outcome protocol; implementation and unit fixtures are authorized, but no
full 35/42 kg/h MPC rollout may be opened until the implementation gate below is frozen
on a clean revision**

## Purpose and claim boundary

The replay-verified future-information planner has established actionable exact-simulator
headroom on four of the six inspected development cells. It does not establish that an
online controller can recover that headroom from information available at an operating
boundary. This protocol freezes the first causal test of that question.

The implementation is called **service MPC** because it must solve a declared receding-
horizon mixed-integer model of delivery, inventory, energy, and electrolyser commitment.
It is not economic MPC: the environment has no calibrated product price, electricity
tariff, shortage contract, or degradation cost. It is also not an oracle, stochastic MPC,
globally optimal plant schedule, validated digital twin, or evidence that RL is unnecessary.

The controller is allowed to be an imperfect internal model. Its planned trajectory is not
evidence of feasibility; only the action applied through `PlantEnvV2`, the independent
guard, and the exact one-minute simulator determines the controller outcome.

## Prior-outcome boundary

The 72-hour, synthetic-weather cells at delivery rates 35 and 42 kg CH4/h and seeds 0--2
are already development-contaminated. Their exact A0, best-known feasible `F`, and relaxed
`U` values are recorded in `FEASIBLE_PLANNER_RESULTS.md`. They may be used for this
engineering decision and may never be relabelled prospective, validation, test, field, or
commercial evidence. Seed 3 and the disclosed rate-40/seed-0 probes also remain
contaminated.

No uninspected seed, weather regime, site, demand regime, fault tape, or plant parameter
will be opened while implementing or debugging this controller. A later protocol must
freeze those outcomes before generation.

## Questions and predeclared development gate

### M1 -- causal forecast reconstruction

Can the current public mean forecast be converted into a deterministic 15-minute solar
trajectory that:

- uses no future truth or future forecast vintage;
- is zero whenever the public clear-sky geometry is dark;
- preserves each issued band's energy exactly within numerical tolerance; and
- is invariant when private future truth changes behind an identical public issuance?

### M2 -- multi-period control

Can a HiGHS-backed mixed-integer model spanning the remaining episode or 24 hours,
whichever is shorter, return a validated first action at every 15-minute boundary while
representing the coupled energy, feedstock, product, delivery, and commitment decisions?

### M3 -- causal capture

Let `M` be exact delivered methane from the service MPC and use the already frozen `A0`
and `F` values for the same exogenous cell. For cells where `F - A0 > 10^-9 kg`, define

```text
causal capture = (M - A0) / (F - A0)
```

The first implementation passes the **development utility gate** only if all of the
following hold:

1. all six MPC rollouts and every expected solve are evidence-valid, with no hidden
   fallback or surviving-subset aggregate;
2. at 42 kg/h, `M > A0` on at least two of three seeds and mean causal capture is at
   least 20%;
3. total 42 kg/h commitment service is no worse than A0's frozen 26/36;
4. at 35 kg/h, mean delivery fill is no more than one percentage point below A0 and
   neither A0-saturated seed loses a complete commitment;
5. every rollout has zero reserve breach, shutdown step, non-finite/invalid action, and
   forced electrolyser stop;
6. every terminal battery SOC is at least 25%, and no rollout makes more than four cold
   starts; and
7. solver decision latency has p99 at most 2 seconds and maximum at most 5 seconds on
   the recorded development host.

These thresholds are engineering gates, not estimates of generalization. Failure is a
result. A later amendment may change the formulation, but it must preserve this first
complete matrix and identify the change before another registered matrix is opened.

Terminal H2, CO2, CH4, CH4-equivalent feedstock, battery energy, curtailment, battery
throughput, starts, stops, guard intervention, and accepted process energy are mandatory
reported outcomes even when the threshold does not attach a benefit sign. The existing
endowment-preserving relaxation remains a separate sustainability diagnostic; the current
`A0 <= F <= U` bracket is not redefined.

## Legal information boundary

At decision boundary `k`, the controller may receive only:

- the detached `ControllerObservationV2` for `k`;
- the one `PublicForecastIssueV2` embedded in that observation;
- static `PlantConfig`, active `Constraints`, and the active commitment profile;
- immutable, serialized controller settings; and
- controller-local history created from earlier public observations and its own earlier
  solutions.

It may not receive or retain `PlantEnvV2`, `PlantSimulator`, `EpisodeV2`, the episode seed,
private audit manifest, private delivery schedule, weather tape, truth factor, issue-stream
identity, replay hash, future forecast vintage, offline planner schedule, `F`, or `U`.
References `A0`, `F`, and `U` enter only the post-rollout evaluator.

The task publicly declares a fixed recurring delivery cadence. The model may infer future
requests by repeating the currently announced amount at
`DeliveryObservation.delivery_period_steps`; it must label those requests as inferred and
must never inspect the private schedule. If no next request is announced, it may not invent
one. The task horizon is an immutable controller setting and must match the evaluation
contract; progress and local decision count determine the public remaining horizon.

The controller must reject a missing forecast, issue time inconsistent with the current
boundary, commitment snapshot/profile mismatch, unsupported timestep, non-finite public
state, or unsupported component fault/degradation. The first registered matrix is scoped
to the existing fault-free commitment-alpha cells. Later fault handling requires an
explicit robust-model amendment rather than silently treating degraded equipment as
nominal.

## Frozen controller and solver settings

The first registered controller identity is `service-mpc-issued-mean-v1`. Its immutable
default settings are:

| setting | value |
| --- | ---: |
| task horizon | 72 h |
| receding planning horizon | 24 h |
| optimization interval | 900 s |
| forecast statistic | issued mean only |
| terminal battery SOC target | 25% |
| HiGHS presolve | enabled |
| relative MIP gap | `1e-6` |
| time limit per optimization stage | 2 s |
| solver randomness | none exposed by SciPy `milp` |
| controller fallback | none inside the MPC identity |

The implementation may reduce the final horizon below 24 hours at the declared intrinsic
episode end. It may not extend an issue beyond its public 24-hour support. Every numerical
setting, model version, forecast-projection version, objective version, SciPy version, and
linked HiGHS version is serialized into the controller/evidence identity.

The model terminal-energy lower bound is 25% of effective capacity while current SOC is at
least 25%. If forecast error has already left the observed battery below that target, the
bound becomes its current energy, so the optimizer cannot deepen the deficit but also does
not become structurally infeasible merely because it cannot undo past error inside a short
final horizon. The unchanged evidence gate still requires actual terminal SOC of at least
25% for a successful registered rollout.

## Deterministic solar reconstruction

The public windows are the canonical half-open offsets `[0,1)`, `[1,6)`, `[6,12)`, and
`[12,24)` hours. Each `mean_peak_power_fraction` is an average available-PV fraction of
nameplate over its complete band, not a weather/resource factor.

For optimization interval `i` within band `b`:

1. analytically integrate the public `PlantConfig.solar` half-sine across the interval to
   obtain its mean clear-sky nameplate fraction `c_i`;
2. compute `s_b = issued_mean_b / mean_i(c_i)` over the complete band;
3. require `0 <= s_b <= 1` within a declared numerical tolerance; and
4. set forecast available PV to `peak_kw * s_b * c_i`.

Dark intervals therefore remain zero, the clear-sky shape is coherent across each band,
and the arithmetic mean across the full band equals the issued mean. The implementation
must fail closed rather than clip a materially impossible issuance. A final shortened
optimization horizon takes a prefix of this already reconstructed 24-hour trajectory; it
does not renormalize a partial band.

The current p10 values are not used. They are marginal band summaries, not a coherent
scenario path. A controller using them can be a deterministic risk-adjusted sensitivity,
but no implementation is called stochastic MPC until public, causal scenario members are
defined.

## Mixed-integer planning model

### Resolution and variables

The model uses one 15-minute interval per held controller action. For interval `t` it has
continuous variables for:

- solar routed to battery, electrolyser, DAC, and methanator, plus curtailment;
- battery support to the electrolyser;
- accepted productive and startup electrolyser power;
- hydrogen, CO2, and methane production;
- boundary battery energy and H2, CO2, and CH4 inventories;
- methane delivered at each announced or causally inferred settlement; and
- minimum commitment fill and terminal CH4-equivalent feedstock potential.

Binary variables prevent simultaneous battery charge/discharge and represent cold start,
powered startup, productive running, stop, and the declared minimum up/down locks. The
first model deliberately selects only off, cold-starting, and running trajectories; it
does not proactively schedule hot standby. A public hot-standby or hot-starting initial
snapshot is bridged conservatively and explicitly. That approximation must be identified
in every result.

### Physical constraints retained

The model retains:

- forecast solar-bus balance and component/safety power caps;
- charge/discharge efficiencies, power limits, effective capacity, active reserve, and
  the 25% terminal battery floor;
- no simultaneous battery charge and support;
- battery support only to the electrolyser;
- current H2, CO2, and CH4 inventories, storage capacities, stoichiometry, and
  production-before-settlement ordering;
- DAC and methanator specific energy and current observed availability;
- cold-start delay and power, minimum stable electrolyser load, minimum up/down time,
  current commitment mode/timers, and a conservative H2 yield using the worst declared
  part-load multiplier; and
- declared next and inferred recurring settlement amounts and order.

The 15-minute unit-commitment approximation does not claim minute-exact ramp, hot-start,
shortfall-grace, part-load, dispatch-priority, or storage-refusal behavior. The exact
simulator remains authoritative, rolling re-optimization observes model error at the next
boundary, and first-step planned-versus-accepted power and inventory residuals are reported.

### Objective

Each decision uses two explicit solver stages:

1. maximize total delivered methane across the causal horizon;
2. lock stage-one delivery within `10^-6 kg`, then maximize worst commitment fill and
   normalized terminal CH4-equivalent inventory and battery energy while penalizing cold
   starts, curtailment, and battery throughput with bounded, serialized tie-break weights.

No secondary term may buy more than `10^-6 kg` of primary delivery. Solver status for both
stages is mandatory. The second-stage plan supplies the action. Failure, infeasibility,
invalid output, or missing optional optimization runtime raises a typed controller error;
the causal evaluator records a failed rollout and suppresses aggregate claims.

The frozen stage-two utility to maximize is:

```text
100.00 * minimum commitment fill
 10.00 * terminal CH4-equivalent inventory / maximum CH4-equivalent capacity
  1.00 * terminal battery energy / effective battery capacity
 -0.10 * cold starts / number of optimization intervals
 -0.01 * curtailed energy / (PV nameplate * horizon hours)
 -0.01 * battery charge-plus-discharge energy
         / (maximum battery power * horizon hours)
```

All denominators must be finite and positive. These are bounded scheduling tie-breakers,
not prices or degradation costs. Changing a coefficient, normalization, priority, or
delivery-lock tolerance creates a new objective version and requires a prospective
amendment before another registered matrix.

## First-action realization

Only the first 15-minute plan is applied. Its solar-route kW values are divided by the
first interval's reconstructed available PV to form the four action shares. The same
downward binary64 simplex closure used by existing transparent controllers ensures the
request is already valid before `Action.sanitised`. The optimized battery-support kW is
sent as the fifth action field.

If the first forecast interval is dark, every solar share is zero and only explicitly
planned support may be requested. If the boundary itself is dark but the first interval
contains sunrise, the nonzero interval forecast provides the routing denominator. The
independent guard still runs before every one-minute physical step and the exact plant may
refuse power; neither effect is written back into the evidence as if it had been planned.

## Decision evidence and failure handling

Every attempted decision must retain or hash-bind:

- public observation digest and current forecast digest;
- reconstructed trajectory and digest;
- model dimensions, coefficients/bounds digest, and inferred-delivery schedule;
- both solver statuses, messages, objectives, bounds, MIP gaps, node counts, and elapsed
  times;
- selected plan digest, first-step physical targets, requested action, and notes;
- planned-versus-observed first-interval process/energy residuals when the next boundary
  becomes available; and
- whether the exact guard changed the request or the plant refused guarded power.

An operational deployment may place a separately identified fallback supervisor around
the MPC. This experiment does not: fallback under the same controller label would make
solver reliability and controller attribution unknowable.

## Development matrix and reporting

The frozen execution order is rate-major then seed-minor:

```text
35/0, 35/1, 35/2, 42/0, 42/1, 42/2
```

Each cell runs A0, the existing issued-mean stateless controller A2, and the service MPC
on one paired exogenous episode. The primary equal-forecast contrast is `M - A2`; `M - A0`
measures practical improvement over the persistent legacy baseline. The postprocessor may
attach the previously frozen `F` and `U` values only after exact manifest/cell identity is
matched. It must not rerun, alter, or expose the privileged schedule to the controller.

Reports include every physical metric already required by the causal evaluator, causal
capture for eligible cells, delivery/service effects, complete terminal balances, action
treatment expression, solver reliability/latency, plan-to-plant residuals, and all gate
decisions. Population summaries are descriptive (`ddof=0`) because three inspected seeds
are not an inferential sample.

## Implementation gate before full rollouts

Before any registered 72-hour MPC cell is opened, a clean commit must pass:

1. forecast reconstruction fixtures for darkness, sunrise/sunset, midnight wrap, every
   band, shortened horizons, impossible issuance, and exact energy preservation;
2. counterfactual tests proving identical public inputs yield byte-identical requests when
   private future truth, episode seed, or future vintages differ;
3. hand-solvable abundant/zero-resource, battery-only, delivery-order, storage,
   stoichiometry, and terminal-SOC MILP fixtures;
4. unit-commitment fixtures for each initial mode/timer, startup energy and delay, minimum
   up/down behavior, and prohibited simultaneous charge/discharge;
5. malformed/unsupported input, absent SciPy, infeasible model, timeout, non-optimal
   status, non-finite result, and invalid first-action fail-closed tests;
6. deterministic model/plan/action digests across repeated construction and solves;
7. one-step and short-horizon exact-environment tests that compare planned first-step
   quantities with actual guard/plant accounting without treating mismatch as feasibility;
8. evaluator tests for complete decision evidence, failed-solve retention, no fallback,
   source start/end identity, pairing, immutable controller descriptors, and no aggregate
   from an incomplete matrix;
9. formatter, strict type check of the new surface, compilation, dependency-lock check,
   focused tests, and the complete repository suite; and
10. an independent source audit mapping every item above to executable evidence.

Unit and deliberately tiny synthetic fixtures are authorized before this gate. Any
accidental full development-cell result must be disclosed and that invocation excluded
from the registered matrix.

## Decision after the matrix

- If the utility and validity gates pass, freeze an unseen-regime protocol and implement
  sustainable-service beta before PPO training.
- If solves are reliable but capture fails, inspect forecast resolution, first-action
  realization, and model-to-plant residuals before adding complexity. Preserve the failed
  matrix before any model amendment.
- If the five-field share action is the dominant loss, version an explicit bounded
  mode/absolute-setpoint action rather than hiding the mismatch in tuning.
- If deterministic forecast error dominates, add causal scenario paths and risk-aware MPC
  before asking RL to absorb unmeasured uncertainty.
- If no causal method can improve without endowment liquidation, recalibrate demand and
  terminal contracts; do not train against an ill-posed horizon artifact.

RL is admitted only after this comparator exists. It earns a product role by improving a
measured residual in delivery, tail risk, robustness, or latency under equal information
and identical safety—not merely by obtaining a larger private reward.
