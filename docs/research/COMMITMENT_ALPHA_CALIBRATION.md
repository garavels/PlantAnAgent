# Commitment-alpha delivery-rate calibration

Status: **development calibration only; never formal validation or held-out
evidence.**

## Decision

Use **35 kg/h** as the primary commitment-alpha development target. It is mostly
feasible for the legacy heuristic with commitment enabled, but it exposes both a
controller spread and a measurable commitment-on/off separation:

- the commitment-on heuristic achieves 97.7% mean delivery fill, with two of three
  seeds perfect;
- its remaining episode leaves 172.0 kg unmet, giving 57.3 kg mean unmet methane and
  97.2% mean commitment service;
- the commitment-on fixed production and balanced mixes achieve 86.2% and 59.0%
  fill, respectively;
- the same heuristic reaches 100% fill with commitment disabled.

Use **42 kg/h** only as a named development stress point. At that rate the heuristic
falls to 84.97% fill, 454.5 kg mean unmet methane, 641.6 kg worst-episode unmet methane,
and 72.22% service with commitment enabled. With commitment disabled it retains 98.20%
fill, 54.4 kg mean unmet, 163.2 kg worst-episode unmet, and 97.22% service. This creates
a useful commitment-sensitive failure regime, but it is not a held-out stress test.

The 35 and 42 kg/h choices were made after inspecting these development results.
Neither rate, seeds 0--2, nor this weather regime may later be relabelled as validation
or held-out evidence.

## Evidence scope and provenance

The completed screen used:

| item | value |
| --- | --- |
| source revision | `9d94483832b5a1a2ed80e995f36c99edfdfa2c93` |
| source state at execution | clean |
| dependency-lock SHA-256 | `df2bfa50a09d20137e6360634c66acfa93b5dcfec91b221aab369d20d8e45afa` |
| seeds | 0, 1, 2 |
| horizon | 72 h per rollout |
| delivery period | 6 h |
| commitment modes | frozen alpha profile enabled; compatibility-off `None` ablation |
| controllers | legacy forecast-blind heuristic, fixed production mix, fixed balanced mix, idle |
| evaluation path | maintained `plant_backend.rl.v2.evaluation` baseline runner |
| statistics | three-seed population means; report precision retained below |

Every result in this note is a small-sample, in-distribution development diagnostic.
It is not a confidence estimate, controller tournament, optimality bound, RL result, or
commercial result. The evaluated arms also have unequal information sets: the fixed
arms are open loop, and the legacy heuristic consumes one-minute telemetry and safety
constraints but no V2 delivery context or issued forecast. No result-artifact checksum
is asserted in this note.

An earlier 20-seed, 168-hour sweep was interrupted before it produced complete cells.
Its partial cells are not pooled with, substituted for, or promoted above the completed
screen. It was replaced with the deliberately bounded 3-seed, 72-hour development
screen so every reported rate and ablation has the same completed seed set.

## Pairing semantics

Commitment-on and compatibility-off rollouts use common random numbers. For a fixed
delivery rate and seed, they share the same `exogenous_episode_sha256`; their complete
manifest hashes are expected to differ because the equipment dynamics differ.

The reported pairing counts use **one controller-specific on/off comparison** as the
unit of a pair:

- the low-rate segment has 36/36 matched comparisons: 3 rates x 3 seeds x 4
  controllers. Those comparisons cover 9 `(rate, seed)` episode cells and comprise
  72 controller rollouts;
- the high-rate segment has 48/48 matched comparisons: 4 rates x 3 seeds x 4
  controllers. Those comparisons cover 12 `(rate, seed)` episode cells and comprise
  96 controller rollouts.

Thus “48 pairs” does not mean 48 independent weather episodes, nor does it mean 48
single rollouts. It is an audit of matched on/off comparisons across controller arms.

## Low-rate screen

Values are mean aggregate delivery fill over seeds 0--2. The precision shown is the
precision retained by the completed screen; no extra digits are reconstructed.

| rate [kg/h] | commitment | heuristic | production mix | balanced mix | idle |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 30 | on | 100.0% | 96.9% | 67.7% | 7.1% |
| 30 | off | 100.0% | 100.0% | 96.7% | 7.1% |
| 35 | on | 97.7% | 86.2% | 59.0% | 7.1% |
| 35 | off | 100.0% | 99.2% | 84.5% | 7.1% |
| 40 | on | 88.9% | 76.3% | 52.5% | 7.1% |
| 40 | off | 99.7% | 92.4% | 74.9% | 7.1% |

The 30 kg/h heuristic cell remains saturated and is therefore too easy as the primary
development target. At 40 kg/h, commitment-on heuristic fill has already fallen to
88.9%, with 320.7 kg mean unmet methane, 507.4 kg worst-episode unmet methane, and 75.0%
mean service. The 35 kg/h cell lies between those regimes: mostly feasible, but no
longer universally perfect.

This choice supersedes the earlier provisional 40 kg/h setting only for subsequent
commitment-alpha development. It does not alter or reinterpret the clean, pre-commitment
weather-alpha result at 40 kg/h.

## High-rate boundary extension

The high-rate extension is summarized only for the heuristic because those are the
controller figures retained for this decision.

| rate [kg/h] | commitment-on fill | compatibility-off fill |
| ---: | ---: | ---: |
| 42 | 84.97% | 98.20% |
| 45 | 79.78% | 94.32% |
| 48 | 75.23% | 88.87% |
| 50 | 72.51% | 85.60% |

Rates 45--50 increasingly overload even the compatibility-off heuristic. They are
useful boundary observations, but they are less clean commitment diagnostics because
delivery scarcity is already material without stateful equipment dynamics. Rate 42 is
therefore the named stress point: the off arm remains close to feasible while the on
arm has a pronounced failure tail.

## Operating behavior is invariant to the contract rate

The commitment-on operating means below are unchanged across every screened rate at the
reported precision. The table intentionally omits the supplied ambiguous `guard` field:
the evaluator exposes both intervention counts and an energy-valued guard metric, and
the retained summary did not identify which one it meant.

| controller | cold starts | hot starts | aborted starts | forced stops | nonproductive energy [kWh] | guard-to-plant dispatch gap [kWh] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy heuristic | 1.0 | 10.3 | 0.0 | 0.0 | 3,760 | 27,299 |
| fixed production mix | 5.0 | 9.0 | 2.0 | 5.0 | 2,880 | 18,568 |
| fixed balanced mix | 4.7 | 25.3 | 1.7 | 4.7 | 3,894 | 37,330 |

`nonproductive energy` is startup plus hot-standby energy. `dispatch gap` is the
integrated physical guard-to-plant refusal, not the safety guard's request correction.
Idle is omitted because no retained nonzero operating-exposure summary was needed for
the decision.

This invariance is a controller-contract finding, not evidence that demand is irrelevant
to the environment. The 57-field V2 observation includes causal delivery context, but
none of these evaluated baselines consumes that contract target: the legacy heuristic is
called through its older telemetry interface and the fixed arms do not observe the
environment. Consequently, raising the target changes settlement outcomes without
giving these controllers a direct way to change their plan. The unchanged operating
metrics confirm that they did not adapt indirectly over this screened range.

The present comparison therefore measures how fixed controller behavior is scored under
different obligations and how the commitment kernel changes that behavior. It does not
measure the attainable value of delivery-aware scheduling.

## Next experiment

The next comparator must first be a **delivery-aware, forecast-blind** rule under the
same causal observation and action protocol. That isolates the value of reacting to the
known contract without granting future weather information. The following comparator is
an otherwise matched **issued-forecast-aware** rule. Only after that causal pair is stable
should deterministic MPC, risk-aware planning, a feasible future-information planner,
or RL be interpreted.

Both comparators should use the same 35 kg/h primary development target and 42 kg/h
development stress point initially, with identical safety routing, commitment dynamics,
forecast vintages, controller cadence, and exogenous pairing. Any formal validation or
held-out stress suite must then use newly predeclared seeds and regimes that were not used
to choose these rates or controllers.
