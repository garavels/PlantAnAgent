# Environment V2 weather-alpha baseline result

Status: **formal baseline-only calibration; not an RL, MPC, forecast-value, or
product-performance result.**

## Question

Does seeded renewable uncertainty break the universal delivery ceiling observed in the
delivery alpha and produce paired, reproducible outcome variation across transparent
dispatch strategies?

This experiment does not ask whether RL is better. None of the evaluated controllers
uses the issued forecast, the arms do not share an equal information set, and the process
equipment remains memoryless. The narrow acceptance condition is that weather creates
nontrivial episode-to-episode risk without making the nominal task broadly infeasible.

## Frozen experiment

Command:

```text
.venv/bin/python -m plant_backend.rl.v2 --seed-start 0 --count 20 --hours 168 \
  --delivery-rate 40 \
  --json docs/research/results/v2_weather_alpha_baselines_seed0_19.json
```

Provenance:

| item | identity |
| --- | --- |
| source revision | `129624ad6668d3286b431a76cb5e616fc87fa034` |
| source dirty | `false` |
| dependency lock SHA-256 | `cb346b6cace28625107c93cc446181b8f85a6916741393c98e8e5ab00f91e3af` |
| environment | `plant-env-v2-weather-alpha`, specification 2 |
| interface | 42 observations, 5 actions |
| interface digest | `2f664d7fc5c2aaf37e2bfef62697cf5aa48200597799afcb6ad349ea1ec30ec3` |
| full checkpoint-contract digest | `89bcd56a211f3228419d7fce530c1039bd9902c8d7ab7d08b82f8cba7b08476d` |
| result JSON SHA-256 | `d5af5bee853dd5729792753fe8d5c47d50dd45e6a56ef588f1b6b3b47c7f6721` |

The artifact contains 80 runs: four arms on seeds 0 through 19. Every seed has all
four arms, each arm uses the same episode-manifest identity within a seed, and all 20
seed manifests are distinct. The independent audit regenerated all manifests, replayed
all four arms on seeds 0, 6, and 19 to numerical tolerance `1e-12`, recomputed every
summary, and found no accounting, reward, reserve, or conservation inconsistency.
Population standard deviation (`ddof=0`) is reported.

## Result

Every episode contains 28 six-hour commitments totalling 6,720 kg CH4. Available solar
varies from 193.4 to 213.7 MWh across the synthetic weather seeds.

| controller | delivery fill mean +/- std [min, max] | service level mean +/- std [min, max] | unmet kg mean +/- std [min, max] | CH4 produced kg mean +/- std | curtailment kWh mean +/- std | reward mean +/- std [min, max] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy forecast-blind heuristic | 98.58% +/- 2.15% [93.70%, 100%] | 96.43% +/- 5.30% [85.71%, 100%] | 95.4 +/- 144.7 [0, 423.7] | 6561.5 +/- 273.9 | 403 +/- 83 | 25.92 +/- 3.04 [19.04, 27.96] |
| fixed production mix | 86.36% +/- 4.76% [78.03%, 93.87%] | 77.86% +/- 7.46% [64.29%, 89.29%] | 916.9 +/- 320.1 [412.1, 1476.5] | 5583.6 +/- 321.5 | 24523 +/- 2090 | 8.66 +/- 6.67 [-3.00, 19.18] |
| fixed balanced mix | 68.37% +/- 4.44% [60.85%, 75.18%] | 51.25% +/- 7.43% [35.71%, 60.71%] | 2125.6 +/- 298.4 [1667.9, 2630.8] | 4374.9 +/- 299.8 | 63573 +/- 3279 | -16.65 +/- 6.22 [-27.19, -7.12] |
| idle | 3.27% +/- 0.90% [1.86%, 4.89%] | 1.25% +/- 1.70% [0%, 3.57%] | 6500.5 +/- 60.7 [6391.5, 6595.0] | 0 | 203579 +/- 7143 | -108.11 +/- 1.27 [-110.11, -105.87] |

The heuristic achieves perfect fill and service on 12 of 20 seeds, rather than the
20 of 20 observed in the delivery alpha. Its eight non-perfect seeds are 1, 2, 4, 6,
13, 16, 17, and 19. Seed 6 is the principal worst case: it has the lowest solar resource
(193.4 MWh), 93.70% aggregate fill, 423.7 kg unmet, 89.29% service, and only 22.06% fill
on the worst individual commitment. Seeds 1, 16, and 19 share the lowest service level
of 85.71%.

The result therefore contains both a high-performing region and a failure tail. The
mean alone hides this distinction: although aggregate heuristic fill is 98.58%, its
mean worst-commitment fill is only 78.81%, with a range from 22.06% to 100%.

## Paired comparison with the strongest open-loop arm

The fixed production mix is the strongest of the three open-loop diagnostics. Because
the episode manifests are paired, per-seed differences are more informative than two
unpaired summary means.

| metric | heuristic minus fixed production mix, mean +/- std [min, max] | paired outcome |
| --- | ---: | ---: |
| delivery fill | +12.22 +/- 3.11 percentage points [+6.13, +16.35] | higher on 20/20 seeds |
| service level | +18.57 +/- 4.60 percentage points [+10.71, +32.14] | higher on 20/20 seeds |
| unmet CH4 | -821.5 +/- 208.7 kg [-1099.0, -412.1] | lower on 20/20 seeds |
| CH4 production | +977.9 +/- 89.5 kg [+815.6, +1099.0] | higher on 20/20 seeds |
| curtailment | -24120 +/- 2113 kWh [-27994, -20637] | lower on 20/20 seeds |
| reward | +17.26 +/- 4.32 [+8.77, +23.01] | higher on 20/20 seeds |

This establishes that closed-loop inventory-aware operation matters in the alpha. It
does not establish that the legacy heuristic is a strong benchmark for the complete
V2 information set.

## Safety and action-authority finding

| controller | minimum SOC mean +/- std [min, max] | guard events mean +/- std | request-to-guard L1 gap kWh mean +/- std |
| --- | ---: | ---: | ---: |
| legacy forecast-blind heuristic | 15.009% +/- 0.007% [15.000%, 15.023%] | 2508 +/- 193 | 3727 +/- 465 |
| fixed production mix | 53.112% +/- 9.401% [32.624%, 64.553%] | 3463 +/- 157 | 13030 +/- 1280 |
| fixed balanced mix | 53.112% +/- 9.401% [32.624%, 64.553%] | 3455 +/- 155 | 13021 +/- 1275 |
| idle | 53.112% +/- 9.401% [32.624%, 64.553%] | 0 | 0 |

No arm breaches the configured 15% controller reserve, enters shutdown, or emits an
out-of-range decision. The heuristic's lower SOC is not by itself inferior: it uses the
battery while the fixed arms leave their initial energy unused. Its reserve is nevertheless
active in effectively every episode. Guard projection also remains material: the heuristic
request is altered on about one fifth of physical steps, with a 3.7 MWh mean L1 adjustment.
This reinforces the need to make controller authority explicit before interpreting an
advanced policy's requested actions as its realized operating strategy.

## What changed relative to delivery alpha

Delivery alpha produced 100% heuristic fill and service on every seed under deterministic
solar, even with a 55 kg/h contract. Weather alpha misses deliveries on eight seeds at
40 kg/h. This is strong stage-to-stage evidence that the benchmark is no longer universally
saturated.

It is **not** an isolated causal weather ablation. The contract rate, observation contract,
simulator version, interval aggregation, and guard/accounting semantics also changed.
Consequently, the numerical delta between the two alpha results must not be attributed to
weather alone. A same-revision, same-contract compatibility-off mode is still required.

## Gate decision

The weather increment passes its narrow calibration condition:

- the clean run is exactly replayable and paired;
- outcome variation is large enough to expose a failure tail;
- the task separates a feedback heuristic from weak open-loop dispatch;
- the nominal demand remains feasible on most seeds.

Gate B remains **not passed**:

- no evaluated controller consumes the issued forecast;
- diagnostic arms have unequal information sets;
- equipment is memoryless, so there is no costly startup, ramp, or commitment decision;
- a perfect controller can improve mean fill by at most 1.42 percentage points in this
  regime, so a forecast-aware rule may restore saturation;
- one uncalibrated stationary weather regime and 20 in-distribution seeds do not establish
  tail risk or generalization;
- no MPC or feasible perfect-information planner bounds the remaining gap;
- neither reward nor physical parameters yet support a commercial cost claim.

The next scientific increment is therefore one opt-in, technology-labelled electrolyser
commitment model, with exact compatibility-off replay. It should introduce a powered
startup with no production, minimum stable load, ramp limits, minimum up/down time,
part-load efficiency, observable mode/timers/ramp headroom, and operational metrics. The
contract rate must be recalibrated after this change.

Before RL is promoted as a result, the frozen comparison must include a delivery-aware
forecast-blind rule, its forecast-aware counterpart, deterministic MPC, a risk-aware or
scenario MPC where tractable, and a feasible perfect-information planner. All achievable
controllers must receive equivalent causal information; only the offline planner may see
future truth. The current four forecast bands are adequate for a simple policy input but
too lossy for a credible 15-minute MPC, so a controller-neutral causal trajectory/scenario
interface is also required.

Weather generation currently adds about 0.54 seconds per seven-day reset and constructs
roughly 31 million innovations. Caching paired episode bundles and benchmarking a faster
equivalent implementation are worthwhile before large RL sweeps, but missing commitment
physics—not reset throughput—is the current scientific bottleneck.
