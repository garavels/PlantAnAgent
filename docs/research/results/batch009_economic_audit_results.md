# Batch 009: saved-trajectory economics and reward timing

The observed operating savings were already rewarded. Both policies remain below OperatingMPC on the existing eight development cases under both undiscounted weekly euros and the recorded training discount factor. Economic assumptions can reverse mean value on fixed trajectories, but no rescoring restores the lost deliveries or daily resources. No policy is selected and no operational-benefit claim follows.

This audit follows the [prospective plan](batch009_economic_audit_plan.md), with [reproducible analysis code](batch009_economic_audit.py), [small-fixture tests](batch009_economic_audit_test.py) and [compact evidence](batch009_economic_audit_results.json). It reads the frozen batch 008 report and 56 digest-bound gzip episodes without importing simulator or model code. Every available prefix reward/component sum reconciles with the frozen summary. All 40 complete episodes are retained; the eight M24 and eight E-beta stopped prefixes remain recorded and excluded from performance means.

## Component accounting

Candidate minus OperatingMPC, EUR per simulated week, averaged over all eight complete paired cases. Positive means an economic improvement.

| Component | Scalar A | PI B |
| --- | ---: | ---: |
| Battery throughput charge | +81.72 | +127.19 |
| Electrolyser wear charge | +22.33 | +9.69 |
| Solar opportunity cost | +109.16 | +59.71 |
| Delivered-service value | -45.92 | -45.49 |
| Unmet-delivery penalty | -229.59 | -227.47 |
| Battery endowment accounting | +1.64 | -1.10 |
| Methane-equivalent endowment accounting | +0.25 | -0.06 |
| **Net operating value** | **-60.40** | **-77.54** |

The full delivered-service value and shortfall penalty remain separate. Each additional missed kilogram currently loses EUR 4 of service value and incurs EUR 20 of penalty. Battery wear, electrolyser wear and solar opportunity cost are not omitted from the reward. Their economic representativeness is a separate calibration question.

## Fixed-trajectory sensitivity

The prospective grid multiplies one charged component at a time by 0, 0.5, 1, 1.5 or 2; it also includes all 25 battery-charge × shortfall-penalty combinations. These are illustrative mathematical sensitivities, not calibrated credible ranges or modified plant/controller runs. All other components and all actions, physics, deliveries and resource boundaries stay fixed.

| One-factor mean break-even multiplier | Scalar A | PI B |
| --- | ---: | ---: |
| Battery throughput charge | 1.7391 | 1.6096 |
| Solar opportunity cost | 1.5533 | 2.2986 |
| Electrolyser wear charge | 3.7048 | 8.9986 |
| Unmet-delivery penalty | 0.7369 | 0.6591 |

Battery break-even corresponds to approximately EUR 0.0988/kWh throughput for A and EUR 0.0915/kWh for B, versus the current EUR 0.0568/kWh. Doubling the battery charge makes mean differences positive, but the worst paired economic loss remains negative for both policies and becomes worse than under the base charge. An adjusted average therefore would not establish reliable superiority. The delivery penalty break-even rescales only the penalty; EUR 4/kg delivered-service value remains unchanged. No calibration recommendation is made.

## Physical intensity

Ratios of panel totals, with all eight per-case ratios and terminal inventories retained in the compact evidence. Accepted process/solar energy normalization was an additional descriptive extension recorded after the original plan commit; it is not a preregistered selection test.

| Measured proxy | OperatingMPC | Scalar A | PI B |
| --- | ---: | ---: | ---: |
| Battery throughput, kWh/kg CH4 produced | 1.8678 | 1.6222 | 1.4835 |
| Electrolyser nonproductive energy, kWh/kg CH4 produced | 0.9075 | 0.8768 | 0.8736 |
| Accepted process energy, kWh/kg CH4 produced | 30.8763 | 30.8224 | 30.8344 |
| Accepted solar energy, kWh/kg CH4 produced | 30.5629 | 30.5001 | 30.5033 |

The battery-intensity reduction survives normalization by produced output. Accepted process energy per produced kilogram falls only about 0.17% for A and 0.14% for B. This supports a battery-use improvement more strongly than a broad plant-efficiency claim. Starting and ending H2/CO2/CH4/battery inventories differ from a steady-state test; these ratios are not thermodynamic efficiency or measured equipment lifetime.

## Reward timing

The recorded training gamma is 0.995 per 15-minute decision. Each saved trajectory is rescored as the sum of gamma^t times its raw EUR reward, with the first reward at t = 0. The gamma = 1 sum reconciles with undiscounted economic value.

| Gamma | Scalar A minus MPC, discounted EUR | PI B minus MPC, discounted EUR |
| --- | ---: | ---: |
| 0.99 | -8.97 | -3.31 |
| 0.995 | -33.58 | -26.72 |
| 0.999 | -70.01 | -69.08 |
| 1.0 | -60.40 | -77.54 |

OperatingMPC remains above both policies at every prespecified discount factor. The A/B ordering changes at gamma = 1, but there is no hidden PPO-over-MPC discounted win. Gamma 0.995 gives a reward 24 hours later 61.8% of the current weight and a reward 168 hours later 3.44%; this is a substantial horizon effect. It does not, by itself, prove that temporal credit assignment caused the trained policies to lose delivery or that changing gamma will help. Realized start-of-episode returns do not reconstruct GAE advantages, learned value functions or optimizer updates.

The compact JSON retains separate component differences for all seven 24-hour blocks at every gamma. Both scalar and PI policies lose primarily through service accounting despite positive modeled operating savings. Any gamma-alignment experiment should keep economic coefficients and physical constraints fixed and compare equal training budgets; it is an additional controlled hypothesis, not an established repair.

## Validation and custody

Twenty-four small deterministic fixtures passed; Black and strict MyPy passed for the analysis source. The final streamed analysis rechecks each input path, regular-file/single-link identity, logical size, SHA-256 and stability while reading. Gzip decompression and per-decision objects are bounded. Every history has contiguous decision indices, and every reward reconciles with its component sum; episode component totals reconcile to the frozen report. Empty zero-decision E-beta prefixes are explicitly retained with zero accumulated components, not treated as completed episodes.

Preparation failures were retained: an initial summary extractor did not handle the zero-decision empty component map, and the initial history extractor looked for `reward_breakdown` where the frozen evidence uses `info.reward`. Both were corrected before the final run, with fixtures covering the final extraction paths. Neither affected source episodes or generated a policy result.

Frozen input report SHA-256: `9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c`. Final full analysis SHA-256: `2360c2a843c8aadbb61d0c050f105994736b91e91009e9d0baa05f984480aab8`. Full-analysis runtime: 100.78 seconds. Source and all 56 episode records are included in the compact evidence.

No retraining, economic coefficient change, policy promotion, held-out evaluation or new inference was performed by this audit. Existing single-optimizer-seed and reused-development-panel limits remain.
