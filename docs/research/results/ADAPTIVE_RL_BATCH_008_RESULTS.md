# Adaptive batch 008: evaluation completed; neither trained arm improves on OperatingMPC

Batch 008 closed normally and passed its frozen audit after evaluating the exact batch 007 scalar (A) and PI (B) final checkpoints. It performed no training. All 56 evaluator and 56 replay processes closed; 40/56 actor-scenario episodes completed in full. The 16 stopped M24/E-beta prefixes remain in the evidence, outside performance means.

The untrained residual matches OperatingMPC in all eight full physical projections. Training therefore moved both arms away from a verified MPC-equivalent initialization. Both finals preserve strict hard safety in 8/8 cases, but lose mean economic value and delivery service versus OperatingMPC and fail individual-delivery and daily-resource noninferiority. Neither is selected or supports an operational-benefit claim.

The [batch 008 protocol](../ADAPTIVE_RL_BATCH_008_PROTOCOL.md) governs this descriptive evaluation. The [compact JSON](adaptive_rl_batch_008_results.json) retains exact values, every actor/seed outcome, A/B paired differences, all individual delivery regressions and all eight training diagnostic/optimizer summaries.

## Complete-panel outcomes

| Actor | Complete/replayed | Strict safety | All daily returns | Mean delivery | Mean EUR/week |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | 8/8 | 7/8 | 0/8 | 86.4336% | -13,499.22 |
| OperatingMPC | 8/8 | 8/8 | 0/8 | 89.4478% | -6,611.81 |
| M24 | 0/8 | 0/8 | 0/8 | unavailable | unavailable |
| E-beta | 0/8 | 0/8 | 0/8 | unavailable | unavailable |
| initial-mpc-residual | 8/8 | 8/8 | 0/8 | 89.4478% | -6,611.81 |
| A-mpc-residual-scalar | 8/8 | 8/8 | 0/8 | 89.2770% | -6,672.21 |
| B-mpc-residual-pi | 8/8 | 8/8 | 0/8 | 89.2786% | -6,689.35 |

H has one forced stop (seed 9700302). M24 stops in all 8 cases because its service model does not model degraded/offline DAC; E-beta stops in all 8 with an infeasible service-maximization solve. Their stopped-prefix fill values are not complete service results.

## Trained policies versus OperatingMPC

| Metric (candidate minus MPC) | Scalar A | PI B |
| --- | ---: | ---: |
| Mean EUR/week | -60.40 | -77.54 |
| Value wins (out of 8) | 5 | 3 |
| Worst EUR/week difference | -973.42 | -558.60 |
| Mean delivered kg/week | -11.48 | -11.37 |
| Worst delivered kg/week difference | -44.53 | -34.77 |
| Mean battery-throughput kWh/week | -1,438.31 | -2,238.54 |
| Mean battery-throughput change | -13.32% | -20.73% |
| Individual deliveries with less delivered(out of 224) | 27 | 24 |
| Worst individual delivery difference, kg | -40.98 | -32.35 |

A’s five value wins do not outweigh its three losses; its worst value loss is seed 9700304. B wins three and loses five, with its worst value loss at seed 9700307. Mean fill falls from 89.4478% to 89.2770% (A) and 89.2786% (B), differences of −0.1708 and −0.1692 percentage points.

A’s worst single delivery is commitment 23 of seed 9700301: 164.1176 kg versus 205.0995 kg (−40.9819 kg). B’s is commitment 23 of seed 9700304: 198.2058 kg versus 230.5529 kg (−32.3471 kg). Sign counts use exact recorded differences, including tiny numerical differences; these worst losses are separately substantial in physical units.

Battery-throughput reductions are averages: throughput is worse in 2/8 cases for A and1/8 for B. The modeled battery-throughput charge improves by €81.72/week (A) and €127.19/week (B), but lower delivered-service value plus higher shortfall penalties cost €275.51/week and €272.96/week respectively. Other operating savings do not offset the service loss. These are modeled costs, not measured battery lifetime gains.

| Seed | A ΔEUR/week | B ΔEUR/week | A Δdelivered kg | B Δdelivered kg |
| --- | ---: | ---: | ---: | ---: |
| 9700300 | 182.05 | -120.17 | 8.01 | -5.49 |
| 9700301 | 364.55 | -27.80 | 2.44 | -16.26 |
| 9700302 | -286.40 | -464.47 | -16.60 | -9.62 |
| 9700303 | -445.80 | 319.11 | -40.02 | -15.43 |
| 9700304 | -973.42 | -290.61 | -44.53 | -34.77 |
| 9700305 | 112.46 | 345.57 | 4.20 | 9.48 |
| 9700306 | 511.22 | 176.67 | 1.14 | 6.17 |
| 9700307 | 52.13 | -558.60 | -6.48 | -25.07 |

PI versus scalar is also mixed: B−A mean value is −€17.13/week, with 3/8 value wins; mean delivery is only +0.1061 kg/week. B has lower mean battery throughput, but higher throughput in 5/8 paired cases. Neither dominates the other or supports an algorithm-level conclusion.

## Daily resource restoration

Both arms fail the exact every-delivery and every-daily-resource directional checks; all-three-condition eligibility is false. OperatingMPC and both trained arms have 0/8 episodes passing all daily returns. Positive entries below are the maximum worsened candidate-minus-MPC deficit across 56 daily boundaries; the accompanying counts use exact positive differences.

| Resource | A worst increase | A worsened/56 | B worst increase | B worsened/56 |
| --- | ---: | ---: | ---: | ---: |
| battery (kWh) | 709.495159 | 31 | 587.640018 | 24 |
| h2 (kg) | 52.787790 | 20 | 22.121683 | 33 |
| co2 (kg) | 315.713124 | 28 | 158.556373 | 28 |
| ch4 (kg) | 37.499903 | 8 | 12.877729 | 5 |

## What the training evidence supports

Both imported trainings completed 65,536 transitions, 8 rollouts and 2,560 minibatch updates. Their reward value explained variance rises from −0.3108 to 0.9443 (A) and 0.9453 (B). The policies remain close to their narrow initial action distribution: all eight diagnostics report zero observed action-bound clipping, with standard deviations roughly 0.19–0.21. PPO probability-ratio clipping is distinct and remains approximately 5.6–9.9%. These data do not support action-bound saturation as the observed failure mechanism, and higher training explained variance did not imply an improved policy.

At the final pre-update diagnostic, common gradient clip factors are 0.1046 (A) and 0.1015 (B); reward-critic gradient norms are 4.3647 and 4.6622. B’s weighted-cost actor gradient norm is 1.2728 versus 0.3624 for its reward actor gradient. Its final PI multipliers are [1.0744, 1.1381, 1.1466, 1.1473] for battery/H2/CO2/CH4, while reported mean completed-episode costs remain [0.4217, 0.8537, 0.8998, 0.8984]. Cost signals were present, but did not produce daily-resource noninferiority. These sampled diagnostics are not actual shuffled minibatch/Adam update traces and establish no causal explanation for underperformance.

The recorded nominal MPC prior uses no scheduled-mode decisions in any evaluated case; it runs in recovery mode with occasional fallback. The next investigation should inspect existing delivery-boundary trajectories, inventory changes and recovery/fallback transitions, then check residual action authority, credit assignment, cost weighting and common clipping. This is a diagnosis proposal; no further training, tuning or evaluation has been started.

## Provenance and limits

Execution source `d0eab3667312454dbb42c1d667239593539cbbcf`, tree `9d35e79c5fc21e14d209caf2797882e5f2763a70`. Training source remains `61b97851637a0f5a16e56c479e1a436cd30cf68d`. Batch 007 remains aborted. Its historical allocation-custody failure is preserved; prospective import verifies path/logical size/SHA and records allocation as storage observation, without retroactively passing the old audit.

| Frozen artifact | SHA-256 |
| --- | --- |
| analysis.json | `9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c` |
| analysis.md | `3a4747efc8dc0235944df2b79b8c64f964065f828cde5a4016e7582e83d9b84d` |
| audit.json | `22aff1c0aa3b7c4a3b9bf78d6ec9a7825c14b88534dbe77537021bd66f6d9536` |
| batch report.json | `afbf5bd9ed918b877985713eaa22acf972a88502ff024596098adc9cd51e8052` |
| auditor source | `6d384ee50ab9f3aea78c981ec5728c4368b6d1ca2945edadcce802982d45f366` |
| reporter source | `c455ca9d32bc80be6d04c1d98a2b1396be051223eecb6753b30a6e3c6157d9f5` |

Internal evidence root: `/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-008-analysis`. The compact JSON records exact paths, sizes and hashes. One optimizer seed 101 and eight adaptive development cases support a descriptive within-batch comparison only: no held-out, multi-seed, site-generalization, materiality or commercial benefit claim. No policy is selected.
