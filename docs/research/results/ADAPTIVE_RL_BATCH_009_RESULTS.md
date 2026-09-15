# Adaptive batch 009: positive average PPO value, unresolved delivery and resource regressions

Batch 009 closed normally at **2026-09-15T03:05:42.172437+00:00**. Both fresh scalar PPO endpoints completed 16,384 transitions and all 16 new evaluations and 16 independent replays passed the frozen custody audit. The two policies improve mean simulated operating value versus the reused OperatingMPC reference by **€143.87 and €145.48 per 168-hour episode**, reducing its mean net operating loss by **2.18% and 2.20%**. Both pass strict hard safety in 8/8 cases. This is a positive average result on the eight-case development panel; individual delivery and daily-resource regressions prevent a qualified operational-benefit claim or policy promotion.

Changing reward gamma from .995 to 1.0 yields only **+€1.61/week in the matched B−A mean**, with 3/8 value wins, median −€40.77 and a worst loss of −€939.95. This small screen does not establish discounting as the cause of earlier underperformance or demonstrate a successful gamma fix.

The [prospective protocol](../ADAPTIVE_RL_BATCH_009_PROTOCOL.md) freezes both final endpoints, optimizer seed 101, two 8,192-transition rollouts, 640 minibatch updates, uniform .001 reward scaling, GAE λ=.95, 28 allocated training resets and reused development seeds 9700300–9700307. Both initializers matched each other and the inherited initial policy before simulation. Seven batch-008 actors are **reused reference evidence**, not newly trained or evaluated here. The [compact JSON](adaptive_rl_batch_009_results.json) preserves all nine actors, all fixed paired contrasts, every worsened individual delivery/resource row, all eight scenario tails, exact checkpoint pins and artifact digests.

## Nine-actor outcomes

| Actor | Evidence | Complete/replayed | Strict safety | All daily returns | Mean fill | Mean EUR/week |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H | reused 008 | 8/8 | 7/8 | 0/8 | 86.4336% | -13,499.22 |
| OperatingMPC | reused 008 | 8/8 | 8/8 | 0/8 | 89.4478% | -6,611.81 |
| M24 | reused 008 | 0/8 | unavailable | unavailable | unavailable | unavailable |
| E-beta | reused 008 | 0/8 | unavailable | unavailable | unavailable | unavailable |
| initial-mpc-residual | reused 008 | 8/8 | 8/8 | 0/8 | 89.4478% | -6,611.81 |
| A-mpc-residual-scalar | reused 008 | 8/8 | 8/8 | 0/8 | 89.2770% | -6,672.21 |
| B-mpc-residual-pi | reused 008 | 8/8 | 8/8 | 0/8 | 89.2786% | -6,689.35 |
| A-scalar-gamma0995 | fresh 009 | 8/8 | 8/8 | 0/8 | 89.4389% | -6,467.95 |
| B-scalar-gamma1000 | fresh 009 | 8/8 | 8/8 | 0/8 | 89.4373% | -6,466.34 |

M24 stops in all eight scenarios because its service model does not support degraded/offline DAC. Its retained prefix lengths are 6, 72, 35, 165, 143, 46, 25 and 7 decisions in seed order. E-beta stops before its first decision in all eight scenarios when service maximization is infeasible. Those prefixes are excluded from full-week performance means. Their recorded-prefix safety flags are true; **complete-episode safety is unavailable**, not eight demonstrated unsafe episodes. H completes all cases but has one forced electrolyser stop at seed 9700302.

All complete actors, including OperatingMPC, have 0/8 episodes satisfying every daily return condition. This means each episode has at least one failed return condition, not that every daily boundary fails. The paired checks still require PPO not to worsen any individual return.

## Fresh policies versus OperatingMPC

| Candidate-minus-MPC metric | A: γ=.995 | B: γ=1 |
| --- | ---: | ---: |
| Mean EUR/week | +143.866379 | +145.476729 |
| Mean delivered kg/week | -0.599030 | -0.702388 |
| Mean battery-throughput kWh/week | -1,706.549411 | -2,025.669708 |
| Economic wins / 8 | 6 | 5 |
| Worst economic difference, EUR | -717.334099 | -695.912134 |
| Battery-throughput reduction | 15.8039% | 18.7592% |
| Worsened individual deliveries / 224 | 23 | 28 |
| Worst individual delivery difference, kg | -32.878762 | -42.854947 |

Mean delivery falls by only 0.5990 kg (A) and 0.7024 kg (B) out of the 6,720 kg contractual week, but averages conceal individual losses.
A (γ=.995)'s worst delivery is seed 9700304, commitment 12: 124.371353 kg versus 157.250115 kg (Δ -32.878762 kg).
B (γ=1)'s worst delivery is seed 9700305, commitment 27: 196.503130 kg versus 239.358078 kg (Δ -42.854947 kg).

Counts use exact recorded signs, including numerical-scale differences. No tolerance was fitted after observing outcomes; the much larger worst losses are shown separately. Lower throughput is reduced battery cycling, not proof of thermodynamic efficiency or measured life extension. Closing inventory differences remain visible.

| Economic contribution, candidate minus MPC (EUR/week) | A | B |
| --- | ---: | ---: |
| Delivered service value | -2.396118 | -2.809553 |
| Shortfall penalties | -11.980591 | -14.047765 |
| Battery wear reserve | +96.963035 | +115.094870 |
| Electrolyser wear reserve | +8.925000 | +15.693750 |
| Solar opportunity cost | +51.651972 | +30.020955 |
| Battery endowment liability | +0.907685 | +1.857209 |
| Methane-equivalent endowment liability | -0.204603 | -0.332737 |

The modeled operating and wear credits outweigh the small mean service loss in the weekly ledger. Daily inventories are not preserved, so these raw gains are not validated sustained savings and cannot be attributed wholly to efficiency. Economic coefficients remain unchanged synthetic benchmark assumptions; this is not site-calibrated commercial savings.

## Every scenario and daily-return regression

| Seed | A−MPC EUR | B−MPC EUR | B−A EUR | A−MPC delivered kg | B−MPC delivered kg |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9700300 | +253.848293 | +251.227689 | -2.620605 | +17.101827 | +13.751199 |
| 9700301 | +232.701488 | -34.375873 | -267.077361 | -7.663531 | -25.443589 |
| 9700302 | -717.334099 | +25.981295 | +743.315394 | -18.467836 | +3.234296 |
| 9700303 | +840.067314 | +902.082170 | +62.014857 | +16.210768 | +27.151092 |
| 9700304 | -478.302698 | +246.454274 | +724.756972 | -32.611986 | +5.094267 |
| 9700305 | +244.034163 | -695.912134 | -939.946297 | +17.384283 | -25.429886 |
| 9700306 | +555.255204 | +476.343418 | -78.911786 | +7.655952 | -0.340792 |
| 9700307 | +220.661369 | -7.987010 | -228.648379 | -4.401714 | -3.635693 |

Positive resource differences mean larger deficits at the same daily boundary. Counts are out of 56 boundaries for each resource.

| Resource | A worsened / 56 | A largest increase | B worsened / 56 | B largest increase |
| --- | ---: | ---: | ---: | ---: |
| battery (kWh) | 31 | 613.651361 | 29 | 540.711908 |
| h2 (kg) | 20 | 17.295688 | 25 | 17.258927 |
| co2 (kg) | 34 | 318.866335 | 34 | 318.866335 |
| ch4 (kg) | 6 | 23.246826 | 6 | 36.593494 |

Both fresh policies fail every-individual-delivery and every-individual-resource noninferiority despite passing hard safety. A lower mean H₂ deficit does not waive worsened individual H₂ boundaries. The JSON records every regression with seed, delivery ordinal or cycle, and signed magnitude.

## All fixed paired contrasts

Resource count order below is battery/H₂/CO₂/CH₄, each out of 56; deliveries are out of 224. Every available row fails strict service/resource preservation. The initial residual matches the inherited OperatingMPC reference exactly; its duplicated contrasts are retained.

| Candidate | Comparator | Mean ΔEUR | Value wins / 8 | Worst ΔEUR | Worse deliveries | Worse resource counts |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B-scalar-gamma1000 | A-scalar-gamma0995 | +1.610349 | 3 | -939.946297 | 28 | 35/32/24/9 |
| A-scalar-gamma0995 | H | +7031.271875 | 7 | -533.766953 | 15 | 55/0/56/19 |
| A-scalar-gamma0995 | OperatingMPC | +143.866379 | 6 | -717.334099 | 23 | 31/20/34/6 |
| A-scalar-gamma0995 | M24 | unavailable | unavailable | unavailable | unavailable | unavailable |
| A-scalar-gamma0995 | E-beta | unavailable | unavailable | unavailable | unavailable | unavailable |
| A-scalar-gamma0995 | initial-mpc-residual | +143.866379 | 6 | -717.334099 | 23 | 31/20/34/6 |
| A-scalar-gamma0995 | A-mpc-residual-scalar | +204.268143 | 6 | -430.932215 | 27 | 28/27/24/9 |
| A-scalar-gamma0995 | B-mpc-residual-pi | +221.403110 | 5 | -252.866299 | 25 | 32/20/23/14 |
| B-scalar-gamma1000 | H | +7032.882224 | 7 | -762.415332 | 15 | 55/0/56/20 |
| B-scalar-gamma1000 | OperatingMPC | +145.476729 | 5 | -695.912134 | 28 | 29/25/34/6 |
| B-scalar-gamma1000 | M24 | unavailable | unavailable | unavailable | unavailable | unavailable |
| B-scalar-gamma1000 | E-beta | unavailable | unavailable | unavailable | unavailable | unavailable |
| B-scalar-gamma1000 | initial-mpc-residual | +145.476729 | 5 | -695.912134 | 28 | 29/25/34/6 |
| B-scalar-gamma1000 | A-mpc-residual-scalar | +205.878493 | 4 | -808.367348 | 20 | 33/27/27/6 |
| B-scalar-gamma1000 | B-mpc-residual-pi | +223.013459 | 6 | -1041.486628 | 25 | 32/21/28/10 |

The new 16,384-transition endpoints have higher mean value than both older 65,536-transition endpoints, but those older runs used four times the budget. These are descriptive endpoint comparisons on reused development cases, not a controlled claim about training duration, an interim-checkpoint selection, or evidence that more training causes deterioration.

## Discounted returns and the gamma hypothesis

Both endpoints rank above OperatingMPC under the reported undiscounted weekly ledger and below it under a γ=.995 return measured from the episode start. Discounting therefore changes trajectory ordering for these new endpoints. The older batch-008 policies remained behind OperatingMPC throughout the earlier audited gamma sweep.

| Controller | Mean undiscounted EUR | Mean γ=.995 return EUR | γ=.995 Δ vs MPC |
| --- | ---: | ---: | ---: |
| OperatingMPC | -6611.812620 | 1489.587558 | +0.000000 |
| A-scalar-gamma0995 | -6467.946240 | 1470.162919 | -19.424639 |
| B-scalar-gamma1000 | -6466.335891 | 1486.776006 | -2.811551 |
| A-mpc-residual-scalar | -6672.214384 | 1456.008546 | -33.579011 |
| B-mpc-residual-pi | -6689.349350 | 1462.865660 | -26.721897 |

These are realized trajectory returns, not learned critic estimates or discounted accounting profits. A positive discounted return can coexist with a negative weekly ledger because the time weights differ. Changing the training gamma produces B−A mean +€1.610349 undiscounted and +€16.613088 under γ=.995, but B loses undiscounted value in five of eight paired scenarios; its best/worst changes are +€743.315394/−€939.946297. It worsens 28 individual deliveries, with the largest B−A loss −64.730469 kg. One optimizer seed and two rollouts do not resolve this heterogeneous contrast or establish convergence.

The MPC baseline accepted **5,327 optimized recovery plans** and used 49 heuristic fallbacks, across 5,376 decisions (99.09% optimized plans). A has the same mode counts; B has 5,343 recovery plans and 33 fallbacks. Recovery is a valid optimized plan labelled for current/projected inventory deficits or forecast delivery shortfalls; only Fallback replaces the rejected plan with the heuristic. Zero `scheduled` labels therefore does not mean optimization was inactive. These results compare PPO against an actively optimizing MPC baseline, predominantly under recovery conditions. The individual delivery/resource regressions and development-only evidence still prevent a qualified superiority claim.

## Decision, custody and limits

No arm is selected or deployed. The existing trained models remain intact as the user-authorized fallback, and the live OperatingMPC mode is not replaced. The positive average result is retained alongside the service/resource failures. No held-out confirmation, protected seed access, registered three-family/five-seed experiment, site calibration, algorithm-level superiority or commercial-benefit claim follows from this screen.

Execution source `a8602d9256182469f6abb9ced2d62012e8cdc592`, tree `520cd4e0d6462e216ddcd875ed29842fe34c063b`. Fresh models start from matched seed-101 initialization and preserve all checkpoint, allocator, optimizer, supervision and independent replay bindings. The older 65,536-transition endpoint files remain under batch 007 and were evaluated in batch 008. Batch 007 remains aborted; the historical batch-008 reference admission is not a retroactive reclassification.

| Artifact | SHA-256 |
| --- | --- |
| analysis_json | `285f53d1ae6f3a07a4b6c657071711b338c9ef206fa9a0575fc122f34ccb7f31` |
| analysis_markdown | `731edc4c495ba973f9d74087b5629d2b1412159810a152e4f0cfd7a0d246a143` |
| audit | `9fc2e4b6fb74f4252079f94903b34831875281cf62a621e10cec0dd024f69b31` |
| batch_report | `86db89596756b2c23708625c98667ff60733ba1a0416698af838e48dfa126d0f` |
| plan | `5c70448278468d37a7ac99fdacc604f03d9bae30f8f08d5aca337c9b6ea8f00e` |
| gate | `1e471a67cfcec5fe9eccd7be105156c4afcaa528c2d6811f32fd273226b8921f` |
| reference_admission | `f3525ce4c73b31345e6aa2e1a04be21b7213cc85c8c0e422ec8003035417aac8` |
| analysis_tools_manifest | `db7b5981a706ec2461e723f3fc8da659bb0c7425c8244edac22479817186545f` |
| A-scalar-gamma0995 final checkpoint (815,891 bytes) | `2d9115acb3563f3a352aa7f6a241673002095bac8845f141a544cc4c9e49866f` |
| B-scalar-gamma1000 final checkpoint (815,887 bytes) | `be134c0809b77896695d7482db4e432afe730c7e75cf89780a454b12e2bbe907` |

The retained older scalar/PI checkpoint SHA-256 values are `4031802ee0d8b7042de4b5193ab82797479dfa800d6943f62b134aa0e415e524` and `881140cb3ddfa52d975017bb6cbbcbcecb8fbeb147a91cdd6a7db8430fd8e05d`; the JSON binds their paths, sizes and historical audit/report sources.

Raw checkpoints and full evidence remain outside Git under `/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-009` and `/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-009-analysis`. The compact JSON includes exact artifact paths and sizes. No frozen audit, report, simulator or policy was rerun to prepare this publication.
