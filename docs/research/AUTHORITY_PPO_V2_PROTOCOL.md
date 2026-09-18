# Matched PPO action-interface experiment

Authorized by the user on 17 September 2026 (local time): “Okay, do this, you have full research autonomy.” Execution began at 2026-09-16 23:26:17 UTC and must finish by 2026-09-17 11:26:17 UTC. This is a new twelve-hour research campaign, not execution or amendment of the original registered fifteen-policy beta study.

## Question and fixed task

Does a broader learned action interface improve complete plant operation relative to the existing narrow coverage-residual controller at an equal training budget? Both are hybrid controllers. Changing the action mapping also changes effective exploration; this study does not isolate authority from exploration or establish dispatch learning from scratch.

The binding contract is `benchmarks/authority_ppo_v2.json`. Equipment, minute physics, causal observations, weather and DAC-outage distribution, opening inventories, 25 kg/h methane demand, 150 kg six-hour deliveries, independent daily opening-stock returns, economic coefficients and hard guards remain exactly those of `feasible25_offgrid_v1`. Dedicated PV has no purchase/export settlement. Solar capital recovery and fixed O&M remain unestimated and excluded from operating contribution.

## Controllers and learning

- Coverage reference: the unchanged `daily-coverage-offgrid-v1` controller, including the development-selected 100 kg CO2 and 50 kg CH4 buffers.
- Narrow PPO: the exact historical ±10% multiplicative residual mapping, including preservation of zero and low/startup electrolyser targets.
- Broad PPO: additive changes up to ±30% installed target rating before explicit projection to equipment caps and permissible commitment/startup targets. It can request nonzero production when the coverage command is zero. It preserves in-progress starts and mandatory minimum-up operation. Projection events and raw actions are recorded; guards and simulator limits are unchanged.
- Strict mission MPC: supplementary first-eight-case comparison, with the existing five-second solve-stage limits and at most two concurrent workers, run without training load. Failures and partial execution are retained.

Both PPO arms use the same 139 inputs (actual 135 public observation fields plus four coverage targets), four outputs, PI-Lagrangian learner, two 64-unit tanh layers, fresh network weights with exactly zero actor means, initial Gaussian standard deviation 0.2, learning rate 0.0003, reward gamma 1, GAE lambda 0.95, 2,048 rollout steps per environment, four environments, minibatch 256, ten epochs, clipping 0.2, entropy coefficient zero, value coefficient 0.5 and gradient cap 0.5. Reward is actual interval EUR times 0.001. Four existing normalized daily resource-cost channels have zero limits, gamma 1 and the unchanged PI settings. These costs are not a mathematical safety or feasibility guarantee.

Optimizer seeds are 101, 211 and 307 in each arm. Each final run has exactly 262,144 transitions, 32 rollouts and 10,240 minibatch updates. All six use the same scenario sequence: 392 resets, comprising 388 complete weeks and four partial final weeks. Initial deterministic policies must reproduce coverage exactly. Intermediate checkpoints at 65,536-transition intervals are descriptive learning-curve evidence only; final-checkpoint selection is fixed and every final endpoint is reported.

## Scenario separation

Training: 9940000–9940391. Development: 9950000–9950007. Fresh test: 9960000–9960031. The MPC subset is prospectively the first eight test seeds. These ranges do not overlap the previous benchmark or original registered beta namespaces. Prior exposed failures 9930001 and 9930011 are diagnostic cases and cannot be called new held-out evidence. No fresh test scenario may be opened before all final endpoints, evaluation source and comparison rules are frozen. No scenario is screened or removed based on its outcome. Engineering optimizer fixtures use training seeds only, with discarded weights and explicit fixture identity.

The code expands only the seed allocation and research identities. It preserves the parent benchmark checksum and verifies equality of every physical-task field before environment construction. It keeps historical source, artifacts and negative outcomes intact.

## Endpoints and analysis

Report all six final endpoints on all eight development and 32 fresh test cases. Primary outcomes are full uninterrupted mission qualification, all 28 delivery settlements, every daily return of battery/H2/CO2/CH4, and full exact-minute safety checks including operator reserve and forced stops. Retain incomplete and failed evaluations in denominators. Report deficit magnitudes alongside strict counts. Economics is descriptive across all complete cases and paired separately on mutually qualified full missions; partial-run contribution is not weekly income. Also report wear, curtailment, power/inventory trajectories, raw/proposed/applied action changes and inference latency.

Environmental uncertainty uses paired resampling of the same 32 scenarios, preserving all optimizer seeds within a scenario; three training seeds do not create 192 independent weather replications. Show each optimizer seed and per-arm aggregates. Confidence intervals are descriptive, with no post-hoc choice of favorable metric. No general real-site, calibrated-profit or algorithm-wide superiority claim follows from this synthetic finite panel.

The default demonstration is narrow optimizer 101 on test seed 9960000, regardless of its outcome. Additional successes or failures may be shown only with their selection stated. At least one existing fault trajectory is explained dynamically. A source/checkpoint/contract-pinned replay verifies the demonstration and a prespecified broader-policy case. Existing plotting and playback mechanisms may be reused without modifying research evidence.

## Time and failure policy

The broader interface has a two-hour engineering gate, ending 01:26:17 UTC. Diagnose old failures for no more than one hour; a constructive valid schedule would establish feasibility only for the tested case, while an unsuccessful search does not establish infeasibility. Do not change the production prior based on that search.

Training has a firm cutoff at 05:26:17 UTC (six elapsed hours). Stop tuning then and freeze exact final endpoints. If a controller fails the engineering gate, retain its failure and complete the proven path; do not silently rename a different arm. A production error is retained; a repaired attempt requires a distinct path and explicit amendment before execution. Interrupted checkpoints are not represented as full-budget endpoints. Evaluation, figures and writing receive reserved time; no late new algorithm or open-ended planner repair displaces them.

Use local CPU with one Torch intra/inter-op thread per learner, up to three learners concurrently. Use internal active storage, compact training diagnostics and compressed evaluation traces with at least 2 GiB free headroom. Preserve all attempts and source snapshots. No paid compute is required. Source revisions and dirty tracked state are captured because the workspace contains unrelated ongoing product edits; source hashes bind the actual executed files. A publication-ready code package includes ignored research dependencies explicitly rather than assuming Git alone captures them.

## Deliverables

Trained and reopenable checkpoints; immutable configuration, source and attempt manifests; complete comparison and failure accounting; learning curves and paired operational figures; a two-page commission write-up draft; reproducibility instructions; and an inspectable trained-policy trajectory. Improved reliability or economics is a hypothesis, not a promised outcome.
