# Coverage-residual PPO on the feasible off-grid mission

This is a new training experiment on the separately versioned **25 kg/h methane
mission**, with the original equipment, minute physics, stochastic physical
distribution, guard and daily stock-return requirements. It preserves every
earlier experiment and checkpoint. The binding task is
[`feasible25_offgrid_v1.json`](benchmarks/feasible25_offgrid_v1.json), contract SHA256
`deae125d30f30cf63860aeaec76db566dffee18e6523de2fa508b26f57dafe7e`.

Dedicated PV has zero marginal electricity charge and no export alternative.
The actual reward and public price observations use that boundary. The original
synthetic market is private audit data. Solar capital recovery and fixed O&M are
uncalibrated and excluded; positive operating contribution is not commercial
profitability. No equipment capacity, weather realization, delivery deadline or
inventory return requirement is relaxed by training.

## Fixed learner and initial policy

Three fresh PI-Lagrangian PPO learners use optimizer seeds **101, 211 and 307**.
Each completes exactly **65,536 physical transitions**, using four synchronous
environments, eight rollouts of 8,192 transitions and 2,560 minibatch updates.
The existing constrained-PPO implementation is reused without modifying its
algorithm or claiming registered-study admission.

The policy starts from an exactly zero actor output head around the causal
`daily-coverage-offgrid-v1` prior, with CO2 buffer 100 kg and CH4 buffer 50 kg;
all other original coverage settings are unchanged. This prior was adopted
before training after the original coverage V2 failed two of the eight fresh
development cases and the declared reserve variant passed all eight. Both
development results remain recorded. Its exact version and settings are bound by
`make_coverage_prior()` in the new
[`coverage_ppo_v1.py`](tools/coverage_ppo_v1.py) and in the frozen training plan.
The prior is fixed before production training and keeps the original
coverage controller's results intact. The existing learned MPC-residual weights
are not imported. This is fresh PPO training with a rule-based initial policy.

The network receives the actual float32 135-field public observation plus four
cached coverage targets, each divided by its installed rating. A bounded residual
changes target effort by at most 10%; zero targets remain zero. Electrolyser
commands at or below the stable operating floor are preserved. Larger commands
are perturbed only above that floor. Existing operator caps remain authoritative,
and every projection is recorded. Zero residual returns the original command
exactly. The deterministic initial policy must reproduce coverage commands,
physical states, reward and stock costs before training.

| Setting | Fixed value |
| --- | --- |
| Network | Separate actor/reward MLPs, two 64-unit tanh layers; existing four-head cost critic |
| Initial Gaussian standard deviation | 0.2 |
| Learning rate | 0.0003 |
| PPO rollout steps per environment | 2,048 |
| Minibatch size / epochs | 256 / 10 |
| Reward gamma / GAE lambda | 1.0 / 0.95 |
| Reward units | Actual interval EUR multiplied uniformly by 0.001 |
| PPO clip / entropy coefficient | 0.2 / 0 |
| Value coefficient / gradient norm cap | 0.5 / 0.5 |
| Cost channels | Actual battery, H2, CO2 and CH4 daily deficits; original normalization |
| Cost limits / gamma / GAE lambda | Four zeros / 1.0 / 0.95 |
| PI multipliers | Existing fixed PI settings, with initial integrals of one |
| Runtime | CPU; one Torch intra-op and one inter-op thread per process |

Four stock costs do not mathematically guarantee delivery or hard safety. The
actual EUR delivery penalty, common guard and complete mission checks remain
necessary. No failed learned command is silently replaced by coverage. The
learned policy must show actual actor-parameter changes, not merely critic updates.

## Data separation

All three optimizer seeds see the same physical training sequence
**9,910,000–9,910,099**, assigned as
`base + 4 × local_episode_ordinal + environment_index` by the existing allocator.
Each learner uses 100 resets: 96 complete weeks and four partial final episodes.
The final partial episodes remain counted in the exact transition and cost
ledgers; they are not reported as completed weeks.

Development uses **9,920,000–9,920,007**. Held-out testing uses
**9,930,000–9,930,015**. The previously inspected development seeds
9,700,300–9,700,307 and demo 9,000,034 remain historical development evidence.
They are not held-out cases. Original registered-study seed namespaces remain
unopened. Construction and learning fixtures use training seeds only and are
separately labeled; their weights are discarded.

## Endpoints, comparison and claims

All three fixed final endpoints are saved, reopened and checked against their
embedded constraint state, exact transition counts, optimizer ledgers and actor
tensors. Their source, task, configuration and checkpoint hashes are frozen
jointly **before any held-out scenario is opened**. No interim checkpoint is
selected. All three final policies are reported on development and held-out
panels, including failures, together with their exact initial policy and the
declared comparison controllers.

Full mission success requires complete 168-hour execution, all 28 deliveries,
all seven daily returns for all four stocks, and the unchanged hard-safety
checks. Economic comparisons retain every service/resource failure and terminal
inventory. A trained policy that fails these checks is not promoted, regardless
of average operating contribution. Three optimizer seeds and a finite synthetic
panel do not establish site generalization or algorithm-wide superiority.

The runner writes an exclusive plan and separate per-seed attempt, initial
weights, rollout diagnostics, optimizer ledgers, progress, final checkpoint and
completion/failure records. It checks task/source identity before and after
training. A failed or interrupted attempt is preserved; it is not overwritten or
silently resumed. Further design changes require a new versioned plan and must
not adapt to the held-out results.

The source set includes backend and simulator Python code except the exact file
`backend/src/plant_backend/mission_mpc.py`. That separately developed comparator
is not imported or executed by coverage PPO and receives its own frozen
evaluation evidence. All actual coverage-PPO runtime sources remain bound.

## Local execution estimate and checks

A training-seed throughput probe executed 512 zero-residual transitions in
6.11 seconds, approximately 83.8 transitions/second. This projects about 13
minutes of physics per 65,536-transition learner. Allow roughly **15–25 minutes
per learner** for actual optimization, resets and diagnostics; shared CPU load
can increase wall time. No paid infrastructure or external data transfer is used.

The focused suite checks exact zero-residual day trajectories, invalid-action
rejection before physics, bounded command projection, task identity across
training/development, and an actual small PI-PPO optimization with changed actor
weights and exact checkpoint reopening. These are engineering fixtures, not
production training or held-out performance evidence.
