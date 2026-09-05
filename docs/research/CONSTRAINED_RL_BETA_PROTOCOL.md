# Constrained-RL beta protocol

Status: **frozen before implementation, training, development evaluation, or held-out opening**

Date: **2026-09-05**

Protocol identifier: **`plant-beta-constrained-rl-protocol-v1`**

## Decision

The first learned controller for the cyclic stochastic-market beta will not be
selected from ordinary scalar-reward PPO alone. The primary learner is a
PPO-compatible proportional-integral Lagrangian controller with one reward
critic, four resource-specific cost critics, and four independent non-negative
multipliers. Plain integral-only PPO-Lagrangian and the existing scalar PPO are
mandatory matched controls.

The learner optimizes expected constraint cost. It does **not** replace either
of the two stronger operational boundaries:

1. the common dispatcher and independent guard remain the authority for hard
   instantaneous plant safety; and
2. a policy is admitted for value comparison only after exact simulator evidence
   shows complete execution, every daily resource return, hard safety, and
   comparable delivered service.

No CMDP or Lagrangian convergence statement is treated as a deployment safety
certificate. This document authorizes only bounded optimizer smokes whose every
environment reset uses the fixture domain. It does not authorize any of the 15
registered training runs or any development or held-out reset.

## Why this method is the first baseline

The beta exposes a separate non-negative cycle-constraint signal and an unchanged
EUR reward, but stock Stable-Baselines3 PPO ignores the constraint information.
Penalty-tuning the EUR reward would collapse two distinct questions—economic
value and sustainable resource return—back into one arbitrary coefficient.

PI/PID Lagrangian control was introduced to damp the oscillation and overshoot of
ordinary dual ascent while retaining a PPO-family implementation
([Stooke, Achiam, and Abbeel, 2020](https://proceedings.mlr.press/v119/stooke20a.html)).
The primary protocol uses proportional plus integral control and fixes the
derivative term to zero because the authoritative signal is sparse and arrives
only at 24-hour boundaries. The integral-only arm recovers the conventional
PPO-Lagrangian mechanism; scalar PPO isolates the effect of using a constraint at
all.

CPO is not the first implementation. Its expected-cost update bound is useful,
but its second-order approximation, conjugate-gradient solve, line search, and
infeasible-step recovery would add a second optimizer stack before this
repository has validated its first cost-aware rollout path
([Achiam et al., 2017](https://proceedings.mlr.press/v70/achiam17a.html)).
FOCOPS and the newer nu-PI multiplier update remain registered follow-up
families, not outcome-dependent substitutes
([Zhang et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/af5d5ef24881f3c3049a7b9bfe74d58b-Abstract.html),
[Sohrabi et al., 2024](https://proceedings.mlr.press/v235/sohrabi24a.html)).

Zero training cost is an objective, not a guarantee. Empirical safe-RL work has
shown that constrained algorithms can retain violations even when their cost
limit is zero; denser learned or shaped costs may help but change the learning
problem
([He, Zhao, and Liu, 2023](https://doi.org/10.1609/aaai.v37i12.26734)).
The first result therefore keeps the exact sparse settlement cost. Any dense
margin shaping requires a new named ablation and cannot be introduced after
observing this protocol's outcomes.

## Frozen plant and information contract

All controllers use `plant-env-v2-cyclic-market-beta-v1` and its existing:

- 135 ordered finite observations;
- four normalized absolute process targets;
- 15-minute policy decision and one-minute physical dispatch cadence;
- seven-day, 672-decision episode;
- causal public solar and price forecast vintages;
- private realized future weather, price, market regime, and DAC schedule;
- non-ratcheting opening-resource reference; and
- exact 24-hour battery, H2, CO2, and saleable-CH4 settlement.

The constrained trainer may read only the transition's public observation,
scalar EUR reward, termination flag, and the new versioned cost record derived
from `cycle_constraint.settled_this_step`. It may not read the cumulative cost,
private manifest, seed, future truth, future forecast vintage, disturbance
schedule, simulator object, or another controller's result.

The policy observation and action contracts do not change. Constraint-training
settings are bound to training and checkpoint provenance, not smuggled into a
new observation under the old environment identity.

## Canonical transition-cost contract

Implementation must add an immutable `CycleConstraintTransitionV1` record at
every controller transition. Resource order is fixed:

```text
battery, h2, co2, ch4
```

For resource `r` at transition `t`:

```text
raw_cost[t,r] =
    max(0, deficit[r] - tolerance[r]) / max(1, target[r])
```

at an exact cycle settlement, and zero otherwise. The existing
`total_constraint_cost` must equal the ordered sum of the four raw values.
Training uses:

```text
training_cost[t,r] = raw_cost[t,r] / episode_cycle_count
```

so one completed episode's undiscounted sum is its mean per-cycle normalized
violation. Under the frozen seven-day task, `episode_cycle_count = 7`.

Every record must contain its schema, decision index, whether a cycle settled,
optional cycle index, fixed resource order, raw four-vector, scaled four-vector,
raw/scaled sums, and a canonical digest. Non-settlement records must contain
exact zeros and no invented cycle index. Settlement records must reconstruct
exactly from `CycleSettlementV1`.

The cost is non-negative and each channel has budget zero. Four channels are
retained instead of optimizing only their scalar sum so a multiplier cannot
silently trade one resource's recurring feasibility against another. The
existing scalar sum remains a reporting checksum.

The cumulative episode cost must never be used as the transition cost: doing so
would charge every earlier violation again at every later decision. Current
cycle margins also cannot become the authoritative cost. They are early-warning
state features; only the exact boundary settlement decides feasibility.

The final transition's settlement is collected before terminal handling. A true
episode termination bootstraps every value head with zero. A PPO rollout ending
inside an episode bootstraps reward and cost values from that same boundary. The
vector environment's automatically reset observation must never replace the
terminal observation before its cost is recorded.

## Algorithms and exact optimization contract

The three core families are:

| Identifier | Policy update | Constraint use |
|---|---|---|
| `plant-beta-ppo-pi-lagrangian-v1` | PPO with PI-controlled multipliers | primary |
| `plant-beta-ppo-i-lagrangian-v1` | PPO with integral-only multipliers | mandatory algorithm ablation |
| `plant-beta-sb3-ppo-scalar-v1` | ordinary PPO on unchanged EUR reward | mandatory unconstrained control |

All three use the same actor, reward critic, initialization, stochastic training
actions, environment sequence, optimizer schedule, and PPO settings:

```text
policy hidden layers       [64, 64], tanh
reward critic layers       [64, 64], tanh, separate from actor
learning rate              3e-4
n_steps                    2048 per environment
n_envs                     4, synchronous
batch size                 256
epochs per update          10
reward gamma               0.995
reward GAE lambda          0.95
clip range                 0.2
entropy coefficient        0
reward value coefficient   0.5
maximum gradient norm      0.5
device                     cpu
```

The constrained families additionally use one separate `[64, 64]` tanh cost
network with four output heads and:

```text
cost gamma                 1.0
cost GAE lambda            0.95
aggregate cost-value loss  0.5 * mean(four channel MSE losses)
cost limits                [0, 0, 0, 0] mean normalized cost per cycle
multiplier initial values  [1, 1, 1, 1]
K_I                        0.01
K_P                        0.25 for PI; 0 for integral-only
K_D                        0
proportional EMA alpha     0.95
multiplier bounds          [0, 100] independently
update cadence             once after every complete 8192-transition rollout
update statistic           mean completed-episode sum of scaled cost, per resource
```

Actor and reward-critic parameters are initialized first through the unchanged
SB3 seed path. The cost network is then initialized inside an isolated Torch RNG
scope with `cost_network_seed = 10,000,000 + optimizer_seed`; leaving that scope
must restore the actor/action-sampling RNG state exactly. Tests compare the
actor/reward parameter bytes of all three families before the first rollout.

If a rollout contains no completed episode, the multiplier state is unchanged;
partial-episode accumulators cross rollout boundaries without being counted
twice. The default rollout contains multiple complete episodes, but the behavior
remains defined for short implementation fixtures.

One completed episode's scaled-cost sum is exactly its mean raw normalized cost
over the seven cycles. For each resource, let `e[r]` be the mean of those
completed-episode sums in the rollout minus its zero limit. The PI update is:

```text
p[r] = 0.95 * p[r] + 0.05 * e[r]
i[r] = clip(i[r] + 0.01 * e[r], 0, 100)
lambda[r] = clip(i[r] + K_P * p[r], 0, 100)
```

with `p` initialized to zero and `i` initialized to one. There is no derivative
state and no outcome-dependent anti-windup rule beyond the declared projection.
The integral-only arm differs only by `K_P = 0`.

After a rollout, the implementation first computes reward/cost returns and the
completed-episode cost statistic, then updates the multipliers, then optimizes
that same rollout using the new multiplier vector. On every minibatch, reward
advantage and each cost advantage are normalized independently as
`(x - mean(x)) / (sample_std(x) + 1e-8)`, where `sample_std` uses denominator
`N - 1`. An exactly constant advantage is explicitly replaced by an exact zero
vector before division. The actor uses the rescaled combined advantage:

```text
A_combined =
    (A_reward - sum(lambda[r] * A_cost[r]))
    / (1 + sum(lambda[r]))
```

and otherwise preserves the declared PPO clipped-surrogate update. Cost critics
are optimized only from their own returns. Scalar PPO must still record the
four cost channels and all constraint statistics, but none may affect its
gradient.

Every rollout record retains reward return, per-resource raw and scaled cost,
complete-cycle count, four value losses, four multiplier values, four PI states,
KL, entropy, clipping fraction, optimizer steps, elapsed compute, and a digest.
NaN, infinity, missing settlement cost, unknown resource order, multiplier
saturation without disclosure, or rollout/reset discontinuity fails the run.

## Outcome-blind seed custody

Every episode seed below `1,000,000` is conservatively development-contaminated.
This includes the earlier alpha studies, beta seed-34 smoke, and 901--906
fixtures. The new domains are disjoint:

| Purpose | Frozen domain |
|---|---|
| Training episodes | `1,000,000` through `1,999,999` |
| Development/selection | exactly `2,000,000` through `2,000,031` |
| Held-out test | exactly `3,000,000` through `3,000,031` |
| New implementation fixtures | `9,000,000` through `9,999,999` |

Merely listing the development and held-out integers does not authorize an
environment reset. Before the implementation gate closes, tests and dry runs
may use only the fixture domain. No development or held-out manifest, inventory,
weather, price, market regime, fault schedule, observation, or controller output
may be inspected.

Training separates optimizer randomness from environment randomness. Optimizer
and network seeds are exactly:

```text
101, 211, 307, 401, 503
```

Every family/optimizer-seed run receives the same deterministic episode stream.
For vector-environment index `j` in `0..3` and its local episode ordinal `k`
starting at zero:

```text
global_episode_ordinal = 4 * k + j
episode_seed = 1,000,000 + global_episode_ordinal
```

The allocator must pass an explicit seed at every reset, ledger the seed and
manifest identity, and reject duplicate or out-of-domain use. Its exact cursor
save/restore behavior is exercised only by fixture-domain resume tests;
registered mode rejects `resume_from` and always starts from four zero cursors.
`ControllerNeutralTrainingConfigV1.seed` cannot continue to mean both optimizer
seed and first environment seed in these runs.

At the exact registered budget, every environment receives 262,144 decisions:
390 complete 672-decision episodes and 64 decisions in episode ordinal 390.
Therefore a run consumes exactly 1,564 episode seeds, `1,000,000` through
`1,001,563`; each environment ends with `current_episode_ordinal = 390`,
`decision_offset = 64`, and `next_episode_ordinal = 391`. The final ledger and
checkpoint must match those values exactly.

The private `mixed` market regime remains naturally sampled. Seeds cannot be
replaced to balance regimes, outages, inventories, or weather. Those strata are
reported after opening; imbalance limits interpretation rather than authorizing
substitution.

## Finite training budget and checkpoint rule

Each of the 15 core runs has exactly:

```text
total decisions            1,048,576
rollout size               8,192
optimizer updates          128
eligible checkpoints       final checkpoint only
```

The total registered core budget is therefore 15,728,640 controller decisions.
There is no early stopping, outcome-dependent extension, hyperparameter search,
or best-intermediate-checkpoint selection. Intermediate crash-recovery material
is not an eligible policy. An output directory must be new, absolute, external
to the repository, and non-overwriting.

Training order is family order from the table above, then ascending optimizer
seed. Once the first gradient update occurs, that `(family, optimizer seed,
source revision)` is consumed. There is no same-revision retry, restart,
`resume_from`, or seed substitution after a registered training failure. A
failure before model construction, output creation, or the first gradient update
may be repaired only after it is journalled as infrastructure-only. Resume
remains a fixture-tested software capability for checkpoint-custody validation;
it is disabled for all 15 registered runs.

Every final checkpoint must remain an SB3-loadable PPO inference artifact but
must identify its actual algorithm. It must bind:

- environment and task contracts;
- policy and all reward/cost critic architecture;
- complete PPO and PI/I settings;
- cost schema, resource order, scaling, budgets, and terminal semantics;
- optimizer seed and exact decision count;
- episode allocator identity, all consumed seed/manifests, and terminal cursors;
- complete rollout-ledger digest and final multiplier/PI state;
- source revision, dependency lock, Python, NumPy, Torch, SB3, and plant-sim
  versions;
- policy bytes; fixture-only resumed artifacts additionally bind their exact
  parent identity and are permanently ineligible for registered evaluation.

The existing `sb3-ppo` container format may be reused only when ordinary
`PPO.load` can reconstruct the exact custom policy and embedded Gym spaces.
`CheckpointMetadata.algorithm` and constrained-training metadata must distinguish
PI-Lagrangian, integral-only, and scalar policies. `trained=True` alone is not
evidence that a constrained optimizer was used.

## Outcome-free implementation gate

Only fixture-domain tests are permitted before this gate closes. They must prove:

1. the four transition costs reconstruct exact settlement deficits/tolerances,
   are zero off-boundary, use fixed order, and sum to the historical scalar;
2. the terminal cycle cost is collected before reset and a rollout cut uses the
   correct cost-value bootstrap;
3. vectorized auto-reset cannot lose or duplicate a cost or terminal observation;
4. the allocator gives all three families the same ordered episode sequence,
   separates optimizer seeds, rejects split crossing and duplicate use, and
   restores cursors exactly;
5. reward and cost GAE are independently hand-checkable across ordinary,
   rollout-cut, and terminal transitions;
6. each cost head and multiplier changes only from its own resource channel;
7. `K_P=0` reproduces the declared integral-only update exactly;
8. scalar PPO gradients are invariant to changed cost evidence while retaining
   that evidence;
9. actor loss implements the frozen multiplier rescaling and remains finite at
   zero variance and multiplier bounds;
10. checkpoint save/resume/load rejects algorithm, cost, allocator, cursor,
    environment, source, and artifact substitution;
11. a genuine tiny beta PI/I/scalar train-save-load-deterministic-inference smoke,
    plus a separately marked fixture-only resume smoke, crosses the common
    four-target dispatcher and guard without making either artifact eligible;
    and
12. historical scalar PPO, alpha tasks, beta task identity, benchmark evidence,
    and V1 runtime behavior remain unchanged.

The complete Python repository, strict touched-module typing, formatting,
compilation, dependency-lock cleanliness, CLI dry-run/import checks, and frontend
build/lint are required before a clean implementation commit. These are software
tests, not policy outcomes.

## Development evaluation and selection

Development seeds may be opened only after the implementation protocol, source,
dependency lock, 15 run specifications, expected output destinations, and
training order are committed. All 15 final policies are then evaluated on all 32
development seeds with deterministic actions.

The existing beta four-arm runner remains a development integration instrument,
but its all-four value eligibility cannot become the primary constrained-RL
estimand. H and M24 deliberately lack full market/cycle objectives and may be
cycle-infeasible. A separately versioned campaign must therefore retain every arm
while defining pairwise eligibility:

```text
primary pair      constrained policy versus E-beta
secondary pair    constrained policy versus scalar PPO
context arms      H and M24, always reported but not blockers for the primary pair
```

A pairwise value is eligible only when both named arms complete, pass hard
safety, meet every cycle return, and deliver requested/delivered/unmet service
within the existing `1e-6 kg` tolerance. No failure, missing arm, or ineligible
cell enters value aggregation. Constraint, completion, safety, service, and
latency outcomes are still reported for every arm without pairwise filtering.

A policy is development-admissible only if all 32 policy/E-beta pairs satisfy the
full pairwise gate above. Thus both arms' reports and detached verifications must
succeed; both must contain every expected decision, minute, and cycle; neither
may have an unexpected hard-safety event, forced stop, or out-of-range decision;
both must satisfy all four returns at every daily settlement; and their service
must be comparable. Champion statistics consume exactly one complete 32-value
paired vector. A surviving subset is never scored.

Only the final checkpoint can be considered. Among admissible policies in one
family, select one development champion by this descending lexicographic key over
paired `policy net_operating_value_eur - E-beta net_operating_value_eur`:

1. interquartile mean;
2. lower-20% CVaR;
3. median;
4. fraction strictly greater than zero;
5. lower optimizer seed.

For 32 seeds, IQM removes the lowest eight and highest eight values before taking
the mean. Lower-20% CVaR is the arithmetic mean of the seven smallest values.
Exact zero is a tie, not an improvement. All five policies, including failed and
non-selected policies, remain visible in family summaries.

Held-out opening is authorized only if:

1. all five PI-Lagrangian training runs and development reports complete and
   verify;
2. at least four of five PI policies are development-admissible;
3. the PI champion's development IQM delta versus E-beta is positive and its
   lower-20% CVaR is non-negative;
4. every core policy, champion choice, source/config/artifact digest, comparator,
   statistic, and report destination is frozen; and
5. the economic sensitivity rules below are implemented and verified without
   rerunning a controller.

Failure is a valid negative development result and leaves held-out seeds sealed.
Any learning-rule or hyperparameter repair creates a new protocol version, new
training and development domains, and a separate audit trail.

## Economic sensitivity before held-out opening

Physical trajectories are evaluated once. A new generic, versioned pairwise
rescorer and detached verifier must always orient differences as `policy -
E-beta`; the existing M-minus-H-labelled helper cannot be reused unchanged. From
retained additive ledger components, report the existing outcome-blind one-factor
multipliers `0`, `0.5`, `1`, and `2` for:

- methane endowment liability, leaving unmet-delivery penalty unchanged;
- joint battery-throughput and electrolyser-life reserves; and
- solar opportunity cost.

Also report no endowment liability, no wear reserve, and service plus solar
opportunity cost only. Controllers are never rerun and no coefficient is fitted.
Every ranking reversal is retained. These are synthetic assumption sensitivities,
not calibrated confidence intervals or alternative primary objectives.

## One-shot held-out campaign

The held-out campaign requires a new format and detached verifier; the current
development-labelled mean-only runner cannot be relabelled. It retains baseline
evidence once per episode seed and references it by digest across policy pairs.
For every held-out seed it contains:

```text
H
M24
E-beta
five PI-Lagrangian policies
five integral-only policies
five scalar-PPO policies
```

All environments are independently constructed and identity-matched. Controller
failures are never replaced. The first held-out environment reset consumes all
32 seeds for this protocol. There is no retry, seed replacement, source repair,
coefficient change, controller repair, or inspection before exclusive report
closure and complete detached verification. A software defect discovered after
opening remains the final failed result for this task/protocol version.

Primary uncertainty uses episode seed as the paired resampling unit. Family
uncertainty uses a hierarchical bootstrap: resample optimizer seeds, then paired
episode seeds. PI-versus-scalar family contrasts match optimizer seeds exactly
(`101` with `101`, through `503` with `503`); Cartesian cross-seed comparisons
are prohibited, and any champion-to-champion result is separately labelled. The
frozen analysis uses:

```text
bootstrap seed       20260905
replicates           10,000
interval             two-sided 95% percentile
```

Report every seed value, missingness, mean, standard deviation, median, IQM,
lower-20% CVaR, extrema, sign counts, probability of improvement, bootstrap
intervals, and empirical performance profiles. Five policy seeds times 32
episodes must not be presented as 160 independent policies or episodes. This
follows the few-run uncertainty and robust-aggregate recommendations of
[Agarwal et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
and the reproducibility cautions of
[Henderson et al., 2018](https://doi.org/10.1609/aaai.v32i1.11694).

The hash-bound analysis implementation must be committed before the first
development reset. It must use `numpy.quantile(..., method="linear")` for the
2.5% and 97.5% percentile endpoints, define every 32-sample IQM by sorting and
averaging positions 8 through 23 inclusive even when bootstrap values repeat,
use two-sided 95% Wilson intervals for binomial proportions, and freeze the
performance-profile threshold grid in its source contract. No ineligible cell
is resampled. A policy missing any of its 32 required pairs has no paired value
aggregate; family missingness retains that policy in the denominator of five.

Mandatory scorecards include:

- training-run and checkpoint admission rate;
- per-resource per-cycle feasibility with binomial uncertainty;
- worst and distributional resource deficits;
- training cost exposure and multiplier trajectories/saturation;
- execution, hard safety, guard intervention, forced-stop, and invalid-command
  rates;
- delivery, unmet demand, inventory return, outage recovery, energy, battery
  cycling, start behavior, and wear proxies;
- net operating value and every additive component;
- paired economic differences and all frozen coefficient sensitivities; and
- controller, solver, dispatcher, and end-to-end p50/p95/p99 latency on registered
  hardware.

Natural outage/no-outage strata are reported at cycle level, with episode seed
retained as the bootstrap cluster; episode-level outage count and duration are
also reported. Sampled calm/volatile/scarcity regimes remain episode-level
post-opening strata. They are not selectable subsets, and no minimum count
authorizes seed substitution.

Latency is not a development-selection variable. Before development opens, the
analysis registration binds the CPU model, OS, dependency versions, Torch intra-
and inter-op thread counts of one, absence of concurrent benchmark jobs,
`time.perf_counter_ns` clock, 1,000 deterministic warm-up calls, and one sample
per subsequent policy inference. The held-out p99 gate uses the nearest-rank
99th percentile over all timed calls on that registered host.

## Frozen simulated-controller success rule

The primary deployment contrast is the selected PI-Lagrangian champion versus
E-beta. A simulated-controller win requires all of:

1. all 32 policy/E-beta pairs pass the complete pairwise gate: both arms complete
   and verify, remain hard-safe, have no forced stop or out-of-range decision,
   satisfy every resource return, and provide comparable service;
2. zero unexpected hard-safety violations, forced stops, or out-of-range
   decisions for the learned controller;
3. every learned-controller cycle return is feasible;
4. exact service comparability in all 32 primary pairs;
5. the lower endpoint of the registered 95% bootstrap interval for IQM paired
   net-operating-value delta is above zero;
6. at least 24 of 32 paired value deltas are positive;
7. lower-20% CVaR of paired value delta is non-negative;
8. deterministic policy inference p99 is at most 100 ms on registered hardware;
9. at least four of five independently trained PI policies each have all 32
   policy/E-beta pairs eligible and a positive held-out median delta; and
10. all scorecard and coefficient-sensitivity fields are published, including
    any materially worse tail.

Failure of any item means **no constrained-RL superiority demonstrated**. It does
not erase the descriptive result and does not authorize a different held-out
subset.

Even a complete pass permits only this claim:

> Under the frozen synthetic cyclic-market beta and declared economic
> assumptions, a preregistered PI-Lagrangian PPO controller improved verified
> risk-adjusted simulated operating value over the beta economic MPC while
> preserving matched service, hard safety, and every cyclic resource return.

It does not establish site savings, field safety, return on investment,
commission superiority, or commercial deployment readiness. Those claims still
require calibrated or uncertainty-bounded plant/economic parameters, hardware-
in-the-loop evidence, site-specific replay, and governed operational validation.

## Deferred layers

HEPA remains a predictive state/risk representation candidate. Natural language
remains a governed operator interface for configuration, diagnosis, explanation,
and approval. Neither enters this real-time controller protocol. After the core
controller comparison, each must be added through a separately frozen factorial
ablation so predictive or supervisory value cannot be misattributed to the RL
policy.
