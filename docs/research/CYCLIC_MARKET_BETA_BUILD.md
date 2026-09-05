# Cyclic stochastic-market beta build

Status: **implemented as a separately identified controller-neutral task; no
performance experiment or controller claim has been opened**

Date: **2026-09-05**

## Result

The repository now contains a runnable Environment V2 beta whose optimal operation
cannot be reduced to filling tanks against one perfectly repeating day. The new task
combines the existing delayed physical plant with a stochastic and revisable electricity
market, uninterrupted daily resource-return obligations, and private finite DAC outages.
It retains the same four absolute process targets and the same minute dispatcher, safety
guard, and simulator used by the controller-neutral alpha tasks.

The beta environment identifier is:

```text
plant-env-v2-cyclic-market-beta-v1
```

Its Gym interface is 135 observations and four actions:

```text
57 unchanged physical fields
+ 73 public market fields
+  5 cyclic-resource fields
= 135 observations

4 normalized absolute process targets
```

The production PPO configuration, environment constructor, checkpoint contract,
deterministic inference adapter, and paired heuristic/policy evaluator recognize this
task. A beta policy therefore has a real training and execution path through the common
physical authority; this is not a stand-alone simulator demonstration or a placeholder
policy API.

No research-scale PPO training, protected comparison, controller ranking, HEPA fit, or
language-layer experiment was run. In particular, the build does not report that RL
solves the beta or outperforms a baseline.

## Why this task exists

The economic-wear alpha established controller-neutral physical authority and a
reconcilable EUR ledger, but it retains three ceilings that make an RL/MPC comparison
less informative:

1. electricity value is a perfectly known 24-hour schedule that repeats forever;
2. a finite-horizon controller can consume the episode's starting inventories without
   demonstrating repeatable operation; and
3. process availability is deterministic after reset.

The beta adds a decision pressure for each weakness without changing the frozen alpha
contracts. A controller must now respond to revised price information, preserve four
distinct resource inventories for the next operating day, and absorb an unannounced
conversion outage. These are synthetic but causally meaningful pressures: they create
reasons to value storage flexibility, forecast revision, inventory form, and recovery
behavior rather than only total production.

## Architecture

```text
episode seed
  |-- children 0..2 --> unchanged alpha inventories, weather, contracts
  |-- child 3 -------> private market truth + public issue at each decision
  `-- child 4 -------> private per-cycle DAC-outage schedule

public controller boundary
  |-- 57-field physical observation
  |-- current price + 24 hourly mean/p10/p90 bins
  `-- cycle progress + battery/H2/CO2/CH4 return margins
               |
               v
      135-field checkpoint-bound vector
               |
               v
      four normalized absolute targets
               |
               v
 common command -> minute dispatcher -> independent guard -> simulator
               |                                      |
               |                                      `-> public plant effects
               `-> realized EUR ledger

authoritative simulator state at each 24 h boundary
               `-> separate feasibility and constraint-cost record
```

The implementation is deliberately layered rather than embedded into the plant model:

- `market.py` owns stochastic price truth, causal forecast issues, replay identities,
  and the typed projection of public hourly means;
- `beta_disturbances.py` owns the private seeded outage schedule and its replay
  evidence;
- `beta_config.py` owns the immutable 24-hour return contract, opening reference, and
  settlement schema;
- `beta_control.py` owns the typed controller-visible market/cycle observation;
- `beta_spaces.py` owns the exact Gym field order and interface digest;
- `beta_env.py` composes those contracts over `PlantEconomicSetpointEnvV2` and routes
  both normalized Gym actions and typed controller commands through the existing common
  setpoint path; and
- the existing training, policy-controller, and policy-evaluation modules now admit the
  beta only when its full task identity matches the checkpoint.

Two protected extension seams were added to inherited code. `PlantEnvV2._sample_episode`
delegates exactly to the old alpha sampler by default; beta overrides it to attach market
and fault schedules before simulator load. The economic environment similarly exposes
protected opportunity-price and public-price-context methods whose default remains the
old deterministic tariff. Beta overrides them with the current market issue. The alpha
field order, actions, stream allocation, and default pricing semantics are therefore not
relabelled as beta behavior.

## Stochastic market

### Realized truth

The alpha day-ahead tariff remains the deterministic civil-time baseline. At controller
cadence, a bounded AR(1) residual adds serial correlation and rare signed arrivals add
decaying excursions. The versioned profiles are `calm`, `volatile`, and `scarcity`;
`mixed` privately selects one of them. The default configuration uses `mixed`, 64
forecast ensemble members, and a forecast innovation scale of 1.0.

Truth and forecasts are bounded to the declared `[-300, 300] EUR/MWh` domain. The
realized price at decision `k` applies to the physical interval `[k, k+1)` and enters the
existing additive economic ledger. Consistent with the frozen economic objective, a
negative realized price remains observable but accepted renewable opportunity cost is
floored at zero; the build does not add an export-payment model.

The generator also accepts an explicit realized-price tape as a strict test/replay seam.
That changes truth while retaining the seeded regime and issue streams; supplied values
must already be complete, finite, correctly sized, and in range.

### Public information and causality

Every controller boundary receives one `PublicPriceForecastIssueV1` containing:

- the currently realized price;
- the issue decision and episode-relative time;
- 24 consecutive one-hour forecast bins; and
- mean, p10, and p90 for every bin.

Each issue starts from the price observed at that boundary and uses its own deterministic
forecast stream. It never reads a later element of the realized truth tape. Consequently,
replacing prices strictly after an issue leaves that issue unchanged, while the next issue
may revise the forecast after observing the new current price. The issue can project its
24 hourly means piecewise-constantly onto any controller grid that is a whole subdivision
of an hour, including the 96-step, 15-minute MPC horizon.

The public issue contains no selected regime, seed, random-stream state, future truth, or
future issue. Those fields exist only in the private audit manifest. The public `market_id`
identifies the generator configuration, baseline, cadence, and start hour without becoming
a label leak for the sampled episode.

## Continuous cyclic operation

At reset, the environment records authoritative simulator values for battery energy and
H2, CO2, and saleable CH4 inventory. Every exact 24-hour boundary is checked against
configured fractions of that same opening reference. The defaults require a full return
for all four resources, with separate absolute tolerances.

The reference does not ratchet down after a deficit. The plant is not reset at a cycle
boundary: equipment state, weather, delivery accounting, price process, inventories, and
fault lifecycle continue. H2, CO2, and CH4 remain separate because identical methane-
equivalent mass is not identical operating flexibility when conversion takes time or the
DAC is unavailable.

The five online cyclic fields are cycle progress followed by signed battery, H2, CO2, and
CH4 margins to their respective targets, normalized by installed capacity and clipped to
the declared range. At a boundary, `CycleSettlementV1` retains target, actual, deficit,
tolerance, feasibility, and a dimensionless aggregate violation cost. That cost is a
separate constraint output; it is not silently converted into EUR or added to the scalar
reward.

The current generic policy evaluator retains every settlement, aggregate feasibility, and
cumulative constraint cost. A normally terminated trajectory that violates any cycle return
is recorded with failed task status and a typed `CycleConstraintViolation`; normal physical
termination, completed decision count, and the constraint failure remain separate evidence.
Future scientific aggregation must still apply its preregistered eligibility rule rather
than silently dropping such runs.

## Private finite disturbances

The version-one disturbance generator supports one mechanism: a complete DAC outage.
When enabled, each 24-hour cycle has an independent draw and may contain at most one
outage. The defaults use probability 0.5 and a uniformly sampled duration from one to four
hours. Start and duration are aligned to the plant's physics lattice, the fault is finite,
and the final physical dispatch in a cycle is always healthy so boundary telemetry cannot
carry stale offline state into the following cycle.

The sampled schedule is installed through the simulator's existing `FaultSpec` and fault
registry. A controller sees only effects that become visible through ordinary plant
telemetry and health state. It receives no advance warning, schedule, onset countdown, or
outage random stream. Disabling disturbances produces an exactly empty schedule while
preserving the task's explicit disturbance configuration and identity.

## Randomness, replay, and identity

Beta sampling uses five deterministic children from one root `SeedSequence`:

| child | owner | visibility |
|---:|---|---|
| 0 | starting inventories | existing alpha contract |
| 1 | weather truth and forecast | existing alpha contract |
| 2 | delivery commitments | existing alpha contract |
| 3 | market regime, truth, and issue streams | public issues; private replay detail |
| 4 | outage generator and per-cycle streams | observable effects; private schedule |

The original three alpha children are neither renumbered nor regenerated by beta code.
Market and disturbance inputs are cloned before spawning so a caller-supplied
`SeedSequence` is not consumed as an incidental side effect.

Identity is intentionally split by purpose:

- the interface digest covers the 135/4 field order and online semantics;
- the checkpoint contract additionally covers plant, physical episode, safety,
  diagnostic reward, economic assumptions, market generator and profiles, cyclic-return
  settings, disturbance generator, random-stream allocation, and execution doors;
- the retained alpha physical exogenous digest identifies the unchanged inventory,
  weather, and commitment draw;
- the beta exogenous digest combines that alpha identity with market episode,
  disturbance episode, and cycle-task identities; and
- the private audit manifest binds selected regime parameters, initial RNG states,
  truth/forecast/schedule digests, opening cycle reference, and evolving delivery,
  economic, and cycle accounting.

Source revision and dependency-lock identity remain the responsibility of the experiment
runner and are named as provenance gaps in the environment manifest. An alpha checkpoint
cannot be admitted to beta, and a beta checkpoint cannot be admitted to an alpha task,
because environment ID, online field order, interface digest, and full checkpoint digest
must all agree before policy execution.

## Controller-neutral RL path

`ControllerNeutralTrainingConfigV1` now accepts the beta environment only when a validated
`CyclicMarketTaskConfigV1` is bound. The strict JSON codec rejects unknown market, cycle,
and disturbance fields and preserves the requested market regime enum, ensemble settings,
return fractions/tolerances, outage probability, and duration range. Selecting beta at
the training CLI binds the version-one default task when no beta settings were loaded;
selecting an alpha task removes an otherwise stale beta extension. The normal
Stable-Baselines3 construction path creates independent beta environments and retains the
same checkpoint/resume machinery used by the 57/4 and 62/4 tasks.

That path is standard scalar-reward PPO. The cycle signal is observable and retained in
`info`, but the learner does not yet enforce it as a constraint or optimize a separate
cost budget. The build is therefore ready for beta policy training at the interface level,
not yet a constrained-RL implementation.

At inference, `NormalizedSetpointPolicyControllerV1` requires the beta checkpoint
contract, economic-assumption digest, and beta-task digest. It concatenates the typed
57-field physical observation, exact 73-field public issue, and five cycle fields, checks
the result against the checkpoint field count, performs deterministic inference, decodes
four normalized values into physical targets, and issues `ProcessSetpointCommandV1` with
checkpoint- and observation-bound evidence. The command then crosses the common minute
dispatcher, independent guard, and plant.

The paired heuristic/policy evaluator can instantiate identity-matched beta copies. The
heuristic is intentionally the existing physical A0-style baseline and therefore does not
use the additional market/cycle fields; PPO receives the declared full beta vector. The
evaluator retains rewards, economic accounting, cycle constraints, typed decisions,
minute executions, failures, and timing without installing a fallback policy.

## Verification boundary

The implementation includes outcome-free contract tests for:

- deterministic market and disturbance replay, immutable serialization, and strict
  canonical hashes;
- all three explicit market regimes and private mixed-regime selection;
- finite bounded truth, ordered forecast quantiles, fixed 73-field issue shape, and exact
  hourly-mean projection;
- causal invariance of an issued forecast to changed strictly future truth, followed by
  later-vintage revision;
- seed/configuration/baseline/truth identity sensitivity and private/public field
  separation;
- exact empty disturbance ablation, horizon-prefix invariance, at-most-one aligned finite
  outage per cycle, and simulator fault-lifecycle semantics;
- fail-closed configuration, seed, cadence, duration, geometry, and identity validation;
- exact 135/4 interface shape and prefix equality with the alpha physical draw;
- realized-price reward attribution to the opening issue rather than a baseline or later
  vintage, with public/private manifest separation;
- one settlement per exact 24-hour boundary across a continuous two-cycle plant run,
  unchanged opening targets, and no scalar constraint penalty;
- parity between a normalized Gym action and the same fixed checkpoint action through the
  typed common command door; and
- bidirectional alpha/beta checkpoint rejection plus beta training-config round-trip and
  exact environment construction;
- direct CLI beta selection with a complete default task and pre-reset rejection of invalid
  disturbance geometry; and
- failed task status for a normally terminated policy trajectory that violates a 24-hour
  return constraint.

The focused post-composition command was:

```text
.venv/bin/pytest -q \
  backend/tests/test_rl_v2_beta_env.py \
  backend/tests/test_rl_v2_controller_neutral_matrix.py \
  backend/tests/test_rl_v2_policy_integration.py \
  backend/tests/test_rl_v2_training_config_strict.py
```

It completed with **105 passed and 2 skipped**. Subsequent hardening added CLI-default,
pre-reset geometry, and hard cycle-failure checks before the release gate below. This is
implementation evidence, not a policy rollout or performance result.

These are software and causal-contract checks. They deliberately do not train a useful
policy or inspect a protected controller outcome. The final pre-commit release gate passed:

- complete Python repository: **1,461 passed, 4 skipped**, with the existing
  Starlette/httpx deprecation warning and trailing HiGHS diagnostics;
- strict typing of the 11 touched/new production modules: passed after excluding only
  third-party missing-stub/import diagnostics and inherited unused override-ignore markers;
- Black on all 17 touched/new Python files, `compileall`, and Git whitespace validation:
  passed;
- frontend production TypeScript/Vite build: passed, retaining the existing large-chunk
  advisory; and
- frontend `oxlint`: passed.

## What is complete, and what is not

Complete in this increment:

- a separately versioned, runnable 135/4 cyclic stochastic-market environment;
- realized stochastic prices in the authoritative economic reward path;
- causal revisable price forecasts in one typed public representation;
- non-ratcheting four-resource cycle accounting without plant resets;
- private replayable finite DAC outages through existing simulator physics;
- full beta task, episode, observation, manifest, and checkpoint identities;
- PPO training/checkpoint/inference support through the common setpoint authority; and
- heuristic-versus-policy execution and evidence capture on beta tasks.

Not complete and not implied:

- a trained beta PPO policy or evidence that any controller masters the task;
- a constrained-RL learner that treats cycle feasibility as more than episode metadata;
- a beta-native market-aware economic MPC or stochastic MPC;
- extension of the existing H/M24/E/PPO four-arm benchmark to this 135-field task;
- a frozen outcome-blind train/development/held-out beta split;
- demand or product-price uncertainty, grid import/export, logistics, or multiple units;
- sensor delay, missingness, drift, remaining-useful-life state, or degradation physics;
- historical price calibration, site calibration, equipment-lifetime calibration, or
  validated economic coefficients;
- HEPA training or an action-conditioned risk forecast; or
- language-supervisor integration, operator study, field deployment, savings, ROI, or a
  commission-readiness claim.

The alpha tariff-aware economic MPC is deliberately not passed the beta environment under
an old type. Generic economic access returns the richer beta observation, causing an
alpha-only MPC input check to fail instead of silently optimizing the deterministic tariff
and receiving privileged or incorrect information.

## Next direction

The next research increment should finish comparator parity before opening an RL outcome:

1. implement a separately identified beta economic MPC that consumes only the current
   `PublicPriceForecastIssueV1`, uses the same projected hourly means available to PPO,
   represents the four cycle-return constraints explicitly, and reports uncertainty and
   model mismatch;
2. implement and freeze a constrained-RL training contract that consumes the separate
   cycle-cost signal, with a declared feasibility budget and checkpoint-selection rule,
   while retaining ordinary PPO as an explicit ablation;
3. extend the four-arm runner and detached verifier to H, rolling M24 or its beta
   successor, beta economic MPC, and checkpoint-bound PPO, retaining cycle-infeasible and
   failed arms without fallback;
4. calibrate or uncertainty-bound economic coefficients and freeze an outcome-blind beta
   regime/seed split plus a finite policy-training-seed and checkpoint-selection rule; and
5. only then train and compare policies, using episode as the experimental unit and
   requiring comparable delivery, zero hard safety/cycle violations, and complete pairs
   before value or tail-risk comparisons.

Further realism should be introduced as separately switchable contracts after this first
comparison identifies the next limiting decision pressure. High-value candidates are
demand/logistics uncertainty, constrained grid energy, sensor-quality state, gradual
degradation and maintenance, and calibrated multi-unit bottlenecks. HEPA should then be
tested as a predictive warning or action-conditioned risk layer across diverse controller
trajectories. Natural language should remain an auditable operator-governance layer that
can explain, propose, and request approval, not an unverified source of minute physical
actions.

## Claim boundary

This build makes the repository capable of asking a harder and commercially relevant
control question: how should a renewable-powered, storage-coupled process schedule
conversion under revised prices, delayed physical dynamics, recurring service/resource
obligations, and unannounced loss of a unit? It does not answer that question yet. The
commercial proposition must ultimately be earned through calibrated operational metrics
such as reliable product delivery, avoided expensive energy use, preserved flexibility,
recovery from outages, equipment-life reserve, and bounded fallback behavior against
credible model-based and operational baselines.
