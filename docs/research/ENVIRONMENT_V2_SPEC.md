# Environment V2 research specification

Status: **accepted direction; implementation parameters remain provisional until the
corresponding experiment is committed.**

Implementation status: the weather alpha now provides recurring contracts, a 15-minute
decision cadence, service reward and metrics, explicit boundary/preceding-interval
`ControlState` semantics, seeded solar-resource truth, four causal forecast bands, and a
private replay manifest under an explicit V2 namespace. It still uses a synthetic,
uncalibrated weather fixture and memoryless V1 process equipment. Commitment dynamics,
empirical profiles, model-based comparators, and controller training/evaluation remain
open increments; this is not the completed V2 benchmark described below.

## 1. Benchmark question

Environment V2 is designed to answer one operationally meaningful question:

> Can a bounded supervisory controller meet recurring methane delivery commitments
> under imperfect renewable forecasts at lower risk-adjusted operating and equipment
> cost than credible heuristic and model-predictive baselines?

This replaces Environment V1's implicit task—produce methane until a large tank is
full—with a continuing service obligation. It creates value for forecasting,
inventory planning, and equipment commitment without pretending that the current
simulator is already a calibrated plant twin.

The objective is not to make the simulator “as complex as possible.” Complexity is
accepted only when it introduces a real decision pressure, has traceable parameters,
can be tested, and can be removed in an ablation.

## 2. Place in the full architecture

```text
scenario/data plane
    truth profiles + issued forecasts + delivery contract + plant parameters
                              |
                              v
routine controller ---- proposed action / schedule
    heuristic | MPC | RL | hybrid
                              |
                              v
independent guard ------ applied action + intervention record
                              |
                              v
control-oriented simulator ---- truth state ---- accounting / benchmark score
             |
             +---- instrument model ---- telemetry ---- controller
             |
             +---- trajectory ---- HEPA risk estimate ---- supervisor
                                                        |
                                                        +-- bounded changes to
                                                            constraints, mode,
                                                            fallback, or controller

natural-language layer: scenario authoring, operator intent, explanation, and reports
                        (never direct unvalidated actuator authority)
```

The four intelligence-related parts have deliberately different scopes:

| Part | Scope in V2 | Objective | Explicit non-objective |
| --- | --- | --- | --- |
| Controller, including RL | Frequent dispatch and equipment-commitment decisions within a declared envelope | Minimise delivery/cost/wear objective using available telemetry and forecasts | Diagnose root cause, rewrite safety limits, or converse with an operator |
| Safety guard | Deterministically project or reject unsafe commands and invoke fallback | Preserve hard invariants independently of controller quality | Optimise long-horizon economics |
| HEPA predictor | Estimate calibrated future event risk and warning lead time from recent trajectory | Warn before delivery failure, reserve loss, thermal excursion, or forced shutdown | Apply actions directly or compensate for a weak benchmark |
| Supervisor | Use risk, health, and operator policy to select bounded strategy changes | Coordinate recovery, derating, fallback, and controller switching | Replace the fast control loop or bypass the guard |
| Natural-language layer | Translate operator intent into schema-validated scenarios/policies and explain evidence | Make commissioning and investigation accessible and auditable | Supply hidden future information or directly actuate equipment |

The first V2 benchmark freezes HEPA, the supervisor, and natural-language behavior.
It compares routine controllers through the same safety path. Those layers are added
only after the base environment produces sufficient event prevalence and controller
separation to measure their incremental value.

## 3. Compatibility and versioning

Environment V1 remains reproducible. Default `PlantConfig`, existing scenarios,
`PlantEnv`, observation contract version 1, reward, and saved checkpoint behavior must
not silently acquire V2 semantics.

The backend therefore introduces an explicit `plant_backend.rl.v2` namespace. V1's
global `spaces`, `observation`, `actions`, `reward`, `metrics`, and `episodes` modules
remain the legacy contract because checkpoints, the live runtime, and HEPA currently
bind to them. V2 checkpoint loading requires an environment identifier and contract
digest; an artifact without those fields is V1-only. Cross-version loading fails
loudly—vectors are never padded and weights are never silently reused.

V2 has three deliberately different identities. The static descriptor digest covers
only observation/action layout and normalization. `PlantEnvV2.checkpoint_contract` is
the policy-compatibility identity: it covers that interface plus timing and guard
semantics, dynamics implementation versions, episode distribution, complete plant and
constraint parameters, objective version, and reward weights. The per-reset private
audit manifest adds the sampled episode seed hierarchy, scenario, commitments, weather
stream states and content hashes, while the experiment runner adds source revision,
dependency-lock digest, runtime, and artifact/training provenance. Keeping these scopes
separate avoids pretending a checkpoint digest identifies one sampled episode or a Git
tree. Future V2 checkpoint writers/loaders must store and require the compatibility
digest; no V2 loader exists in the weather alpha, so V1's loader must not be used for it.

Across those three records, V2 evidence contains at least:

- environment contract identifier and semantic version;
- simulator implementation/package versions plus experiment source revision;
- complete plant-parameter serialization and digest;
- observation/action field names, order, units, and version;
- objective/reward configuration and version;
- scenario/profile generator version, episode content hashes, and seed streams;
- physics timestep and policy decision interval;
- forecast model and issue-time convention;
- controller/checkpoint and experiment-run provenance.

Any changed meaning, order, normalization, timing, or control authority invalidates a
checkpoint. V1 policies may be evaluated only in the frozen V1 environment unless an
explicit adapter is scientifically justified and labelled.

## 4. Time convention

All V2 flows use half-open physical intervals `[t, t + dt)`. The controller observes
state and forecasts issued at `t`, selects an action, and that action is held during
the interval. Production and consumption are integrated over the interval; deliveries
whose deadline is `t + dt` are settled after interval production. The resulting state
is stamped `t + dt` and reports end-of-interval levels plus flows from the interval
that just ended.

This convention must be tested at reset, profile boundaries, forecast issue times,
delivery deadlines, and episode termination. It resolves the current ambiguity in which
V1 flows are evaluated at the old timestamp and exposed on a state stamped with the
new timestamp.

The initial target is:

- physics step: 1 minute, preserving current integration resolution;
- controller decision interval: 15 minutes, with one action held for 15 physics steps;
- nominal episode: 7 days, long enough to contain repeated commitments and weather
  regimes without making early experiments prohibitively expensive.

The discount factor must be derived from a declared economic time constant and policy
interval, then sensitivity-tested. It must not be inherited from V1's per-minute
`gamma = 0.995`, whose effective horizon is only a few hours.

The configured V2 horizon is an intrinsic finite-horizon MDP boundary, not an external
wall-clock time limit. The final transition therefore returns `terminated=True` and
`truncated=False`. It settles the last delivery and pays the bounded terminal-inventory
value once. PPO and other value-based learners must not bootstrap through that terminal
observation; doing so would add a fictitious continuation value to a state with no next
contract or newly issued forecast. This termination convention is part of the full
checkpoint identity.

V2 constructs an explicit `ControlStateV2` for policy observation at each 15-minute
decision. Its timestamp and inventories are current at the boundary. Solar,
health, and online fields are the final one-minute dispatch sample in the completed
interval, while power/load/production fields are time-weighted means over that complete
interval. Reset uses the initial boundary sample because no preceding interval exists.
Before every V2 physics step, the guard combines causal telemetry with a privileged,
read-only preview of the exact solar bus that step will dispatch. That preview never
enters the controller observation; it is the fast interlock's actuation-time input for
enforcing absolute kW caps. Reward and physical accounting integrate the complete
interval. Existing `PlantState` fields are not silently reinterpreted as 15-minute
aggregates.

Environment V1 deliberately retains its established telemetry-based guard path and
transition identity for replay compatibility. Moving the live operational runtime to
the exact-bus interlock requires an explicitly versioned adoption of the V2 semantics;
the weather-alpha implementation does not silently rewrite V1 trajectories.

## 5. Minimal discriminative feature bundle

Every feature has an on/off ablation and deterministic replay test.

### 5.1 Recurring methane delivery commitments

The physical simulator gains scheduled product offtake. A commitment declares a
deadline and requested kilograms. At settlement, methane is withdrawn from storage;
the simulator records requested, delivered, and unmet kilograms for the step and
cumulatively. Mass balance becomes:

```text
ending CH4 = starting CH4 + produced CH4 - delivered CH4
```

The controller can observe the amount and time of announced commitments, but never
the benchmark's private accounting state through a normalized proxy that leaks future
random draws. Phase-one commitments are known contractual data, not stochastic demand
forecasts.

Provisional challenge design:

- one batch delivery every 6 hours;
- contractual average below sustainable nominal production, leaving scheduling
  flexibility but insufficient margin for careless cycling;
- initial methane inventory expressed as hours of contracted demand;
- terminal inventory valued at the same marginal product value and capped, preventing
  both end-of-episode dumping and hoarding;
- deadline alignment with the physics grid validated at configuration time.

The exact rate and buffer size are selected by a calibration sweep. The acceptance
criterion is a nontrivial feasible region: naive policies miss deliveries, a credible
controller can meet most of them, and perfect foresight is measurably better without
being unattainable.

### 5.2 Stochastic truth and controller-visible forecasts

The weather alpha replaces repeated clear-sky output with a seeded, autocorrelated cloud process
multiplied by the deterministic solar envelope. Separate random streams generate:

1. physical weather innovations;
2. forecast/measurement uncertainty;
3. initial inventories and commitments.

The stream identities are recorded so modifying one subsystem does not silently
change all later random draws.

At issue time `t`, the forecast is generated from information available at or before
`t` and independent future forecast innovations. It may expose mean and uncertainty
at selected horizons or a fixed scenario ensemble. It must not read realized future
weather. A required leakage test changes physical innovations after the issue time and
proves that the already-issued forecast is unchanged.

The implemented synthetic fixture is a stationary Gaussian AR(1)/discrete-OU latent
transformed through a logistic function. Its provisional median resource factor is 0.70,
stationary latent standard deviation is 0.75, and autocorrelation half-life is one hour.
These are transparent challenge parameters, not European climatology. At every controller
boundary, 32 independent conditional trajectories are generated from current truth and an
issue-specific forecast stream. They never read future truth. Each member is multiplied by
known future clear-sky/nameplate power and aggregated before mean and p10 are taken over
the half-open `0–1`, `1–6`, `6–12`, and `12–24` hour bands.

Forecast uncertainty must be evaluated for calibration as well as point error. Truth,
measured current output, the one current immutable issuance, all future issuances, and
perfect-future oracle data remain distinct interfaces. The environment keeps the full
weather episode private and exposes only `current_forecast`; Gym observations and public
info contain no tape, seed, future issue, stream identity, or private hash.

Generating all 672 forecast vintages for a seven-day default episode currently costs
roughly half a second on the development machine and draws about 31 million ensemble
innovations. This is acceptable for alpha evaluation but material for large RL sweeps.
Controller-resolution scenarios, lazy deterministic issuance, or conditional moments
must be benchmarked before expensive training; optimization may not change forecast
semantics without a generator-version and golden-digest change.

Agent-facing reset and step information also omits episode seeds, future fault
schedules, private truth tapes, and split labels. Reproducibility metadata is written
to a separate audit manifest, and only an explicitly typed offline oracle interface
may access future truth.

### 5.3 Electrolyser commitment dynamics

The hydrogen bottleneck gains a technology-labelled operating state machine:

```text
off -> starting -> running -> standby/cooldown -> off
```

The initial model includes startup time and energy, minimum stable load, ramp rate,
minimum up/down commitment, part-load efficiency, and cumulative start/low-load
exposure. Mode, timers, prior load, and admissible ramp headroom are observable so the
controller-facing state remains Markov.

Parameters begin as documented archetype priors and carry a calibration status; they
are not presented as vendor-specific truth. Literature values cannot be mixed across
PEM and alkaline systems without an explicit equipment profile. Thermal and purity
states are deferred until the commitment experiment shows a need and suitable
calibration evidence exists.

## 6. Observation and action contracts

V2 uses a separate observation specification rather than appending fields to the
global V1 vector. The target mature contract includes:

- cyclic time and episode/contract phase;
- latest completed one-minute renewable dispatch sample and recent trend;
- forecast mean/scenarios and uncertainty at declared horizons;
- battery, H2, CO2, and CH4 inventory with reserve/buffer margins;
- next commitment amount, time-to-deadline, and announced horizon aggregate;
- equipment mode, load, ramp headroom, commitment timers, and health;
- active supervisor constraints and previous guard intervention magnitude.

The implemented weather-alpha contract has 42 fields. It uses V1's named fields only
where their meaning is explicitly redeclared, replaces process rates/loads with preceding-
interval means, adds normalized next-delivery time/quantity/inventory coverage and
cumulative fill rate, and appends forecast mean and p10 for `0–1`, `1–6`, `6–12`, and
`12–24` hour windows. It exposes active power/reserve constraints but does **not** yet
contain equipment commitment mode, ramp headroom, startup/cooldown timers, forecast
scenarios, or previous guard intervention magnitude. Those arrive only with the
corresponding versioned increments. The exact field table committed with the encoder is
authoritative; this prose does not substitute for the machine-readable contract.

The initial action keeps the current five control dimensions to bound the experiment,
but the benchmark records requested physical kW, applied physical kW, and routing or
guard intervention magnitude. If materially different policy outputs still collapse
onto the same dispatch, V2 will replace solar shares with absolute equipment setpoints
and explicit common-bus battery charge/discharge authority.

The controller must never receive:

- future realized solar or unissued future forecasts;
- truth hidden by an instrument fault;
- private reward components unavailable to a real operator;
- an episode seed or scenario label that encodes the future regime.

## 7. Objective and accounting

Raw methane production is not positively rewarded in V2. Product has value when it is
delivered on specification or retained as explicitly valued terminal inventory.

The initial scalar training objective combines:

- delivered-product value;
- unmet-delivery penalty, including a tail-risk-sensitive term;
- terminal inventory value within a target band;
- renewable curtailment or purchased-energy cost;
- battery throughput cost;
- electrolyser startup, cycling, low-load, and degradation-proxy cost;
- guard-intervention magnitude and fallback cost.

Hard physical limits, safety reserve, forbidden modes, and thermal/purity constraints
are not exchangeable reward terms. They are enforced and reported separately. Reward
components remain visible for diagnosis, but final comparisons use the primary
scorecard in `STATE_OF_THE_ART.md`.

## 8. Scenario and evaluation splits

Each episode manifest records immutable profile identity and belongs to exactly one
split:

- training: ordinary regimes and parameter variation;
- validation: disjoint profile blocks used for model/hyperparameter selection;
- test: held-out profile blocks and seeds opened only after selection;
- stress/OOD: extreme cloud duration, forecast bias, delivery surge, capacity loss,
  and parameter shift outside the training envelope.

Random adjacent timesteps from one profile may not be split across training and test.
All controllers are compared on paired manifests. RL uses multiple training seeds and
the selected checkpoint rule is fixed before test evaluation.

Required summaries include median, interquartile mean, dispersion, bootstrap
confidence interval, probability of improvement, worst case, CVaR, and paired delta
against every principal baseline. Training compute and p95/p99 decision latency are
reported alongside plant outcomes.

## 9. Baselines and oracles

The first V2 release is not complete without:

- a simple delivery-aware rule controller;
- a tuned forecast-aware heuristic;
- deterministic economic MPC over the same forecast;
- a perfect-foresight offline optimiser receiving future truth and clearly labelled
  as an unattainable bound;
- simple constant/greedy ablations used to diagnose benchmark saturation;
- retrained PPO with multiple training seeds.

Stochastic MPC and hybrid RL-MPC follow once the deterministic formulation and
forecast uncertainty are stable. Controller tuning budgets and information access are
declared to avoid giving the learned policy a hidden advantage.

## 10. Layered-autonomy experiment

Only after the routine-controller benchmark passes its separation gate:

1. collect trajectories from all controller families and stress regimes;
2. define operational events from V2 accounting and physical state, then version the
   HEPA event schema;
3. train HEPA on behaviorally diverse data and report calibration, discrimination,
   false-alarm burden, and useful lead-time distribution;
4. compare controller only, controller + deterministic supervisor, controller +
   supervisor + HEPA, and controller + HEPA without supervisor;
5. add LLM reasoning only as a separately reproducible supervisor arm, never as the
   source of hard safety.

This sequence distinguishes predictive value from supervisory value. It prevents an
LLM or predictor from being credited for improvements actually caused by the guard or
a different controller.

## 11. Implementation increments and acceptance gates

### Increment 0 — correctness and contracts

- correct the lossless supervisory battery-reserve calculation;
- define and test `[t, t + dt)` semantics;
- align simulator version metadata and strengthen checkpoint provenance;
- freeze a V1 manifest and diagnostic output.

### Increment 1 — delivery kernel

- schedule, settle, and account for methane commitments in the simulator;
- use integer tick indices for boundaries; do not rely on floating-point timestamp
  equality;
- add mass-balance, deadline-boundary, reset, serialization, and no-delivery V1 tests;
- expose V2 delivery observations, reward, and metrics behind a separate contract;
- sweep contract rate and storage buffer to establish feasibility and separation.

### Increment 2 — weather and forecasts

- add reproducible autocorrelated truth profiles and issued forecasts;
- add leakage, calibration, split-integrity, replay, and time-alignment tests;
- implement forecast-blind and forecast-aware heuristic ablations.

### Increment 3 — electrolyser commitment

- add the parameterized operating state machine and wear exposures;
- add unit/energy/ramp/timer invariants and compatibility-off tests;
- measure whether commitment dynamics widen the oracle-to-naive gap.

### Increment 4 — controller tournament

- implement deterministic MPC and perfect-foresight oracle;
- train PPO under immutable manifests and multiple seeds;
- publish paired distributions, tail risk, ablations, and compute/latency.

### Increment 5 — prediction and supervision

- version V2 HEPA events and collect balanced trajectories;
- evaluate HEPA, supervisor, and natural-language assistance by factorial ablation;
- advance to PLC/protocol HIL only if the software benchmark passes.

Gate B passes when the benchmark is feasible, reproducible, and discriminative: a
naive policy is materially worse than the tuned baseline, the perfect-information
bound is materially better, and outcomes are not dominated by randomized initial tank
headroom. No percentage is fixed until the initial sweep makes the achievable scale
visible; thresholds are then predeclared before training RL.

## 12. Explicit non-goals for the first release

- claiming a validated physical digital twin without calibration data;
- direct low-level actuator control by RL or an LLM;
- replacing deterministic safety with reward shaping;
- adding every chemical, thermal, market, and regulatory detail simultaneously;
- optimizing PPO before demonstrating a benchmark gap;
- presenting one seed, best checkpoint, or scalar reward as product evidence;
- presenting technical evidence as proof of customer willingness to pay.
