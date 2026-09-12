# Research journal

## 2026-09-02 — Programme initiation

### Mandate

The user granted broad research autonomy to evolve the repository toward a product
that is technically competitive, commercially relevant, and demonstrably addresses a
real industrial problem. Work is isolated on `codex/environment-v2-research` and will
be organized as small, auditable commits.

### Starting evidence

A colleague has trained an RL policy against Environment V1 and reports that it
performs similarly to the shipped heuristic. The working hypothesis is that this is
environment saturation rather than evidence that the two policies are behaviorally
equivalent: the deterministic, mostly linear plant admits a broad plateau of adequate
allocations, and the heuristic already encodes the dominant process priorities.

This report has not yet been independently reproduced. The first research phase must
locate or request the checkpoint, its training configuration, learning curves, seeds,
and evaluation output before treating parity as a quantified result.

### Initial audit observations

- Environment V1 uses a deterministic clear-sky curve plus scheduled `solar_drop`
  disturbances. Once an episode is sampled, the trajectory is deterministic for a
  given action sequence.
- Default RL episodes last 48 simulated hours and draw from `clear_day` and
  `cloud_cover`, randomized starting inventories, and optionally one additional solar
  disturbance.
- Conversion is predominantly linear. The electrolyser has a minimum load, but there
  are no startup delays, ramp limits, minimum up/down times, delivery commitments,
  prices, or physical wear state.
- The observation has 30 values and includes the current solar value, clear-sky
  potential, one-step trend, storage, unit condition, controller constraints, and
  episode progress. It contains no exogenous forecast or demand commitment.
- The five-dimensional action allocates contemporaneous solar shares and battery
  support. It expresses dispatch, not equipment commitment or scheduling.
- The reward is methane production minus curtailment, battery drain, reserve breach,
  tank saturation, shutdown, safety correction, and cycling penalties. Its weights
  were scale-selected before any documented training evidence existed.
- The existing evaluation compares controllers on identical episode seeds, but it
  does not estimate an optimality gap, separate in-distribution validation from
  out-of-distribution stress, or report confidence intervals and tail risk.

### Main falsifiable hypothesis

`H1`: On Environment V1, the heuristic and trained RL policy are both close to a
perfect-information upper bound, and many materially different actions lie on a broad
performance plateau.

Evidence against `H1` would include a large gap between both controllers and an oracle,
or a large train/evaluation discrepancy for RL. In that case, reward design, training,
or observability should be investigated before increasing simulator complexity.

### Immediate research actions

1. Audit the current environment and training/evaluation contracts.
2. Locate the colleague's training artifacts and establish exact provenance.
3. Research current control, safe-RL, and renewable hydrogen/e-fuel benchmarks.
4. Define an Environment V1 ceiling study with heuristic, RL, simple-policy,
   model-based, and perfect-information comparisons.
5. Specify the smallest Environment V2 slice that adds genuine temporal commitment
   and uncertainty without prematurely building a high-fidelity chemical simulator.

### Tooling state

At the start of the programme, the repository had no local `.venv` or frontend
`node_modules`. Python source parsed successfully in the earlier audit, but the full
test and build suites could not yet be executed without installing locked dependencies.

### Reproducibility and dependency setup

The locked Python dependencies, including the optional `train` group, were installed
with:

```text
UV_CACHE_DIR=/tmp/plant-an-agent-uv-cache uv sync --group train --frozen
```

The first sandboxed attempt could not resolve `files.pythonhosted.org`; the identical
locked command was then authorized with network access and completed successfully. It
installed Stable-Baselines3 2.9.0, Torch 2.13.0, Gymnasium 1.3.0, and the workspace
packages into `.venv`. No dependency specification or lockfile changed.

Baseline verification:

```text
UV_CACHE_DIR=/tmp/plant-an-agent-uv-cache uv run pytest -q
321 passed, 1 warning in 16.03 s
```

The warning is an upstream Starlette/FastAPI TestClient deprecation concerning `httpx`;
it is not a failing behavior test. The branch therefore begins from a green Python
baseline.

### Artifact audit

Neither this repository nor the adjacent `latent-world-models` checkout contains the
reported trained PlantAnAgent policy. There is no PlantAnAgent `policy.zip`, checkpoint
metadata, training configuration, learning curve, or evaluation JSON available locally,
and no Git ref contains an otherwise-unmerged training result. The adjacent repository's
artifacts concern Atari and Doom experiments and contain no `PlantEnv` or plant-simulator
integration.

The minimum provenance required from the colleague is:

- the complete checkpoint directory (`policy.zip` and metadata);
- the exact source commit and dependency-lock digest;
- full training configuration and policy seeds;
- learning curves and training duration/hardware;
- per-seed paired heuristic/RL evaluation output;
- any uncommitted reward or environment modifications.

Until those artifacts arrive, this branch will create its own reproducible baseline and
will label the colleague's reported parity as unverified external evidence.

### Initial saturation evidence

A 100-seed, 48-hour exploratory heuristic audit exposed a structural ceiling: the CH4
tank has no offtake, starts at a randomized 0--20% fill, and can only fill. The heuristic
deliberately stops methanation near 97% fill. Over seeds 1--100, 73 runs ended at or above
96% fill; within those ceiling-limited runs, the correlation between starting CH4
inventory and reported CH4 production was approximately `-0.99994`.

This means the headline production metric is, for most nominal episodes, nearly the tank's
remaining headroom rather than a sensitive measure of dispatch quality. Mean heuristic
solar utilization was approximately 99.4%, no shutdown step occurred, and the positive
methane reward contribution was roughly two orders of magnitude larger than the combined
penalties. These observations strongly support `H1`, but they will be reproduced through
a committed diagnostic command before being treated as a branch result.

### Committed saturation diagnostic

`plant_backend.rl.evaluation.ceiling` now formalizes the preceding exploratory audit. It
can evaluate the heuristic alone or pair it with any repository-compatible checkpoint on
identical seeds, prints an explicit evidence scope, and writes every rollout plus
population distributions to JSON. `make rl-ceiling` is its maintained entry point.

The independently repeated baseline over seeds 0--99 produced:

```text
72/100 runs at or above 96% terminal CH4 fill
Pearson r(initial CH4, produced CH4) = -1.000 on those 72 runs
mean terminal fill                  = 94.7%
mean CH4 production                 = 2538.69 kg
mean solar utilization              = 99.4%
mean shutdown steps                 = 0
```

The exploratory 73/100 result used the shifted seed range 1--100. Both are retained in
the record. The conclusion is insensitive to that choice: roughly three quarters of
default episodes terminate against storage headroom, and their production is nearly a
deterministic function of randomized starting inventory.

## 2026-09-02 — Battery reserve consistency correction

### Finding

The V1 audit found a mismatch between the supervisory control envelope and the
physical battery. `Battery.discharge_headroom_kw` correctly converts stored energy
into bus-side power through discharge efficiency, but both the safety validator and
the heuristic divided usable stored energy by timestep duration without applying the
efficiency loss. Near the configured reserve, they could therefore request more
bus-side support than the battery could deliver without crossing that reserve.

The correct relationship is:

```text
deliverable bus power = usable stored kWh * discharge efficiency / timestep hours
```

### Change and verification

The validator and heuristic now use the same relationship as the physical component.
Two regression tests cover different failure modes:

- the validator clamps a deliberately excessive request to the loss-adjusted boundary
  and the projected ending SOC is exactly the declared reserve;
- the heuristic refuses a minimum-stack-load top-up for which the lossless estimate
  appears sufficient but the physically deliverable power is insufficient.

Focused verification before commit:

```text
backend/tests/test_control.py                         22 passed
simulator battery-focused tests                       5 passed
```

This fix preserves existing nominal behavior away from the boundary. The envelope
still operates on reported SOC and reported effective capacity by design; a positive
SOC sensor bias can consequently make it optimistic. Sensor uncertainty and robust
reserve estimation are separate V2 research questions and are not silently folded
into this correctness fix.

## 2026-09-02 — External benchmark and product-thesis review

### Research question

What technical claim would be competitive with current research and relevant to a
commercial or commission evaluation, and which simulator capabilities are necessary
to test it honestly?

The review screened recent electrolyser scheduling and dynamic-operation reviews,
process-control comparisons of RL and MPC, safe-learning work, process-control
benchmark design, power-to-methane process studies, digital-twin verification and
validation literature, and official Clean Hydrogen JU and EU RFNBO sources. The
decision-relevant bibliography and the boundary between supported and unsupported
claims are recorded in `STATE_OF_THE_ART.md`.

### Findings

- “RL works” or “RL matches a heuristic” is below the relevant evidence bar. A recent
  green-hydrogen study already combines multi-year weather, higher-fidelity PEM
  behavior, PPO, rule-based and MPC baselines, sizing, and interpretability.
- MPC is a mandatory comparator, not an intentionally weak baseline. Its explicit
  treatment of prediction and constraints also makes an RL-MPC hybrid commercially
  more defensible than assuming standalone RL superiority.
- Reward penalties cannot stand in for hard safety. Proposed and applied actions,
  intervention magnitude, fallback activation, and physical violations must be
  reported separately.
- Commission- and customer-relevant outcomes are delivered on-spec product, cost,
  energy consumption, reliability, degradation exposure, carbon qualification, and
  maturity of validation—not private cumulative reward.
- Without physical calibration and automated data exchange, the present model is a
  control-oriented simulator or digital-twin prototype. The journal will not silently
  promote it to a validated twin.

### Product decision

The working product is a safety-bounded supervisory optimisation and
virtual-commissioning platform for power-to-X developers, integrators, and operators.
It begins with sizing/dispatch studies, controller qualification, and shadow-mode
recommendations above the PLC layer. The first benchmark problem is recurring methane
delivery under uncertain renewable supply and equipment commitment.

The target claim is deliberately algorithm-neutral: at equal safety and delivery
reliability, the selected Plant An Agent controller should reduce risk-adjusted cost
and wear relative to tuned forecast-aware heuristics and stochastic MPC on held-out
conditions, with the effect surviving hardware-in-the-loop evaluation. If MPC wins,
the product should use MPC; if a hybrid wins, it should use the hybrid.

Customer willingness to pay and the best initial buyer remain unvalidated. They will
require design-partner evidence rather than a simulator result.

### Environment V2 contract decision

`ENVIRONMENT_V2_SPEC.md` turns the thesis into a staged, falsifiable program. V1 is
frozen; V2 gets a separate versioned observation/environment contract. The minimal
bundle is recurring delivery settlement, stochastic renewable truth with forecasts
that cannot read future innovations, and a technology-labelled electrolyser commitment
state. Physics remains one minute while the initial controller interval is 15 minutes.
HEPA, the supervisor, and natural-language scenario authoring are frozen during the
routine-controller tournament and introduced later through factorial ablations.

## 2026-09-02 — Frozen V1 interface identity

### Problem

Checkpoint compatibility previously compared the integer specification version and
ordered field names. A change to a field's normalization, source, bound, target, or
conversion could therefore retain the same names and accidentally load a policy under
different semantics. Checkpoints also did not explicitly identify Environment V1.

### Change

The V1 observation/action interface now has the stable identity `plant-env-v1` and a
SHA-256 digest over its canonical machine-readable descriptor. The digest covers field
order, names, sources/targets, units, normalization/conversion, bounds, sizes, and the
specification version. It is serialized into all newly written checkpoint metadata and
validated when loading.

Metadata written before this change has neither field. It remains explicitly treated
as legacy V1 and is checked using the prior specification-version and ordered-name
rules; it cannot retroactively prove its exact descriptor semantics. A checkpoint that
declares a different environment or a nonempty mismatched digest now fails loudly.

The frozen digest at this commit is:

```text
3a0f8b9c812e0937f48c5c6a700f1628d8d6d66d845df345c33720001a9ac085
```

This digest intentionally does not cover reward, episode generation, plant parameters,
or simulator dynamics. V2 requires a broader environment manifest before training;
the limited scope is documented rather than overstated.

Verification before commit:

```text
focused RL tests                              60 passed
full backend suite                           279 passed, 1 existing warning
```

## 2026-09-02 — Methane delivery kernel

### Research objective

Remove the V1 terminal-tank objective without changing V1 behavior. The first physical
V2 seam must turn methane into a recurring service obligation and provide accounting
that later rewards, metrics, controllers, and oracles can consume.

### Design decision

Delivery is an opt-in business ledger beside `PlantState`, not a silent extension of
the V1 observation or scenario schema. A `DeliveryCommitment` has a unique identifier,
an integer episode-relative physics tick, and requested kilograms. Integer ticks avoid
floating-point deadline comparisons.

At each simulator step:

1. the selected action produces over `[t, t + dt)`;
2. the physics tick advances;
3. every commitment due on the new tick settles against available CH4;
4. delivered methane is withdrawn from storage;
5. the end-of-interval plant state is constructed.

This means product made during the deadline interval is available to that delivery.
Each commitment settles exactly once. Same-tick commitments preserve declared order,
and every settlement satisfies:

```text
requested kg = delivered kg + unmet kg
ending CH4    = starting CH4 + produced CH4 - delivered CH4
```

The simulator exposes only the current result and next pending commitment through its
ordinary delivery seam. Full schedules and settlement history are available through an
explicit audit serialization for research provenance. `PlantState`, `Scenario`, and
their serializations remain unchanged, so installed V1 controllers receive no new
fields or future contract tape.

Schedule installation validates the complete input before mutation. Invalid IDs,
deadlines, masses, duplicates, or already-passed deadlines leave an existing schedule
untouched. Reset and scenario load clear both schedule and accounting. A successful
mid-episode replacement is allowed only with future deadlines and intentionally begins
a new ledger; evaluation manifests will prohibit such replacement during a benchmark.

### Verification

Sixteen focused tests cover deadline ordering, shortfall and tank conservation,
same-tick obligations, exactly-once settlement, reset/load behavior, invalid and atomic
replacement, immutable/detached views, and equivalence of absent and explicitly empty
schedules.

```text
delivery tests                                 16 passed
complete simulator suite                       72 passed
delivery + existing simulator behavior         49 passed in local cross-check
```

This commit provides accounting, not yet a new RL task. The next increment constructs
an explicit V2 environment with delivery observations, service-based reward, decision
cadence, and an audit manifest; stochastic weather and equipment commitment remain
separate later ablations.

### Post-correction V1 replay

The 100-seed V1 ceiling diagnostic was repeated at `ed368e8` after the battery-reserve
fix and delivery-kernel commit. The opt-in delivery seam is inactive in this command.

```text
72/100 runs at or above 96% terminal CH4 fill
Pearson r(initial CH4, produced CH4) = -1.000 on those 72 runs
mean terminal fill                  = 94.7415%
mean CH4 production                 = 2538.6667 kg
mean reward                         = 987.1140
mean minimum SOC                    = 18.1671%
```

The original pre-fix values were 2538.6929 kg, 987.1268 reward, and 18.1641% minimum
SOC. This small, intentional drift is attributable to correcting loss-adjusted reserve
support; the saturation count and conclusion are unchanged. The generated replay is
`/tmp/plant-an-agent-v1-ceiling-post-kernel.json` and remains outside Git.

## 2026-09-02 — Commercial and commission context check

An official-source update checked the product thesis against current IEA, European
Commission, and CORDIS material.

- The IEA's 2026 review reports that low-emissions hydrogen has only just exceeded 1%
  of global production, investment momentum slowed in 2025, the announced 2030
  pipeline contracted, and renewable hydrogen remains more expensive than fossil
  production in most regions. Cost and credible offtake are therefore central product
  risks, not secondary presentation metrics.
- The IEA's 2025 review reports that established industrial hydrogen uses and
  hydrogen-based fuels represented almost all announced firm private-sector offtake.
  Customer discovery should begin with projects that have a real industrial or fuel
  buyer and a measurable shortage/operating cost.
- The current EU Innovation Fund regular-grant criteria are greenhouse-gas avoidance,
  degree of innovation, project maturity, replicability, and cost efficiency. An RL
  benchmark can support only part of this case; it cannot substitute for project,
  financial, physical-validation, or emissions evidence.
- TURN2X already reports an EU-supported renewable-methane reactor digital twin used
  for design optimization. “We have a digital twin” is consequently not a defensible
  differentiator. The research must demonstrate whole-chain operational decisions,
  comparative performance, bounded safety, transfer, and deployment evidence.

These findings reinforce the delivery-contract benchmark and the algorithm-neutral
platform thesis. They also raise the priority of finding a design partner and exact
target call after the first discriminative software benchmark exists.

## 2026-09-02 — Delivery-only Environment V2 alpha

### Implemented contract

The first runnable V2 task now lives under the explicit
`plant_backend.rl.v2.PlantEnvV2` namespace. The parent package's `PlantEnv` and its
30-value V1 observation remain unchanged. The alpha identity is
`plant-env-v2-delivery-alpha`; its 34-value observation/action descriptor digest is:

```text
7fc3808d1a393fb59ccd34c7b6cd31114530348a37fcb66f054604877951316f
```

The four appended observations expose only current contractual context: normalized
time to the next announced delivery, next quantity relative to tank capacity, current
inventory coverage of that commitment, and cumulative fill rate. Full future schedules,
seeds, scenario details, fault tapes, and split labels are absent from Gym reset/step
information. They live in an explicit audit manifest with a canonical SHA-256 digest.

One policy decision is held for 15 one-minute physics steps. The independent safety
guard is rerun before every inner step rather than assuming that a safe command remains
safe as inventories change. Both vector policies and normal `Action` controllers share
one transition, reward, and metrics path.

The default episode is seven days with a commitment every six hours. Inventory,
weather, and contract generators receive named child seed streams so adding a future
weather draw cannot silently alter initial inventories. Weather and contracts are
deterministic in this alpha, but their stream identities are already reserved.

### Objective and reporting

Raw methane production is retained as a diagnostic signal but has no positive reward
component. Positive value comes only from delivered methane and a bounded terminal
inventory value. Shortfall is penalized four times the delivery scale, while
curtailment, battery discharge, reserve breach, shutdown, and guard correction remain
visible costs. Final evidence will use physical delivery and risk metrics rather than
this provisional scalarization.

Episode metrics now include commitment count and service level, requested/delivered/
unmet kilograms, worst individual delivery, product and terminal inventory, renewable
use, battery use, minimum SOC, shutdown, and guard-intervention frequency.

### Exploratory feasibility sweep

Before fixing the provisional default, the uncommitted alpha was rolled out over seven
days using identical seeded initial-condition distributions. At a 45 kg/h contract,
20-seed means were:

| controller | delivery fill | unmet kg | service level | CH4 produced kg | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped heuristic | 100.0% | 0.0 | 100.0% | 9355.5 | 21.60 |
| fixed production mix | 100.0% | 0.0 | 100.0% | 7903.9 | 14.18 |
| fixed balanced mix | 86.6% | 1014.3 | 85.0% | 6298.7 | -9.87 |
| idle | 3.3% | 7313.1 | 1.3% | 0.0 | -145.23 |

Delivery alone therefore removes the terminal tank ceiling and rejects clearly poor
dispatch, but 45 kg/h still leaves a broad competent plateau. A ten-seed rate sweep
found:

| rate kg/h | heuristic fill | production-mix fill | balanced fill |
| ---: | ---: | ---: | ---: |
| 45 | 100.0% | 100.0% | 85.3% |
| 50 | 100.0% | 96.2% | 77.1% |
| 55 | 100.0% | 87.7% | 70.4% |
| 60 | 95.5% | 80.7% | 64.8% |

A finer 20-seed heuristic sweep gave 100% mean fill at 55 kg/h, 99.96% at 56,
99.65% at 57, 98.87% at 58, 97.63% at 59, and 96.17% at 60. The alpha default is
therefore provisionally 55 kg/h: the shipped heuristic is feasible on the explored
seeds but has limited surplus, while simple fixed allocation is materially worse.
This value must be recalibrated when stochastic weather changes the available-energy
distribution.

The sweep is exploratory, not the formal benchmark: it has no MPC/oracle, stochastic
weather, equipment commitment, training seeds, or held-out regimes. It already confirms
that product delivery is a better objective than tank headroom, and also confirms that
delivery alone is insufficient to establish a state-of-the-art control problem.

### Verification

The alpha tests cover coexistence with V1, descriptor identity, configuration/time
alignment, independent seed streams, public-information leakage, action holding,
vector/controller path parity, delivery-only reward, terminal settlement ordering,
observation bounds, exact replay, audit separation, and the Gymnasium environment
contract.

```text
V2-focused plus existing RL contract tests            78 passed
complete repository Python suite                     369 passed, 1 existing warning
```

Known limitation: decision reward and metrics aggregate all 15 physical steps, but the
base telemetry fields still sample V1's most recent `PlantState` flow values. The
explicit boundary `ControlState` and unambiguous forecast-validity convention remain
part of the weather increment and must be completed before training a policy presented
as a V2 result.

## 2026-09-02 — Independent alpha review: non-finite action correction

An independent read-only review of the delivery alpha found that the shared action
adapter replaced `NaN` with vector value zero before checking validity. Zero is not a
neutral physical action in the `[-1, 1]` policy space: it maps to 50% on every channel,
then share normalization yields 25% for each solar consumer and requests half of the
battery-support limit. An all-`NaN` policy output could therefore become a strong action
without being marked out of range.

Non-finite policy outputs now map to the `-1` zero-authority endpoint on their channel
and always set the invalid/clipped flag. An all-`NaN`, all-`+inf`, or all-`-inf` vector
therefore produces `Action.idle()`, zero requested power, and the normal correction
penalty/metric. Finite out-of-range behavior is unchanged. This is a shared V1/V2
safety correction because both environments use the same adapter.

### Invalid experiment configuration

The same review identified two invalid configurations that previously reached runtime.
A delivery could exceed the methane tank capacity, making full instantaneous settlement
physically impossible while the normalized observation clipped away the excess amount.
Reward weights could also be negative or non-finite, allowing a negative shortfall
weight to reward missed delivery or a `NaN` to poison both reward and manifest output.

V2 now rejects a non-finite or over-capacity commitment amount during episode validation,
and requires every reward weight to be finite and non-negative at environment construction
and direct reward computation. These are experiment-contract errors, not conditions a
controller should learn around.

### Immutable delivery episodes

Successful mid-episode delivery-schedule replacement previously began a fresh ledger,
which intentionally suited an interactive simulator but could erase benchmark settlement
history without changing V2's sampled manifest hash. The delivery ledger now has an
opt-in freeze. V2 freezes the complete schedule immediately after reset; replacement then
raises without altering schedule or accounting. A simulator reset clears the ledger and
its freeze, preserving the interactive/V1 default. The ledger audit explicitly records
its frozen state.

### Full checkpoint compatibility identity

The static V2 digest initially covered only the ordered observation/action descriptor.
Changing the 15-minute cadence to 30 minutes, the delivery rate, plant sizing, safety
envelope, or reward weights therefore retained that digest even though it changed the
task a policy had learned. There is not yet a V2 checkpoint loader, making the
specification's stronger compatibility promise premature.

V2 now exposes a second, canonical checkpoint-contract digest over the interface,
physics and decision timing, time/flow semantics, complete episode distribution,
plant configuration, constraints, objective version, and reward weights. A validation
function refuses a missing, wrong-environment, or mismatched full digest. The static
digest remains explicitly labelled interface-only. The evaluator and private replay
manifest record both identities. Any future V2 trainer/loader must use the full digest;
the existing V1 loader remains V1-only.

### Additive energy and duration costs

The initial curtailment signal divided curtailed energy by solar energy separately in
every 15-minute decision. Fully wasting 1 kWh and 1,000 kWh therefore incurred the same
penalty, and splitting an identical energy trajectory into more decisions changed its
total cost. Battery, shutdown, reserve, and correction terms had related cadence effects.

Objective version 2 normalizes absolute curtailed and discharged kWh, shutdown physics
steps, guard adjustments, and reserve-deficit duration against fixed one-delivery-period
references. Their sums are now invariant to partitioning the same physical interval,
and wasting 1,000 times more energy costs 1,000 times more. Delivery and bounded terminal
inventory semantics are unchanged. The full checkpoint digest makes policies trained
under objective version 1 incompatible rather than silently reusing them.

### Interval action and guard accounting

The alpha reported requested kW only at the start of a 15-minute decision and the
share-based action applied in its final minute. Solar, storage, and the guard can change
during the other fourteen physics steps, so those fields could not substantiate the
specification's requested-versus-applied power claim. The original raw policy vector was
also discarded after decoding, hiding action aliasing caused by clipping or share
normalization.

Every V2 transition now integrates requested and physically applied kWh for battery
charging, the electrolyser, DAC, methanator, and battery support over all inner physics
steps. It reports interval mean kW and an L1 request/application energy gap across the
four dispatch channels; episode metrics accumulate that intervention magnitude. The
exact raw policy vector is retained in step diagnostics, with non-finite values encoded
as JSON-safe strings. This accounting is observational and does not alter plant physics.

## 2026-09-02 — Auditable delivery-alpha baseline evaluator

The exploratory calibration loops have been replaced by a maintained paired evaluator,
available as `python -m plant_backend.rl.v2` or `make v2-delivery-eval`. For every seed it
constructs a fresh environment for the shipped heuristic, idle control, a fixed balanced
mix, and a fixed production-oriented mix. It verifies that all four arms carry the same
immutable episode-manifest hash before accepting the result.

The JSON includes every rollout; population count, mean, standard deviation, minimum,
and maximum for every numeric physical metric and reward component; the full episode,
plant, constraint, objective, interface, and checkpoint contracts; Git revision and dirty
state; Python/platform identity; and the SHA-256 digest of `uv.lock`. The command's human
report leads with a baseline-only disclaimer: it contains no trained RL policy, MPC, or
perfect-foresight oracle and cannot establish optimality or commercial superiority.

A two-seed, six-hour smoke run verified CLI and JSON wiring after the independent-review
corrections. It is not a calibrated result because initial methane coverage dominates the
first delivery and the worktree was intentionally dirty. The formal 20-seed, seven-day
delivery-alpha result will be generated only from the clean evaluator commit.

### Formal delivery-alpha result

The planned 20-seed, seven-day evaluation was run from clean revision `1200a61` at the
provisional 55 kg/h contract rate. The committed raw JSON hash is
`5d5b34f538f5ec8bb6ff015b80ccfec494a50f01b772606e81e745a5aa8b3711`; its full
provenance and analysis are in `ENVIRONMENT_V2_ALPHA_RESULTS.md`.

The heuristic achieved 100% delivery fill and service with zero unmet kilograms on all
20 inventory seeds. Idle, fixed-balanced, and fixed-production mixes achieved mean fill
rates of 3.3%, 71.4%, and 88.8%, respectively. The delivery feature therefore creates a
real task gap, but the heuristic again saturates the current deterministic regime.

The new accounting also quantified substantial guard action: the heuristic averaged
3,216 requested-to-guard L1 kWh per episode, versus 30,083 and 37,979 kWh for the two
fixed mixes. This strengthens the action-aliasing concern and establishes intervention
magnitude as a required comparator metric.

Decision: do not train a policy against delivery-only alpha as a headline result. Add
causal stochastic weather and forecasts, recalibrate the contract, then introduce an
equipment-commitment consequence and model-based/oracle comparators before evaluating
whether RL has a meaningful advantage.

### Weather data source decision

The next increment begins with a seeded synthetic process so truth/forecast separation,
future-leakage tests, and feature ablations are exact. Its interface will also accept an
immutable external profile. The European Commission Joint Research Centre's documented
PVGIS hourly service provides multi-year solar-radiation and PV-output data and is the
first planned empirical adapter. PVGIS timestamps and database versions must be retained;
hourly values will not be misrepresented as native one-minute measurements.

## 2026-09-02 — Opt-in exogenous solar-forcing kernel

The simulator now accepts an immutable, episode-relative `SolarFactorTape` at reset or
scenario load. A tape contains one bounded resource factor at every physics boundary:
boundary zero initializes the reported reset state, and the factor at boundary `k + 1`
is consumed by physical interval `[k, k + 1)`. Integer indexing, an exact timestep
contract, and explicit exhaustion errors make stochastic episodes replayable without
depending on floating-point wall-clock lookup. Exhaustion is checked before faults,
events, equipment, inventories, or counters advance, so a malformed episode cannot leave
the simulator partially mutated.

Resource availability and equipment condition are deliberately separate. The tape factor
multiplies the solar array's clear-sky power; an active solar-equipment fault multiplies
that result through component health. Ordinary cloud variability therefore does not
fabricate equipment degradation, while simultaneous weather and equipment loss compose
physically and remain distinguishable through resource and health fields.

This is an opt-in seam rather than a V1 behavior change. With no tape, or with an all-ones
tape, state trajectories, events, and delivery accounting are exactly equal to the prior
deterministic simulator. The clear-sky curve also retains its existing sampling convention:
the state at boundary `k + 1` reports flows over the interval just completed using the
clear-sky value sampled at its start. That legacy convention is now documented and will be
paired with explicit interval aggregation in the V2 controller view.

### Verification

Focused forcing and boundary-semantics tests cover hand-indexed factors, source-copy
immutability, strict scalar validation, scenario-clock independence, exact reset replay,
all-ones equivalence, preflighted exhaustion, and multiplicative fault composition. The
simulator package's complete test suite passed independently (`90 passed`); the combined
new forcing and boundary tests passed locally (`25 passed`). Static type checking also
surfaced two pre-existing errors in delivery/event modules, neither in this increment;
they remain recorded technical debt rather than being hidden in a feature commit.

This kernel is not itself a weather model and makes no empirical fidelity claim. The next
layer owns the seeded latent process, causal forecast issuances, clear-sky-envelope digest,
and private replay manifest. Keeping generation outside the plant physics lets future
PVGIS/site traces replace the synthetic fixture without changing downstream equipment or
dispatch logic.

## 2026-09-02 — Solar truth and causal-forecast evidence review

A targeted primary-source review compared JRC PVGIS, Copernicus CAMS, BSRN, ERA5,
ECMWF issued forecasts, TIGGE, ENTSO-E, and peer-reviewed forecast-verification
practice. The central finding is negative but consequential: no preferred open source
is simultaneously pan-European, plant-level pre-curtailment AC truth, native 15-minute,
and paired with a long archive of causal forecast vintages. Dataset convenience must not
be mistaken for operational ground truth.

The resulting data hierarchy is:

- site SCADA with curtailment/outage/inverter/availability flags for eventual plant truth;
- native 15-minute CAMS all-sky and clear-sky radiation as the first open regional proxy;
- BSRN one-minute ground observations where available to validate that proxy;
- PVGIS hourly output for climatology and a convenient modelled-PV adapter, without
  inventing one-minute ramps;
- archived issued ECMWF/ENTSO-E products for causal forecast experiments; and
- ERA5 only as retrospective covariate/reanalysis, never as a historical issued forecast.

The synthetic alpha remains scientifically useful only as a transparent engineering
fixture. It uses a stationary Gaussian AR(1)/discrete-OU latent transformed by a logistic
function, with a one-hour autocorrelation half-life and coherent conditional ensemble
paths. Each ensemble member is first multiplied by the known future clear-sky/nameplate
envelope and aggregated over a supervisory window; mean and p10 are computed across
those member-level aggregates. Averaging pointwise quantiles would destroy the intended
energy-window semantics. Numerical parameters are sensitivity hypotheses, not European
climatology or site calibration.

The leakage contract now requires that an issued forecast may use the current resource
state, deterministic future solar geometry, and an independent forecast RNG, but never
future truth innovations. Evaluation will use rolling origins, a purged maximum-horizon
gap, daylight and all-hours scores, and per-lead calibration. Point metrics are bias,
capacity-normalized MAE/RMSE, and skill against smart persistence. Probabilistic metrics
include CRPS or pinball loss, reliability/coverage and sharpness, while coherent paths
also require energy and variogram scores. A p10 feature alone is not evidence of a
calibrated probabilistic model.

Decision: land the causal synthetic fixture and its future-leakage invariant first, then
add CAMS/PVGIS adapters as separately versioned profiles. Do not tune a learned controller
to the fixture and describe that as weather generalization; empirical and stress-profile
splits remain a later evidence gate.

## 2026-09-02 — Causal weather and decision-boundary primitives

The first weather primitive is now implemented independently of the Gym environment. A
stationary logistic-latent discrete-OU process produces an immutable resource-factor tape
at the one-minute physics grid. The default median factor (0.70), latent standard deviation
(0.75), and one-hour autocorrelation half-life are declared challenge parameters. They are
not fitted weather statistics. Truth and forecast streams are named children of a caller's
seed hierarchy, snapshotted rather than consumed, and retain canonical initial-state
descriptors plus SHA-256 identities for replay.

At every 15-minute decision boundary, an issue-specific child stream draws 32 coherent
conditional trajectories through 24 hours. Only the current resource factor and known
clear-sky geometry enter the forecast; the constructor never reads truth after the issue
boundary. Available PV/nameplate fractions are aggregated within each member before mean
and p10 are computed for `0–1`, `1–6`, `6–12`, and `12–24` hour windows. A counterfactual
test changes every future truth value and proves the already-issued forecast is byte-for-
byte unchanged. A one-hot clear-sky envelope test locks half-open interval indexing.

The accompanying `ControlStateV2` keeps current boundary levels, time, online status,
health, and measured solar separate from time-weighted mean powers, loads, and production
rates over the preceding controller interval. A strict accumulator integrates every
physics sample and rejects invalid durations before mutation. This removes the delivery
alpha's misleading use of the final one-minute flow as if it represented a 15-minute
decision.

### Independent review and corrections

An independent review confirmed the causal `k -> k+1` forecast mapping and found no direct
future-truth read. It also reproduced a numerical failure for valid, very short correlation
half-lives: the vectorized AR closed form divided by underflowed powers. Forecast blocks now
adapt to the process coefficient, with an exact recurrence for the near-memoryless edge.
The regression configuration (`0.015 h` half-life at a 15-minute grid) remains finite.

The review also found that MT19937 state arrays and NumPy scalar configuration values were
accepted but not JSON-hashable, and that reusing a `SeedSequence` advanced only the forecast
side. Canonicalization now handles NumPy arrays/scalars, all input random sources are cloned
before use, and tests cover MT19937 plus repeated-source purity. Private manifests embed
initial RNG states while stating explicitly that synthetic profiles still require the
matching code/lock and deterministic envelope, and external profiles require retained
tape/envelope artifacts matching their digests. They no longer imply that a hash alone can
materialize an arbitrary external episode.

A fixed seed now pins truth, forecast, and complete episode digests to weather-generator
version `synthetic-logit-ou-v1`. Any stochastic algorithm, stream mapping, or quantile
change must bump that version and deliberately update the golden evidence. Strict tests
also cover alignment, immutable issue cadence, bounds, autocorrelation, p10 ordering,
source separation, envelope provenance, and JSON-safe private metadata.

### Remaining limitation

Precomputing a seven-day episode creates 672 vintages × 32 members × 1,440 minute steps,
about 31 million innovations. Independent review measured roughly 0.54 seconds per reset,
material beside roughly 0.27 seconds for 10,080 bare simulator steps. Keep this transparent
implementation for alpha correctness, but set a throughput gate before large policy sweeps:
benchmark controller-resolution forecast scenarios, deterministic lazy issuance, or a
validated conditional-moment approximation without changing the versioned semantics.

## 2026-09-02 — Weather-alpha environment integration and release review

### Integrated task contract

The causal weather primitive is now connected to Environment V2 as
`plant-env-v2-weather-alpha`, specification version 2. A reset samples independent named
inventory, weather, and reserved contract streams. The private weather episode drives an
immutable simulator forcing tape; only the forecast issuance valid at the current decision
boundary is exposed. The full truth tape, future vintages, seeds, stream states, and private
hashes stay out of Gym observations and public step/reset information and remain available
only through the explicit audit manifest.

The controller observation grows from 34 to 42 fields. It retains current boundary time and
inventories, reports solar/health/online state as the latest completed one-minute dispatch
sample, replaces instantaneous process flow/load/rate fields with time-weighted means over
the preceding 15-minute decision, and appends mean and p10 available-PV/nameplate energy
fractions for the half-open `0–1`, `1–6`, `6–12`, and `12–24` hour forecast windows. Reset
uses the initial boundary sample because no preceding decision interval exists. The exact
ordered descriptor and its digest are the authority; prose is not allowed to silently
reinterpret a field.

The checkpoint identity now covers the weather configuration and generator version, solar
forcing convention, clear-sky envelope identity, multi-rate sampling rules, causal forecast
timing, and guard semantics as well as the existing plant, objective, constraints, and
interface. The current 42-field interface digest is
`2f664d7fc5c2aaf37e2bfef62697cf5aa48200597799afcb6ad349ea1ec30ec3`; the default full
checkpoint-contract digest is
`89bcd56a211f3228419d7fce530c1039bd9902c8d7ab7d08b82f8cba7b08476d`.

### Provisional load calibration

Before freezing a formal run, a dirty-worktree heuristic sweep tested five common-random
seeds over seven days. This was a design calibration, not benchmark evidence:

| contracted rate | mean fill | worst-seed fill | mean service level |
| ---: | ---: | ---: | ---: |
| 32 kg/h | 100.0% | 100.0% | 100.0% |
| 35 kg/h | 100.0% | 100.0% | 100.0% |
| 38 kg/h | 99.8% | 99.0% | 99.3% |
| 40 kg/h | 98.0% | 94.2% | 95.0% |
| 42 kg/h | 94.6% | 89.8% | 87.1% |
| 45 kg/h | 88.5% | 84.1% | 78.6% |

The provisional default moved from the delivery-alpha's 55 kg/h to 40 kg/h. At 55 kg/h,
the stochastic resource made the legacy heuristic average only 72.1% fill in a two-seed
probe, confounding controller discrimination with broad infeasibility. At 40 kg/h the
legacy heuristic is usually feasible but no longer saturated. A five-seed four-arm probe
then produced mean fill rates of 98.0% (legacy heuristic), 3.6% (idle), 65.6% (fixed
balanced), and 83.6% (fixed production mix). These values selected a challenge point only.
The run preceded the final realized-bus guard/accounting correction, came from a dirty
revision, lacked equal-information forecast-aware and optimizer arms, and must not be cited
as a formal comparison.

### Independent review findings and corrections

Two independent read-only integration reviews were used as adversarial release gates.
They produced the following corrections before commit:

1. Share-based requested/guarded energy was initially multiplied by the previous boundary's
   solar bus. It now uses the solar bus actually realized in every physical interval, so an
   exogenous weather ramp is not mislabeled as controller or dispatch error.
2. The guard itself also used the previous interval's bus to convert absolute component caps
   into shares. A `0.1 -> 1.0` noon resource jump reproduced a 300 kW electrolyser envelope
   producing 3,000 kW for one interval. The simulator now exposes a side-effect-free preview
   of the exact next dispatch bus, including forcing and a fault beginning or ending on that
   boundary. V2 passes it to the guard before every one-minute step. Tests cover electrolyser,
   DAC, methanator, and combined solar-plus-battery support caps.
3. Applying that correction to the frozen V1 environment changed legacy transitions and
   correction rewards without changing V1 identity. The global adoption was therefore
   rejected. V1 retains its historical telemetry-based guard path for replay; V2 alone uses
   the exact-bus interlock. A future live-runtime migration must be explicit and versioned.
4. The first full checkpoint digest omitted weather/forcing implementation identities. The
   digest now includes all stochastic and clear-sky dynamics versions plus the exact actuator
   guard convention.
5. A horizon could end with an un-settled fractional delivery period. V2 now requires the
   horizon to contain whole delivery periods and reports the weather-alpha stage from the
   shared contract rather than a stale delivery-only string.
6. The evaluator coerced booleans, fractional floats, and numeric strings into integer seeds,
   allowing silent common-random-number aliasing. Seeds are now strictly unique,
   non-negative integer values before casting.
7. The baseline report could imply a fair controller tournament even though the shipped
   heuristic sees neither V2 delivery/aggregate fields nor forecasts and fixed policies see
   no state. Every arm now records its information set and is labelled diagnostic. Equal-
   information forecast-aware, MPC, oracle, and RL comparisons remain open.
8. Descriptor prose called the final one-minute solar sample a simultaneous boundary value.
   Field sources and timing text now state the simulator's established start-of-interval
   sampling convention exactly; no physics were silently shifted to make the prose true.
9. The module-level simulator version used by checkpoint metadata was `0.4.0`, while
   package metadata still declared `0.1.0`. The package declaration is now aligned and a
   regression equates the installed distribution and module versions, making the dynamics
   fingerprint unambiguous.
10. V2 paid a finite-horizon terminal inventory value while returning a time-limit
    truncation. PPO commonly bootstraps truncated transitions, which would add a value for
    a nonexistent continuation and potentially double-count salvage inventory. The V2
    horizon is now an intrinsic termination (`terminated=True`, `truncated=False`), tested
    with positive salvage value and fingerprinted in the checkpoint contract. V1's distinct
    continuing-plant convention remains frozen.
11. Weather alpha added information-set metadata and changed study semantics but initially
    retained the delivery result's schema version. The evidence payload is now schema 2;
    historical delivery-alpha JSON remains schema 1 rather than being rewritten.
12. An empty/dark interval aggregate reported solar output factor 1 even though the V2
    controller convention masks an unobservable nighttime factor to 0. Aggregate and
    boundary conventions now agree, with an explicit dark-interval regression.

### Verification and decision

Final focused simulator, guard, V1-isolation, V2, evaluator, and termination checks passed
(`169 passed`). The complete
repository command `.venv/bin/pytest -q` passed (`479 passed`) with one pre-existing
third-party Starlette/httpx deprecation warning. Preview tests prove no mutation of state,
time, fault lifecycle, event log, or forcing position before dispatch. Existing all-ones
forcing equivalence continues to protect the V1 dynamics kernel.

Decision: commit weather alpha, then run the maintained 20-seed/seven-day paired diagnostic
from that clean revision and preserve its raw JSON separately from the historical
delivery-alpha result. This is still not the benchmark on which to claim RL value. The next
substantive physics increment remains technology-labelled equipment commitment, followed by
a forecast-aware heuristic, deterministic MPC, stochastic/risk-aware MPC where tractable,
and a perfect-information oracle before any headline multi-seed PPO campaign.

## 2026-09-02 — Formal weather-alpha baseline evidence

### Frozen run

The maintained evaluator ran from clean revision
`129624ad6668d3286b431a76cb5e616fc87fa034`:

```text
.venv/bin/python -m plant_backend.rl.v2 --seed-start 0 --count 20 --hours 168 \
  --delivery-rate 40 \
  --json docs/research/results/v2_weather_alpha_baselines_seed0_19.json
```

The run evaluated four diagnostic arms over seeds 0--19, 168 hours per episode, for
80 runs and 806,400 one-minute physical transitions. It took roughly two minutes on the
development machine. Most of that overhead is consistent with the already-measured eager
construction of 672 forecast vintages and approximately 31 million innovations at every
reset; throughput must be addressed before large training sweeps.

The committed artifact's SHA-256 is
`d5af5bee853dd5729792753fe8d5c47d50dd45e6a56ef588f1b6b3b47c7f6721`.
It records result schema 2, Python 3.12.7, the clean source revision, dependency-lock
digest `cb346b6cace28625107c93cc446181b8f85a6916741393c98e8e5ab00f91e3af`,
the 42-field interface digest
`2f664d7fc5c2aaf37e2bfef62697cf5aa48200597799afcb6ad349ea1ec30ec3`,
and full checkpoint digest
`89bcd56a211f3228419d7fce530c1039bd9902c8d7ab7d08b82f8cba7b08476d`.
All arms share one manifest identity within each seed; all seeds have complete arms and
distinct manifests.

### Result and interpretation

The forecast-blind heuristic produced 98.58% +/- 2.15% mean delivery fill, 96.43% +/-
5.30% service, and 95.4 +/- 144.7 kg unmet methane. Twelve of 20 episodes achieved
perfect fill and service; eight did not. Seed 6 was the principal tail case at 93.70%
aggregate fill, 423.7 kg unmet, and only 22.06% fill on its worst commitment. The strongest
open-loop diagnostic, the fixed production mix, achieved 86.36% +/- 4.76% fill and left
916.9 +/- 320.1 kg unmet. In paired comparisons, the heuristic improved fill by 12.22
percentage points and reduced unmet methane by 821.5 kg on average, winning both metrics
on all 20 seeds.

The heuristic also reached the 15% battery reserve in effectively every episode and its
requests accumulated 3.7 MWh of guard projection on average. These are important control-
authority findings rather than hard-safety failures: no reserve breach, shutdown, or
out-of-range action occurred.

This passes the weather increment's narrow calibration condition. It breaks universal
heuristic saturation and creates a meaningful seed-dependent failure tail while retaining
nominal feasibility. It does not demonstrate forecast value, RL value, optimality, or
commercial value: no arm uses the forecast, controller information sets are unequal,
equipment remains memoryless, and the 20 seeds come from one synthetic in-distribution
regime. The comparison with delivery alpha is also confounded by the changed contract
rate and several interface/guard/accounting corrections; it is stage-to-stage evidence,
not a controlled weather ablation.

### Independent evidence audit

An independent read-only audit regenerated all 20 episode manifests and matched every
identity. It replayed all four controllers on seeds 0, 6, and 19 and matched every stored
metric to `1e-12`, recomputed all population summaries, confirmed the interface and full
checkpoint digests against current code, and found no non-finite values or violation of
delivery identities, reward decomposition, solar utilisation, reserve SOC, or inferred
methane conservation. The only archival limitations are that the raw JSON itself does not
store its generation timestamp, exact command, or its own external checksum; this journal
entry supplies those identities.

### Decision

Gate B remains **not passed**. The next scientific bottleneck is an opt-in, technology-
labelled electrolyser commitment model with a compatibility-off replay mode. A formal
controller tournament is deferred until that stateful physics exists and forecast-blind,
forecast-aware, MPC, risk-aware, and feasible perfect-information comparators share a
controller-neutral information contract. Weather generation can be optimized in parallel,
but speed alone would only make the present near-ceiling task cheaper to overfit.

### Pre-change compatibility fixture

Before touching equipment physics, a 24-hour seed-101 replay was frozen at revision
`e0a0ada`. The deterministic tape cycles four declared actions through all 96 controller
decisions and hashes the initial plus post-decision 42-field observations, complete legacy
plant states, rewards, delivery settlements, guard reports, and requested/applied energy
accounting. Its canonical SHA-256 is
`5858920bf829d5c5bc1c755552ec22943e8c8338b0f67b5c7ce67c34b02324e6`.
The test intentionally excludes task labels and checkpoint identities, which must change
when the observation and dynamics contract changes. It includes the numerical behavior
that a compatibility-off implementation is required to preserve exactly.

## 2026-09-02 — Electrolyser commitment evidence freeze

### Technology choice

The first commitment profile is frozen as
`ael_pressurised_fixed_bop_alpha_v1`, a system-level pressurised alkaline scheduling
archetype. This choice is scientific rather than algorithmic: AEL thermal startup and
low-load gas-purity constraints can affect a 15-minute dispatch decision, whereas PEM
electrical response is commonly sub-minute and an invented hour-scale PEM commitment
would manufacture a learning advantage.

Clean Hydrogen JU's 2020 state-of-the-art AEL table anchors a 3,600-second cold start,
60-second hot-idle ramp, 50 kWh/kg nominal system consumption, and 0.12%/1,000 h steady
nominal degradation at an AC-power/tap-water to 30-bar purity-5 boundary. JRC reports a
20--25% practical AEL minimum and 1--5 minute warm response. A 250 kW experiment reports
42.0% minimum load with lye-flow-only control, falling to 21.2% with pressure control and
15.6% only with coordinated flow and pressure. Since this simulator has no pressure,
lye-flow, or gas-purity states, the alpha uses a conservative 40% running minimum and
retains the existing 52 kWh/kg nominal value.

The source review also prevents three tempting but invalid shortcuts: stack-DC curves
cannot calibrate an AC system model; electrical current ramp cannot stand in for
saleable-hydrogen ramp; and steady nominal degradation cannot be converted into an
invented per-start damage coefficient.

### Declared benchmark assumptions

Cold-start power (20% nameplate), hot-start power (20%), standby power (2%), 15-minute
unpowered cooldown, 60-minute minimum up, 30-minute minimum down, +/-25% nameplate/min
system-settling ramp, two-minute power-shortfall grace, 60% low-load reporting threshold,
and the provisional part-load specific-energy multipliers are explicitly synthetic
challenge parameters. They are fingerprinted, switchable, and require sensitivity tests.
At 3 MW, the chosen one-hour start consumes 600 kWh; the project will not present that
derived benchmark number as a universal equipment fact.

No cycling event changes physical efficiency. The alpha records an exposure vector for
later economic sensitivity: cold/hot starts, aborted starts, forced and controlled stops,
mode time, startup/standby energy, operating/low-load time, ramp limiting, and absolute
load movement.

### Architecture decision

The equipment profile is opt-in and lives outside `PlantConfig` and `PlantState`, which
protects V1 configuration/state serialization and HEPA input. The legacy allocation
arithmetic remains a distinct branch when no profile is supplied. The committed branch
must retract unused battery support before discharge, spill refused solar through the
existing battery/curtailment route, and conserve both bus energy and hydrogen mass.

The first integration retains the five-dimensional action: any positive electrolyser
power request is implicit commitment intent. That makes early loss of supplied power a
recorded trip rather than pretending minimum-up is an enforceable command when no energy
exists. Mode, timers, shortfall grace, previous load, and ramp headroom become observable.
An explicit `(desired mode, target load)` action remains the next interface candidate if
component refusal and guard projection continue to obscure policy authority.

The detailed frozen profile, source links, state-transition timing, limitations,
compatibility gate, and required experiment are recorded in
`ELECTROLYSER_COMMITMENT_PROFILE.md` before implementation begins.

## 2026-09-02 — Opt-in electrolyser commitment kernel

### Implementation

The frozen alkaline alpha profile is now implemented as an opt-in simulator feature.
`PlantSimulator(..., electrolyser_commitment=profile)` constructs a six-mode equipment
state machine (`off`, `cold_starting`, `running`, `hot_standby`, `hot_starting`, and
`cooldown`) while omitting the keyword or passing `None` retains the historical
memoryless path. The immutable profile, internal state, public snapshot, and interval
result live outside `PlantConfig` and `PlantState`; this keeps the V1 state/API and HEPA
telemetry shapes unchanged.

The one-minute transition kernel makes startup electricity physically visible on the
plant bus but produces no deliverable hydrogen during the final startup minute. A cold
start therefore consumes exactly 600 kWh at the frozen 3 MW/20%/60-minute settings, and
hydrogen can begin only in the next physical interval. Minimum-up time starts at that
running boundary rather than elapsing during cold startup. Hot standby consumes 2%, a
return from standby requires the five-minute powered hot start, shortfalls receive the
declared two-minute grace, and trips enter unpowered cooldown plus minimum-down lockout.
The running branch applies the frozen piecewise specific-energy curve, external fault
health, storage headroom, and the +/-25% nameplate/minute settling envelope.

The committed dispatch path reserves possible battery support before asking the equipment
what it can accept. It discharges only the accepted remainder after solar priority, sends
refused solar back through the existing spill path, and preserves both the solar-bus and
electrolyser-bus identities. This ordering closes a subtle failure mode in which a
ramp-limited or starting component could otherwise discharge energy into a load that had
already refused it.

`plant-sim` advances from 0.4.0 to 0.5.0 in module metadata, package metadata, and the
workspace lock so future checkpoints can identify the new dynamics explicitly.

### Adversarial corrections

Review during implementation found and corrected five boundary/audit defects before the
increment was committed:

1. Minimum-up time originally elapsed during the one-hour cold start. It now begins only
   when cold startup completes; an already-running commitment clock continues across a
   hot restart.
2. Every declared duration is required to align exactly to the physics timestep, and the
   grid must permit the first producing interval to reach minimum stable load from startup
   power without violating the running ramp.
3. Powered startup and standby originally exposed a nonzero conversion efficiency through
   generic telemetry. Committed efficiency is now zero whenever no saleable hydrogen is
   produced.
4. Public ramp headroom is mode-aware and respects both nameplate and minimum-running
   floors.
5. If routed power falls below the feasible down-ramp trajectory, the simulator does not
   invent the missing electricity: it falls back to hot standby. That exceptional movement
   is now counted as a ramp-limit exposure, with a dedicated regression, so it cannot look
   like a free transition in later controller comparisons.

### Verification and limits

The focused commitment suite contains 36 cases covering frozen-profile validation,
duration and energy boundaries, hot restart, grace and cooldown, minimum up/down,
controlled versus forced stops, ramping, part-load efficiency, fault degradation and
outage, storage headroom, reset/replay, battery-support retraction, bus conservation,
small-tank mass conservation, and explicit-`None` legacy equivalence. The simulator suite
passes 128 tests. A separate deterministic randomized probe replayed 25 day-long action
sequences through paired simulators: 36,000 paired samples/72,000 physical transitions
preserved exact replay, finite non-negative commitment state, monotonic cumulative
counters, tank bounds, and both electrical identities.

This is still a control-oriented equipment kernel, not a calibrated thermochemical twin.
Temperature, pressure, lye flow, product purity, venting, compression, and an equipment-life
law remain absent. The current five-field action still expresses commitment only implicitly
through requested electrolyser power. Most importantly, commitment is not yet active in
Environment V2: observations, objective terms, checkpoint identity, metrics, and paired
off/on evaluation remain the next integration increment. No controller or commercial claim
is supported by this simulator commit alone.

## 2026-09-02 — Commitment-alpha Environment V2 integration

### Versioned task contract

The opt-in alkaline equipment kernel is now wired into the default Environment V2 task.
`EpisodeConfigV2.electrolyser_commitment` defaults to the frozen
`ael_pressurised_fixed_bop_alpha_v1` profile, validates it against the physics timestep,
serializes the complete profile into the episode/checkpoint contract, and passes it to the
shared simulator. Passing `None` is the explicit compatibility-off ablation; it does not
silently select some second equipment approximation.

This semantic change creates a new policy contract:

```text
environment id          plant-env-v2-commitment-alpha
interface spec          3
stage                   equipment-commitment-alpha
observation size        57
action size             5 (unchanged)
checkpoint schema       3
private manifest schema 3
objective version       3
```

The prior 42 observations remain the exact prefix. Fifteen Markov fields are appended:
seven one-hot modes in the fixed order `legacy`, `off`, `cold_starting`, `running`,
`hot_standby`, `hot_starting`, `cooldown`; previous accepted power; upward and downward
ramp headroom; and startup, cooldown, minimum-up, minimum-down, and shortfall timers.
Power/headroom values are normalized by equipment nameplate (3 MW in the default plant)
and timers by the corresponding frozen profile duration. All are bounded in `[0, 1]`.
Compatibility-off sets only the `legacy` one-hot field to one and every other suffix
value to zero.

`ControlStateV2` carries the immutable equipment snapshot beside the current
`PlantState`; no commitment fields were inserted into the shared/V1 state schema. The
snapshot is exposed in public reset/step information because it is causal controller
state, while the full fixed profile remains in the checkpoint/audit identity rather than
being repeated as observation data. Full future weather, seed streams, private hashes,
and future forecast vintages remain private.

The full checkpoint identity now includes the simulator's commitment dynamics version
and complete equipment profile. Separately, each reset derives an
`exogenous_episode_sha256` from sampled initial inventories, weather identity, contracts,
seed streams, and timing. The latter deliberately excludes the commitment profile: paired
off/on arms must have the same exogenous digest but different checkpoint and full-manifest
digests. This creates an auditable pairing key without falsely claiming that two arms have
identical plant dynamics.

### Exact exposure accounting and objective boundary

At every 15-minute decision boundary, V2 differences the simulator's monotonic cumulative
counters rather than reconstructing events from final samples. The immutable interval
record contains exact deltas for:

- cold starts, hot starts, aborted starts, forced stops, and controlled stops;
- startup and hot-standby energy;
- productive operating seconds and low-load seconds;
- ramp-limited interval count and absolute load movement;
- seconds spent in each equipment mode.

Compatibility-off records the interval as `legacy` mode time with every equipment
counter at zero. Mode-time deltas must sum to the physical interval, counter regressions
are rejected, and profile enablement/identity cannot change within a decision. The same
interval record is returned in step information and accumulated into the human-facing
episode metrics, making every exposure independently inspectable.

Only two of these new quantities affect objective version 3. `nonproductive_energy`
penalizes startup plus standby kWh after normalization by electrolyser nameplate times
one delivery period, with provisional weight `0.05`. `forced_stop` penalizes the exact
event count with provisional weight `0.10`. Both are additive across interval partitions.
All other exposure fields remain diagnostic. In particular, the implementation does not
invent a monetary wear law, a per-start health decrement, or a low-load damage coefficient.

### Focused validation evidence

The integration was checked from the working tree based on simulator commit `d3aa615`:

```text
pytest -q backend/tests/test_rl_v2.py \
  backend/tests/test_rl_v2_boundary.py \
  backend/tests/test_rl_v2_weather_observation.py
```

Result: **54 passed in 0.68 seconds**. These tests cover the exact 57-field order and
bounds, one-hot/timer boundaries, immutable sidecar sampling, per-decision exposure
deltas and mode-time coverage, additive reward signals, profile-sensitive checkpoint
identity, profile-independent exogenous pairing, prior-contract rejection, public/private
information separation, controller/vector parity, deterministic replay, and V1 isolation.
The compatibility-off fixture hashes only the original 42-field prefix plus its established
physical/reward/accounting payload and retains
`5858920bf829d5c5bc1c755552ec22943e8c8338b0f67b5c7ce67c34b02324e6`.

This was the first focused integration checkpoint. Subsequent whole-repository regression
from the same integration working tree ran:

```text
.venv/bin/pytest -q
```

Result: **526 passed in 8.74 seconds**, with one pre-existing third-party Starlette/httpx
deprecation warning. The final integration commit and contract digests must still be
recorded after adversarial review. One review item is to keep the exposed ramp-headroom
horizon unambiguous if a future task uses a non-60-second physics grid; the current frozen
alpha uses the one-minute grid.

### Interpretation and next decision gates

The repository can now pose the intended scheduling question—whether to spend energy
starting or keeping an alkaline unit warm under uncertain future solar and recurring
delivery obligations—and measure the resulting operational exposures. It has not yet
shown that the question is well calibrated or that any sophisticated controller answers
it better.

Next, run paired commitment-off/on feasibility sweeps on identical exogenous digests,
then the forecast-blind/issued/perfect information factorial with a delivery-aware
heuristic, deterministic MPC, and a feasible future-information planner. Inspect physical
refusal and safety projection before deciding whether the five-field share command must
be replaced by an explicit equipment mode/setpoint interface. Only after the achievable
gap and held-out thresholds are predeclared should multi-seed PPO training begin.

The remaining claim limits are material: the weather and equipment parameters are
synthetic and not site/vendor calibrated; commitment intent is implicit; no MPC, oracle,
or RL tournament exists; HEPA, supervisor, and natural-language increments have not been
factorially evaluated; and no technical result yet establishes customer value or
willingness to pay.

### Adversarial contract hardening before release

An independent read-only review found that the first integration was test-clean but not
yet fit for continuous-policy training. A controller had to make both commitment-bearing
commands exactly zero to request an orderly stop. Because the normalized action mapping
turns only a raw value of exactly `-1` into zero, an arbitrarily close continuous output
could instead be interpreted as continued commitment, experience a power shortfall, and
be counted as a forced trip. The review also found that step information called the
guard-approved command `applied_action` even when equipment physics subsequently refused
it, that the observation compared only a human profile label, that invalid non-finite
internal values could be clipped or masked, and that the ramp-headroom horizon was
ambiguous under a non-one-minute V2 physics grid.

The release-blocking action issue was corrected in simulator commit `7ccb1a8`. Commitment
dynamics version `electrolyser-commitment-alpha-v2` declares a 1% zero-command deadband:
intent is false when the electrolyser solar share is at most 1% and battery support is at
most 1% of electrolyser nameplate. This keeps the off region finite for Gaussian/tanh
policies while remaining below the frozen 2% hot-standby draw. The constant is part of the
V2 action descriptor and full task identity; it is controller-interface semantics, not a
vendor equipment claim. A paired regression proves that near-endpoint and exact-endpoint
commands produce the same controlled stop once minimum-up time has elapsed. `plant-sim`
is version 0.5.1.

Every immutable equipment snapshot now carries a canonical digest over the complete
profile. The encoder and interval-delta boundary compare this digest in addition to the
human label, closing a case where `replace(profile, cold_start_s=...)` could silently use
the wrong timer normalization. Simulator-owned commitment fields, interval outcomes, mode
times, and cumulative deltas now reject non-finite, negative, incomplete, duplicated, or
backward-moving values. Commitment-alpha V2 explicitly requires a 60-second physics grid;
the compatibility-off task remains free to use other aligned grids.

Control authority is now reported as three distinct energy distances. The existing
weather-alpha `interval_power` keys retain their historical interval-total definitions so
the frozen compatibility SHA remains
`5858920bf829d5c5bc1c755552ec22943e8c8338b0f67b5c7ce67c34b02324e6`.
New `control_authority` values integrate the absolute per-physics-step request-to-guard,
guard-to-plant, and request-to-plant gaps, so opposite routing errors cannot cancel inside
one controller decision. Episode metrics retain the exact latter two values as
`dispatch_gap_kwh` and `request_application_gap_kwh`. The final guard-approved command is
now named `guarded_action`, not physical application.

One attempted implementation initially redefined the historical `dispatch_gap_l1_kwh`
key as the non-cancelling integral. The frozen weather-alpha trace rejected that change
with a different SHA. The implementation was revised to add the exact integral under a
new semantic name while preserving the archival key and replay. This failure is evidence
that the compatibility gate is detecting accounting drift rather than only physical-state
drift.

The tightened focused set completed with **127 passed**. Refreshing the editable 0.5.1
package metadata with a default offline dependency sync removed the optional training and
HEPA groups; the resulting `500 passed, 2 skipped` run was rejected as incomplete evidence.
After restoring all locked groups offline, the authoritative full regression was:

```text
.venv/bin/pytest -q
545 passed, 1 warning in 8.34s
```

The warning remains the pre-existing Starlette/httpx deprecation notice. No test was
skipped. The frozen default identities after hardening are:

```text
profile SHA-256             551be0136b714758cef7067e44d416cc9df5b69ff5576f21a50c74938c504c60
V2 interface digest         6a4943d9350ba7c98c749b46a309f478b692c173cbb568809e34b5b6d5fee155
V2 checkpoint-task digest   ef851793314414c19d3f7442664bf8d09b2b3f9637aa7c1698988bb14625c42f
```

The review's economic concern is deliberately not papered over: objective version 3 is
still service-oriented and does not claim to price stack wear. Starts, low-load time,
cycling, ramping, and movement remain Pareto scorecard outputs until a defensible economic
model and sensitivity range are supplied. That limitation is part of the next controller
tournament rather than an excuse to invent degradation coefficients.

## 2026-09-02 — Commitment-alpha delivery-rate development screen

### Replacement of the interrupted sweep

The first attempted calibration was a 20-seed, 168-hour sweep. It was interrupted before
complete cells were available, so none of its partial cells is treated as formal evidence
or combined with later results. It was replaced with a bounded screen over seeds 0--2,
72 hours per rollout, from clean revision
`9d94483832b5a1a2ed80e995f36c99edfdfa2c93`.

This replacement is deliberately labelled development work. The seeds, weather regime,
35 kg/h target, and 42 kg/h stress point were all inspected during calibration and are
therefore neither validation nor held out.

### Paired result and target decision

Commitment-on and compatibility-off arms shared exogenous episode hashes for every
controller-specific `(rate, seed)` comparison. The low-rate screen completed 36/36 such
pairs (3 rates x 3 seeds x 4 controllers); these cover 9 rate-seed episode cells and
72 rollouts, not 36 independent weather episodes. The high-rate extension completed
48/48 pairs (4 x 3 x 4), covering 12 rate-seed episode cells and 96 rollouts. Full
manifests appropriately differ across the on/off dynamics.

At 30 kg/h the commitment-on legacy heuristic remained saturated at 100% delivery fill.
At 35 kg/h it achieved 97.7% mean fill, two perfect episodes out of three, 57.3 kg mean
unmet methane, 172.0 kg worst-episode unmet methane, and 97.2% mean service. The fixed
production and balanced mixes reached only 86.2% and 59.0% fill, while the compatibility-
off heuristic remained perfect. At 40 kg/h the on heuristic fell to 88.9% fill and 75.0%
service, compared with 99.7% fill off.

The primary development target is therefore **35 kg/h**: mostly feasible, but able to
separate both controller behavior and commitment-on/off dynamics. **42 kg/h** is a named
development stress point. There the heuristic achieved 84.97% fill, 454.5 kg mean unmet,
641.6 kg worst-episode unmet, and 72.22% service on, versus 98.20%, 54.4 kg, 163.2 kg,
and 97.22% off. Rates 45, 48, and 50 kg/h reduced heuristic fill to 79.78%, 75.23%, and
72.51% on and 94.32%, 88.87%, and 85.60% off; those cells increasingly stress delivery
feasibility even without commitment.

### Controller-contract finding and next comparator

Cold/hot starts, aborted/forced stops, nonproductive energy, and physical dispatch gap
were invariant by delivery rate within each commitment-on controller at reported
precision. This is expected from the information boundary: the legacy heuristic is
called through its older telemetry interface and the fixed arms are open loop, so none
consumes the V2 contract target even though a causal delivery representation exists in
the 57-field task observation. The rate sweep changes obligations and scores, not their
plan. It consequently does not bound the value available to a delivery-aware controller.

The next experiment is an otherwise matched causal comparison: first add a delivery-aware
forecast-blind rule, then an issued-forecast-aware version under the same safety,
commitment, cadence, action, forecast-vintage, and exogenous-pairing protocol. The complete
tables, exact pairing semantics, operating-exposure means, provenance, and evidence limits
are recorded in `COMMITMENT_ALPHA_CALIBRATION.md`.

## 2026-09-02 — Pre-outcome causal-comparator adversarial review

### Why execution was paused

The initial three-arm causal controller harness was integrated and its focused tests were
green, but no 72-hour comparator output was opened. An independent read-only adversarial
review was run first because a clean test suite does not establish that the experiment measures
the estimand it claims. The review found no direct future-truth leakage and no release-
blocking hard-invariant failure, but it identified several benchmark-invalidating design
and reporting defects. They are being repaired before the development experiment is
frozen; none will be concealed as an after-the-fact adjustment to a favorable result.

The findings were:

- the evaluator showed only marginal arm summaries rather than the predeclared paired
  `A1-A0` and `A2-A1` effects;
- controller exceptions aborted the matrix instead of remaining visible failed outcomes;
- dirty-tree provenance recorded only a Boolean and therefore could not reconstruct the
  executed source;
- the first planner could starve feedstock production by routing scarce power to a
  methanator that had no immediately usable H2/CO2;
- its one-minute electrolyser ramp headroom incorrectly capped a command held for 15
  minutes;
- startup/cooldown time was not removed from productive time before a deadline;
- a zero boundary solar sample could produce a held zero-share action even as PV rose,
  curtailing new solar while contingent support discharged the battery;
- targeting only one delivery could empty methane at an evening settlement immediately
  before a sunless overnight commitment;
- prorating complete-window forecast aggregates over partial windows assumed uniform PV
  inside a window, which the forecast product does not provide;
- contingent solar/support routing intentionally creates part of the reported physical
  routing distance, so the old generic “dispatch gap” label could be misread as pure loss
  of control authority;
- accepted conversion-process energy and energy per delivered kilogram were not yet
  retained at episode level; and
- commitment intent compared a dimensionless PV-bus share with a stack-nameplate
  fraction. On a 120 MW PV bus, a 0.6% route can offer more than the 600 kW required to
  start the 3 MW stack, yet the old test classified it as no intent because 0.6% is below
  1%.

The reviewer used a 24-hour seed-0 development reproduction while diagnosing the
inventory-horizon defect. The provisional A1 and A2 implementations delivered 467.7 kg
and 603.1 kg respectively from 840 kg requested, with the midnight commitment empty.
Those values are contaminated development observations, not evidence, but they are
retained here because they motivated the explicit multi-delivery reserve correction.

### Dimensionally correct commitment intent

The simulator now forms commitment intent from the physically resolved power offered to
the electrolyser: routed solar plus feasible battery support after intake capping and
charge/support overlap cancellation. This kW value is compared strictly with 1% of stack
nameplate in kW. Equality and smaller offers remain an explicit off region; infeasible raw
support cannot create fictional intent. A regression with a 120 MW PV bus proves that an
adequate sub-1%-share offer starts the stack, and a boundary regression proves that exactly
30 kW remains off.

This behavioral correction is explicit task evolution, not a silent bug fix:

```text
plant-sim                         0.5.2
commitment dynamics              electrolyser-commitment-alpha-v3
default commitment profile SHA   c6123ad4c30ef29445fa68e37d7f3d55bdeec05129d8535ab9dca2bc4715c4f9
V2 interface digest              6a4943d9350ba7c98c749b46a309f478b692c173cbb568809e34b5b6d5fee155
V2 checkpoint-task digest        127a63d3af8aa9e17b7da191aa9f9fae866ccfec203ac811475435fbb7273204
```

The unchanged interface digest is intentional: the 57 observation values and five action
fields did not change. The full checkpoint/task digest and profile hash did change because
their dynamics did. Commitment and all simulator tests passed locally (`39 passed` and
`131 passed`). A broader 205-test V2/simulator run also passed with the package metadata
test temporarily excluded: changing the local package version requires rebuilding the
editable distribution, and the isolated offline build cache did not contain `hatchling`.
The first offline rebuild failed for that exact reason. After fetching the locked build
dependency and rebuilding the editable package, the package-metadata test and the same
focused surface completed with **206 passed**. The complete repository regression remains
required before the complete comparator increment is committed.

## 2026-09-02 — Final pre-outcome causal-protocol hardening

### Scope and frozen-outcome boundary

A second source-level review was performed before executing the frozen development matrix.
No frozen 72-hour causal-controller outcome for the final core arms was generated or
opened during this review. The previously disclosed 24-hour seed-0 diagnostic and the
six-hour seed-3 CLI smoke remain contaminated development observations; this entry neither
reopens them nor promotes them to evidence. In particular, seed 3 is ineligible for any
future holdout under the current generator, regardless of horizon.

### Additional findings

The review found that the first repair was still not a clean forecast-information design:

- The analytic clear-sky arm and issued-aggregate arm shared physical planning code, but
  not temporal resolution. A direct contrast would combine forecast content with a smooth
  clear-sky integral versus coarse aggregate-band interpretation.
- Treating only wholly contained aggregate bands as available was conservative but could
  collapse a short or band-straddling deadline horizon to the one-physics-step floor. The
  resulting discontinuity was an estimator artifact, not evidence of physical scarcity.
- Issued mean and downside-band weighting were bundled, so a favorable difference could
  not be attributed to weather-conditioned mean information or to p10 use.
- The requested controller action was not retained. Outcomes alone therefore could not
  establish whether an information treatment ever changed the pre-guard request.
- Source identity was not yet enforced as a complete start/end audit. A beginning-of-run
  dirty-tree hash cannot detect source changes during a long execution, and a Boolean
  dirty flag cannot identify unreadable or omitted content.
- A controller failure could remain visible while a successful subset still supplied an
  aggregate-looking contrast, and the CLI could finish successfully despite an incomplete
  planned study.
- The CLI accepted arbitrary development seeds without requiring an explicit declaration
  that doing so consumes their future holdout status.
- Terminal methane alone was not a sufficient balance check. Delivery and energy intensity
  can both look better by liquidating initial H2, CO2, CH4, or battery energy.
- `dispatch_gap_kwh` and `request_application_gap_kwh` mix routing, equipment refusal, and
  deliberate charge/support overlap cancellation. Assigning either an automatic
  lower-is-better benefit sign would overstate their interpretation.
- The running-stack floor check originally consumed scarce solar as charge/support overlap
  before securing the direct complement. Because overlap substitutes for rather than adds
  to battery discharge, a physically feasible 600 kW direct-solar plus 600 kW
  battery-support command for the 1.2 MW stable floor was falsely rejected and all 600 kW
  solar was sent to storage.

### Controller and estimand remedies

The final pre-outcome design now has four ordered core arms: legacy A0, delivery-aware
fixed-climatology A1, otherwise identical issued-mean A2, and otherwise identical
risk-adjusted issued A3. The analytic clear-sky delivery controller remains available as
a non-core diagnostic. It causally integrates the configured `PlantConfig.solar`
half-sine from the current boundary time across sunrise, sunset, midnight, and multi-day
intervals; it never queries realized future weather.

All delivery-aware arms use one deficit, commitment, allocation, and action path. They
also share the complete availability-to-deadline-horizon and next-delivery transform. The
same availability source is evaluated over the next inferred delivery period, and
`clip(1 - equivalent_hours / period_hours, 0, 1)` sets carry between zero and one inferred
recurring delivery. The stale fixed 50% carry and clock-time-only horizon no longer exist.
Startup, cooldown, and minimum-down unavailability are removed before computing the
electrolyser's productive horizon.

For the causal information contrasts, A1--A3 share the same canonical 0--1, 1--6, 6--12,
and 12--24-hour bands and the same piecewise-constant overlap calculation. Every query
receives `overlap_duration * band_fraction`. A1's fraction is the analytic clear-sky
average over the full band multiplied by the frozen unconditional synthetic-prior
expectation `0.6803843775134744`. That constant is
`E[sigmoid(N(logit(0.70), 0.75²))]`, evaluated once with 128-node Gauss--Hermite
quadrature rather than fitted to episode truth. A2 uses the current issued mean, and A3
uses `0.50 * mean + 0.50 * p10`. Uniform-within-band disaggregation is explicit: it fixes
the horizon-collapse pathology and matches resolution across the causal arms, but it is
still an estimator assumption rather than observed sub-band PV timing.

This does not erase the first review's objection to prorating aggregates. That objection
was to silently treating an aggregate as sub-band truth in only one arm. The final design
records the approximation as an estimator assumption and applies the identical
piecewise-constant rule to fixed-prior and issued-information arms.

The planner retains the earlier feedstock-aware methanator cap, commitment-aware
productive time, zero-boundary-solar forward shares, sustainable battery support, and
downward closure of floating-point action-simplex residue. It now also secures
`max(0, running_floor - sustainable_support)` as direct solar before assigning residual
solar to storage, so the feasible direct-solar plus battery-support floor is not rejected.
A deterministic running-mode fixture is the required regression for that exact mechanism.

### Evaluator, provenance, and reporting remedies

Every attempted core arm is retained. Copy, reset, descriptor-integrity, computation,
output-validation, and action-application failures carry their phase, bounded exception,
partial metrics, completed time, and requested-action prefix. A retained controller
failure makes the complete core study ineligible: per-seed diagnostics remain visible,
but no aggregate contrast may be reported from a successful subset. That invalid
diagnostic artifact is still writable, and the CLI then exits nonzero.

Missing or duplicated arms, undeclared seed/controller runs, and full/exogenous manifest
mismatches are structural pairing failures instead. Evaluation aborts without returning
an artifact, and `write()` revalidates mutable results so such evidence cannot be
serialized. They are not mislabeled as controller failures.

Each run now retains the typed pre-guard request and controller-side invalid/clipping flag
at every decision, plus a canonical action-trace SHA-256. Pairwise mechanism reporting
aligns decision indices and separately reports solar-share L1 distance, battery-support
differences, invalid-flag disagreements, unequal-request counts, and whether the treatment
was expressed. The environment counts the invalid flag even after a policy adapter has
converted its raw vector to a finite action. A treatment that never changes a request is a
negative mechanism result, not evidence that forecasts or quantile bands have no value.

Source provenance now covers the reproducible full-index binary tracked diff plus every
non-ignored untracked path, kind, byte count, and content hash. It publishes whether the
identity is complete, captures that identity at both execution boundaries, and invalidates
the audit if the identity is incomplete or changes. Unlike a retained controller failure,
either source-audit condition is a fail-closed infrastructure abort: no result is returned
or written, and the CLI exits nonzero. Normal execution requires clean source at both
boundaries. `--allow-dirty-development` is an explicit forensic development exception and
cannot support a prospective claim.

The normal CLI seed allowlist is 0--2. Any other seed requires
`--acknowledge-seed-contamination`, which explicitly consumes its unseen status; it is not
a prospective-evaluation switch. Seed 3 remains contaminated from the earlier smoke
output.

Episode accounting now retains initial and terminal H2, CO2, and CH4 mass, battery SOC,
and battery stored energy, plus every inventory change. Accepted electrolyser, DAC, and
methanator energy remains separated and summed without double-counting battery support;
the per-delivered-kilogram ratio stays `null` when nothing is delivered. Human summaries
show no intensity mean unless every planned run defines it, and paired intensity aggregates
are suppressed unless every paired episode defines both values. Delivery and energy
intensity may be interpreted only together with service and the complete inventory change
vector. Battery discharge and both action-application gaps are raw diagnostics with no
benefit sign. Reset-time minimum SOC is initialized from the sampled state so an immediate
controller failure still has truthful partial accounting.

The additive metrics, summary-scope declaration, and pairing fields materially extend the
baseline evidence payload, so its result schema advances from 3 to 4 instead of silently
changing version 3. The causal controller artifact remains schema 5 and additionally
contains execution, action-trace, source-audit, and predeclared-contrast sections. Older
checked-in evidence is not rewritten.

### Validation gate

The implementation was split into two auditable commits: `04798fc` adds the causal
delivery-controller ladder and its deterministic mechanism suite; `8d210d6` adds the
failure-safe paired evaluator, evidence validation, provenance, accounting, CLI, and
regressions. The final focused controller/evaluator/CLI/V2 surface completed with
**165 passed**. After both code commits, the complete repository completed with
**657 passed** and one pre-existing Starlette/httpx deprecation warning in 12.41 seconds.
`git diff --check` was clean.

An independent final pre-outcome source audit found no remaining blocker after adversarial
regressions for duplicate-seed pseudo-replication, post-hoc contrast replacement, causal
evidence mutated into baseline shape, descriptor mutation, failed-arm survivor bias,
source-audit abort/no-write behavior, treatment-expression tracing, and the running-stack
direct-solar/support floor. The audit explicitly confirmed the A0--A3 order and exact
adjacent contrasts. Its remaining limitations are carried forward: three development
seeds per rate are descriptive only; the two rates require separately named artifacts;
piecewise-constant band disaggregation and the climatology are synthetic assumptions;
equal-representation RL/MPC, site calibration, and commercial evidence remain future work;
and the installed environment is identified by lock hash plus runtime metadata rather than
a complete installed-package snapshot.

The protocol and journal will be committed next so the worktree is clean before execution.
**No frozen 72-hour final-arm outcome or holdout was opened during implementation,
validation, or review.**

## 2026-09-02 — First causal controller development outcomes

### Execution and integrity

After commit `f6cb663` froze the protocol on a clean worktree, the two predeclared
development cells were executed separately: 35 and 42 kg CH4/h, seeds 0--2, 72 hours,
and the ordered A0--A3 controller ladder. Both strict-JSON artifacts completed with all
12 planned rollouts, valid start/end source audits, exact per-seed full and exogenous
pairing, all three adjacent contrasts, and expressed controller treatments. The 35 kg/h
artifact is 1,411,840 bytes with SHA-256
`2e1200c397c2aa5b23762a551d949e3787fd1bec5a73e7ee183fcd2d7b99fd55`; the 42 kg/h
artifact is 1,426,064 bytes with SHA-256
`e7b27f0269b2097b92bace6f435821b1088d8dc7548713546a65625cf7114de1`.

The result is negative and important. At 35 kg/h the legacy A0 mean fill was 97.72% with
57.3 kg unmet, while the climatology A1 achieved 84.55% with 389.2 kg unmet. At 42 kg/h
A0 achieved 84.97% with 454.5 kg unmet, while A1 achieved 81.43% with 561.5 kg unmet.
A1 was worse on fill in all six rate-seed pairs. It changed all 1,728 paired requests, so
the comparison is not a dormant-controller failure.

Issued mean information changed 694/864 decisions at 35 kg/h and 768/864 at 42 kg/h, but
A2 improved mean fill over A1 by only 0.20 and 0.19 percentage points and did not improve
service. Seed-level unmet-mass effects changed sign. The p10 mixture changed 711/864 and
769/864 decisions. A3 improved mean fill by 2.18 points at 35 kg/h but worsened mean
worst-commitment fill by 6.83 points; at 42 kg/h it was effectively neutral on fill and
increased forced stops. Neither incremental forecast hypothesis is supported.

### Failure mechanism

The controller family converts current mass deficits into instantaneous target power but
does not optimize a commitment/inventory trajectory. A0 cold-started once and never
forced-stopped. A1--A3 averaged 9.0--11.7 cold starts and 4.7--6.3 forced stops, ended
near the 15% battery reserve rather than A0's 91%, and curtailed 17.5--30.1 MWh rather
than A0's 3.3--6.4 MWh. At 35 kg/h they also ended with hydrogen stranded while CO2 was
nearly empty. The apparent 2.3--3.0 kWh/kg intensity improvement is not a benefit: it
coincides with worse service and liquidation of battery and feedstock endowments.

Exact action replay then localized the failure. In the 35 kg/h cell all delivery-aware
arms miss commitments 4, 8, and 12--the 06:00 pre-sunrise settlement on each complete
day--for every seed. They request no electrolyser, DAC, or battery support during the
first local 06:00--12:00 solar window. Across the first 12 elapsed hours, A1 curtails
22.367 of 28.620 MWh mean available solar while A0 curtails only 0.174 MWh. The shared
planner can target at most 426.3 kg: one 210 kg delivery, its 3% buffer, and one inferred
210 kg carry. It ignores that the productive solar window must provision three later
settlements before the next sunrise despite 3,000 kg of methane storage, and it never
uses the declared 12--24-hour forecast band. This is a structural horizon defect rather
than evidence against forecast value.

A read-only post-outcome replay instrumented every one-minute dispatch to test one
alternative explanation: solar curtailment coincident with battery discharge. The
per-rollout overlap lower bound was only 0.7--38.0 kWh, orders of magnitude below total
planner discharge and curtailment. The dominant defect is the short planning horizon,
compounded by commitment cycling and battery/feedstock coordination, not same-step bus
netting.

### Research decision

The current delivery-aware rule is rejected as a production baseline, and parameter
tuning is not an adequate next step. The next work package will construct a trajectory-
level model-based benchmark: first a separately labelled future-information best-known
feasible planner to establish that better sequences exist, then a causal deterministic
MPC under the public forecast and finally a risk-aware extension. RL will be evaluated
only after it has equal information and these comparators exist. The complete per-run
tables, paired effects, balances, artifact identity, limitations, and revised direction
are preserved in `CAUSAL_CONTROLLER_RESULTS.md`.

## 2026-09-02 — Two-sided controllability benchmark selected

No new simulator outcome was opened in this design step. The negative causal-controller
result leaves two quantities unidentified: a certified physical delivery ceiling and a
best known delivery that the exact five-field action interface can actually realize. A
single “oracle” label would conflate them, so the next benchmark is deliberately
two-sided.

The upper side will be a minute-resolution linear relaxation with perfect solar truth,
the exact delivery schedule, bus balance, battery efficiency/reserve, storage,
stoichiometry, component caps, and deadline ordering. It relaxes commitment, minimum
load, ramps, part-load penalty, and share-routing restrictions. An optimal solution is
therefore a delivery upper bound; an endowment-restoring variant is reported separately
and is not mislabeled as a bound on the current objective. The implementation is scoped
to fault-free episodes with initial energy above reserve. A bound is publishable only
after an optimal HiGHS status, independently validated residuals, exact solar-vector
replay, and an exact-rational box-Lagrangian certificate reconstructed from projected
solver marginals; the numerical primal objective is never itself called a certificate.

The lower side will be a privileged exact-simulator search whose selected schedule is
freshly replayed. The complete legacy trajectory is the mandatory incumbent, so the
reported best-known feasible trajectory cannot regress under the frozen lexicographic
score. It is not claimed optimal. Its action library, beam/search budget, ranking,
tie-breaking, and replay schema must be frozen in a protocol amendment before full
development outcomes are viewed.

Only if this bracket shows actionable exact-simulator headroom will a causal service MPC
be built. That controller will receive the same `ControllerObservationV2` boundary as
other online policies and only the current public forecast. The p10 aggregates are not
coherent scenario paths, so stochastic-MPC language is deferred. SciPy/HiGHS is selected
as a locked research dependency for sparse LP now and a possible MILP comparator later;
solver failures will remain explicit rather than falling back under an optimizer label.
The full pre-outcome design and proof obligations are recorded in
`PLANNING_BENCHMARK_PROTOCOL.md`.

### Split-boundary incident during implementation review

An independent code-review performance probe exceeded the intended unit-fixture-only
boundary and solved two previously unopened default-configuration cells: 40 kg CH4/h,
seed 0, at 24 and 72 hours. The reviewer used the fault-free synthetic weather and
commitment-alpha profile with `PlantConfig()`, `Constraints()`, and otherwise default
`EpisodeConfigV2`; it built and solved the relaxation directly through the optional
SciPy/HiGHS runtime. No file or result artifact was written, but terminal output exposed
the 24-hour 960 kg and 72-hour 2,880 kg certified ceilings, both with zero displayed gap.
The 72-hour model contained 56,176 variables, 21,604 equalities, and 43,200 inequalities;
build time was about 2.4 seconds and solve-plus-certificate time about 15.9 seconds on the
review host.

This is retained as an audit incident, not promoted as a finding. Both rate-40/seed-0
horizon cells are now explicitly development-contaminated and barred from prospective,
held-out, or commercial claims. No other new cell was authorized or opened. The incident
also tightened delegation instructions: implementation review must distinguish
unit-sized performance fixtures from any full-horizon outcome-producing solve.

The same review identified a proof-scope distinction that is now part of the protocol.
The exact-rational box-Lagrangian calculation certifies the serialized LP, but Python's
stepwise floating-point simulator can leave machine-precision residuals when its trace is
inserted into those rational equations. Until a per-operation interval error envelope is
derived and included in the relaxation, the artifact must say “dual-certified relaxation
bound with a numerically validated simulator bridge,” not “bit-exact simulator upper
bound.” This correction changes no optimizer equation; it prevents the strength of the
mathematical claim from exceeding the implementation proof.

## 2026-09-02 — Dual-certified delivery relaxation implemented

### Research instrument and mathematical boundary

Part I of the two-sided controllability benchmark is now implemented as an isolated
research instrument in `backend/src/plant_backend/rl/v2/upper_bound.py`. It is not a
controller and cannot enter the causal controller path. Given a complete fault-free V2
episode, it builds a sparse, physics-step linear program with perfect solar truth and the
declared delivery schedule. The model retains solar-bus balance, battery efficiencies and
reserve, battery-to-electrolyser-only support, component caps, storage capacities,
stoichiometry, production-before-settlement order, transient pre-conversion storage
headroom, and all energy/mass recurrences. It deliberately relaxes commitment state,
minimum load, ramps, part-load losses, action shares, dispatch priority, and simultaneous
battery charge/discharge exclusion. Its objective is cumulative delivered methane.

Every physical variable is represented by an affine finite box
`x = lower + width * z`, with `0 <= z <= 1`. This makes the complete normalized program
serializable and permits an independent box-Lagrangian calculation. SciPy/HiGHS supplies
only a candidate primal solution and marginal vector. Inequality marginals are projected
to the valid non-positive sign convention, then the Lagrangian minimum over every box is
evaluated with exact `Fraction.from_float` arithmetic over the stored IEEE-754
coefficients. The resulting delivery ceiling and requested-mass cap are converted back to
binary floating point only with checked upward rounding. The exact rational certificate
is compared directly with the numerical primal before any ceiling field is exposed.

The claim boundary is explicit in code and serialization. The certificate formally
applies to the encoded finite-box LP. Its bridge to the stepwise floating-point simulator
is numerically validated, not bit-exact certified. Failed solver, primal-validation,
objective-validation, certificate, or numerical-tightness paths suppress every ceiling
field and the nested certificate. The endowment-restoring solve is marked as a diagnostic,
not as the current task's upper bound.

### Provenance and adversarial hardening

Evidence-mode construction defaults to regenerating `sample_episode` from the episode's
seed, `EpisodeConfigV2`, and `PlantConfig`, then requiring exact canonical audit-manifest
equality. This covers the initial state, commitments, stream spawn keys, timing, weather
truth and forecast issues, clear-sky identity, and hashes. Complete checkpoint-contract
identity is recomputed against the current V2 runtime. Hand-built unit fixtures require
the explicit `require_canonical_episode=False` opt-out; both model and result then
serialize `noncanonical research fixture; not evidence-verified`.

The immutable model validates its block layout, affine bounds, solar digest, commitment
boxes and ordering, objective, physical limits, state/endowment boxes, row counts,
checkpoint digest, and episode manifest. Solve results retain all four acceptance
tolerances, canonical solver options, SciPy version, and the linked HiGHS version when the
SciPy build exposes it. They also disclose that git revision and dependency-lock digest
must be attached by the future evaluation runner. Therefore the current function can
produce a mathematically certified development result, but not by itself a publishable
benchmark artifact with a complete start/end source audit.

An independent adversarial review initially found nine issues: overclaiming the simulator
bridge, a nested ceiling on failed results, a rounded-float certificate gate, incomplete
solve provenance, incomplete episode/checkpoint verification, unexercised trace embedding,
tolerant invalid initial states, a misleading minute-resolution identity, and insufficient
model-construction invariants. Each material issue was either fixed or narrowed explicitly
to the defensible encoded-LP claim before this milestone. The final delta review added
direct regressions for standby embedding, post-certificate failure redaction, the exact-
rational comparison boundary, and full-runtime checkpoint validation. It also found two
documentation defects: the affine lower offset is now stated accurately, and the protocol
status now distinguishes disclosed contaminated probes from retained or prospective
outcomes.

### Verification and remaining limits

The proof-oriented suite includes hand-solvable zero-resource, abundant-resource,
battery, deadline, and storage-order cases; exact marginal-sign and rational-rounding
cases; malformed-input and solver-failure paths; deterministic solves; and actual
simulator traces for startup/standby with zero production, full-tank refusal,
charge/support cancellation, same-step production and settlement, sunrise/day wrap, the
active reserve boundary, and the guarded `PlantEnvV2.step_action` path. Each trace is
inserted through the model's affine coordinates and every LP row is recomputed
independently. Delivery-bearing cases, plus the explicit standby regression, also solve
the relaxation and check observed delivery below its certified ceiling.

After the final-review regressions, the focused suite completed with **66 passed**. The
complete repository completed with **679 passed, 2 skipped**, plus one pre-existing
Starlette/httpx deprecation warning, in 39.36 seconds. Python compilation,
dependency-lock validation, formatter checking, and `git diff --check` were clean. No new
full-horizon outcome-producing cell was opened; only unit-sized fixtures were solved. The
previously disclosed rate-40 probes and 35/42 development cells remain contaminated
exactly as recorded.

The implementation still does not provide a formal rounding-error enclosure for every
Python simulator operation, support scheduled faults, attach a start/end source audit, or
construct a realizable five-field action schedule. It therefore answers only the relaxed
physical side of the bracket. The next work package must freeze and implement the exact-
simulator feasible-planner search before any additional full-horizon outcome is opened.

## 2026-09-02 — Exact-simulator feasible search frozen before implementation

### Method selection

No new rollout, solver objective, terminal metric, or full-horizon planner outcome was
opened in this design step. Three independent read-only reviews compared template beam
search, rollout search, coordinate descent, discretized dynamic programming, MILP
projection, MCTS, and evolutionary search. The selected version-1 method is a
deterministic budgeted template beam followed by fixed block-coordinate polishing. Every
candidate crosses the real five-field `PlantEnvV2.step_action` boundary; the relaxed LP
is used only for one proposal template. The complete legacy schedule remains an external
incumbent and exact ties return legacy.

The primary beam is frozen at width 24 and advances four 15-minute decisions per edge.
This makes the one-hour cold-start precursor expressible while reducing the 72-hour tree
from 288 cuts to 72. Its 15 ordered semantic templates include legacy-live, the relaxed
witness projection, idle, charge, explicit standby/start offers, isolated H2/DAC/CH4
operations, feedstock recovery, and three nominal mixed-process loads. Each semantic
action is regenerated at the native decision boundary, routed against the maximum private
solar truth within that hold, and then subjected to the unchanged minute-level guard and
exact simulator.

The beam retains the A0 prefix plus one best state in every active electrolyser
commitment mode before global filling. Version 1 deliberately uses no coarse state bins:
only byte-identical complete same-depth state-signature payloads deduplicate. Intermediate
ranking includes next-deadline readiness so a startup/feedstock precursor is not judged
only by immediate production. Final selection is delivery-first, then service, and
lexicographic: delivered methane, exact commitment service, worst fill, terminal
feedstock potential, battery endowment, forced/aborted/cold starts, curtailment, then
charge-plus-discharge throughput. Shaped RL reward is excluded.

One coarse six-hour substitution sweep is run independently from A0 and the best beam
trace, followed by one one-hour sweep from their winner. All templates and blocks are
visited in frozen order, only strict final-score improvements are accepted, and unchanged
suffixes remain concrete actions. This gives at most 1,440 coordinate trials in a
72-hour cell. The final trace must reproduce exact canonical outcome bytes on a brand-new
reset; search state itself is never evidence.

### Clone-performance finding and safety response

A clone-only timing probe used the already contaminated default seed-0 configuration and
never stepped an episode or exposed an outcome. One hundred ordinary full-environment
deep copies took 4.490410 seconds (44.9 ms each). One hundred simulator copies took
0.550676 seconds (5.5 ms each). A prototype memo-sharing environment copy took 0.953945
seconds (9.5 ms each), while a simulator copy sharing only frozen configuration and its
immutable solar tape took 0.101059 seconds (1.0 ms each). Generic copying would make even
the reduced beam unnecessarily expensive.

The protocol therefore permits a structural-sharing clone only behind a strict proof
obligation: every operational object and mutable nested container must detach; only
validated immutable episode/configuration data may share. Equal-action behavior must
match ordinary deep copy byte for byte, sibling/parent mutation tests are mandatory, and
the optimization is rejected if a complete deterministic checkpoint cannot be
constructed. `EpisodeV2` is not treated as transitively immutable because its frozen
record contains a mutable stream-key dictionary.

### Integrity and claim boundary

The complete specification is now recorded in `FEASIBLE_PLANNER_PROTOCOL.md`: exact
template equations and order, 25,575 beam-edge ceiling, rankings, tie rules, mode lanes,
clone and state-signature contracts, polishing schedule, final score, throughput
definition, replay hashes, immutable result schema, 5,400-second/4-GiB emergency failure
limits, and pre-outcome adversarial tests. Core results remain explicitly not evidence-
ready until a later runner attaches clean start/end source and dependency-lock audits.

This planner is privileged and non-deployable. A feasible improvement would demonstrate
that the current interface admits operational value which a causal controller could try
to capture; it would not validate RL. Failure to improve would diagnose only this frozen
library/search budget, not global unreachability. The protocol remains restricted to the
already declared fault-free commitment-alpha development scope. No new full-horizon cell
was opened.

### Pre-commit adversarial protocol review

Two independent reviewers attacked the draft before it became the registered method.
They found three changes capable of altering or invalidating outcomes. First, the initial
share-closure rule could loop forever when rounded process shares summed one ULP above one
and the battery residual was already zero. Closure is now a proportional downward scale
with an eight-iteration ceiling and exact hex reference vectors. Second, eliminating
templates that shared only their first action was invalid because semantic templates can
diverge later in a four-decision macro. Every template edge is now completed before exact
state/trace deduplication. Third, terminal candidates could have been pruned by an
intermediate readiness rank after its purpose had ended. The terminal cut now streams all
generated children through the full final comparator.

The review also removed implementation discretion: coordinate trials use best improvement
against one immutable block-start trace; the A0 cursor and source-lineage grammar are
explicit; no-pending-deadline readiness is zero; result, settings, failure, comparison,
runner, hash-domain, counter, resource, clone-allowlist, and exact state-signature schemas
are enumerated; and an exact replayed `F>U` remains a valid feasible finding while
invalidating the upper bracket rather than being clamped or discarded. The relaxed guide
must come from exactly one preregistered Part-I solve and is bound by model, result, and
witness hashes. It cannot be selected among repeated degenerate witnesses.

Registered execution is now restricted to seeds 0--2 and rates 35/42 with default plant,
constraints, weights, weather/inventory configuration, and commitment profile. Their
configuration and checkpoint digests are frozen in the protocol. Clone review identified
the exact memo-sharing allowlist and confirmed that the outer `EpisodeV2`, checkpoint
mapping, health dictionaries, metrics, encoder, spaces, RNGs, components, ledgers,
registries, and event state must detach. A graph audit and ordinary-deepcopy equivalence
are mandatory. Exact-simulator tests may use at most eight explicitly noncanonical
decisions; longer sequence mechanics use outcome-free test doubles.

A final delta audit found four more specification-level defects. State deduplication had
conflicting keys in two sections; version 1 now uses the complete state payload alone
after every full macro, with source/action bytes only as survivor tie-breaks. The
no-improvement result had required every discarded search trace to equal A0 before it
could omit a duplicate finalist; omission now depends only on the final post-polish trace.
Early input/binding failures could not populate mandatory model identities, so those
fields are explicitly phase-dependent and nullable on failure. Finally, an external
watchdog could kill the child before a core result existed, but the wrapper had no
failure form; it now has an exact runner-failure envelope and permits a null core only in
that failed state.

The streaming-frontier memory proof was also reviewed. Rather than depend on a subtle
online-selection induction in version 1, the registered ceiling was conservatively raised
from 50 to 57 live environments. That directly accommodates 24 current parents, a union
of global top-24 plus six distinct mode winners and A0, the rolling A0 cursor, and one
transient child. Score extraction now names one authoritative source for every field and
requires exact agreement at duplicated simulator/metrics/ledger boundaries. The action
library and legacy-controller canonical payloads and hashes were frozen explicitly.

The last reproducibility audit then showed that naming canonical domains was insufficient:
two implementations could choose different root wrappers for the same trace. The
request/source/action, outcome, replay-binding, deterministic-counter, source-snapshot,
runtime, result, and runner-artifact preimages are now field-exact, including explicit
nested paths for every timing/resource exclusion. The core embeds the complete canonical
Part-I result, not only its digest, so a writer can independently prove that the reported
upper-bound scalar came from the bound result it hashes.

The protocol freeze itself is machine-bound to this documentation-only commit: settings
carry its future anchor commit and protocol-file SHA-256, require the anchor's frozen
parent and exact four documentation paths, and compare current bytes with the anchor.
Final score fields now map to exact metric/state attribute names, every state-signature
nested object has a frozen private-field projection, failure strings have one bounded
sanitizer, and the registered clock begins before episode/relaxation work. Registered
artifacts must be written outside Git until all cells finish, so one output cannot
invalidate the next cell's clean starting boundary.

These corrections were made without opening a solver objective or plant outcome. The
method is still pre-implementation at this boundary.

## 2026-09-02 — Feasible-planner evidence schema implemented

### Research purpose

The first implementation slice turns the frozen feasible-planner protocol into an
executable evidence contract before any registered search is allowed to run. This is a
deliberate ordering constraint: a favorable planner outcome must not be able to influence
which fields are retained, how ties are resolved, which failures disappear, or how the
relaxed upper bound is connected to the exact-simulator result.

### Implementation

`planner_schema.py` now defines immutable versioned records for canonical payloads,
action provenance, exact service fractions, terminal task scores, planner settings,
search and runner failures, deterministic diagnostics, replay evidence, incumbent
comparisons, core results, and runner artifacts. Every serialized payload has an exact
field set and a domain-separated SHA-256 preimage. Strict reconstruction functions reject
missing and additional JSON fields rather than silently accepting schema drift.

The final comparator implements the registered ten-field delivery-first ordering. Service
fractions use integer cross-products, binary64-sensitive fields are compared and bound by
their exact packed representation where identity matters, and A0 wins every complete-score
tie. A successful core can select a searched finalist only when its full score strictly
exceeds the freshly replayed incumbent. Request, source, action-record, outcome, replay,
settings, counter, result, source-snapshot, runtime, and artifact identities are connected
as one validated graph rather than a collection of independent hashes.

Registered development records receive additional fail-closed checks. The frozen default
plant configuration is re-hashed before its methanation ratios are used to recompute the
terminal methane-equivalent inventory from exact terminal truth. The embedded Part-I LP
result, witness, certificate, normalized residuals, solve contract, and guidance vectors
are structurally and numerically cross-bound to settings and episode identities. A small,
explicit noncanonical-fixture scope remains available for unit tests because detached
noncanonical JSON does not contain an arbitrary plant configuration from which the same
physical equivalence can be reconstructed.

Search diagnostics encode the registered edge, frontier, A0, replay, resource, and
coordinate schedules. Review found that final finalist presence cannot reveal whether the
pre-polish beam trace duplicated A0. The schema therefore admits exactly the two legal
coordinate totals implied by the frozen algorithm: one or two coarse origins followed by
one fine origin (1,260 or 1,440 trials at 72 hours), while retaining 1,440 as the ceiling.
It does not infer an execution fact that the persisted schema cannot prove.

Runner records enforce phase-dependent progress: a start-source failure cannot claim a
child or runtime observation, a child failure requires complete start provenance, an
end-source failure requires a validated core, and an artifact-serialization failure
requires the complete graph. Evidence readiness additionally requires a successful
privileged core, a valid physical bracket, identical clean start/end source snapshots,
identical dependency-lock hashes, and runtime resource methods matching the registered
settings. The future runner must still establish the external Git ancestry, working-file,
and actual lock-file facts; JSON validation alone cannot observe the filesystem that
produced it.

### Adversarial review and validation

Adversarial review closed cases in which a result could select A0 despite retaining a
strictly better finalist, invent terminal feedstock value, mark an incomplete/over-budget
search successful, admit arbitrary registered relaxation payloads, preserve invalid
controller flags on generated actions, or retain A0 after a failure without binding it to
the same settings and episode. Absolute POSIX and Windows paths are rejected in source
snapshots, failure phase/code progress is checked, and registered successful payloads must
carry the complete canonical identity graph.

The final independent gate found three further cases before commit. An internally
re-hashed successful LP graph could carry a primal--dual gap above the registered
tolerance and an arbitrary numerical objective; successful validation now requires the
gap to be at most `1e-5` kg and the objective to agree with the negative numerical primal
within the same frozen objective-tolerance construction used by Part I. A one-origin
polishing run could over-report accepted blocks under the two-origin ceiling; accepted
blocks are now bounded by the origin count derived from its legal attempted-trial total.
Source inventory paths now also reject traversal, Windows drive-relative/rooted forms,
and the dot path. Finally, a successful search must have observed at least one live
environment. Self-consistently re-hashed adversarial fixtures cover each rejection.

The combined pure schema, action, state, and upper-bound suite passed 164 tests. Formatting,
Python compilation, static type checking of the schema slice, and whitespace validation
also passed. These checks did not call the LP solver and did not open any new full-horizon
plant outcome. The registered six-cell development bracket therefore remains unopened.

## 2026-09-02 — Frozen planner action generation implemented

### Scope and architecture

The second implementation slice isolates privileged proposal generation from search and
simulation. `_planner_actions.py` cannot reset, clone, or step an environment. It owns
only the frozen 15-template catalogue, live legacy-action capture, the Part-I witness
projection, the shared future-truth router, binary64 closure, and conversion to immutable
`ActionRecordV1` provenance. This boundary makes it possible to test every proposal rule
without observing a plant rollout and prevents the relaxed LP from being mistaken for a
feasible controller.

The committed template order covers the legacy action, one relaxed-witness proposal,
idle/charge, warm and cold-start holds, stable/full hydrogen operation, isolated DAC and
methanation, joint feedstock production, and three clipped stoichiometric operating
levels. Every nontrivial template is regenerated at its real decision boundary. It uses
the maximum private solar truth over the exact remaining half-open hold, current active
component caps, current boundary battery capacity and energy, the registered reserve and
efficiency limits, and the common proportional binary64 closure. Generated actions are
sanitised and must remain bit-identical to the closed request before they can be recorded.

### Relaxation binding and tamper resistance

One `RelaxationBindingV1` is created from the sole Part-I model/result pair. Canonical
admission requires the exact concrete model and result types, model/result/witness hashes,
checkpoint and exogenous episode identities, task-upper-bound status, solver method,
frozen solve tolerances and empty options, runtime version/provenance-gap coherence, exact
solar truth, all four projection arrays, and the complete guidance payload. Synthetic
stand-ins are accepted only through an explicit noncanonical unit-fixture scope.

Review showed that an ordinary frozen dataclass was still forgeable by direct construction,
copying, replacement, or mutation through supplied payload objects. The final design uses
a private factory token and a process-local attestation registry. Full graph validation is
performed once at admission; every public consumer then verifies object identity plus a
compact seal over all scalar identities, exact binary64 contents of the five routing
vectors, and all canonical payload domains/hashes. A constructed, copied, replaced, stale,
or mutated binding cannot be used to generate evidence actions.

### Adversarial review and validation

The action review also corrected three behavioral risks: merely recording solver settings
without enforcing them, using nominal rather than boundary battery capacity in discharge
sustainability, and accepting stale or arbitrary projection vectors under otherwise valid
hash labels. Tests now independently cover signed zero and one-ULP identity changes,
half-open left-to-right projections, a within-hold sunrise maximum, each active projection
cap, degraded-capacity energy limitation and directed `nextafter`, proportional share
closure, every template/reference vector, repeated fresh legacy-controller instances,
descriptor drift, source lineage, and all binding-tamper paths.

An independent read-only delta review approved the layer with no remaining protocol defect.
Its 39 focused tests passed; the combined schema/action/state/upper-bound gate passed 164
tests. Formatting, compilation, and whitespace checks passed. No solver call, environment
step, or new full-horizon outcome was opened by this slice.

## 2026-09-02 — Exact planner state and structural-sharing clone implemented

### Scope and state identity

The third implementation slice creates the privileged boundary through which the future
planner may observe, compare, and copy exact simulator state. `_planner_state.py` is the
only planner module allowed to name private `PlantEnvV2` state. It provides the complete
same-depth signature, optimized environment cloning, clone-graph auditing, battery
throughput accumulation, and authoritative final-score extraction; it contains no search
policy.

The signature binds the manifest, exogenous episode, checkpoint, settings, action library,
macro depth, environment/simulator/metric counters, all simulator totals and truth,
component and commitment snapshots, the complete fault/event/delivery state, control
boundary state, every metrics dataclass field and history, observation-encoder memory, and
planner-owned throughput. The environment recomputes its audit manifest at capture time,
must retain the exact sampled weather tape used by the simulator, and must agree on every
redundant step counter. Values use the frozen serializer: native scalars, enum strings,
sorted string-key maps, ordered records, and hexadecimal finite binary64 values. No state
binning or approximate equality is introduced.

Terminal score extraction is intentionally redundant. It requires a complete horizon,
checks environment/simulator/metric counts, checks terminal H2, CO2, CH4, battery SOC, and
battery energy against truth with exact binary64 identity, and checks delivered methane
against the ledger. Only then does it construct the delivery-first score and terminal
methane-equivalent feedstock value from the active plant ratios. Throughput accumulates
bus-side battery charge and support energy separately in fixed left-to-right order.

### Clone implementation and isolation proof

The optimized clone recreates the outer episode and its stream-key dictionary, recursively
rebuilds every checkpoint dictionary, list, and tuple container, and uses `deepcopy` memo
sharing only for the exact frozen path list registered in the protocol. A direct audit of
the test fixture found 97 checkpoint containers (81 dictionaries, seven lists, and nine
tuples) and zero shared container identities. Alias topology and exact scalar types/values
are preserved, while cycles, unsupported values, non-finite floats, altered shapes, and
allowlist/checkpoint overlap fail closed.

Adversarial review discovered that path validation alone was insufficient: a mutable
metrics dictionary assigned to the allowlisted ledger-current path could poison the memo
and become shared by parent and child. Every non-null allowlist value now must have its
path-specific exact repository class and pass a transitive closure attestation containing
only finite scalars, enums, tuples, and frozen dataclasses. Strong process-local references
make cached attestations identity-safe against object-ID reuse. Standalone dictionaries,
operational aliases, and exact outer records containing list or frozenset drift are
rejected. Null paths add no memo entry, preserving the protocol's compatibility behavior.
The executable path list is re-hashed on every use and must retain the frozen allowlist
digest.

The same review found that Python `deepcopy` legitimately reuses a tuple when all of its
children are immutable, contrary to the protocol's stronger checkpoint-container identity
rule. Checkpoint tuples are therefore rebuilt explicitly and audited recursively instead
of relying on generic deepcopy behavior. Fault-free eligibility was also tightened from a
value-equivalent integral zero to the raw built-in integer `0`; booleans, floats, NumPy
integers, nonzero sequences, and nonempty registries are rejected rather than normalized.

### Validation and performance boundary

The 23 focused tests independently perturb every registered inventory, commitment timer,
previous-power field, ledger field, interval summary, metric/history field, event/log field,
encoder value, and throughput value. They mutate every known nested operational collection,
exercise checkpoint tuple aliases and allowlist poisoning, and advance optimized siblings
plus an ordinary deepcopy through the complete two-decision noncanonical fixture while
proving the parent and untouched sibling remain byte-identical.

The final independent gate approved the layer. A clone-only timing check on a reset default
72-hour environment (no step and no outcome inspection) measured approximately 35 ms for
the first transitive attestation and 26 ms after caching, with 48 immutable roots retained.
The focused suite passed 23 tests; formatting, compilation, strict bounded type checking,
and whitespace validation passed. All step-based tests remain explicitly noncanonical and
limited to two decisions, and the registered six-cell bracket remains unopened.

## 2026-09-02 — Deterministic feasible-planner search implemented

### Scope and separation from the plant runtime

The fourth implementation slice encodes the frozen search and polishing algorithm without
importing `PlantEnvV2`, the simulator, the Part-I solver, or the action generator. The
module instead depends on a typed runtime port whose candidate objects remain opaque. This
keeps the registered algorithm independently testable and makes the future plant-specific
runtime solely responsible for reset, cloning, exact stepping, fresh replay, resource
measurement, and release of live environments.

The primary pass expands the 15 semantic templates in their registered order over
four-decision macros. It preserves a separately generated A0 prefix at every nonterminal
cut, performs exact same-depth state deduplication only after a complete macro, and streams
candidates into a bounded reservoir rather than materialising all children. A cut is
assembled in the frozen order: unchanged A0, the best candidate for each of the six
commitment modes, then the globally ranked fill up to beam width 24. Terminal candidates
never pass through the intermediate rank: they are compared by the full delivery-first
score and deterministic action lineage, then their environment is released immediately.

The post-search pass performs exactly one left-to-right coarse sweep in 24-decision blocks
for A0 and, when request-distinct, the beam winner. It then performs one left-to-right
four-decision sweep from the better coarse result. Every block evaluates all 15 templates
against the same immutable block-start trace, accepts only a strict full-score improvement,
and retains the incumbent on equality. A complete request trace with A0 lineage receives
the registered equal-score priority only in final selection; it is not a general heuristic
bonus inside search.

### Exact identity, accounting, and ownership

All boundary comparisons of action records use canonical binary64-exact bytes rather than
Python dataclass equality. This distinction is operationally important because Python
considers positive and negative zero equal, while the frozen protocol treats their request
and provenance payloads as different evidence. The byte comparison protects A0 injection,
parent prefixes, terminal lineage, and coordinate fixed prefixes and suffixes. Same-state
dedup additionally requires identical state-derived rank fields and otherwise fails closed;
it cannot silently choose between inconsistent renderings of one claimed state.

Counter semantics now match the preregistration at the precise failure boundary. An edge is
attempted before expansion, but becomes completed and generated only after returning a
valid complete state signature. The resource checkpoint follows immediately while the
child is live and precedes rank or terminal extraction. A coordinate trial becomes
completed only after returning a structurally valid full trace; its resource checkpoint
then precedes fixed-region and lineage validation. Consequently a later validation failure
does not retroactively erase executed simulator work from the evidence counters.

Candidate ownership is explicit across every frontier transfer. Losing duplicates,
evicted reservoir entries, terminal candidates, abandoned next frontiers, and exceptional
paths are released exactly once. Insertion is protected even against a synthetic
`MemoryError` after a candidate has entered the exact-state table: the candidate is
released and its live table reference is cleared. A 500-candidate stress proxy confirmed
the conservative reservoir bound while keeping current parents separate, consistent with
the registered maximum of 57 live environments once the runtime's rolling A0 cursor and
one transient child are included.

### Adversarial review and validation

The first review found four material gaps: signed-zero aliases at lineage boundaries,
completion counters that were delayed until after rank evaluation, a potential ownership
leak during exceptional frontier insertion, and insufficient independent coverage of the
terminal comparator and two-macro search. Each was corrected before commit. Tests now
force terminal final-score selection to disagree with intermediate rank, verify the exact
A0/mode/global cut order and fill, inject failures on both sides of each accounting
boundary, and compare the algorithm with an independent exhaustive enumeration of all
reachable choices in a two-macro synthetic problem.

The final independent review approved the search layer with no remaining blocker. Its 28
focused tests passed, and the combined search, schema, action, state, and upper-bound gate
passed 195 tests. Formatting, Python compilation, strict bounded type checking, and
whitespace validation also passed. These are synthetic algorithm tests only: no real
environment was imported or stepped, no solver was called, and no registered or new
full-horizon outcome was inspected. The six-cell development bracket therefore remains
unopened.

## 2026-09-02 — Pre-runtime binding-attestation boundary exposed

The runtime API audit found one ordering hazard before plant integration: relaxation
bindings were fully re-attested only inside action generation. That was sufficient to
protect each generated action but could not enforce the protocol's stronger requirement
that a copied, constructed, replaced, or mutated binding fail before A0 generation begins.
The action layer now exposes a validation-only `require_attested_binding` function. It is
a direct delegate to the existing private identity, process-local registry, canonical
payload-identity, and full-seal checks; it exposes neither the registry nor its factory
token and performs no environment or solver operation.

An independent review approved the boundary and required the runtime tests to prove call
ordering with a forged binding and A0 spies, not merely validation behavior in isolation.
The action test accepts the exact factory object and rejects both a copy and a one-ULP
mutation. The focused action suite passed 40 tests and the combined search/schema/action/
state/upper-bound gate passed 196 tests. No plant step, solver call, or outcome inspection
was introduced.

The subsequent cross-module acceptance audit found another signed-zero boundary in the
result schema. A searched trace was allowed to retain `source_code=-1` only when its row
request equalled the corresponding A0 request, but that comparison used Python dictionary
equality and therefore collapsed `+0.0` and `-0.0`. The check now compares the canonical
request JSON bytes, which preserve the binary64 sign of zero. A regression rebuilds every
dependent request, outcome, replay-binding, comparison, and result identity around the
adversarial trace, proving that the rejection is semantic rather than caused by a stale
hash. This closes an evidence-acceptance hole that exact runtime replay by itself could
not repair.

The wider binary64 and canonical-identity audit then examined every remaining equality
operation in the planner schema, action generator, state boundary, and search. It found
four related schema risks. Replay-binding dictionaries could accept Python scalar aliases
such as `True == 1`; score-source integers could accept booleans or integral floats;
recomputed gain, residual-gap, fraction-closed, and A0 self-comparison fields could hide a
signed-zero substitution; and embedded relaxation witness/guidance objects used mapping
equality instead of their registered canonical bytes. The final result also compared its
recomputed bracket as a dataclass, inheriting the same float-equality problem.

All of those boundaries now validate canonical JSON or explicit binary64 identity. The
five integer metric sources require actual non-boolean integers, and the optional residual
gap is normalized back to a float consistently with the other comparison fields. The
tests rebuild dependent hashes and bindings around bool/int and signed-zero substitutions,
exercise every integer score source, and mutate both the embedded witness and guidance.
An independent review approved the resulting contract; the combined adjacent planner
gate passed 204 tests before the two final review hardenings. No solver, plant step, or
outcome was opened by this audit.

## 2026-09-03 — Planner runtime ownership core implemented

### Scope and architectural boundary

The fifth feasible-planner implementation slice establishes the plant-specific ownership
primitives needed by the previously committed plant-independent search. The new
`_planner_runtime.py` layer does not yet execute A0, expand a macro, replay a schedule, or
write evidence. It owns detached terminal/A0 records, live `PlantEnvV2` candidate handles,
logical environment reservations, exact wall/RSS sampling, and fail-closed lifecycle
errors. Keeping those responsibilities separate lets the next port translate the frozen
search protocol into real plant operations without placing environment internals in the
generic search algorithm or allowing the privileged planner to implement `ControllerV2`.

`TerminalSnapshotV1` cross-validates the complete action records against request, source,
and full-record canonical payloads; reconstitutes and canonically compares the final score;
binds every score component to the complete metrics, terminal truth, delivery audit, and
battery-throughput sources; recomputes methane-equivalent inventory; and reconstructs the
complete outcome payload. `A0GenerationV1` accepts only exact A0 source phase, code, and
controller binding, revalidates its nested terminal snapshot on every construction, and
requires one non-empty complete-state prefix per four-decision macro.

Each `PlantPlannerCandidateV1` owns exactly one reset environment and seals its environment
identity, canonical action-record trace, binary64 throughput, macro depth, and termination
flags. Validation checks all environment/simulator/metrics counters and rejects truncation,
stale depth, signed-zero substitution, metadata mutation, or reuse after release. Ordinary
release validates first. A separate cleanup-only disposal path returns the originally
attested environment and clears both strong references without endorsing corrupted metadata;
the future port must pair it with the independently retained guard token and preserve the
primary integrity exception.

The resource guard reserves capacity before construction, separately tracks unbound tokens
and exact environment identities, freezes its limits against subsequent settings mutation,
and enforces the registered 57-environment beam ceiling and two-environment non-beam ceiling.
The wall comparison is strict `>` against an injected monotonic start, Darwin RSS is treated
as bytes and other supported platforms as KiB, and unavailable/nonintegral measurements fail
closed. Binding is exception-safe, foreign/double tokens are rejected, and a normal close
reports any leaked reservation rather than silently clearing a success path.

### Adversarial corrections and retained port obligations

The first independent review rejected the initial core and demonstrated seven classes of
failure: an unbound terminal score, a self-consistent wrong A0 binding, limits mutable after
guard construction, admission of unreset or later-mutated candidates, incomplete trace
projection, reusable/suppressible guard cleanup, and clock failures assigned to the wrong
phase. All were corrected with targeted regressions. A second review found that A0 could
accept a nested snapshot mutated after construction; `A0GenerationV1` now reruns the complete
snapshot validator.

Two integration risks were then closed before this commit. Terminal snapshots now carry the
raw canonical `PlantConfig` SHA-256 and expose `validate_for_config`, which first revalidates
the full snapshot, requires the exact plant-config type, recomputes its canonical identity,
and binary64-compares both H2/CH4 and CO2/CH4 ratios. This prevents a caller from inventing
conversion factors that change terminal feedstock value. Candidate cleanup can now detach
the originally attested environment after the candidate validator itself fails, preventing
that failure from making its guard reservation unreachable.

The next real-plant port remains obligated to call binding attestation before A0, call
`validate_for_config(candidate.env.config)` before admitting any terminal snapshot to A0,
ranking, or replay, and retain each candidate's guard token outside mutable candidate
metadata. Cleanup must release that token/environment pair even on validation failure while
leaving the original failure authoritative.

### Verification and evidence boundary

The final independent read-only review approved the slice with no remaining core blocker.
The focused runtime suite passed **16 tests**. The combined runtime, schema, action, state,
search, and upper-bound gate passed **228 tests**. Formatting, strict targeted type checking,
Python compilation, and whitespace validation passed. Tests reset only the declared
two-decision noncanonical fixture and do not step it in this slice. No LP solver, registered
episode, new full-horizon plant rollout, terminal benchmark outcome, or holdout was opened.
The six preregistered 35/42 kg/h development cells therefore remain unopened.

## 2026-09-03 — Exact planner admission and episode-evidence binding

### Why this slice was required

Before connecting the plant-independent search algorithm to `PlantEnvV2`, I audited the
earliest point at which the planner could reject an invalid experiment. The result schema
already rejected unregistered evidence, but that happened after a rollout existed. For the
preregistered six-cell experiment this ordering was unacceptable: a wrong rate/seed pair,
an oversized fixture, or a settings object mutated after construction could expose an
outcome before the run was declared inadmissible. The runtime also had an attested
relaxation result and exogenous digest, but did not retain the exact Part-I episode-manifest
bytes needed to prove that every later reset regenerated the same stochastic episode.

### Implementation and architectural effect

The schema now provides one fail-closed `require_planner_admission` gate shared by both
pre-rollout runtime construction and final result validation. It reconstructs the complete
frozen settings record, including exact nested canonical-payload objects, and separates two
non-overlapping scopes: registered 72-hour development runs must use one of seeds 0–2,
the exact registered rate-specific episode/checkpoint pair, the canonical plant,
constraints, reward, cadence, and horizon; noncanonical unit fixtures must explicitly deny
canonical verification and may contain at most eight decisions. The raw canonical-JSON
SHA-256 helper makes the pre-existing static-configuration hashing rule explicit and avoids
confusing it with the planner's domain-separated evidence hashes.

The factory-only relaxation binding now seals the exact canonical episode-manifest JSON and
its raw SHA-256 alongside the exogenous digest. Construction parses and canonicalizes those
bytes and checks their embedded exogenous identity; every later attestation rechecks the
bytes, digest, cross-binding, and binding seal. The frozen relaxation-guidance payload was
deliberately left unchanged, so this strengthens the process-local runtime handoff without
silently revising the preregistered public evidence schema.

### Adversarial checks and evidence boundary

Regressions cover detached nested settings, illicit post-construction mutation, boolean
seed aliases, canonical/noncanonical scope confusion, oversized fixtures, unregistered
seeds, mismatched rate/checkpoint pairs, noncanonical manifest bytes, and manifests bound to
the wrong exogenous episode. The schema and action suites pass **98 tests** together; strict
targeted type checking, formatting, compilation, and whitespace validation are performed
again at the commit boundary. These checks instantiate only structural two-decision
fixtures. No environment was stepped, no optimisation solver was called, and no registered
or full-horizon outcome was inspected. The six preregistered development cells remain
unopened.

## 2026-09-03 — Exact reset, stepping, ranking, and terminal runtime completed

### Trusted pre-rollout identity

The plant runtime now has a single detached `PlannerRuntimeAdmissionV1` constructed before
any `PlantEnvV2` reset. Admission requires exact static input types and identities, the
attested Part-I relaxation binding, the registered/noncanonical scope gate, an independently
regenerated episode from configuration plus seed, the authoritative rebuilt checkpoint
contract, and the exact physics-step solar truth used by the relaxation. It stores canonical
episode, checkpoint, and immutable environment-manifest bytes and their hashes. Revalidation
reconstructs these contracts rather than trusting frozen Python objects, so a low-level
mutation or a digest recomputed over changed but semantically inadmissible bytes fails.

Every fresh reset is checked against that admission before it can be stepped. The check
covers the exact sampled episode and seed; static configuration hashes; rebuilt and cached
checkpoint bytes; commitment-profile identity; horizon, cadence, simulator, metric, and
environment counters; weather-tape identity and coverage; every solar-truth coordinate;
the fault-free registry; initial control boundary and previous-interval state; initial
metric values; frozen delivery schedule and zero accounting; and the complete immutable
environment manifest. This creates the concrete bridge from Part-I upper-bound evidence to
each exact-simulator branch without learning episode identity from the branch being checked.

### Exact plant operations

`step_action_record` reconstructs each exact action record, enforces its zero-based decision
index, applies the five requested values and stored controller-invalid flag through the real
`PlantEnvV2.step_action` safety/dynamics path, checks the exact decision and physics-step
transition including final partial holds, rejects truncation or post-terminal use, and updates
the shared left-to-right battery-throughput accumulator only after a successful step.

Intermediate rank extraction now implements the frozen next-deadline readiness formula from
authoritative simulator truth, delivery ledger, active constraint cap, exact commitment
snapshot, episode metrics, and planner throughput. Source counters and physical values are
validated before arithmetic, including bool/int alias rejection, finiteness, non-negativity,
positive conversion constants and remaining time, and exact horizon agreement.

At true termination, `capture_terminal_snapshot` detaches the complete request/source/action
trace, metrics, simulator truth, telemetry, ordered delivery audit, task score, throughput,
counts, termination flags, and outcome fingerprint. The existing terminal validator binds
every score component back to its authoritative source and the exact plant conversion ratios.
The runtime now additionally rejects subclasses of protocol records and payloads, preventing
an object with valid-looking fields from overriding `to_dict()` or `value()` during evidence
construction or later revalidation.

### Ownership cleanup and adversarial review

`PlannerCandidateRegistryV1` retains the authoritative candidate, reservation token, and
original environment outside mutable candidate metadata. Normal validation still detects a
swapped environment, altered trace, throughput, depth, flags, or counters; cleanup deliberately
does not re-endorse those fields and releases the original guard lease even after candidate
corruption. Counter checks also reject non-integral aliases and any state beyond the declared
horizon. This closes the remaining lifecycle obligation identified in the prior runtime-core
review and gives the forthcoming concrete search port one cleanup authority.

The focused runtime suite passes **21 tests**, and the adjacent runtime/schema/action gate
passes **119 tests**. The full repository passes **849 tests with 2 skips**; the sole warning is
the already-known Starlette `TestClient` deprecation. Formatting, strict targeted type checking,
Python compilation, and whitespace validation pass. The tests reset and step only an explicit
two-decision noncanonical exact-simulator fixture built around a structural relaxation witness;
they call no optimisation solver. No registered seed/rate cell, 72-hour trajectory, new
full-horizon terminal result, or holdout outcome was opened. The six preregistered development
cells remain unopened.

## 2026-09-03 — Exact A0 replay and concrete plant search port implemented

### End-to-end control path now connected

The sixth feasible-planner slice connects the previously separate components without
weakening their boundaries. A new private `_planner_port.py` implements the generic
`PlannerSearchPortV1` protocol over real `PlantEnvV2` instances. The generic search still
owns ordering, beam retention, exact duplicate handling, terminal selection, and coordinate
sweep policy; the port alone owns plant resets, structural clones, template regeneration,
exact stepping, state/rank extraction, terminal capture, replay, and resource leases. This
keeps the privileged perfect-future planner out of the causal `ControllerV2` interface.

`PlannerPlantInputsV1` now cross-binds exact plant, episode, constraint, and reward objects
to a reconstructed runtime admission before any reset. Every fresh environment reserves a
logical slot before construction, is bound to that reservation before reset, regenerates the
admitted seed, passes the complete reset validator, and crosses wall/RSS checks. Failed
construction, reset, validation, controller, clone, action generation, step, signature, or
terminal capture releases the original reservation while preserving the primary exception.
The guard and planner must also share the exact same counter object, preventing resource and
algorithm counters from being split across two superficially equal records.

### A0 generation, replay, and rolling cursor

A0 is generated by one persistent exact `LegacyHeuristicControllerV2`. Its descriptor is
compared with the admitted frozen canonical payload before reset, after controller reset, and
after every computation. Each decision receives only the forecast-free controller
observation plus the exact constraints, plant configuration, and commitment profile; the
returned exact `ControlDecision`, five requested values, and invalid flag are captured before
being applied through `step_action`. Complete state-signature bytes are retained after every
four decisions and at the final partial macro.

The detached schedule is then applied from a second fresh reset. Every replay decision gets a
resource check, every macro state must be byte-identical to generation, and terminal request,
metrics, simulator truth, telemetry, ordered delivery audit, score, throughput, counts,
termination flags, and outcome fingerprint must match exactly. The replay snapshot—not the
generation environment—is exposed as A0 evidence. A third fresh reset becomes the rolling A0
cursor during beam search; it validates each stored replay prefix, injects only a structural
clone at nonterminal cuts, and releases itself at the terminal cut while the evidentiary A0
remains detached and external to pruning.

### Beam, coordinate search, replay, and ownership

Primary expansion now reserves and structurally clones each registered parent, regenerates
the selected semantic template at every evolving decision, applies the resulting exact
record, and admits a candidate only after the full macro succeeds. Candidate state bytes,
deadline-readiness rank, action lineage, and terminal trace all pass through the audited
runtime functions. The shared registry is exception-safe and keeps candidate, original
environment, and guard token together outside mutable candidate metadata.

Coordinate trials require all beam environments to be released, start from a fresh admitted
reset, apply stored prefix and suffix records unchanged, regenerate the selected template
only inside the exact frozen 24- or 4-decision block, and expose their source phase to the
subsequent resource checkpoint. Fresh finalist replay similarly refuses outstanding search
or coordinate ownership. Before replay, every record is reconstructed and its semantic source
binding is checked: A0 and `legacy_live` bind the A0 descriptor, projection binds the exact
relaxation guidance, and templates 2–14 bind the frozen action library. Replay validates
behavioral bytes rather than pretending to rediscover source lineage.

### Adversarial tests, findings, and remaining obligations

The real two-decision integration executes A0 generation/replay, all 15 exact plant template
edges, terminal selection, both coordinate stages, and fresh finalist replay. A separate
five-decision integration crosses a nonterminal cut, proves rolling-cursor state identity and
A0 clone injection, exercises live frontier retention, and releases every terminal and parent
candidate. Direct coordinate fixtures separately preserve an unchanged suffix and unchanged
prefix. A controller whose descriptor mutates immediately after computation is rejected before
its action can step the plant, with no leaked environment. A record carrying a syntactically
valid but semantically wrong source hash is rejected before fresh replay. The tests also drove
an additional registry correction: candidate/environment insertion is now rolled back if the
second ownership index cannot be installed.

The focused runtime/port suite passes **26 tests**; the adjacent planner schema, action,
state, search, runtime, and upper-bound gate passes **241 tests**. The full repository passes
**854 tests with 2 skips** and only the existing Starlette `TestClient` deprecation warning.
Formatting, strict protocol-conformance type checking, Python compilation, and whitespace
validation pass. All new plant outcomes are explicit noncanonical fixtures of two or five
decisions, safely below the eight-decision preregistration limit, and their relaxation inputs
are structural test witnesses rather than solver results.

This slice does not yet translate failures into the public result schema, assemble the
`A0 <= F <= U` bracket/result graph, capture start/end source provenance, isolate a worker,
or write evidence externally. The remaining adversarial gate must also cover exact startup,
settlement, sunrise, charge/discharge-throughput, and final-partial-hold replay cases before
code freeze. No optimiser invocation, registered rate/seed outcome, 72-hour trajectory, or
holdout was opened; the six preregistered development cells remain unopened.

## 2026-09-03 — Replay evidence and fail-closed planner core assembled

### Evidence boundary and selection

The seventh feasible-planner slice adds the private `_planner_core.py` orchestration
boundary. Search states and terminal candidates still cannot become evidence directly.
`build_replay_evidence` reconstructs and revalidates the admitted plant context, validates
both exact terminal snapshots against the same plant configuration, requires generation and
fresh-replay outcome bytes plus complete action-record bytes to agree, and builds the frozen
settings/guidance/manifest/exogenous/checkpoint replay binding. The resulting
`ReplayEvidenceV1` contains only detached immutable records and canonical payloads.

`execute_feasible_planner` now performs the complete admitted control flow: exact A0
generation and independent replay; primary macro beam; the frozen coarse/fine coordinate
sweeps; one fresh replay only when the post-fine concrete request bytes differ from A0; exact
ten-field score selection; and construction of the Part-I incumbent comparison. A0 wins every
complete-score tie. A byte-identical post-fine trace is omitted and reuses the already
verified A0 replay, while a distinct searched finalist remains recorded even if A0 ultimately
wins. The bracket reads the non-null dual-certified ceiling and the frozen optimality
tolerance from the exact embedded relaxation-result payload. It never clamps a bridge
violation or presents the relaxation witness as a feasible schedule.

The successful result embeds the complete validated relaxation graph, settings, episode and
checkpoint identities, both applicable replay records, deterministic counters, dynamic
resource observations, and the immutable comparison. It is canonically serialized, checked
against the final core resource boundary, reconstructed with the sampled dynamic diagnostics,
serialized again, and checked once more. Source provenance is intentionally still absent:
the core truthfully retains `evidence_ready=False` and the declared provenance gap for the
external runner to close.

### Stable failures and lifecycle findings

Every operational exception is now translated into a frozen phase/code/message/counter
record instead of escaping as an implementation-specific exception. Runtime resource errors
retain their exact observed value; search-layer errors retain their stable code; memory
failures have a dedicated code; and remaining exceptions use a phase-specific fail-closed
default. A failed result has no selected role, feasible-planner value, searched finalist, or
comparison. A completely replay-verified A0 may remain diagnostic after a later search
failure, but partial A0 work is never published. Inputs that fail revalidation publish no
settings, manifest, relaxation, or A0 identity rather than laundering values that were never
admitted.

Port and guard cleanup runs on success and failure. The original integrity error remains
authoritative if cleanup also fails. A clean success is returned only after the search port
has released every candidate and rolling cursor, the resource guard has observed no remaining
reservation, and the complete result graph has passed its own constructor plus canonical
serialization. Timing samples now require an explicit failure phase: A0 replay, finalist
replay, and result serialization can no longer turn a backwards or broken clock into a
misleading `input_validation` failure. The generic coordinate layer also carries the exact
coarse or fine stage through algorithm, memory, and otherwise untyped implementation errors.

### Verification and evidence boundary

New exact-plant regressions exercise successful core assembly, outcome-byte parity, selection
and bracket construction, finalist replay-counter agreement, search-failure translation with
retained verified A0, pre-admission input mutation with null identities, partial-A0 failure,
A0 descriptor mutation, and replay-clock phase attribution. An outcome-free algorithmic test
proves that a failure beginning in the fine sweep is labeled `fine_polish`, not
`coarse_polish`. The focused search/runtime suite passes **60 tests**; the adjacent planner
schema, action, state, search, runtime, and upper-bound gate passes **245 tests**. The full
repository passes **860 tests with 2 skips** and only the existing Starlette `TestClient`
deprecation warning. Formatting, strict targeted type checking, Python compilation, and
whitespace validation pass.

All newly observed plant outcomes remain the explicit two-decision noncanonical fixture and
its structural no-solver relaxation witness. No LP solver was called, no registered seed/rate
cell or 72-hour trajectory was opened, and no holdout was inspected. The next slice must put
this core behind a fresh child process, collect exact clean start/end source and `uv.lock`
boundaries, record runtime versions, apply the external 5,400-second watchdog, and atomically
write then reopen a validated artifact outside the repository. The remaining physical
boundary regressions must pass before the registered six-cell code freeze.

## 2026-09-03 — Source provenance and validating external publication

### Complete boundary capture

The eighth feasible-planner slice adds `_planner_evidence.py`, still without executing a
planner child or optimisation solver. `capture_source_boundary` delegates repository state
collection to the existing audited `evaluation.collect_provenance` implementation, but admits
only its complete content identity. The planner payload deliberately removes repository
absolute paths and Git command arguments, preserves the full 40-hex revision, binary tracked
diff identity and byte count, path-sorted untracked file/symlink identities, dirty flag, and
complete worktree-content identity, then reconstructs the exact
`planner-source-snapshot-v1` domain record.

The dependency lock is not trusted through the collector alone. The boundary resolves the
reported repository, reads repository-root `uv.lock` directly, computes its raw SHA-256, and
requires both the collector path and digest to match those bytes. Callers identify whether the
sample is the start or end boundary, so an unavailable end audit cannot be mislabeled as a
start failure. Registered execution can require a clean starting snapshot before any child or
protected outcome exists; the later end sample need only be complete, allowing a genuine
changed-source comparison to remain an explicit non-evidence-ready artifact rather than an
invented partial snapshot.

`capture_runtime_versions` fills every frozen runtime field: Python version,
implementation/compiler, NumPy, SciPy, linked HiGHS when exposed, Gymnasium, OS release and
version, machine/processor, pointer width, byte order, and the exact clock/RSS methods frozen
in planner settings. Completed artifact assembly checks those resource methods against the
core settings and checks the parent-linked HiGHS identity against the solver settings embedded
by the child. This prevents two individually well-formed records from claiming one execution
while referring to different numerical runtimes.

### Artifact state machine and writer

The layer constructs both normal completed wrappers and phase-specific runner failures. An
ordinary failed core remains embedded when both runner boundaries complete; runner failure is
reserved for unavailable source, killed/invalid children, unavailable end source, or artifact
serialization. Evidence readiness is computed only for a successful canonical privileged
core with a valid bracket, clean byte-identical source snapshots, equal lock hashes, and
matching resource methods. The schema recomputes this eligibility and the complete artifact
digest independently.

`write_planner_artifact` is the only publication path. It requires an absolute destination
whose resolved path lies outside the resolved Git repository and whose parent already exists.
It serializes only after strict detached graph reconstruction, writes one uniquely named
temporary sibling, flushes and `fsync`s the file, atomically replaces the destination,
`fsync`s the containing directory, then reads the exact bytes back. Reopened bytes must be
canonical UTF-8 JSON, must rebuild through `planner_artifact_from_dict`, and must reproduce the
same artifact SHA-256. A leftover temporary file is removed on any pre-replace failure.

### Verification and unchanged evidence boundary

Six new tests cover field-exact path-free source capture, independent raw-lock verification,
correct end-boundary failure attribution, complete runtime fields, a normal runner wrapper
around an ordinary failed core, atomic reopen/revalidation, temporary-file cleanup, and
rejection of relative or in-repository destinations. The evidence plus schema gate passes
**63 tests**. The complete repository passes **866 tests with 2 skips** and the one known
Starlette `TestClient` deprecation warning. Formatting, strict targeted type checking,
compilation, and whitespace validation pass; strict type checking uses a disabled cache to
avoid a tool-internal serializer defect observed with the host mypy cache.

No environment was stepped by this slice, no solver was invoked, and no registered episode or
holdout outcome was opened. The six preregistered cells remain untouched. The next commit will
add the minimal registered-cell child entry point and a parent process that samples start
provenance before spawning, enforces the external 5,400-second watchdog, validates canonical
child output, captures the end boundary after core serialization, and hands the resulting
wrapper to this writer.

## 2026-09-03 — Registered planner isolated behind a bounded evidence runner

### Fresh worker and exact preparation path

The ninth feasible-planner slice adds the private `_planner_worker.py` entry point and the
public `plant_backend.rl.v2.planner` parent runner. The worker records
`time.perf_counter_ns()` as the first statement of its entry point, admits only the frozen
35/42 kg/h and seed 0--2 matrix, constructs the canonical 72-hour commitment-alpha task,
samples the episode once, constructs the unconstrained Part-I relaxation once, and invokes
`solve_delivery_relaxation` exactly once with the frozen tolerances and empty solver-options
mapping. A non-successful solve is rejected explicitly before guidance can be bound.

Preparation failures before complete runtime admission now use
`build_unadmitted_failure_result`. They produce a schema-valid privileged core failure with
zero deterministic counters, no invented episode/settings identities, and a stable
`input_validation` or `relaxation_binding` phase. Once the model result, witness and guidance
are cross-bound, the worker constructs the frozen settings and runtime admission and hands the
same initial clock sample into `execute_feasible_planner`. The worker emits exactly one
canonical JSON core object on stdout and strictly reconstructs it before emission; an
unexpected failure outside the ordinary core state machine exits through a dedicated protocol
failure rather than emitting a partial object.

### Parent watchdog, provenance order and publication

`run_registered_planner_cell` validates the requested registered cell and resolved external
destination before protected execution. It then captures a clean complete start-source and raw
lock boundary plus the parent runtime identity before spawning the worker with the current
Python executable. The end boundary is sampled only after a child has exited successfully,
stdout has been parsed as one exact canonical JSON object, the complete core graph has rebuilt,
and the requested seed/rate/scope bindings have been checked. A normal core failure is retained
inside an otherwise completed runner wrapper; killed, nonzero, malformed or unavailable
children have no invented core; an end-source failure retains the already validated child
core. Completed artifacts and runner failures use the same atomic external validating writer.

The external watchdog starts immediately before process creation and subtracts process-startup
time from the 5,400-second communication allowance. Review exposed three cases that the first
draft did not independently close: non-finite limits, a backwards injected monotonic clock,
and a child abstraction that ignored its timeout and returned after the deadline. The final
runner rejects all three, samples completion time after communication, and converts abnormal
spawn/communication exceptions into `child_protocol_error`. Timeout and clock-error cleanup
kill and reap the child best-effort without allowing a secondary cleanup exception to replace
the primary evidence failure. The production path remains the frozen `subprocess.Popen` plus
`time.perf_counter_ns` implementation.

The Makefile exposes one deliberately narrow, dependency-frozen operational target:
`make v2-planner DELIVERY_RATE=<35|42> SEED=<0|1|2> OUTPUT=<absolute external JSON>`.
The external output remains mandatory so one result cannot dirty the next cell's start-source
boundary. It uses `uv run --frozen`, and the CLI returns success only for an evidence-ready
artifact; a valid core whose source boundaries changed remains published but exits nonzero and
is labelled `completed-not-evidence-ready`. A dry command expansion was inspected for rate 42,
seed 2; it did not start a worker.

### Verification and protected-outcome boundary

The evidence/runtime focus passes **47 tests**. The complete schema, action, state, search,
runtime, evidence and upper-bound gate passes **263 tests**. The full repository passes
**876 tests with 2 skips** and only the existing Starlette `TestClient` deprecation warning.
Formatting, strict type checking of all four runner/core source modules and their evidence
test, Python compilation, dependency-lock validation, Makefile dry expansion, and whitespace
validation pass. The first lock check could not initialize the sandboxed host `uv` cache; the
approved host-cache rerun resolved the existing 92-package lock without changing its bytes.

Every new worker test either rejects an unregistered rate before episode sampling or replaces
sampling, relaxation construction, the solver, binding, admission, and the exact planner with
test doubles. Parent tests supply synthetic canonical failed cores and fake processes. No test
invoked the real relaxation solver, sampled a registered episode for execution, advanced a
registered plant, or observed a new full-horizon outcome. The six declared 72-hour development
cells remain unopened. The remaining pre-freeze work is the exact-plant boundary matrix for
startup, settlement, sunrise, charge/discharge throughput, and a final partial macro, followed
by a clean committed-source audit before any registered worker is allowed to run.

## 2026-09-03 — Feasible-planner pre-freeze physical boundary closed

### Exact multi-boundary replay

The final pre-outcome plant slice adds a seven-decision noncanonical integration beginning at
local 05:45. Its first four-decision macro selects the frozen full-stack template, crosses the
06:00 solar boundary, and accumulates a real cold-start transition and startup energy. Its
second macro contains only the three remaining decisions and selects the charge template. The
single delivery settles at the exact terminal physics step 105. The terminal candidate is
released before a fresh reset replays all seven actions; request and action-record traces,
metrics, simulator truth, telemetry, ordered settlement audit, score, throughput, and complete
outcome fingerprints match through the existing replay validator. This combines startup,
sunrise, settlement, and final-partial-macro behavior in the same exact simulator trajectory.

A separate two-decision noncanonical sunset fixture exercises physical charge followed by
discharge. The first charge-template decision increases stored battery energy. The second
full-stack decision reduces it, records positive bus-side discharge, and leaves the planner's
throughput larger than discharge alone because the preceding charge is included. A fresh reset
replays the same binary64 throughput. The fixture also verifies that omitted, appended,
reordered, and behaviorally changed action traces fail before they can be equated to the
original terminal snapshot.

All temporary exact environments are released through the authoritative guard even if an
assertion or replay fails. These are seven- and two-decision fixtures, below the frozen
eight-decision limit; neither uses a registered horizon or evidence scope.

### Final isolation and audit map

An explicit regression now passes a privileged planner artifact to the causal controller
runner. It is rejected for lacking the typed `ControllerV2` contract before environment reset,
episode sampling, or a plant decision. This turns architectural separation into executable
evidence and closes item 17 of the frozen checklist.

`FEASIBLE_PLANNER_PREFREEZE_AUDIT.md` maps all 17 pre-outcome obligations to maintained test
names and records the two physical fixture designs, development-only matrix, freeze commands,
and post-freeze execution discipline. A combined strict type check across all ten planner
source modules exposed two redundant casts and several deliberately dynamic reconstruction
calls that had been hidden when modules were checked separately. The redundant casts were
removed; object-valued source payloads now receive an explicit mapping guard; and the exact
schema reconstruction sites carry narrow argument-expansion suppressions. The complete planner
source set now passes one strict check. The dynamic runtime fixture module remains execution-
and compilation-tested rather than being presented as strictly typed production source.

### Candidate-freeze verification and unchanged outcome boundary

The complete planner schema/action/state/search/runtime/evidence/upper-bound gate passes
**266 tests**. The full repository passes **879 tests with 2 skips** and only the existing
Starlette `TestClient` deprecation warning. Formatting, combined strict planner-source type
checking, Python compilation, dependency-lock validation, and whitespace validation pass on
the candidate content.

No registered child was spawned, no valid registered worker request was made, and no new
35/42 kg/h 72-hour plant or relaxation result was observed. Tiny upper-bound tests continue to
solve only their declared hand-solvable noncanonical LP fixtures. The six protected development
cells remain unopened. The next action is to commit this audit slice, reproduce every required
gate from the resulting clean revision, and only then execute the six cells without modifying
source between them.

## 2026-09-03 — First registered planner cell failed closed

### Frozen execution and retained artifact

The candidate audit slice was committed as
`0059626313866d7a2bda2a44f64926a757a9f5d6`. From that clean revision the complete
pre-freeze gate was reproduced: the dependency lock remained unchanged; the focused planner
gate passed 266 tests; the complete repository passed 879 tests with 2 skips and the one known
Starlette `TestClient` deprecation warning; formatting, the combined strict type check,
compilation, and whitespace validation all passed. The first registered development cell was
then invoked exactly once:

```text
make v2-planner DELIVERY_RATE=35 SEED=0 \
  OUTPUT=/private/tmp/plantanagent-planner-0059626/rate35-seed0.json
```

The parent runner completed normally and atomically published a schema-valid failed artifact.
Its start and end source snapshots are clean and byte-identical at the frozen revision. Both
snapshots report raw `uv.lock` SHA-256
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.
The artifact's embedded `artifact_sha256` is
`9323d4c97205706d7c8a8c03f14734bc502f1056386d2ca7378b601a9b456ed1`; the exact
2,182,072-byte file has raw SHA-256
`e1f621689c176ccd50b5433a2df14d41c963f11618f70be5eb3af4821a5f95d7`.
These are intentionally different identities: the former is the schema's domain-separated
payload digest and the latter hashes the serialized file bytes. `runner_failure` is null, so
the failure belongs to the admitted planner core rather than process, watchdog, provenance, or
publication infrastructure.

### Observed failure

Exact A0 generation and fresh replay both completed all 288 decisions and 4,320 one-minute
physics steps. The primary beam search then stopped during its fourth macro-depth region after
961 attempted edges and 960 completed/generated edges. It retained 85 frontier states, had
discarded 133 exact duplicates, and reported a peak frontier of 24 and peak live-environment
count of 51. The admitted failure is:

```text
phase: beam_search
code: search_incomplete
message: action shares must have a binary64 sum no greater than one
```

The core elapsed time was 65,627,813,708 ns and peak reported RSS was 292,945,920 bytes. No
searched finalist, feasible-planner value, bracket, or evidence-ready claim exists. The
verified A0 evidence remains diagnostic context inside the failed core and is not relabelled as
a successful planner result.

### Research decision and contamination boundary

The remaining five registered cells were not executed. Running around a deterministic source
defect would create a selectively observed matrix and could not answer the preregistered
question. The rate-35/seed-0 cell is now development-contaminated and this failed artifact will
remain part of the record after repair; it must never be overwritten, omitted, or presented as
if it came from the repaired revision.

A read-only localization after the failure places the exception at `ActionRecordV1` share-sum
validation while expanding the `legacy_live` template on an evolved search branch. The exact
A0 trajectory did not visit that branch state, which explains why A0 capture/replay and the
pre-freeze fixtures could pass. This is a localization, not yet a root-cause claim. Before any
protected execution resumes, a noncanonical regression must reproduce the exact arithmetic
boundary, the implementation and frozen text must be reconciled without outcome-driven search
changes, the complete gate must pass, and a new clean execution revision must be declared. The
repaired six-cell matrix will be reported separately from this first failed invocation.

### Localization correction and repair frozen before code

The preceding phrase "during its fourth macro-depth region" was imprecise. The artifact
reports **four completed macro depths**; the uncompleted edge belongs to the following region.
The numerical counters remain correct and unchanged. This correction is additive rather than
rewriting the disclosure commit.

An independent telemetry-only reproduction now confirms the mechanism without sampling an
episode or invoking the relaxation solver. At solar bus
`3097.9458568525442 == 0x1.833e4475970c0p+11`, the production heuristic emits shares whose
ordered binary64 sum is `0x1.0000000000001p+0`. The action is an operational sanitizer fixed
point even though the schema's stronger arithmetic sum predicate rejects it. Only
`legacy_live` bypasses the target router's strict proportional closure, completing the
root-cause chain from the observed schema message to the controller arithmetic.

`FEASIBLE_PLANNER_REPAIR_001.md` freezes the correction before implementation. It preserves
the exact A0/legacy request and permits a sum-over-one record only for A0 or `legacy_live`
lineage and only when the simulator's complete sanitizer would be binary64 bit-identical. It
keeps all future-truth templates strictly sum-closed and rejects any materially normalized
legacy request. No performance-sensitive search, plant, task, score, or solver setting may
change in this repair.

Before implementation, a second arithmetic check corrected one detail in that repair text.
The `0x1.0000000000001p+0` value is the protocol's explicit left-to-right accumulator. On
the registered CPython 3.12 runtime, the built-in `sum` used by `Action.sanitised` returns
exactly `0x1.0000000000000p+0`; the sanitizer therefore never enters its division branch for
this witness. The frozen acceptance rule is consequently the complete sanitizer operation:
compute its built-in sum, divide only if that result exceeds one, and require the final four
shares to be bit-identical. This correction was also committed before implementation and does
not broaden the permitted lineage or any search degree of freedom.

## 2026-09-03 — Legacy closure repair implemented and candidate re-freeze verified

### Exact correction

The repair is implemented only in the pure `ActionRecordV1` validation boundary. Every action
share remains finite and individually within `[0, 1]`, and the protocol's explicit
left-to-right sum must still be no greater than one for ordinary records. The sole exception
requires `source_code` -1 or 0, the frozen A0 controller binding, and a locally reproduced
`Action.sanitised` operation whose output is bit-identical to all four input shares. A legacy
oversubscription that the sanitizer would materially normalize still fails; the same one-ULP
fixed point under any of template IDs 1--14 still fails.

The controller output is not rounded, rescaled, or replaced. Consequently the controller,
action-library descriptor and digest, source trace, request bytes, guard input, plant action,
search ordering, score, and solver are unchanged. The fix removes a contradiction between two
validation accumulators at the evidence-record boundary; it creates no new optimization degree
of freedom.

### Executable repair evidence

Three complementary regressions now preserve the defect and fix:

- the shipped heuristic on synthetic telemetry emits the four exact hex floats, has a
  protocol-ordered sum of `0x1.0000000000001p+0`, has a CPython 3.12 built-in sum of exactly
  one, and is unchanged by `Action.sanitised`;
- both A0-capture and live-legacy records accept and reconstruct those five action doubles
  byte-for-byte;
- a wrong A0 binding, every nonlegacy template lineage, and a materially normalized 0.6/0.6
  legacy request are rejected.

The new three-test defect focus passes. The controller/schema/action suites pass **132 tests**,
and the complete planner schema/action/state/search/runtime/evidence/upper-bound gate passes
**268 tests**. The full repository passes **882 tests with 2 skips** and the same sole
Starlette `TestClient` deprecation warning. All ten planner implementation modules pass the
combined strict type check against the repository Python environment; all ten pass formatting
and compilation checks. `uv lock --check` resolves the existing 92-package lock without a
change, whitespace validation passes, and the worktree is clean after the implementation
commit.

No registered episode was sampled or advanced during diagnosis or verification, and no
relaxation solver or full-horizon planner was invoked. The only plant input added by the repair
is a telemetry-only default-plant observation, not an episode. The next commit records this
candidate audit. Every gate must then be reproduced from that clean commit before executing a
new six-cell repaired matrix in rate-major, seed-minor order. Outputs will use a new external
directory; the failed revision-`0059626` artifact remains immutable and separately reported.

## 2026-09-03 — Repaired feasible-planner matrix completed

### Frozen source and execution discipline

The candidate audit was committed as
`564002cd9a49761b7eb78a84424d5325e5b240c6`. From that clean revision the dependency
lock, 268-test planner gate, 882-test full repository suite with 2 skips, formatter, combined
strict planner-source type check, compilation, and whitespace checks were reproduced before
the first repaired cell. No repository file changed while the six registered cells ran.

The cells were opened exactly once in the frozen rate-major, seed-minor order under
`make v2-planner`, with outputs in the new external directory
`/private/tmp/plantanagent-planner-564002c`. The first four calls used the ordinary locked uv
environment. Before rate 42/seed 1, the elevated-command approval service returned a 404 before
process creation; a restricted invocation then failed before process creation because the
default uv cache was outside its writable boundary. Neither attempt sampled an episode or
created an output. Setting `UV_CACHE_DIR=/private/tmp/plantanagent-uv-cache` allowed the same
frozen Make target and existing `.venv` to run inside the restricted sandbox for the final two
cells. The worker command, Python executable, dependency lock, runtime payload, and source
identity remained unchanged.

All six parent runners completed normally. Every retained wrapper has `success=true`,
`evidence_ready=true`, and `runner_failure=null`; every core search has
`termination_reason="completed"`. Their start and end source payloads are byte-identical,
clean, and bound to revision `564002c`; their common source-payload SHA-256 is
`7d54ea4fb59310a19b72279318bf9af200dfc85117720eb2321e9e42b300cade` and common
worktree identity is
`bd64e058d00659e7675519ad83142110bd04ff774ca9ae37a962e4dc74bc1ec1`.
Every boundary observes raw `uv.lock` SHA-256
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.
The shared runtime is CPython 3.12.7, NumPy 2.5.2, SciPy 1.18.1, linked HiGHS 1.12.0,
Gymnasium 1.3.0, and Darwin arm64.

Each cell completes all 72 macro depths, 25,410 attempted/completed/generated beam edges,
1,440 coordinate trials, exact A0 generation and replay, and a fresh 288-decision finalist
replay through all 4,320 one-minute physics steps. Search elapsed time ranges from
3,559,008,295,584 to 3,780,264,533,625 ns and peak reported RSS from 309,641,216 to
351,633,408 bytes. The roughly one-hour runtime per cell confirms this is an offline research
reference rather than a deployable controller.

### Exact controllability bracket

The complete observed bracket is:

| rate | seed | A0 kg | F kg | U kg | gain kg | residual kg | fraction closed | A0 service | F service |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 0 | 2520.000 | 2520.000 | 2520.000 | 0.000 | 0.000 | n/a | 12/12 | 12/12 |
| 35 | 1 | 2347.973 | 2520.000 | 2520.000 | 172.027 | 0.000 | 100.00% | 11/12 | 12/12 |
| 35 | 2 | 2520.000 | 2520.000 | 2520.000 | 0.000 | 0.000 | n/a | 12/12 | 12/12 |
| 42 | 0 | 2552.208 | 3024.000 | 3024.000 | 471.792 | 0.000 | 100.00% | 9/12 | 12/12 |
| 42 | 1 | 2382.421 | 2969.062 | 3024.000 | 586.641 | 54.938 | 91.44% | 7/12 | 11/12 |
| 42 | 2 | 2773.894 | 3024.000 | 3024.000 | 250.106 | 0.000 | 100.00% | 10/12 | 12/12 |

The 35 kg/h task is delivery-saturated for A0 on two seeds; the planner closes the one
172.027 kg shortage and serves all 36 commitments. It is a nominal case, not a sufficient
standalone learning benchmark. At 42 kg/h, exact actionable headroom exists on every seed.
Mean A0 and feasible delivery are 2,569.508 and 3,005.687 kg, a descriptive 436.180 kg gain;
service rises from 26/36 to 35/36. The planner reaches the relaxed ceiling in two cells and
closes 91.44% of the gap in the third. The sole selected-plan shortfall is 54.938 kg at the
final rate-42/seed-1 settlement.

This answers the two frozen questions. The relaxation shows physical/resource headroom above
A0 in four cells, and a fresh-replay-valid exact action sequence realizes a strict improvement
in the same four cells. The earlier controller failure is therefore not evidence that the
task or five-field action contract is intrinsically unsolvable. It is a temporal planning
failure, especially at the stress rate.

### Operating mechanism and sustainability diagnostic

Selected schedules curtail only binary64-scale numerical residue, versus 3.25--6.44 MWh for
A0. They accept 81.7--91.4 MWh of process energy and discharge 7.10--8.79 MWh, versus
72.5--82.5 MWh and about 2.0--2.1 MWh for A0. They also finish with just 1.20--1.33 MWh in
the 8 MWh battery, near the 15% reserve, while A0 finishes at 7.28--7.33 MWh. Every selected
schedule makes four cold starts and one forced stop; A0 makes one cold start and no forced
stop. The task's lexicographic score legitimately prioritizes delivery over these secondary
costs, but this is not yet a commercially sustainable objective.

The planning protocol separately calls for an endowment-preserving relaxation. A one-shot
driver was frozen outside the repository while source remained clean. Its first version,
SHA-256 `41ee77760fef6fdb40628fcdb09084b87d8638ca771a0a224d04819e9c2c6159`,
completed the rate-35/seed-0 solve but failed before artifact publication because it attempted
to construct an unregistered custom `CanonicalPayloadV1` domain. No numeric value or artifact
was printed, and the remaining five solves did not start. This is retained as an
instrumentation failure rather than erased.

The wrapper alone was corrected to apply the public domain-preimage hash directly. The new
driver SHA-256 is
`c98068b8deb380d04fe2e8817c79aa7313494062341068bb7e60886bd09eb3ea`.
Without selecting on the unseen first value, all six diagnostics were run in the same declared
order and wrote source-audited external artifacts. Each has a successful canonical solve,
`preserve_initial_endowment=true`, and `is_task_upper_bound=false`. The results are:

| rate | seed | endowment-preserving relaxed kg | requested fill |
| ---: | ---: | ---: | ---: |
| 35 | 0 | 2520.000 | 100.00% |
| 35 | 1 | 2423.429 | 96.17% |
| 35 | 2 | 2520.000 | 100.00% |
| 42 | 0 | 2577.509 | 85.24% |
| 42 | 1 | 2423.429 | 80.14% |
| 42 | 2 | 2700.873 | 89.31% |

Mean restricted fill is 98.72% at 35 kg/h and 84.90% at 42 kg/h. Because this diagnostic
uses a restricted feasible set, it is not part of `A0 <= F <= U` and may fall below an A0
trajectory that itself spends initial resources. Its value is diagnostic: most of the
rate-42 full-service result cannot be treated as indefinitely repeatable production. The next
commercial task contract needs cyclic terminal constraints or calibrated inventory shadow
values rather than a weak tie-breaker.

The exact planner also accumulates 38.8--45.5 MWh of request-to-application gap. This does not
invalidate feasibility—the selected actions are replayed through the actual guard and
plant—but it means a causal optimizer must model accepted power, not only requested shares.
An explicit mode plus absolute-setpoint interface remains a contingent next change if the
share/router composition prevents causal capture.

### Retained report and custody

The compact retained report is
`docs/research/results/v2_feasible_planner_matrix_564002c.json`. It contains all selected
action records, replay and trace digests, complete metrics and terminal states, delivery
audits, relaxation certificates, search diagnostics, per-artifact start/end identities, and
raw file hashes. Its domain-separated report digest is
`33f7873b0fba1623d816109900f8a5f6a41929924f6dcf99bff5dedd856b1d64`; its exact
1,158,332-byte raw SHA-256 is
`05bd4b66bf71c01847aad369fe3abbea99fe2efe38a5c15be49b131cb551065e`.
The report builder is retained with SHA-256
`6696e09fdf9e05f1507d9356a47693a7c9fbe8b42cdaaac23c75eab5946c3bd9`.

The original failed artifact remains intact at its separately named revision-`0059626`
location and identity. The 12 repaired planner/endowment source artifacts currently remain in
`/private/tmp/plantanagent-planner-564002c`; the compact report records every embedded digest,
raw byte digest, and size. This temporary path is not durable archival storage and must be
copied to permanent research storage before cleanup.

### Research and product decision

The development controllability Gate B is passed. A deterministic causal service MPC is now
authorized and is the next implementation—not PPO training and not another stateless
heuristic tune. Its contract must use only `ControllerObservationV2`, the current issued
forecast, static public configuration, and controller-local history; solve a declared
multi-period model; expose solver failure and latency; span the next renewable window and
intervening deliveries; and model inventories, commitment/ramp state, battery reserve,
delivery slack, guard-compatible applied power, and terminal value. Private future truth,
episode seeds, planner schedules, and future forecast vintages remain forbidden.

The same contaminated cells may be used for MPC development and debugging. Before opening any
new cell, a prospective amendment must freeze demand/weather regimes, sites/seeds, endpoints,
minimum useful gap capture, failure handling, and tuning boundaries. RL is evaluated only
after the equal-information MPC exists and only earns a product role if it improves a measured
residual, tail risk, robustness, or decision latency.

Environment work then proceeds in attributed slices: sustainable terminal accounting,
empirical multi-site renewable/forecast tapes, coherent uncertainty scenarios, calibrated
prices and offtake penalties, wear/degradation, grid/water/compression/quality constraints,
faults and maintenance, sensor imperfections, and multi-unit bottlenecks. HEPA remains a
calibrated predictive-risk layer; language remains an operator/supervisory interface for
versioned goals, explanations, and escalation. Neither layer directly bypasses the controller
or independent safety guard.

## 2026-09-03 — Causal service-MPC protocol frozen before implementation

The completed feasible-planner bracket authorizes causal optimization but does not specify
enough detail to prevent a favorable implementation from choosing its information set,
horizon, objective, fallback, or success threshold after seeing controller outcomes.
`CAUSAL_SERVICE_MPC_PROTOCOL.md` now freezes those choices before any full MPC rollout is
opened.

The first controller uses only the typed public boundary, current issued mean forecast,
static plant/safety/commitment data, immutable settings, and local causal history. It
reconstructs a 24-hour 15-minute solar path by scaling the analytic clear-sky shape within
each public band while preserving issued energy. It infers only the publicly declared
fixed recurring delivery cadence and may not access the private schedule, seed, weather
tape, future vintage, or offline planner result.

The declared model is a two-stage HiGHS-backed MILP over solar routing, battery direction
and energy, feedstock/product inventories, settlements, and a conservative cold-start/
running unit-commitment approximation. Stage one maximizes delivered methane; stage two
locks delivery within `10^-6 kg` and applies bounded service/endowment/operational
tie-breakers. Both statuses and latencies are evidence. Any failure raises and invalidates
the rollout; a hidden heuristic fallback is prohibited.

The development utility gate requires complete evidence, at least two of three rate-42
improvements, at least 20% mean capture of the frozen `F - A0` gap at that rate, no service
regression, narrow nominal-rate protection, zero hard-safety/invalid/forced-stop events,
at least 25% terminal SOC, no more than four cold starts, and bounded solve latency. These
are engineering thresholds on contaminated cells, not estimates of deployment performance.

Only unit and deliberately tiny fixtures are now authorized. Before the registered
rate-major six-cell matrix, the implementation must pass the protocol's forecast,
causality, MILP, commitment, solver-failure, exact-environment, evaluator, provenance,
format/type/compile/lock, full-regression, and independent-source-audit gate from a clean
revision. The next code increment is the deterministic public-forecast projection and its
standalone evidence schema; it does not open an episode outcome.

## 2026-09-03 — Deterministic public-forecast projection implemented

`mpc_forecast.py` now converts one `PublicForecastIssueV2` into the protocol's immutable
96-interval, 24-hour trajectory without accepting an environment, episode, seed, weather
tape, or future issuance. It analytically integrates the configured half-sine clear-sky
geometry over each 15-minute interval, derives one resource scale per canonical forecast
band, and rejects an issuance whose energy materially exceeds public clear-sky support.
The reconstructed band means match the issued mean values within `1e-12`; partial MPC
horizons receive an unchanged prefix rather than a renormalized partial band.

The trajectory schema cross-validates interval powers, peak fractions, band ranges,
resource scales, reconstructed means, and errors before exposing a domain-separated
SHA-256 identity. Tests cover all four bands, darkness, sunset and next-day sunrise,
midnight additivity, prefix behavior, impossible public energy, malformed resolution,
repeatability, and schema mutation. The first fixture incorrectly expected the
19:15--19:30 interval ending at sunset to be dark; the integral correctly found positive
energy, and the test was corrected to place darkness after the sunset boundary.

The focused forecast/weather/controller surface completed with **102 passed** and the new
module passed strict type checking and formatting. The complete repository regression then
completed with **892 passed, 2 skipped**, plus the unchanged third-party Starlette/httpx
deprecation warning. A read-only compatibility diagnostic
projected all 288 already-contaminated seed-0/rate-35 issued forecasts without a failure;
maximum band reconstruction error was `1.11e-16` and maximum inferred resource scale was
`0.97065`. This opened no controller action or plant outcome. The next increment is the
immutable MPC settings/model/result schema and sparse MILP construction.

## 2026-09-03 — Service-MPC settings and evidence schema implemented

`mpc_schema.py` now makes the preregistered numerical surface executable before solver
algebra is added. `ServiceMPCSettingsV1` validates and hashes the 72-hour task identity,
24-hour/15-minute grid, 25% terminal battery target, delivery-lock tolerance, solver
options, and every secondary-objective coefficient. Any setting change produces a new
domain-separated identity; the descriptor explicitly records issued-mean input and no
controller fallback.

Typed immutable records distinguish the one announced settlement from causally inferred
recurrences, require an optimal status and solution digest before a solver stage can be
called successful, and retain every selected interval's energy routes, production,
inventory, commitment, and battery-direction state. `MPCPlanV1` admits a plan only when
both stages succeed in order, steps and delivery ordinals are contiguous, solar and
electrolyser buses close, charging and support do not coincide, selected delivery matches
the interval records and primary delivery lock, and minimum fill matches the declared
settlements. Failure records cannot publish a solution digest.

The settings/schema and forecast suites complete with **23 passed**; both new source
modules pass strict type checking and formatting. No SciPy solver was called and no plant
or controller outcome was opened. The next increment constructs a canonical sparse MILP
independently of the optional solver runtime.

## 2026-09-03 — Solver-independent service-MPC MILP constructed

`mpc_model.py` now constructs the complete preregistered 15-minute mixed-integer model
without importing or invoking SciPy. The separation is deliberate: every variable bound,
integrality flag, objective coefficient, sparse constraint row, public delivery inference,
and initial-condition transformation can be hashed and audited before an optimizer is
allowed to return a favorable answer.

The 24-hour model contains explicit solar routes and curtailment; mutually exclusive
battery charging and electrolyser support; startup and productive electrolyser power;
H2, CO2, and CH4 production; settlement delivery; boundary inventories; cold-start,
warmup, running, on/off, stop, and battery-direction binaries; minimum commitment fill;
and terminal feedstock methane-equivalent value. Its constraints close both electrical
buses, battery energy with directional efficiencies, all three material balances, and
the simulator's pre-conversion and pre-delivery storage-headroom order. The commitment
approximation enforces cold-start occupancy, minimum up/down periods, current public
mode obligations, and prohibits terminal starts whose obligations would fall outside the
finite horizon. Part-load hydrogen yield uses the worst declared specific-energy
multiplier, making it conservative with respect to the profile's steady-state efficiency
curve. The exact minute simulator remains authoritative for execution.

The primary minimization vector is exactly negative delivered methane. The secondary
vector implements the frozen minimum-fill, terminal methane-equivalent, terminal battery,
cold-start, curtailment, and battery-throughput terms; a separately named delivery-lock
row will bind the second solve to the first-stage optimum. The model and every candidate
solution have independent domain identities and residual audits. Inputs now fail closed
on stale forecast physics steps, mismatched solar plants, non-nominal component health,
invalid commitment state, non-finite or non-positive physical constants, misaligned
delivery deadlines, reserve violations, non-canonical metadata, and malformed sparse
algebra.

At the already contaminated rate-35/seed-0 reset, the canonical 24-hour instance has
2,118 variables, including 576 binaries, 3,145 named rows, and 7,915 nonzero
coefficients. This construction check opened no action, solve, or plant outcome. The new
audit suite plus the existing forecast and schema suites complete with **42 passed**;
the source passes formatting and isolated strict type checking. The first formatting
check correctly requested deterministic reformatting. A subsequent system-level `mypy`
invocation used a Python environment without the workspace packages and therefore emitted
missing-`plant_sim` diagnostics plus one genuine local scalar/sequence narrowing issue;
the latter was repaired with an explicit cast and the source then passed using the
repository package paths.

The next increment is the optional SciPy/HiGHS adapter. It must translate only this
immutable sparse representation, execute the two frozen stages with explicit time limits,
independently audit the returned vectors, and publish failure rather than substitute a
controller. Tiny solver fixtures remain authorized; no registered rollout is opened yet.

## 2026-09-03 — Fail-closed two-stage SciPy/HiGHS adapter implemented

`mpc_solver.py` now forms the only optional-runtime boundary in the service-MPC stack.
Importing the forecast, schema, or sparse model still does not import SciPy. A solve lazily
loads NumPy, SciPy, the sparse-array implementation, and the bundled HiGHS version; checks
that the model's interval, boundary, scalar, and binary groups exactly match the frozen
formulation; translates named sparse rows into one `LinearConstraint`; and supplies the
preregistered presolve, relative-gap, and two-second time-limit settings to each stage.

Stage one minimizes negative total delivered methane. Its independently audited solution
defines the positive delivery optimum and the named `10^-6 kg` lock row. Stage two then
minimizes the negative frozen utility subject to that row. Returned values are finite,
canonically snapped only within the declared residual tolerance, and rechecked against all
bounds, integrality conditions, base rows, and (for stage two) the delivery lock. The
reported objective is also recomputed from the returned vector. A successful stage retains
status, bounded message, elapsed nanoseconds, objective, dual bound, MIP gap, node count,
SciPy/HiGHS versions, and a domain-separated solution digest that includes its stage and
additional rows.

The selected second-stage vector is converted into immutable per-interval plan records and
then revalidated by `MPCPlanV1`: both electrical buses, non-negative flows, charge/support
exclusion, ordered delivery totals, minimum fill, terminal inventories, and the primary
lock all cross-check before publication. `ServiceMPCSolveError` now optionally carries the
failed stage evidence. Missing optimization dependencies, a non-optimal status, an
exception, malformed model layout, or an independently invalid status-zero vector cannot
emit a plan or substitute a heuristic action.

A deliberately small eight-interval/no-settlement instance exercised the real two-stage
runtime under SciPy 1.18.1 with bundled HiGHS 1.12.0. Separate fixtures proved fail-closed
behavior for an unavailable optional runtime, an infeasible primary problem, a thrown
solver exception, mismatched settings, a reordered model layout, and a fake status-zero
all-zero vector that violates the independent residual audit. The complete MPC forecast,
schema, model, and solver unit surface passes formatting, isolated strict type checking,
and **50 tests**. No full 24-hour solve, controller action, plant step, or registered cell
was opened.

The next increment turns a valid first plan interval into the existing five-field action
contract, defines the stateful receding-horizon controller and its local evidence ledger,
and proves first-step exact-simulator behavior on deliberately short fixtures before any
development matrix can run.

## 2026-09-03 — Causal service-MPC controller and transition evidence implemented

The semantic plan digest was corrected before controller use. `MPCPlanV1` had included
solver wall-clock elapsed nanoseconds in its hash preimage, so two identical mathematical
solutions could not have the deterministic plan identity required by the protocol. Full
serialized evidence still retains both stage latencies, statuses, versions, bounds, gaps,
and node counts; the plan identity now binds the model, selected trajectory and objective
summaries, plus both stage solution-vector digests, while excluding measurement timing.
Tests prove that changing latency leaves the plan digest unchanged and changing a solution
digest changes it.

The existing transparent-controller binary64 simplex closure was moved to the shared
typed control boundary without changing its algorithm. Both the stateless comparators and
new MPC action realization now use that one implementation. `realize_first_mpc_action`
divides only the four first-interval planned solar routes by reconstructed first-interval
PV, closes any addition residue downward, carries planned battery support as the fifth
field, treats exact darkness as four zero shares, and rejects any request that
`Action.sanitised` would alter.

`ServiceMPCControllerV1` implements the frozen `service-mpc-issued-mean-v1` identity. It
accepts only the current detached observation, current public mean issuance, static plant
and safety data, the public commitment profile/snapshot, immutable settings, and its own
prior public reconciliation. A local task clock binds decision count, physics step,
boundary time, issuance offset, progress, configured 72-hour (or explicit fixture) task
horizon, and the shrinking final planning horizon. It never accepts an environment,
simulator, episode, seed, private tape, delivery schedule, future vintage, or privileged
planner result. Every compute rebuilds the 24-hour projection, takes the remaining-horizon
prefix, constructs and solves a new model, and emits only interval zero. Any failure is
re-raised; there is no action fallback.

Each successful decision now retains immutable, strict-JSON evidence for the public
observation and forecast digests, projection, settings, model dimensions/digest, inferred
settlements, both solver stages, semantic plan, first planned interval, requested action,
deterministic notes, and total decision latency. The next public boundary closes that event
with forecast, battery-net, process-power, curtailment, production, settlement, and four
terminal-inventory residuals. A separate typed failure event preserves the phase and all
identities reached, including failed-stage evidence where available, without manufacturing
a plan or action.

Two independent controller instances produced identical model, plan, action, and semantic
decision digests from the same public input despite different measured runtimes. Missing
forecast, wrong task progress, duplicate-boundary compute, and an injected solver timeout
all failed at their declared phase and retained no fabricated request. An exact one-hour,
four-decision, seed-0 synthetic fixture exercised the real guard and one-minute plant after
each solve; horizons shrank 4, 3, 2, 1 intervals and every decision was reconciled at the
next public boundary, including the terminal transition. A separate 20:00-start fixture
proved exact-darkness action encoding. These are authorized short unit fixtures, not
registered 72-hour development outcomes.

The new and adjacent control/MPC surface passes formatting, isolated strict type checking,
and **131 focused tests**. The next increment extends the paired evaluator so successful
and failed MPC decision events, exact guard changes, applied-power/refusal evidence, source
identity, and latency aggregates are retained and validated independently of the
controller. Until that exists, no full development rollout is authorized.

## 2026-09-03 — Service-MPC evaluator evidence bound to exact execution

The causal evaluator schema advances from version 5 to version 6 so downstream consumers
cannot silently interpret the new optional controller-evidence record as the older causal
artifact; the frozen baseline schema remains unchanged. The evaluator now treats service
MPC as an evidence-producing controller rather than merely another source of five action
fields. After each request passes through the
unchanged `PlantEnvV2.step_action` door, the runner gives the controller only the next
detached public `ControllerObservationV2` so it can close its planned-versus-observed
residual. Evaluator-only `info` is never returned to the controller. In a separate path,
the evaluator binds that reconciled event to the environment's decoded request, last
guarded action, full interval power accounting, exact per-physics-step authority-distance
integrals, intervention messages, clipping flag, and correction counts.

`MPCDecisionApplicationV1` rejects a mismatched action, non-contiguous environment decision
index, policy-vector contamination, disagreement between clipping records, duration
mismatch, and disagreement between the public residual and exact applied battery,
electrolyser, DAC, or methanator mean power. Request, guarded-action, and interval-power
objects receive separate domain hashes. Each application also binds the semantic decision,
requested action, and public residual digests, preserving a reviewable chain from public
input and mathematical plan through the independent guard to exact plant execution.

The new run-level evidence validates ordered controller events, contiguous applications,
the only permissible one-decision failure boundary, pending-transition consistency, and
complete reconciliation before a rollout can be labelled successful. It publishes decision
and solver-stage latency distributions using an explicit empirical-nearest-rank p99,
intervention/correction/range counts, exact integral request-to-guard and guard-to-plant
distances, every underlying event and application, and a content digest. The fallback count
is structurally zero. A controller-side solve failure retains its failed HiGHS stage and
produces no request; a runner-side failure retains the valid evidence prefix. Failure of
evidence collection itself becomes an explicit failed-run phase rather than a successful
metric record.

The generic paired evaluation behavior is unchanged for other controllers. For service
MPC, an incomplete or failed arm now makes the whole paired effect summary ineligible while
retaining the failed cell, its action prefix, and its controller evidence. A deliberate
solver-timeout fixture confirmed zero fallback actions and invalid aggregate summaries.
The successful one-hour fixture confirmed four requests, four public residuals, four exact
applications, eight successful solver stages, matching request/action identities, and
agreement between serialized interval accounting and episode metrics. These remain
authorized synthetic fixtures; no registered 72-hour development cell was opened.

The combined controller, MPC, evaluator, paired-statistics, and comparison-CLI surface
completed **184 focused tests**. Strict type checking passes the evaluator,
execution-evidence module, controller, and new integration tests. Two existing narrowing
weaknesses in the evaluator were exposed by checking the whole module and repaired with a
real-number type guard and explicit post-validation assertions; this changes no runtime
calculation. The next work is to finish the preregistered hand-solvable MILP and solver
failure fixtures, then perform the complete format/type/compile/dependency/full-regression
gate and independent source audit before any registered development matrix is allowed to
run.

## 2026-09-03 — Service-MPC implementation gate closed at fixture level

The pre-rollout audit found that the original solver adapter lost important evidence on
two failure paths. A nonzero HiGHS status retained the failed stage, but a status-zero
result rejected by the independent vector, residual, objective, bound, gap, or node-count
checks raised a plan error without a stage record. A secondary-stage failure also discarded
the already successful primary stage. `ServiceMPCSolveError` and `ServiceMPCPlanError` now
carry an ordered attempted-stage prefix. Status-zero independent rejection is explicitly a
failed stage with no solution digest, while preserving the solver's status, bounded message,
finite objective/bound/gap diagnostics, node count, runtime versions, and elapsed time.
Secondary failure retains the successful primary record; delivery-lock or plan-extraction
failure retains every successful stage reached. Schema validation permits only the declared
primary/secondary prefix, permits failure only at its end, and forbids a typed solve error
whose retained final stage is marked successful.

Because a failure record changed shape from one optional stage to the complete ordered
prefix, its serialized contract advances to `service-mpc-decision-failure-v2`; run evidence
advances to `service-mpc-controller-run-evidence-v2`. Central constants now bind those
identities across controller descriptors, evidence construction, and evaluator validation.
The Python record and mixed event-union names likewise advance to
`MPCDecisionFailureV2` and `MPCDecisionEventV2`; no registered artifact used their v1
predecessors.
Failure-run latency includes every attempted stage rather than only the rejected tail.
Controller action-realization and evidence-construction failures retain both valid solves,
while no failure can manufacture a request, application, plan digest it did not reach, or
fallback action.

A separate causal audit found that trajectory identity had bound the issue time and all
used mean values but not the unused public p10 fields. The numerical mean trajectory could
therefore be transplanted between two different same-time public issuances. Projection
version `issued-mean-clear-sky-scaled-v2` now stores and hashes a domain-separated digest of
the complete detached `PublicForecastIssueV2`. Model construction recomputes that digest
from the current observation and rejects a projection from any other issuance, even when
its mean path is identical. Changing only p10 leaves the numerical issued-mean path
unchanged, as intended, but changes both source and trajectory identities.

The counterfactual causal fixture now creates seed-0 and seed-97 episodes and first proves
that their exogenous episode, private weather truth, complete forecast stream, and next
future-vintage private identities differ. Both controller instances then receive the exact
same detached public observation. Their canonical requested-action bytes, projection,
model, plan, action, and semantic-decision digests are identical. This is direct executable
evidence that episode seed, private future truth, and future forecast vintages cannot alter
the current service-MPC request outside the declared public boundary.

New hand-solvable real-MILP fixtures cover:

- zero solar, reserve-only battery, and empty inventories, yielding exact zero production
  and service;
- a 5 kg H2 / 27.5 kg CO2 endowment producing and settling exactly 10 kg CH4 with no
  same-interval feedstock production;
- a full 5 kg methane tank settling exactly 5 kg and producing zero additional methane,
  proving production-before-settlement headroom;
- otherwise identical low/high battery endowments, where only the higher endowment improves
  service and it finishes exactly at the 25% terminal floor;
- zero solar with an already-running one-interval minimum-up obligation, where the 1.2 MW
  minimum stable load is supplied only by battery, produces H2 at the exact conservative
  58.24 kWh/kg rate, and closes the 96% discharge-efficiency balance; and
- immediate and 30-minute-delayed cold starts, each with four powered warmup intervals,
  four required running intervals, exact one-hour startup energy, and no H2 credited during
  warmup.

The first storage-fixture draft used the default episode's initial delivery coverage with
tiny synthetic tanks and was correctly rejected before solving; the fixture episode now
declares zero initial coverage and its public test settlement is injected explicitly. The
first battery-only draft allowed an unconstrained running unit to stop, so zero discharge
was the rational optimum rather than a defect. Adding the intended one-interval public
minimum-up obligation exposed a second invalid draft: 1.2 MW for 15 minutes produces
5.1511 kg under the conservative rate, exceeding the proposed 5 kg H2 tank. The final 6 kg
tank makes the physical case feasible without changing its isolation. These rejected
fixtures were not controller-performance outcomes.

The structural commitment matrix now names all six simulator modes—OFF, COLD_STARTING,
RUNNING, HOT_STANDBY, HOT_STARTING, and COOLDOWN—and their relevant remaining timers. A
new exact one-interval plant test complements the full four-decision fixture by binding
planned battery net and three process powers to the next public boundary. An injected
sanitizer mutation proves that any first action requiring repair fails in the declared
`action_realization` phase, emits no pending transition, and retains both successful solve
stages and the plan identity.

`CAUSAL_SERVICE_MPC_PREFREEZE_AUDIT.md` maps all ten protocol gate items to these and the
existing evaluator/provenance tests, and separately reviews every source-module authority
boundary. The seven MPC test modules pass **79 tests**. All 14 touched Python files pass
formatting and Git whitespace checks. The seven production modules pass strict type
checking with transitive legacy modules treated as external; both touched integration-test
modules pass the same strict body check with only the external untyped pytest-parameterize
decorator diagnostic disabled. A deliberately broader strict run also surfaced existing
annotation debt in simulator/RL dependencies; no such diagnostic remains on the new
surface.

The wider Environment V2, transparent-controller, paired-provenance/statistics, and MPC
surface passes **237 tests**. Python compilation succeeds. `uv lock --check` first failed
before lock inspection because the sandbox could not initialize the host-wide uv cache;
the same command with an isolated `/private/tmp` cache resolved the existing 92-package
lock in 5 ms without changing `uv.lock`.

No registered 72-hour cell, full 24-hour controller solve, holdout, or new performance
outcome was opened. The next action is to commit this implementation/audit candidate and
repeat formatting, type, compilation, dependency-lock, focused, and full-repository checks
from that clean revision. Only a fully passing clean-source gate may authorize the frozen
six-cell development matrix.

## 2026-09-03 — Registered service-MPC cell boundary isolated before outcomes

Commit `047858c` passed the complete clean-revision implementation gate: formatting on all
14 relevant Python files, strict type checking on the seven production modules and two
integration-test bodies, Python compilation, the frozen 92-package dependency lock, 237
focused Environment V2/controller/MPC tests, and the full repository suite of **961 passed,
2 skipped**. The only warning is the previously known Starlette/httpx deprecation warning.
The worktree remained clean throughout the gate. This closes implementation authorization;
it does not itself reveal any registered controller-performance result.

The generic causal-comparison command intentionally remains flexible for ordinary research,
so it was not a sufficiently narrow execution boundary for the preregistered service-MPC
matrix. A dedicated one-cell runner now admits only rates 35 or 42 kg/h, seeds 0, 1, or 2,
the 72-hour episode, exactly 288 controller decisions, the declared A0/A2/M controller order,
and the two declared A2-minus-A0 and M-minus-A2 contrasts. It constructs the existing default
plant, safety constraints, and reward weights internally and requires clean-source evaluation.
No CLI flag can change the horizon, plant, controller set, controller order, weights, or
contrast definitions.

Before publication, the runner independently validates the episode, schema, descriptors,
run order, retained action traces, common-random-number pairing identities, dependency-lock
hash, repository identity, and clean/unchanged start-to-end source audit. A successful MPC
arm additionally requires all 288 decisions, events, reconciliations, and exact application
records; zero failed decisions; and the structurally fixed zero fallback count. A controller
failure remains a first-class experimental outcome and is published rather than silently
discarded.

Each cell must target a new absolute `.json` path whose existing parent lies outside the Git
repository. The runner serializes to a temporary sibling, checks exact expected bytes and a
canonical-JSON round trip, fsyncs the file, creates the destination with a non-overwriting
hard link, fsyncs its directory, reopens the published bytes, and records their SHA-256.
It recollects provenance both before and after this publication operation so a source or
lock change between evaluation and durable output fails closed. A Make target exposes only
the rate, seed, and mandatory external output path.

All runner tests were kept outcome-free. Synthetic failed and synthetic complete records
prove admission of the two legitimate result shapes, while mutations prove rejection of a
wrong arm order, hidden fallback, dirty source, invalid rate/seed, internal or relative
destination, wrong suffix, and overwrite attempt. An injected evaluator confirms the exact
arguments without instantiating a real episode. The first atomic-publication assertion tried
to compare a deserialized JSON mapping directly with its Python source mapping; JSON changes
tuples to arrays, so this incorrectly rejected a semantically and byte-correct artifact.
The final test instead checks the exact serializer bytes and canonical-JSON equivalence.
That was a test-representation defect, not an experimental outcome.

The new runner suite passes **9 tests**; the adjacent runner/controller/evaluator suite passes
**53 tests**. Both source and test bodies pass their strict type checks, formatting and Git
whitespace checks pass, and a dry Make invocation expands to the frozen one-cell command.
No registered episode, MPC solve, or performance outcome has yet been opened. The last
pre-outcome task is to freeze and test the six-cell admission, aggregation, and gate report;
only then may the cells run in the declared rate-major, seed-minor order.

## 2026-09-04 — Service-MPC reporting and six-cell admission frozen before outcomes

The final pre-outcome reporting audit found one substantive instrumentation omission in the
ordinary causal evaluator: the protocol required battery throughput, while
`EpisodeMetricsV2` retained only discharge inferred from battery power. The frozen feasible
planner had already computed throughput as the ordered bus-side sum of accepted
solar-to-battery energy and accepted battery-support energy. The environment now passes those
two independently accounted interval values into the metric accumulator, which adds them in
the same left-to-right binary64 order. The metric is serialized and included in causal
contrasts as a diagnostic with no fabricated benefit sign. It does not enter observations,
actions, rewards, controller logic, the plant, or any gate. Because the causal artifact shape
changed, its schema advances from 6 to **7 before any registered result exists**.

This metric addition intentionally changes the exact planner-state fingerprint because a
complete environment checkpoint includes accumulated metrics. The broader focused gate first
exposed both that stale expected fingerprint and three stale schema-6 test assertions. The
new fingerprint was reproduced twice from the same authorized two-decision fixture, and the
expectations were updated to schema 7 and the new exact digest. No registered episode or
service-MPC performance was inspected to make those corrections.

The one-cell runner now embeds a domain-separated matrix contract binding the exact ordered
cells `35/0, 35/1, 35/2, 42/0, 42/1, 42/2`, 72-hour horizon, 288 decisions per arm, A0/A2/M
controller order, the two registered contrasts, and one fresh process per cell. Every output
records its sequence index plus UTC evaluation start and completion. Result validation
reconstructs that metadata rather than trusting it. These additions do not introduce a
controller or threshold choice.

A separate frozen postprocessor now admits exactly those six absolute, external,
non-symlink, non-overwriting JSON artifacts in order. It requires strict unique-key UTF-8
JSON in the exact registered serializer form, non-overlapping declared execution windows,
and an identical clean Git snapshot, dependency lock, and runtime across every cell and the
reporting process. It binds each cell's manifest and exogenous-episode hashes to the exact
committed feasible-planner reference, whose raw and internal report digests are hard-coded.
The complete current A0 metric payload must replay the frozen A0 payload exactly; the newly
instrumented throughput must separately reproduce the throughput retained by that reference.

For every successful M arm the postprocessor independently checks 288 contiguous decisions,
288 exact plant applications, the shrinking `min(96, 288-k)` horizon, two ordered successful
HiGHS stages, the frozen controller settings, requested-action equality, public residuals,
and every action/residual/decision/application/request/guarded-action/interval-power/run
digest and cross-reference. It recomputes latency distributions and all intervention,
correction, invalidity, refusal/reroute, and authority-distance aggregates. A failed M arm
remains a retained outcome, but it fails G1 and makes the complete aggregate `null`; G2--G7
become `not_evaluated`, so no successful-arm subset can be reported.

All seven gate calculations are now executable and fixed before outcomes: the exact complete
execution totals; rate-42 improved-cell count and mean feasible-gap capture; rate-42 complete
commitment total; rate-35 mean-fill and saturated-seed preservation; rollout-level safety
and validity; terminal SOC and cold starts; and pooled plus per-rollout nearest-rank decision
p99/max latency. The compact report retains all physical metrics for all three arms,
terminal CH4-equivalent feedstock, A2/A0/M effects and eligible capture, mechanism summaries,
solver reliability and latency, each gate observation, raw artifact hashes, and compact
evidence-ledger hashes. Its own domain digest and raw SHA-256 are published through an
fsynced, hard-link-based, no-overwrite atomic boundary outside Git.

The first synthetic admission test compared Python controller descriptors directly and
rejected legitimate JSON arrays because the source descriptor contained tuples. Admission
now compares canonical JSON values, preserving exact serialized equality without depending
on Python's tuple/list representation. This was an outcome-free representation defect. A
later coverage audit also added an explicit mutation in which every arm agrees on fabricated
manifest and exogenous hashes but disagrees with the frozen planner cell; admission rejects
it before aggregation.

The outcome-free reporting suite now has **16 passing cases**, including a complete invented
matrix that passes all gates; incomplete execution suppressing every aggregate; independent
G2--G7 failures; and rejection of order, source, overlap, fallback, A0 replay, planner
identity, file-form, and overwrite mutations. The full V2/planner/MPC surface passes **395
tests** after the schema/fingerprint corrections. The entire repository passes **986 tests,
2 skipped**, with only the unchanged Starlette/httpx deprecation warning. The new
metric/runner/postprocessor production surface and both runner/report test modules pass
isolated strict type checking; Python compilation, Black formatting, Git whitespace, the
unchanged 92-package lock, and the frozen Make expansion also pass. An initial concurrent
strict-type invocation shared the global mypy cache and hit mypy's internal unresolved-cache
assertion; rerunning with isolated disabled caches produced the clean results above.

No registered 72-hour cell, full service-MPC development solve, held-out result, or RL
comparison has been opened. The next action is to commit this reporting freeze, repeat the
complete gate from that clean revision, and only then execute the six cells sequentially in
their registered order. While a cell is running, the research process will wait silently;
it will not spend model tokens narrating or polling the solve.

## 2026-09-04 — First registered service-MPC matrix fails execution gate

The reporting freeze was committed as `59e377c`. That exact clean revision then repeated
the complete authorization gate: **986 passed, 2 skipped**, all 12 relevant Python files
Black-clean, strict type checks clean for the metric/runner/report production modules and
runner/report tests, compilation clean, dependency lock clean, and no tracked or untracked
source change. Only then was a new external directory created and the six cells executed in
separate processes, sequentially and without source edits. Long-running cells were waited
on silently, per the requested resource discipline.

The first registered matrix is a valid **G1 failure**, not a controller-performance result.
All six cell artifacts were written and admitted in order from the same source, worktree,
lock, and runtime. All twelve A0/A2 arms completed. All six M arms failed without fallback:

- 35/0 completed 114 decisions (28.5 hours), then HiGHS returned an optimal secondary
  solution whose independent audit found a `5e-7` bound violation;
- 42/0 completed 90 decisions (22.5 hours), then the same boundary rejected a `1e-6`
  bound violation; and
- 35/1, 35/2, 42/1, and 42/2 each applied one decision, then transition reconciliation
  rejected `observed_curtailed_kw` as negative.

The two longer cells retained every successful decision/application and the failed solve
stage; the four early failures retained their requested action, partial metrics, pending
transition, and failed evidence state. Across every record the fallback count is zero. The
frozen postprocessor reported 12/18 successful arms and 0/6 complete MPC rollouts, failed
G1, set the aggregate to `null`, and marked G2--G7 `not_evaluated`. It did not compute a
surviving-subset statistic. The external report has raw SHA-256
`4c509fda07965ab92bfbf78a403822cf34b6e3dc4a492d6592f37ed924d379f2` and internal
digest `22fadb2da9c72baf7f61ddc9fa2373c18b317cb494462dd12c126823df1a0961`.
Every cell's raw hash, size, failure, provenance, and custody location is recorded in
`results/v2_service_mpc_matrix_59e377c.md`.

The four early failures were reproduced after the matrix using only one decision and the
already-retained all-to-methanator request. In the registered uv runtime, the public
interval accumulator observed `-1.894780628693601e-15 kW` for seeds 1 and
`-9.473903143468004e-16 kW` for seeds 2, independent of rate. The methanator converts an
offered kW value to kg and back; binary64 round-trip error can return a power one ULP above
the offer, which the allocator exposed as a tiny negative spill. The episode reward metric
already clamps its independently computed curtailment delta at zero, but public interval
telemetry accumulated the raw negative simulator field. The residual schema correctly
requires a physical non-negative observation; the defect is the inconsistent source
invariant, not the check.

The later solve failures expose a distinct scale contract. The adapter's absolute audit
tolerance is `1e-7`, but the installed HiGHS MIP solve accepted lower-bound excursions of
`5e-7` and `1e-6` while row residuals remained below `1e-11`. These values are physically
negligible but greater than the declared independent threshold. The repair must therefore
make solver feasibility settings explicit and hash-bound, and align independent acceptance
with that declared numerical contract; it must not special-case these seeds or relax the
delivery objective after observing performance.

Finally, opening the failed compact report showed that failed-run evidence retained its
entire event/application ledger instead of only validated ledger hashes. That did not alter
admission or gate results but made the report 2,983,662 bytes, contrary to the addendum's
compactness claim. This first report remains immutable. The next implementation revision
will independently validate partial failure ledgers and compact them, while the raw ledgers
remain in their hash-bound cell artifacts.

The immediate next phase is numerical-contract repair plus outcome-independent regression
fixtures. After a new clean gate, a separately labelled replication matrix may test whether
M can finally complete; the failed `59e377c` matrix will remain the first registered result
and cannot be replaced.

## 2026-09-04 — Numerical execution contracts repaired and replication separated

The repair began at the physical source of the four first-transition failures. Two exact
offers retained from the failed records showed a kW-to-kg-to-kW binary64 round trip returning
the next representable power value above the offer; one representative input is
`float.fromhex("0x1.2e35a8ccf9062p+6")`. The methanator now treats its incoming bus offer as
authoritative and reports the minimum of the offer and power reconstructed from actual CH4
production. An allocator-level regression sends only that exact offer to the methanator and
requires exact accepted power, zero battery spill, and zero curtailment. A second regression
uses the registered 72-hour/rate-35/seed-1 boundary, computes the real first MPC action,
advances the exact plant, and closes its public residual with exactly zero observed
curtailment. This changes no reaction yield or material movement; it prevents numerical
representation from manufacturing energy.

The first full-suite run correctly stopped at the weather-alpha exact compatibility hash.
To avoid laundering a behavioral change as a fingerprint update, the 96-decision trace was
generated twice in one process: once with the repaired component and once with a wrapper
that restored the exact legacy reconstructed-power assignment. The wrapper reproduced the
old frozen digest
`5858920bf829d5c5bc1c755552ec22943e8c8338b0f67b5c7ce67c34b02324e6`; the repair produced
`be16187347bf80825f191f392da595c5d7f6619ad617ceb830829e423135fcab`. Recursive comparison
found exactly nine changed telemetry scalars and no structural, action, reward, inventory,
production, delivery, or termination change. The maximum absolute numerical delta was
`2.842170943040401e-14`; affected fields were methanator power/load, curtailment, and their
derived interval authority-distance accumulators. The compatibility constant now names the
repaired invariant, while this journal retains both identities and the exact audit result.

The solver repair was deliberately not a blanket relaxation. The installed SciPy 1.18.1
wrapper accepts unknown options and forwards them to HiGHS, while the HiGHS 1.12.0 option
reference defines `mip_feasibility_tolerance` with default `1e-6` and minimum `1e-10`.
The controller settings now explicitly choose and hash-bind `1e-7`; the adapter suppresses
only SciPy's exact warning for forwarding this named option. A fake-runtime regression
proves the declared value reaches the optimizer. Values outside the documented range are
rejected by settings construction.

The first attempted independent-audit repair used an absolute `1e-7` term plus a relative
`1e-7 * sum(abs(coefficient * value))` row scale. A real rate-35/seed-0 diagnostic exposed
why that formula is incomplete: it failed at decision 66 even though HiGHS returned status
zero. The failing rows were gated continuous variables beside zero binaries. Examples were
`productive_maximum[32]` at `1.4948606019693498e-05 kW` and
`startup_power[33]` at `8.036523313492694e-06 kW`. Multiplying coefficient by the zero
binary erased the 1.2--3 MW row scale, so a numerically ordinary unscaled solver residual
was misclassified. This rejected formula was not used for a matrix result.

The final row audit retains coefficient scale at zero:
`sum(abs(coefficient) * max(1, abs(value)))`, combined with finite row bounds and an
absolute floor. Bounds use `max(1, abs(value), finite bounds)`; integrality remains an
absolute check. Hand-solvable fixtures cover both cancellation near a running binary and
near-zero leakage beside an off binary. The adapter now validates the raw finite vector,
projects only values within the same declared bound tolerance to their nearest finite
bound, and audits the projected vector again before hashing or extraction.

A second real diagnostic then crossed the row-audit failure but stopped at decision 86
during plan extraction. Its future interval 13 contained only
`3.935447649175992e-05 kW` of battery support beside a true charging binary and
`76.80528330204667 kW` of charge. Related startup/productive and trace-material values were
also at coefficient-scaled numerical epsilon. The strict plan schema correctly exposed
that the extraction boundary still used an unrelated fixed absolute tolerance. The final
repair projects bound-level numerical noise and evaluates plan electrical identities with
the explicit absolute-plus-relative interval-power scale. A unit fixture accepts a
sub-milliwatt-scale equivalent at a 10 kW interval and still rejects a material 1 W
simultaneous request. After those changes, the disclosed rate-35/seed-0 diagnostic completed
130 consecutive MPC decisions, with 130 reconciled events, no pending transition, and no
exception. This crosses the first matrix's decision-114 failure but is not a full rollout;
no final performance metric or paired effect was inspected.

After adding the separate raw-vector audit ahead of projection, a final 90-decision
rate-35/seed-0 check completed with 90 reconciled events and no pending transition. It
crossed both the earlier decision-66 row-scale failure and decision-86 extraction failure,
confirming that projection does not conceal an invalid raw vector on those known cases.
This shorter repeat was limited to the newly introduced validation boundary; it did not
open a complete episode outcome.

Failed evidence reporting was independently hardened. The reporter now verifies the full
run digest; exact counts and failure phase; contiguous successful and application prefixes;
at most one terminal typed failure; settings, stage, action, residual, failure, request,
guarded-action, interval-power, and application digests; all decision/action/residual
cross-bindings; pending reconciliation; latency summaries; and every intervention/gap
aggregate. Only then are the large ledgers replaced by canonical event/application ledger
hashes. Synthetic failure evidence now includes a successful applied prefix plus a terminal
failure, and adversarial tests recompute outer hashes after corrupting an event digest,
application count, or application binding; admission rejects all three. The failed compact
report fixture proves neither raw ledger survives in the report.

Because these repairs were selected after the first G1 result, the original version-1
matrix cannot be overwritten or pooled with repaired output. Cell execution, matrix, and
report contracts advance to replication version 2. The new matrix identity binds the exact
failed source/report, four permitted repair classes, unchanged ordered cells and controller
arms, contrasts, horizon, decision count, and repaired settings digest. The separate
`CAUSAL_SERVICE_MPC_REPLICATION_PROTOCOL.md` discloses every partial diagnostic and freezes
the unchanged gates and sequential no-subset execution rule before any repaired 288-step
rollout or complete matrix is opened.

At this checkpoint the combined simulator, MPC controller/model/schema/solver, registered
runner, and report suites pass **130 tests**. The next step is the broader focused and full
repository authorization gate, strict type/format/compile/lock checks, a clean auditable
commit, and only then the six-cell engineering replication. Long executions will be waited
on silently; no model tokens will be spent narrating an unchanged running process.

## 2026-09-04 — Repaired service-MPC replication authorization gate passes

The repair was divided into three reviewable commits:

1. `4a5bc5d` restores the methanator offered-power invariant and advances the audited
   compatibility trace;
2. `5b97149` binds the HiGHS feasibility setting, raw/projected scaled audits, bound
   canonicalization, and scaled plan extraction; and
3. `d5658a7` deep-validates/compacts failed ledgers and freezes the version-2 engineering
   replication contract and protocol.

From clean revision `d5658a7e61a11e95aee274c709b71deda2cfc571`, the complete repository
suite passed **1,000 tests with 2 skips** in 28.72 seconds. The only warning remains the
pre-existing Starlette/httpx test-client deprecation. Twelve changed backend source/test
files pass deterministic Black checking. The five changed backend production modules pass
strict isolated type checking, and the six changed backend MPC test modules pass the same
check with only the external untyped pytest-decorator diagnostic disabled. Python
compilation succeeds across backend and simulator source/tests, the unchanged dependency
lock resolves **92 packages**, Git whitespace validation passes, and the worktree is clean.

A direct strict check that recursively opened the simulator package also reported seven
pre-existing transitive diagnostics in `faults.py`, `delivery.py`, `allocation.py`, and
`simulator.py`; the changed methanator module itself introduced none. Those unrelated
annotations are not hidden as part of this gate: functional coverage for the simulator
repair is included in the full passing suite, while the strict claim is limited to the
new backend surface as it was for the original MPC freeze.

The exact repaired controller-settings digest is
`0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2`; the version-2
matrix-contract digest is
`39a085da394c5469484cfdd4489982f3021b4de029af2e513684f725f6db74eb`.
No complete repaired rollout or performance aggregate has been opened. The authorization
gate is now closed: after this journal-only record is committed and the executable tree is
confirmed unchanged, the next action is the exact six-cell sequential replication. No
source change is permitted after its first cell starts.

## 2026-09-04 — Repaired service-MPC replication completes and fails the controller gate

The authorization record was committed as `3796eb5`. A Git comparison confirmed that its
only difference from executable revision `d5658a7` was this journal, and the worktree was
clean. The six cells then ran in fresh processes, sequentially in the registered order
`35/0, 35/1, 35/2, 42/0, 42/1, 42/2`. The repository was not modified after execution
began. Long-running solves were waited on silently; no progress narration or reasoning was
generated while a cell was active.

All six cells completed. The version-2 reporter admitted the common clean revision,
worktree identity, dependency lock, runtime, non-overlapping order, all 18 arms, all 1,728
MPC decisions, all 3,456 solver stages, and zero fallback. G1 therefore passes: unlike the
initial matrix, this is eligible controller evidence. G7 also passes, with a pooled decision
latency p99 of 0.784 s and maximum 1.094 s. G2--G6 fail.

At 35 kg/h, A0 mean fill is 97.72% and MPC mean fill is 34.17%; MPC loses 1,601.55 kg
against A0. At 42 kg/h, A0 mean fill is 84.97% and MPC mean fill is 29.49%; MPC loses
1,677.72 kg. MPC improves A0 in none of the six cells and meets only 9/36 stress-rate
commitments. It has no reserve breach, invalid action, out-of-range decision, or shutdown,
but every rollout has forced electrolyser stops. Every MPC cell terminates with 100% battery
SOC and approximately full CO2 storage while mean curtailment exceeds 56 MWh. The poor
delivery is therefore not caused by exhausted renewable or feedstock endowment.

The retained transition ledger localizes the mechanism. The MILP uses a 15-minute mean
power balance and treats `battery_support_kw` as expected discharge. The simulator defines
the same action field as contingent top-up authority, applied only when routed solar leaves
electrolyser headroom. `realize_first_mpc_action` copies the expected flow into the authority
field. A startup plan consequently requests exactly 600 kW in the mean, while four solar
shares are held across 15 one-minute physics steps. The exact commitment state machine
requires 600 kW at each minute and trips after two shortfall minutes.

For example, seed-0 decision 192 follows one fully powered startup interval. The new plan
contains 94.150 kW solar and 505.850 kW support, exactly 600 kW in aggregate, but the exact
plant accepts 0 kW over the next held interval after intra-interval shortfall. Across the
matrix the plant records 206 cold starts, 204 aborted starts, and 204 forced stops. Seeds 0
and 2 never reach a productive MPC first interval at either demand rate; seed 1 reaches 34
productive decisions per rate. At every initial boundary the optimizer nevertheless
reports the next four commitments fully deliverable with 100% minimum fill. Solver status,
objective feasibility, and plant execution are each internally valid; their temporal and
actuation semantics are not aligned.

The result rules out objective-weight tuning as the next scientific step. It would also be
premature to train PPO or add more plant complexity: either would be evaluated through a
known lossy actuator seam. The next bounded intervention will version the first-action
realization and distinguish planned discharge, contingent real-time support authority, and
applied discharge. The MILP, forecast, objective, safety reserve, and physical kernel must
remain unchanged for that mechanism test. If the bridge cannot preserve the target without
unbounded energy deviation, the supervisory interface will move to explicit absolute
process setpoints plus a deterministic inner loop.

The external artifacts remain at
`/private/tmp/plantanagent-service-mpc-replication.AiZGHf/`. The compact report has raw
SHA-256 `bab822476d5b1b986ad03bdecbeaad6217727cfc3539e117af38af09bc458905`, internal
digest `5a5acf5f46ea0363bfee90925b1346d33b11ffc0766fb214ee55d8baaee2df13`, and size
207,908 bytes. Every cell hash and outcome is recorded in
`results/v2_service_mpc_replication_3796eb5.md`.

## 2026-09-04 — Actuation-bridge hypothesis frozen before implementation

The post-replication action is now bounded by
`CAUSAL_SERVICE_MPC_ACTUATION_BRIDGE_PROTOCOL.md`. The controller's MILP treats battery
support as expected 15-minute discharge, while the physical action treats it as contingent
instantaneous authority. The registered realization copied one into the other. The frozen
repair will preserve the model, plan, four solar shares, forecast, objective, settings,
safety guard, and plant, but will expose planned discharge and support authority as separate
hash-bound quantities. Active authority is capped at the planned electrolyser target and
the existing constraint; inactive authority remains exactly zero.

Implementation must prove the distinction at a fluctuating one-minute startup boundary,
prove that the reserve guard is unchanged, and preserve deterministic causal identities.
Only after the focused and full repository gates pass may one rate-35/seed-0 72-hour
development probe run. Its mechanism gate permits at most four aborted/forced starts and
requires a productive interval, versus the frozen 41/41 failure. Safety, accounting,
latency, and delivered-product direction are reported separately. A mechanism failure
triggers an explicit absolute-setpoint/inner-dispatcher design rather than another
share-interface patch. No objective tuning, PPO, HEPA/NLP change, or added environment
complexity is permitted in this package.

## 2026-09-04 — Commitment-aware actuation bridge implemented; probe remains unopened

The permitted controller-to-plant change is now implemented without modifying the MILP,
forecast construction, objective, solver settings, safety guard, component sizing, plant
physics, commitment kernel, reward, episode, or seed handling. The first-action schema has
advanced from `service-mpc-requested-action-v1` to
`service-mpc-requested-action-v2`. Each successful decision now retains four distinct,
hash-bound quantities: the MILP's planned battery discharge, the active support limit, the
planned startup-plus-productive electrolyser target, and the contingent support authority.
The authority is exactly zero for a zero target and otherwise exactly the lesser of target
and active support limit. Only that authority enters the existing five-field plant action;
the safety guard can still reduce or remove it, and the simulator still withdraws only the
minute-level shortfall actually present.

The controller descriptor declares the new realization schema and semantics. Its evidence
constructor rejects support above the planned target or active cap and rejects any authority
that does not follow the frozen rule. The matrix reporter now rejects a legacy-schema
substitution, an incomplete bridge payload, a rule violation, a low-level action mismatch,
or a bridge whose planned flow/target does not match first-step plan evidence. This versions
the actuator meaning without relabelling the historical `3796eb5` artifact.

The executable mechanism probe is deliberately narrower than the generic evaluator. The new
`mpc_bridge_probe.py` runner permits only the 72-hour rate-35/seed-0 cell and the current
service-MPC controller. Its contract has digest
`83ad83b07f57e43fb9af25bf80b07c6cb33b149fdb3e82d2321a9003e63b321f`
and binds the frozen protocol raw digest
`653b118a5d0003e0cb5a37a8076a0f9d765c8eaa7db9a9b759616bf501f981c0`,
the prior report/cell identities, exact manifest and exogenous episode, controller settings,
action schema, expected 288 decisions and 576 solver stages, and the no-same-revision-retry
rule. It requires a complete clean source identity before and after evaluation, rechecks the
lock and worktree immediately before and after publication, writes canonical JSON only to a
new absolute path outside Git, flushes and atomically hard-links without overwrite, reopens
the exact bytes, and publishes both an internal domain digest and raw-file SHA-256.

The probe independently reports B1--B6. Its bridge audit checks run, decision, action,
residual, request, guarded-action, interval-power, and application digests and bindings. For
each decision it binds planned support and target back to the first plan interval, recomputes
the authority rule, matches authority across controller evidence, requested-action trace,
and evaluator application, and verifies that guarded support never exceeds authorized
capacity and applied support never exceeds guarded support. It retains planned,
authority-capacity, guarded, and applied kWh separately. Execution failure leaves B2--B6
explicitly `not_evaluated`; otherwise the mechanism, safety, directional delivery,
accounting, and latency gates remain independent.

Outcome-free regressions exercise the exact contaminated boundary. The first two inactive
seed-0 requests reproduce the frozen `3796eb5` plan identities and low-level actions exactly.
At decision two, observation
`3f9bbe6cc62ec3dfaaf486c20a818ee90bed95984ffe9e7bc9e4708f8dee5aac`,
model `25c5505091125fd36d5b8a79d5691e843cfcf12063a6280bc14392ef2dd56355`,
and plan `36af91aee783fe89979a92f4fb0e62b2f0dd31c04b09467d5c4403378cd2054f`
remain unchanged. That plan retains 68.15390458875038 kW expected support but exposes
600 kW contingent authority for the 600 kW startup target. The unchanged reserve guard can
remove that authority and reports why. With the ordinary reserve, the held one-hour startup
receives 600 kW in its first interval while consuming less support energy than the 150 kWh
authority capacity, then reaches `running` after the four held decisions and records one cold
start with zero aborts or forced stops. Synthetic adversarial tests also reject legacy
schema, wrong authority, an unfunded plan, over-application, malformed stage completion, and
unequal accounting-ledger lengths.

The controller/report/probe focused surface passes **45 tests**. The complete repository
passes **1,011 tests with 2 skips** in 31.39 seconds; the sole warning is the unchanged
Starlette/httpx test-client deprecation. Black formatting, Python compilation, Git
whitespace validation, and strict isolated type checking of the four changed production
modules pass. An initial strict invocation omitted the workspace package path and therefore
reported missing `plant_sim` imports plus pre-existing transitive annotation diagnostics;
the repository-path, isolated-import invocation passed with no issue in the changed modules.

No 72-hour bridge rollout, performance outcome, new holdout, PPO training, or environment
change has been opened. The next action is to commit this complete implementation and journal
on a clean revision, confirm the frozen protocol and executable tree are unchanged, and run
the single authorized mechanism probe once. While it runs, the process will wait silently
and generate no progress narration or speculative analysis.

## 2026-09-04 — One-shot bridge probe fails execution and rejects the share seam

The complete implementation was committed as `709e477`. The worktree was clean, the frozen
protocol still had raw digest
`653b118a5d0003e0cb5a37a8076a0f9d765c8eaa7db9a9b759616bf501f981c0`,
and the unchanged lock had digest
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.
The one authorized rate-35/seed-0 probe then ran once in a fresh process. No source changed
during execution, no fallback was used, and no progress reasoning or narration was produced
while it ran.

The probe is a valid B1 failure. It completed 223 of 288 decisions and 446 successful solver
stages before the secondary solve at decision 223 stopped. B2--B6 are `not_evaluated` exactly
as preregistered. The canonical 6,686,495-byte external artifact is at
`/private/tmp/plantanagent-mpc-bridge.gUi8kC/probe.json`; its raw SHA-256 is
`e5fac6425a92ffc511827f12cf47936b8a776cf879e99c8b019ddcc5aa2f4458`
and its internal report digest is
`e49c7866ddd9a56e71b0f9061aaa8d90d634a14b6b44e5fae6aa4a513ba72aa2`.
The source identity
`c63e8224e6585db5c252febecbaa9fa1ef0ae8464322de5231282d2265b9adfe`
was complete, clean, and unchanged at both boundaries. The revision will not be rerun.

The terminal failure is not solver infeasibility. HiGHS returned optimal status and the
independent raw-vector audit passed. Subsequent bound canonicalization independently
projected a near-zero CO2 state, after which the exact audit found a
`3.49581e-05 kg` residual in `co2_balance[1]`. The physical prefix ends with
`3.495807901682279e-05 kg` CO2. Because the state variable has a 6,000 kg finite upper
bound, the bound-scaled zero projection window is `0.0006001 kg`; projection therefore
changed a feasible coupled state without preserving its equality. The correct future
repair is conditional, feasibility-preserving canonicalization or retention of the already
audited raw vector—not a looser material-balance tolerance.

The incomplete trajectory cannot satisfy a complete-episode gate, but it sharply confirms
one part of the causal diagnosis. Through 55.75 hours it meets all nine settled commitments,
delivers all 1,890 requested kilograms, records three cold and five hot starts, accumulates
85,500 seconds of productive operation, and has zero aborts or forced stops. Minimum SOC is
15%, terminal-prefix SOC is 27.62%, and there are no shutdown or out-of-range decisions.
The old controller's full episode delivered only 647.7138 kg and recorded 41 aborts and 41
forced stops. Extra authority therefore removes the startup starvation mechanism.

It does not realize the optimizer's target. The frozen bridge audit correctly reports zero
digest/accounting violations across all 223 comparable decisions, but its separately
retained values reveal 28.499648 kWh planned battery discharge, 42,722.660301 kWh authority
capacity, 23,911.025270 kWh guarded capacity, and 14,410.008667 kWh physically applied
support. Applied support is 505.62 times planned support. The unchanged safety layer bounds
the plant and accounts for 1,218 interventions over 850 corrected physics steps, but it
cannot infer or restore the optimizer's intended target.

A post-outcome audit was therefore run against the already retained successful ledger and is
explicitly not promoted into B2--B6. Among 109 positive-target decisions, only 22 track
within `1e-7 kW`; 81 overshoot and six undershoot. Mean target error is +717.568 kW, its
range is -309.380 to +1,520.782 kW, planned electrolyser energy is 44,360.845 kWh, applied
energy is 63,914.561 kWh, and absolute target error integrates to 20,227.057 kWh. Net
over-application is 19,553.716 kWh.

The underlying interface behavior is now unambiguous. `battery_support_kw` is additive to
the routed solar power and may also cancel a simultaneous battery-charge request by moving
that solar directly to the stack. It is not a target-aware authority from which the plant
draws only `target - instantaneous routed solar`. Setting authority equal to target can
therefore apply approximately the target twice. The original B5 definition proves ledger
closure and applied-at-most-guarded authority, but it did not predeclare a maximum
planned-to-applied energy difference or target error. That gate will not be changed
retroactively; the missing fidelity condition is a result and becomes mandatory in the next
protocol.

The bridge is rejected as a viable long-term supervisory seam. This is not represented as a
failure of unevaluated B2--B6; it is a separately labelled conclusion against the probe's
stated purpose of deciding whether the share interface can express the first MPC target.
Repairing canonicalization merely to rerun the same semantically invalid bridge would be
misdirected. The next work package will instead freeze an absolute-process-setpoint command
and deterministic one-minute dispatcher. The slow optimizer will declare target power and
priority; the inner loop will observe the current bus, use solar first, draw only the target
shortfall from the battery, remain below the existing guard, and bind plan, target, guarded
minute command, and actual power in evidence. Solver canonicalization will be repaired as a
separate disclosed prerequisite. Full details and artifact identities are retained in
`results/v2_service_mpc_actuation_bridge_probe_709e477.md`.

## 2026-09-04 — Absolute-setpoint and inner-dispatch architecture frozen

Before changing code again, the replacement architecture was frozen in
`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_PROTOCOL.md`, raw SHA-256
`21b12eaa245b21d20d9458b9e3d0b8d163c682ef85ad9d5579fde392647e8782`.
It preserves the historical five-dimensional Gym/share-action path and exact simulator,
while adding a new service-MPC command that declares absolute electrolyser, DAC,
methanator, and battery-charge targets. The command retains expected MILP support and all
planned solar routes as evidence; neither becomes implicit actuator authority.

The deterministic minute dispatcher is now specified algebraically. It reserves enough
current solar to keep the electrolyser target reachable under reserve-aware battery support,
shares remaining scarcity proportionally between the two processes that cannot currently use
battery support, then uses surplus solar to reduce electrolyser support and finally charge
the battery. It cannot request charging and support together, cannot over-offer the
electrolyser target, sees only the contemporaneous bus already available to the guard, and
emits an ordinary `Action` through the unchanged guard and plant. This explicitly separates
15-minute forecast reasoning from one-minute actuator realization.

The protocol also freezes the independent solver repair exposed by the failed bridge run.
Raw vectors still must pass the current audit. The identical nearest-bound projection is
attempted and independently audited; it is selected only if the complete projected vector
remains feasible, otherwise the already valid raw vector is retained and must still survive
plan extraction. The CO2-state and tiny gated-support counterexamples are both mandatory,
and no tolerance may change.

The future evidence graph binds plan to setpoint, every one-minute dispatcher input/output,
ordinary request, guard, application, and next public residual. Its predeclared gates now
make the missing conditions explicit: zero target overshoot, exact reachable pre-guard
tracking, zero unexplained deficit, no simultaneous charge/support, complete energy
attribution, and separate controller/dispatcher latency. PPO, HEPA, language-supervisor,
economic-objective, and realism work remains outside this slice. No setpoint controller
rollout or new outcome was opened while freezing the protocol.

## 2026-09-04 — Feasibility-preserving solver selection implemented

The first implementation slice under the frozen setpoint-dispatch protocol repairs only the
solver-output selection boundary. It does not change the MILP, forecast, horizon, objective,
solver tolerances, plan extraction, simulator, action path, or any physical parameter. The
historical `ServiceMPCSettingsV1` retains its strict projection behavior and exact SHA-256
`0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2`.
The new `ServiceMPCSetpointSettingsV1` uses controller identity
`service-mpc-setpoint-dispatch-v1`, settings domain
`service-mpc-setpoint-settings-v1`, and selection policy
`feasibility-preserving-bound-projection-v1`; its default settings digest is
`e418dc37401482cf78036daefa61d80fa23fef0411a7215c3fb549a91085fb1b`.

The adapter still validates the exact raw vector first and rejects it when bounds, rows,
integrality, or any stage-lock row fail the unchanged absolute-plus-relative audit. It then
constructs the identical nearest-bound/near-integral candidate and audits the complete
candidate. Under historical settings an invalid candidate still fails exactly as before.
Under the new settings, a valid candidate is selected; if and only if the raw vector passed
and the candidate fails, the adapter retains the raw binary64 values exactly. Objective and
plan-extraction checks still run on whichever vector is selected, so this rule cannot turn
an invalid solver result into a physical plan.

Successful stages under the new settings now publish typed selection evidence: selected
mode, projection count, maximum proposed delta, and complete raw/projected audit summaries.
Failed stages cannot publish that evidence. The matrix report admission path deeply checks
the optional record and rejects unknown fields, policies, inconsistent validity, impossible
projection counts, and selection evidence attached to a failed solve. The field remains
absent for historical stages, preserving their serialized evidence and plan identities.

Four outcome-free adversarial controls implement the protocol's required claims. In the
observed bridge failure, a raw-feasible `3.495807901682279e-05 kg` state with a 6,000 kg
upper bound would be projected to zero while its coupled CO2 flow remains nonzero; the new
policy retains the raw vector and records `co2_balance[1]` as the projected failure. In the
earlier gated-flow case, `3.935447649175992e-05 kW` support beside an off binary is safely
projected to zero and selected. A `1e-3` material row error is rejected at the raw audit.
A candidate that leaves base rows valid but breaks a supplied lexicographic lock also causes
exact raw retention. A separate control proves the historical settings still reject the
CO2 projection case.

The schema/solver tests and focused report/controller/evaluation regressions pass **78
tests**. The complete repository passes **1,020 tests with 2 skips** in 33.46 seconds; its
only warning is the unchanged Starlette/httpx test-client deprecation. Black formatting,
compilation, Git whitespace checks, and isolated targeted strict type checking of the four
touched production modules pass. No full controller rollout,
registered cell, holdout, PPO training, environment-realism change, or performance outcome
was opened. The next implementation slice is the immutable process-setpoint command and pure
one-minute dispatcher, beginning with shared reserve-aware support availability so the
dispatcher and independent guard cannot diverge.

## 2026-09-04 — Reserve-aware support authority extracted as one safety primitive

Before implementing the dispatcher, the guard's battery arithmetic was extracted into the
pure `battery_support_availability` boundary. Its typed result separately retains the active
support cap, reserve SOC, usable energy, efficiency-adjusted one-step energy limit,
electrolyser headroom, equipment availability, and final available kW. The existing guard now
uses that exact result instead of independently recomputing the same minimum. This is the
shared authority calculation required by the frozen protocol; the future dispatcher can
reason about support without copying safety logic.

Normal reserve, loss, rating, headroom, and operator-cap behavior is unchanged. The helper
also closes an uncovered fail-safe edge: an explicitly offline battery with inconsistent
positive synthetic SOC now has zero support authority and the guard records that reason.
Focused safety and MPC-controller regressions pass **54 tests**. The complete repository
passes **1,027 tests with 2 skips** in 31.65 seconds, with only the unchanged test-client
deprecation warning. Formatting, compilation, Git whitespace validation, and isolated
strict typing of the safety modules pass. No environment contract, controller decision,
optimizer, simulator physics, or experiment was changed. The next slice will consume this
primitive in the pure absolute-setpoint dispatcher.

## 2026-09-04 — Absolute process intent and deterministic minute dispatch implemented

The next outcome-free slice implements the new supervisory/actuation seam without yet
connecting it to `PlantEnvV2`. `MPCProcessSetpointsV1` is the immutable
`service-mpc-process-setpoints-v1` command. It binds the decision, public observation,
forecast issue, projected trajectory and first interval, settings, unchanged model, selected
plan, 15-minute hold duration, resolved operating envelope, and fixed dispatch policy. It
declares absolute electrolyser, DAC, methanator, and battery-charge targets. The MILP's
expected battery discharge and every original planned solar route/curtailment remain
separate diagnostic evidence and do not become actuator authority.

Construction fails closed on non-finite or negative power, targets above the issue-time
envelope, expected support above either the support cap or electrolyser target, an unbalanced
planned solar bus, a target that does not reproduce the selected first interval, an unknown
policy, a model/plan/projection/public-issue identity mismatch, a different first projected
solar interval, or solver stages lacking the new feasibility-selection evidence. The active
envelope is hashed from resolved equipment, charge, support, and reserve limits and must
remain identical during the held command.

`dispatch_mpc_setpoints` implements the frozen one-minute algebra as a pure function. It
uses the contemporaneous solar bus and the shared reserve-aware support primitive, reserves
the minimum solar needed to keep the electrolyser target reachable, allocates remaining
scarcity proportionally to DAC and methanation, uses further solar to displace battery
support, and charges only when no support is requested. Its ordinary five-field `Action` is
closed downward to the unit simplex and remains a sanitizer fixed point. The resulting
`MPCMinuteDispatchV1` binds the command, detached minute boundary, exact bus, support
availability, targets, every requested route, unallocated solar, reachability, and action.
The record independently recomputes the entire allocation policy on construction, so a
balanced but policy-substituted ledger is rejected.

The implementation preserves the key architectural distinctions exposed by the failed
bridge. Expected support is never consulted by the minute actuator. Actual support is only
the remaining absolute target shortfall and therefore cannot duplicate solar power.
Electrolyser solar plus support cannot exceed the target; it equals the target whenever the
declared electrical reachability test passes. Charge and support are mutually exclusive,
unsupported loads never exceed their targets, support never exceeds the current reserve,
and every solar kW is routed or explicitly unallocated. Offline/ramp/storage refusal remains
the responsibility of the unchanged guard and simulator and can therefore be attributed
rather than hidden inside optimizer behavior.

The focused suite passes **20 tests**. It covers surplus, deficit, darkness, exact reserve,
proportional scarcity, a four-minute fluctuating 600 kW cold-start command, offline-stack
guard intervention, deterministic object/digest reproduction, stale-envelope rejection,
policy/reachability substitution, the real tiny MILP-to-command binding, legacy-stage and
digest substitution, and a 54-case SOC/solar/target invariant grid. The production module
and tests pass Black, compilation, Git whitespace, and isolated strict type checking. The
complete repository passes **1,047 tests with 2 skips** in 31.30 seconds; the only warning
remains the test-client deprecation. No Environment V2 step path, historical Gym/share
action, controller, simulator, episode,
registered cell, or outcome has changed. The next slice is the opt-in `step_setpoints`
environment door and ordered minute ledger, followed by the separately versioned MPC
controller evidence that issues these commands.

## 2026-09-04 — Opt-in setpoint execution and physical evidence integrated

The absolute command can now reach the unchanged physical plant through a separately named
`PlantEnvV2.step_setpoints` door. Admission occurs before any state mutation: the environment
requires the exact current decision index, complete public-observation digest, current public
forecast-issue digest, 15-minute hold cadence, and active resolved authority-envelope digest.
A stale or foreign command therefore cannot partially advance time, settlement, inventory,
metrics, or commitment state. The existing Gym vector and `step_action(Action)` doors still
call the historical held-action branch and neither emits a setpoint field nor invokes the
new dispatcher.

For an admitted command, the environment now performs the frozen sequence independently on
each of its 15 one-minute physics steps: obtain the exact contemporaneous actuator solar bus,
dispatch the held absolute targets with `reserve-aware-proportional-unsupported-loads-v1`,
rerun the unchanged safety guard against the same bus and current telemetry, advance the
unchanged simulator once, and record the result through the ordinary reward, settlement,
metrics, telemetry, and power-accounting paths. Only the setpoint branch changes its request
at the minute cadence. The command remains fixed and contains no newly observed weather or
outcome. Measured dispatcher latency is returned as a separate ordered nanosecond list and
is deliberately excluded from deterministic evidence identity.

The new `service-mpc-minute-application-v1` record binds the full canonical before and after
public plant boundaries, typed minute-dispatch evidence, requested and guarded actions,
guard explanations, full before/after commitment snapshots, storage headroom, all applied
solar routes, support and process power, curtailment, and the next-boundary digest. It rejects
noncanonical boundaries, broken step order, a boundary that does not reproduce the dispatch
digest, non-sanitized guard output, guard authority increases, solar/electrolyser bus
imbalance, support beyond the guarded shortfall, or physical power above the guarded or
absolute target. Explicit per-channel guard offers and signed guard-to-plant gaps keep
safety intervention distinct from commitment, ramp, feedstock, storage, and charging
refusal or rerouting.

`service-mpc-setpoint-interval-execution-v1` requires exactly the number of physical records
implied by the command hold and timestep. Ordinals, command identities, full boundary
records/digests, and full commitment snapshots must form one contiguous chain. Its digest
retains the complete ordered ledger plus planned, target, requested, guarded, and applied
energy by channel; requested, guarded, and applied electrolyser over/undershoot integrals;
maximum instantaneous errors; attribution counts; and terminal boundary/commitment
identity. The latest typed immutable record is available from
`PlantEnvV2.last_setpoint_execution`, while the Gym information payload contains its JSON
representation.

Ten new outcome-free tests exercise the opt-in seam. Two identical seeded environments
produce byte-identical command and execution evidence; the 15 records are contiguous and
the nondeterministic timing values remain outside that digest. Stale decision, observation,
forecast, duration, and envelope substitutions fail before mutation. A real DAC outage is
reported as guard intervention, whereas a 1.5 MW command during cold start passes the guard
and is separately reported as component refusal with applied undershoot and zero overshoot.
Changing only private solar truth after the next physical boundary leaves the held command
and first-minute dispatch identical while appropriately changing later dispatches. Direct
boundary, command, physical-flow, guard-authority, and commitment-chain substitutions are
rejected. Existing dispatcher tests continue to cover day/night, surplus/scarcity, reserve,
offline equipment, proportional allocation, the 600 kW fluctuating cold-start target, and
the SOC/solar/target invariant grid.

The focused command/execution suites pass **30 tests**, and the Environment V2, boundary,
weather-observation, and safety regression surface passes **118 tests**. The complete
repository passes **1,057 tests with 2 skips** in 32.33 seconds; the sole warning is the
unchanged Starlette/httpx test-client deprecation. Black formatting, Python compilation,
Git whitespace validation, and isolated strict type checking of the new production evidence
module pass. No simulator, physics, commitment, reward, episode, Gym action, MILP, forecast,
objective, tolerance, controller trajectory, registered cell, holdout, or performance
outcome changed or ran. The next slice is the separately versioned MPC controller that
constructs these commands and closes them with plan/controller evidence, followed by the
complete outcome-free authorization audit. A 72-hour mechanism rollout remains forbidden
until that gate passes.

## 2026-09-04 — Setpoint-producing service MPC implemented; identity gate conflict exposed

`ServiceMPCSetpointControllerV1` now implements the slow supervisory layer under controller
identity `service-mpc-setpoint-dispatch-v1`. It accepts only
`ServiceMPCSetpointSettingsV1`, reconstructs the same issued-mean forecast, builds and solves
the unchanged two-stage MILP, and maps only the selected first interval through
`realize_first_mpc_setpoints`. Its public output is `MPCSetpointDecisionV1`, not
`ControlDecision` and not a five-field share action. A separate runtime-checkable
`SetpointControllerV2` protocol prevents this new authority model from being mistaken for
the historical Gym/share controller contract.

The information boundary remains the detached `ControllerObservationV2`, its current public
forecast issue, static plant/configuration constraints, the public commitment profile and
snapshot, immutable settings, and controller-local prior public residuals. The controller
has no environment or simulator handle and receives no seed, private weather tape, future
forecast vintage, private delivery schedule, privileged planner result, or minute outcome.
The descriptor states that MILP support is expected-plan evidence only; contemporaneous
shortfall remains the deterministic dispatcher's responsibility and the existing guard is
still final. Failures raise, are typed, retain the deepest valid observation/projection/
model/plan/setpoint and solver-stage identities, and never create a fallback command.

Successful planning is recorded as `service-mpc-setpoint-decision-evidence-v1`. It binds the
public observation and issue, projection, versioned settings, model dimensions and digest,
ordered delivery targets, both successful solver stages with feasibility-selection
evidence, selected plan and first interval, absolute command, operator notes, and controller
latency. Wall-clock and solver-stage latency are excluded from the deterministic identity;
the selected solution and selection audit remain included. Evidence construction rejects a
command whose decision, observation, issue, projection, settings, model, plan, or first-step
power differs from the planning record.

Transition closure is intentionally explicit. The controller refuses another solve while a
setpoint is pending. `observe_transition` requires the typed
`MPCSetpointIntervalExecutionV1`, proves that its first public boundary and commitment state
match the original controller observation, proves that its terminal boundary and commitment
match the next public observation, and then attaches both the complete physical ledger and
the existing plan-versus-public residual. Execution and residual must be present or absent
together. This prevents the optimizer from being credited for a plan without evidence that
the declared command actually reached the guard and plant.

Twelve focused tests cover the distinct descriptor/protocol, rejection of historical
settings, first-decision digest closure, deterministic identical-public-input behavior,
private-episode isolation, a complete four-decision short episode, foreign-execution
rejection, decision/stage/closure substitutions, missing-forecast and solver failures,
stable failure identity, and mandatory pending-transition reconciliation. Across the short
episode, every plan closes with 15 physical records per command and zero requested target
overshoot. The combined historical/new MPC controller, setpoint, dispatcher, and execution
surface passes **56 tests** before the added identity fixture. The complete repository with
that fixture passes **1,069 tests with 2 skips** in 34.96 seconds; the sole warning remains
the unchanged test-client deprecation. Black, compilation, Git whitespace, and isolated
strict types for both new production modules pass. No full trajectory, registered outcome,
holdout, simulator change, objective change, or parameter tuning occurred.

The pre-run comparison exposed one inconsistency in implementation gate 9 of the frozen
protocol. At the identical initial rate-35/seed-0 public observation, digest
`d1d2b3de1cf6314d5cfe857c3f49f7fe48744929853439fa77ebd07b0531ec1c`, the new controller's
first complete planned interval is numerically identical to the historical controller's.
The historical model/plan are
`a0e5b312ffa5ddd055d502ce5205c520dc642cd801e4286ed82924108c5d54d9` and
`782220137621a4578abac42ebd36800b028d22f6aea25f86e2c8dbe93b6d08de`; the new model/plan are
`9fc997f369fc7d83d50b5ade834e4ef63bbb1f720a4aab47923a1d997821d120` and
`a26f020ef6e55b4a225f796d940d7329f235db16731e07ce5310362d83b8dbcc`, with setpoint digest
`6066dc1d012b7466eb34eb32baba8b58cdd1b909f6743d5f753c7fd0d1531cdf`.

The difference is mandatory provenance, not a mathematical drift: the model identity binds
the settings digest, and the plan binds both model and settings. The new feasibility-
preserving settings digest is intentionally distinct from the historical strict-projection
digest. It is therefore impossible to require both a separately versioned settings/hash
domain and byte-identical old model/plan digests. The code will not falsify provenance to
make the gate pass. Before authorization, a narrow protocol clarification must replace
byte-identical model/plan identities with an executable structural-equivalence obligation
while retaining exact public-observation identity and freezing the new digests. This finding
blocks the rollout gate only; it does not prevent implementation of the outcome-free run
evidence validator and admission audit next.

## 2026-09-04 — Cross-version identity gate clarified before admission

The impossible part of implementation gate 9 is now resolved without editing the frozen
protocol or weakening its scientific invariant. The pre-outcome addendum
`CAUSAL_SERVICE_MPC_SETPOINT_DISPATCH_CLARIFICATION_001.md`, raw SHA-256
`5aecfc8e87b87deeb8fb03a4f226854f1b6245ec762209325f88c9c08aaea349`, supersedes only
the demand for byte-identical model and plan digests across deliberately different settings,
controller, and solution-selection identities. Every other implementation gate, S1--S8,
rollout rule, plant contract, optimization coefficient, and tolerance remains unchanged.

The clarification freezes an outcome-free three-boundary counterfactual on the already known
historical rate-35/seed-0 path. At all three public boundaries, the historical and new
forecast projections are exactly equal. Replacing only the new model's settings digest with
the historical digest makes the complete typed models exactly equal, including variables,
bounds, integrality, objectives, sparse rows, deliveries, initial state, forecast values, and
metadata. The complete ordered plan steps, delivery targets, horizons, and all six reported
plan scalars are also exactly equal—not merely tolerance-close. The differing model, plan,
and solver-vector digests remain in stored evidence because they correctly disclose the new
settings domain and feasibility-preserving selection policy.

The addendum binds exact public-observation, forecast-issue, projection, historical-model,
historical-plan, new-model, new-plan, and unexecuted-setpoint identities for decisions 0--2.
The new model/plan/setpoint triples are respectively
`9fc997f369fc...` / `a26f020ef6e5...` / `6066dc1d012b...`,
`6ee66de7f236...` / `505669051798...` / `9c3190dac065...`, and
`6119c5ddb426...` / `7acc746a9f63...` / `de427a3d6991...`.
No new setpoint was executed and no physical trajectory or performance outcome was opened.
The clarification explicitly remains insufficient on its own: an executable fixture must
reproduce all identities and equalities before the complete admission audit can authorize a
full mechanism run.

## 2026-09-04 — Three-boundary equivalence and run-level evidence gate implemented

The executable form of clarification 001 now reconstructs the first three historical
rate-35/seed-0 public boundaries without applying any new setpoint. At every boundary it
independently builds and solves the historical and setpoint settings variants. It reproduces
all 24 frozen observation, public-issue, projection, old-model, old-plan, new-model, new-plan,
and setpoint identities. The two complete forecast projections are exactly equal; replacing
only the settings digest makes each complete typed model exactly equal; and every delivery,
horizon, ordered plan step, and reported plan scalar is exactly equal. The test explicitly
requires the solver-vector identities to remain version-separated rather than laundering
them into historical provenance.

`build_mpc_setpoint_run_evidence` is the new fail-closed run boundary. It accepts only typed
setpoint-controller events and exact per-execution dispatcher timing groups under one
`ServiceMPCSetpointSettingsV1`. It validates contiguous successful decisions, at most one
terminal failure, shrinking planning horizons, closed-versus-pending state, the exact public
observation chain, the complete physical boundary and commitment chain, final public
observation identity, one dispatcher timing for every minute record, and no fallback. A
successful run must contain exactly the settings-declared number of fully closed decisions.
An explicitly failed or pending run is retained rather than converted to a partial success.

The validator independently reconciles each physical execution against the controller's
public residual: interval duration; solar, battery-net, electrolyser, DAC, methanator, and
curtailment mean power; terminal battery/H2/CO2/CH4 state; and cumulative H2/CO2/CH4
production. It aggregates planned, target, requested, guarded, and applied energy; requested,
guarded, and applied target errors; empirical nearest-rank controller, solver-stage, and
dispatcher latency distributions; and ordered event, execution, and application identities.
Every minute's applied target deficit is decomposed into resource shortfall, guard reduction,
and plant refusal/rerouting. It reports absolute 1e-7 kW overshoot and reachable-tracking
violations, simultaneous charge/support, applied-support authority violations, and explicit
zero accounting/chain violations. A deterministic evidence-graph digest excludes all timing,
while the outer run digest binds the complete raw payload including measured timing.

Adversarial tests alter the public residual chain, terminal residual state, minute-latency
cardinality, outer run payload, and deterministic graph identity; every substitution fails
closed. Retiming an otherwise identical run changes the outer artifact but preserves the
deterministic graph. A four-decision short run yields 60 ordered minute applications, eight
successful solver stages, no fallback, no requested target overshoot, no reachable-tracking
violation, and no simultaneous charge/support request.

Two previously implicit implementation-gate cases are now explicit. A 900 kW operator-
derated stack with 400 kW support authority reaches its absolute target using exactly 500 kW
solar plus 400 kW reserve. Separate typed minute fixtures show a running stack refusing power
because of its ramp limit while H2 headroom remains available, and because the H2 tank is
full while the guard leaves the request unchanged. The ledger distinguishes these physical
causes through headroom, commitment mode/counters, and guard-to-plant power gaps.

The focused setpoint/run/execution surface passes **38 tests** after these additions. The
complete repository passes **1,077 tests with 2 skips** in 42.78 seconds; the only warning is
the unchanged Starlette/httpx test-client deprecation. No 72-hour setpoint trajectory,
registered mechanism cell, performance metric, holdout, PPO training, simulator change,
objective change, or parameter tuning ran. The remaining prerequisite is a clean committed
revision audit that maps every frozen implementation gate to exact executable evidence and
freezes the one-shot runner contract before any outcome is opened.

## 2026-09-04 — Fixed one-shot setpoint mechanism runner implemented outcome-free

`plant_backend.rl.v2.mpc_setpoint_experiment` now encodes the only permitted first
absolute-setpoint rollout. Its public CLI accepts one argument, an absolute new `.json`
destination outside the repository. Rate 35 kg/h, seed 0, the 72-hour horizon, 288 decisions,
4,320 physical minutes, the default setpoint-controller settings, manifest and exogenous
episode identities, protocol and clarification hashes, dependency-lock hash, eight outcome
gates, and claim boundary are code-bound rather than operator choices. The no-same-revision
retry rule remains a preregistered governance obligation: the stateless CLI enforces a new
non-overwriting destination but cannot discover artifacts deliberately written elsewhere.
`make v2-mpc-setpoint OUTPUT=/absolute/external/result.json` is the corresponding
frozen entry point; it has not been invoked.

The internal execution loop is separately testable with a short task but the public boundary
always constructs the frozen cell. For every decision it computes one typed command, applies
it through the opt-in minute dispatcher, retrieves the exact interval execution, reconciles
the next public controller boundary, and only then admits the dispatch-timing group. It
passes the resulting typed events to the existing deep run validator. Controller,
environment, transition, truncation, and evidence-validation failures are retained without a
fallback. When a failure prevents construction of a valid evidence graph, the artifact
labels the material unvalidated and retains the controller events, dispatcher timings, and
any unreconciled execution rather than presenting a partial success.

S1 requires a complete evidence graph with exactly 288 closed decisions, 576 attempted and
successful solver stages, 288 interval executions, 4,320 physical minute applications,
matching identity-list cardinalities, matching environment metrics, and zero fallback. An
S1 failure mechanically makes S2--S8 `not_evaluated`. After S1, the runner independently
checks target overshoot and simultaneous battery modes (S2), reachable tracking and deficit
decomposition (S3), productive commitment behavior (S4), reserve/SOC/validity/shutdown
safety (S5), digest/chain/accounting/authority closure (S6), the frozen delivery-direction
diagnostic (S7), and separate controller/dispatcher latency limits (S8).

The runner validates the frozen documents and lock before an outcome, requires a complete
clean Git content identity, verifies that revision/content/lock identity is unchanged before
publication, and rechecks it after publication. Output uses a temporary file, byte and
canonical-JSON round-trip verification, an atomic no-overwrite hard link, filesystem sync,
and a final byte digest. The report verifier recomputes its domain digest, deeply verifies
run-evidence digests when present, recomputes all gates and status, and rejects substitution.

Nine focused admission tests pass in 1.19 seconds. They run one real four-decision/60-minute
setpoint episode, reject episode/controller horizon mismatch, prove S1 suppresses downstream
interpretation, exercise every S1--S8 threshold, publish and reopen a typed failure artifact,
reject overwrite, refuse a dirty source before calling the executor, reject report
substitution, and prove the CLI has no seed/rate/tuning surface. The implementation admission
document now maps all twelve outcome-free gates to named executable tests. Final formatting,
types, focused regression, full regression, and clean-revision checks remain pending. No
72-hour setpoint outcome, performance metric, holdout, training, simulator change, or tuning
was opened in this slice.

## 2026-09-04 — Setpoint mechanism implementation gate admits one exact revision

The runner and admission mapping were committed as
`a968a343f50e5724ea74b75cc3debd1509fc1d80`. The worktree was clean immediately after the
commit. From that exact revision, the six focused runner/run/controller/execution/dispatcher/
solver modules passed **79 tests in 9.13 seconds**. The complete repository then passed
**1,086 tests with 2 skips in 40.78 seconds**. The only warning remains the pre-existing
Starlette/httpx `TestClient` deprecation; post-test Highs lines are upstream solver logging.

Black reported both new modules unchanged. Python compilation passed. Isolated strict MyPy
reported no issue in the production runner. Git whitespace validation passed. The lock
remained `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`, and the
protocol/clarification/lock/one-shot-contract identity test passed. The implementation audit
therefore marks all twelve outcome-free gates as passed and admits exactly one registered
rate-35/seed-0 72-hour mechanism cell from `a968a343`.

No cell was started during admission. The exact executable source will be preserved through
a fresh clean worktree pinned to `a968a343`, so this later documentation record does not
silently alter the evaluated revision. The external destination must be preregistered before
execution. S1 failure will suppress S2--S8; any complete result will be retained without a
same-revision retry. No performance conclusion, multi-seed extension, RL training, HEPA
work, realism change, controller tuning, or commercial claim is authorized yet.

## 2026-09-04 — First absolute-setpoint mechanism cell passes; service objective saturates

The exact external destination was committed before execution in
`CAUSAL_SERVICE_MPC_SETPOINT_EXECUTION_REGISTRATION.md`. A second worktree could not be
created because the app's automatic permission review timed out twice. The already-clean
main worktree was therefore switched temporarily to detached implementation revision
`a968a343f50e5724ea74b75cc3debd1509fc1d80`, verified clean and source-resolved, used for the
single registered invocation, and restored to `codex/environment-v2-research` immediately
after publication. No source file changed, no concurrent experiment ran, and there was no
retry.

The runner completed in **166.057 seconds** with status `complete_all_gates_passed`. S1
closed all 288 decisions, 576 solver stages, and 4,320 physical minute applications with no
failure or fallback. S2 found maximum requested overshoot `4.55e-13 kW`, zero overshoot
minutes, and zero simultaneous charge/support. S3 found zero reachable-tracking violation,
zero unattributed deficit, and maximum decomposition error `4.55e-13 kW`. S4 recorded
102,600 productive seconds and no aborted or forced starts. S5 recorded no reserve penalty,
shutdown, or invalid decision; minimum SOC was 57.76% and terminal SOC 99.53%. S6 found no
digest, chain, accounting, decomposition, or authority violation. S7 delivered the full
2,520 kg, 1,872.286 kg above the failed share-MPC result. S8 measured controller p99/max at
0.988/1.401 seconds and dispatcher p99/max at 0.135/0.352 milliseconds.

The external 82,419,626-byte artifact was reopened with the production verifier. Its raw
SHA-256 is `6b90ea6eb907c338fdf1aaa695ace173ad4cbe0b23b6eb7fef18bb3c513b4cf0`;
report SHA-256 `dfb3bac1b19afd78306f40110502fadfacd585ca26d54855a952a28e0b1cb5cc`;
run-evidence SHA-256 `1060a46c275277025add905d2c6aa5345b5c8dfe3b7c2bc1ee7b720a0772a592`;
and deterministic graph SHA-256
`dc0dbf78c4670a41473e9a2ccb8f406e73f009cfbde5f40ea5d81de8056407e9`.
The recorded source revision, manifest, exogenous episode, gates, counts, and identities all
revalidated exactly.

The mechanism result is positive and the controller result is deliberately inconclusive.
The setpoint MPC reaches the same 2,520 kg delivery ceiling as the exact-cell A0 heuristic
and future-information upper bound, so delivery gain is zero. It raises reward by 0.0707 and
reduces curtailment by 3.418 MWh, but accepts 2.628 MWh more process energy, increases energy
intensity by 1.043 kWh/kg (3.49%), and cycles the battery 3.856 MWh more (58.36%). The current
objective does not price the lifecycle or electricity consequence of that trade.

The dispatcher offered every reachable target within numerical precision, while the plant
accepted 1,210.584 kWh less electrolyser energy over 76 ramp/storage refusal or rerouting
minutes. That entire deficit is attributed to the plant, not weather resource or the guard.
The MILP planned only 28.500 kWh battery support, whereas real-time target realization
requested 3,602.453 kWh and physically applied 3,546.863 kWh. Safe target realization is
therefore solved, but sub-interval battery use and component acceptance remain material
model/objective residuals.

The architecture decision is to retain absolute setpoints, deterministic minute dispatch,
independent safety, and typed residual closure. There is no basis to claim MPC superiority:
the environment saturates and A0 still uses a different action abstraction. The next bounded
work must first put a transparent heuristic and future RL policy through the same setpoint
door, then replicate mechanism robustness across seeds. The leading realism increment is
time-varying electricity value plus explicit cycling/equipment wear, because it turns the
observed tied-service/58%-throughput difference into a commercially meaningful decision.
No new matrix, environment change, or training was opened while documenting this result.

## 2026-09-04 — Common setpoint authority phase 1 closes without an outcome

The outcome-free controller-parity phase frozen in
`COMMON_SETPOINT_AUTHORITY_PROTOCOL.md` is now implemented. Its unchanged raw SHA-256 is
`088d6ec4bbf1c1ef1b8743166a09a1c7f7c9cf8203914678498471b99385b801`; the dependency lock
also remains unchanged at
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.

`ProcessSetpointTargetsV1` is a controller-neutral value containing exactly four finite,
non-negative absolute targets: total electrolyser input, DAC input, methanator input, and a
battery-charge ceiling. It contains no optimizer, plan, forecast, solver, controller, or
future-state identity. `ProcessSetpointActionAdapterV1` accepts exactly four normalized
policy coordinates, rejects booleans, strings, wrong shapes, and non-finite inputs, retains
the raw and clipped vectors, and maps `[-1, 1]` linearly into the resolved active equipment
limits. It does not allocate resources or step the plant. Historical Gym spaces and
checkpoints remain untouched.

The existing `reserve-aware-proportional-unsupported-loads-v1` arithmetic has been extracted
once into `allocate_process_setpoint_targets`. The typed result binds the targets,
contemporaneous solar, reserve-aware support envelope, four solar routes, battery support,
unallocated solar, reachability, ordinary plant action, and a deterministic digest. It
requires a sanitizer-fixed-point action, exact route/action and solar-bus closure, bounded
support, no simultaneous battery charge and discharge, target ceilings, and exact tracking
whenever the electrolyser target is reachable. Battery support remains deterministic
dispatcher authority, never a learned fifth action.

The private duplicated MPC allocator was deleted. Both new minute dispatch construction and
the existing `MPCMinuteDispatchV1` self-audit now delegate to the common kernel using a plain
`ProcessSetpointTargetsV1`. A recording-spy test proves both calls cross that shared boundary;
the old day, night, surplus, deficit, reserve, operator-derating, proportional-scarcity, and
dispatch-grid fixtures remain exact. The three frozen rate-35/seed-0 model, plan, setpoint,
and physical-boundary identities also remain exact, and the registered one-shot contract
digest remains
`421561b2ee75650024c11fdf55d806012275b44838a33f6b3714635fa689e304`.

The first focused adapter/dispatcher run found two assertion-message mismatches and no
implementation failure; the assertions were corrected without weakening input rejection.
The final adapter and dispatcher surface passes **41 tests in 0.66 seconds**. The wider
compatibility selection passes **114 tests in 5.14 seconds**, including the frozen
three-boundary equivalence fixture, the real four-decision/60-minute short executor, the
registered-runner admission tests, and historical action/controller paths. The complete
repository passes **1,106 tests with 2 skips in 41.81 seconds**. The only warning is the
unchanged Starlette/httpx test-client deprecation; post-test Highs messages remain upstream
solver logging. Black, Python compilation, strict targeted MyPy, Git whitespace validation,
and lock identity all pass.

No 72-hour controller episode, protected outcome, multi-seed matrix, policy training,
environment modification, objective modification, HEPA fitting, or NLP evaluation ran in
this phase. The next admissible slice is the separately versioned common command and
execution door: an opt-in four-target Gym contract, a transparent heuristic using that same
door, and an explicit adapter retaining the MPC's original provenance. Economic or wear
pressure remains behind that parity gate.

## 2026-09-04 — Observation-bound common command seam implemented

The next controller-parity phase was frozen in
`COMMON_SETPOINT_COMMAND_PROTOCOL.md` before implementation. The architecture audit rejected
adding electricity economics directly to the identity-bound `EpisodeConfigV2` while the
controllers still had unequal execution doors: that ordering would change the task before
isolating action authority. Economic calibration remains pending behind this parity phase.

`ProcessSetpointEnvelopeV1` now resolves the process ratings, battery charge rating,
operator-limited battery support, and reserve SOC. `ProcessSetpointCommandV1` binds one
non-negative decision index, exact public-observation identity, positive hold duration,
issuer ID, source schema and digest, four absolute targets, and that complete authority
envelope. Construction rejects malformed identities and any process or charge target above
the active envelope. Identical values produce identical typed commands and digests.

`dispatch_process_setpoint_command` rejects stale authority, derives contemporaneous support
from the existing reserve-aware safety primitive, and delegates resource allocation to the
single common target kernel. Its typed minute result retains the complete command, physical
boundary identity, support envelope, allocation, ordinary plant action, and deterministic
digest. It has no forecast, optimizer, private episode, reward, or future-state access.

The MPC projection adapter retains the exact original `MPCProcessSetpointsV1` schema and
SHA-256 as source evidence while mapping its values into the common command. The existing
MPC dispatcher now calls the generic dispatcher and wraps that allocation in the unchanged
MPC minute-evidence schema. Its independent constructor audit still recomputes the allocation
kernel. A two-layer spy verifies one common dispatch followed by one MPC evidence audit.

Seventeen new command tests cover operator/reserve resolution, malformed timing and identity,
all four envelope overruns, deterministic source binding, surplus and deficit behavior,
physical-boundary binding, stale constraints, and invalid boundary/solar input. The focused
command/MPC/frozen-identity selection passes **41 tests in 3.99 seconds**; strict targeted
MyPy and Black pass. The complete repository passes **1,123 tests with 2 skips in 42.04
seconds** with only the pre-existing Starlette/httpx deprecation warning. No Gym contract,
physical outcome, training run, economics change, or controller comparison was opened. The
next commit must add the separately identified four-target Gym door and common-door heuristic.

## 2026-09-04 — Controller-neutral RL and heuristic execution door completed

The outcome-free authority-parity boundary is now implemented end to end. The new opt-in
`PlantSetpointEnvV2` publishes environment ID
`plant-env-v2-common-setpoint-alpha` and interface digest
`a192861bdd43d3d822cb99c099450c7dbb3a10cabb69faae70e875ee92016b38`. It reuses
the historical V2 encoder and exact 57-field observation order but replaces the five-field
share/support action with exactly four normalized absolute targets. The default
`PlantEnvV2`, its five-action space, and its checkpoint contract remain unchanged. For the
same seed, both environments sample the same exogenous episode, while their environment and
checkpoint identities differ and reject one another in both compatibility directions.

Every normalized vector is rejected before state movement if it has the wrong shape, a
boolean/string entry, or a non-finite value. Valid entries are retained before and after
clipping, decoded through `ProcessSetpointActionAdapterV1`, bound to the exact public
observation and active envelope in `ProcessSetpointCommandV1`, and executed for fifteen
one-minute physics steps through `dispatch_process_setpoint_command`, the independent
`validate` guard, and the simulator. The Gym path returns the source decision, common
command, reward and plant metrics, and a complete typed minute execution chain. Raw wall-
clock dispatcher timings remain available to dedicated controller runners but are replaced
by a deterministic call count in Gym `info`, because otherwise identical seeded transitions
would violate Gymnasium's deterministic contract.

`ProcessSetpointMinuteApplicationV1` and
`ProcessSetpointIntervalExecutionV1` close the controller-neutral physical evidence path.
They bind each dispatch to a physical boundary, guarded and applied actions, adjustments,
solar/electrolyser bus balances, process target authority, next boundary, contiguous minute
order, and the command hold duration. Evidence construction rejects action substitution,
process overdraw, missing minutes, and broken physical chains. This schema is deliberately
separate from the richer MPC-only execution schema: it requires no optimizer, forecast,
plan, solver, or commitment-transition evidence.

The first smoke execution exposed an incorrect draft invariant rather than a plant defect.
When a starting, ramp-limited, or storage-constrained process refuses offered solar, the
existing physical allocator conserves that power by charging the battery if possible. The
draft generic evidence compared applied charging only with the direct battery share and
therefore rejected legitimate recovered spill. The final invariant follows the already
validated MPC boundary: process and battery-support draws may never exceed guarded target
authority, while the battery target limits only the dispatcher's direct route. Execution
evidence separately records direct dispatch, its ceiling, applied charge including spill,
and recovered/guard-redirected power. This interpretation is now explicit in the frozen
protocol and machine-readable interface contract.

`LegacyHeuristicSetpointControllerV1` is a transparent authority adapter around the existing
`HeuristicController`; it adds no forecast, delivery context, commitment logic, or policy
rule. At each public boundary it retains the exact sanitized A0 action and notes, converts
solar shares into absolute process targets using the measured boundary solar, adds requested
battery support only to the electrolyser target, and binds that evidence into the common
command. If A0's residual battery share would exceed the physical charge rating, only the
direct charge target is capped and that fact is explicit in the evidence. The arm is labelled
as behaviorally distinct from A0-share because absolute targets, rather than shares, are held
as solar changes during the interval.

An exact cross-source fixture constructs the same `(0, 0, 0, 2500 kW)` target through the
heuristic adapter, normalized RL adapter, and MPC provenance adapter. Their issuer and source
digests remain distinct, while target values and active envelopes are equal and the generic
dispatcher produces identical physical allocations. Existing MPC tests continue to prove
that its legacy wrapper delegates to this dispatcher and independently revalidates the common
allocation.

Seventeen new tests cover the distinct Gym/checkpoint identity, identical exogenous sampling,
Gymnasium contract, normalized target mapping, deterministic source binding, complete minute
execution, recovered-spill attribution, invalid/clipped inputs, stale indices/observations/
durations/envelopes, chain tampering, A0 conversion and charge limiting, deterministic reset,
and three-source authority equality. The focused setpoint selection passes **86 tests in
1.28 seconds**; the wider V2 compatibility selection passes **105 tests in 2.05 seconds**.
The complete repository passes **1,140 tests with 2 skips in 41.84 seconds**. The only warning
is the pre-existing Starlette/httpx test-client deprecation; trailing HiGHS messages remain
upstream solver logging. Black, Python compilation, strict targeted MyPy, Git whitespace
validation, and the unchanged dependency-lock digest
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` pass.

No protected rollout, multi-seed comparison, RL training, HEPA fit, economics change,
equipment-wear change, or commercial outcome was opened. This commit establishes equal
actuator semantics, not controller quality. The next task-design slice must make tied service
outcomes economically discriminative with explicit energy value, cycling/equipment wear, and
a sustainable terminal condition before heuristic, MPC, and RL are compared.

## 2026-09-04 — Economic/wear task implemented without opening a controller outcome

The registered setpoint mechanism result established the next task-design requirement: MPC,
A0, and the certified ceiling all delivered 2,520 kg on the only admitted mechanism cell,
while MPC accepted 3.49% more process energy and moved 58.36% more energy through the battery.
The old delivery objective could not decide whether this behavior was economically useful.
This increment therefore adds a separately identified task rather than tuning a controller or
mutating the historical objective.

The evidence review was deliberately narrow. ACER's wholesale-market reporting supports
time-varying, low, and negative day-ahead electricity values as a genuine flexibility signal.
ENTSO-E provides the eventual traceable European price-data boundary. The Clean Hydrogen JU
KPI table provides an AEL 2024 capital target of 480 EUR/kW, alongside explicit system-boundary
and interdependence warnings. DOE storage assessments treat replacement, augmentation, cycle
life, and calendar life separately; the Storage Innovations LFP scenario's 2,640-cycle value
is retained only as a sensitivity anchor. These sources support which variables to expose.
They do not calibrate this plant, justify a vendor-specific degradation law, or turn the alpha
into a financial forecast. All source links, numerical statuses, and non-claims are frozen in
`ECONOMIC_WEAR_TASK_PROTOCOL.md`.

The new opt-in `PlantEconomicSetpointEnvV2` publishes environment ID
`plant-env-v2-economic-wear-alpha`, interface digest
`0b7c71e56a939c1a1acd92e93a2fce072de43fa962a5371d94715c2d3382197a`, and checkpoint
schema 5. It inherits the common four-target action without widening controller authority.
The first 57 observations are unchanged. Five append-only observations publish the current
hourly opportunity price and exact public 1-hour mean plus 6-hour minimum, mean, and maximum,
producing a 62-float Gym observation. Values are divided by 300 EUR/MWh and bounded to
`[-1, 1]`. The default 24-hour stress shape repeats deterministically, ranges from -10 to
150 EUR/MWh, and has explicit low-midday/high-evening structure. It is not labelled as a
historical day or probabilistic forecast. Controller intervals must divide an hour and start
on a matching boundary, ensuring one price applies to every physical minute in a held command.
The detached `EconomicControllerObservationV1` carries the same physical observation and raw
tariff values, plus tariff and assumption identities, to structured heuristic or optimizer
controllers. This closes an implementation-review gap in which only the Gym vector initially
had direct price access; non-Gym controllers no longer need to scrape step metadata.

The economic assumptions have default digest
`dce6ba3082e0e2a16c1e341589afb25d8f1d0b2da9124522499dafd705fe6ee2`. The complete
default checkpoint digest is
`455bde44efc7f15783ad2f28133630136c2055405f075cc6495f2f6532fcde5d`; every tariff
point, coefficient, plant/episode setting, constraint, physical interface, and diagnostic
legacy reward is inside that identity. The same seed retains the exact physical exogenous
episode digest across historical V2, common-setpoint V2, and economic V2. The economic audit
adds a second episode digest over the physical episode reference and tariff so different
economic conditions cannot be called the same paired cell.

The returned scalar is now an additive EUR ledger with seven visible components: delivered
service value, unmet-delivery penalty, measured accepted-solar opportunity cost, bus-side
battery-throughput reserve, electrolyser-life exposure reserve, change in CH4-equivalent
endowment liability, and change in battery-energy endowment liability. The previous objective
version 3 still executes to preserve established physical diagnostics, but it is returned under
`legacy_reward` and cannot affect the scalar. Historical `EpisodeMetricsV2` remains structurally
unchanged; a separate `EconomicEpisodeMetricsV1` accumulates the new components and physical
cost drivers.

Accepted renewable energy is authoritative generated-minus-curtailed kWh, not requested or
guarded power. The plant has no grid port, so the tariff is explicitly an export-opportunity
shadow price. Curtailment remains available and negative prices are therefore floored at zero
for cost: a negative public observation creates cheap flexibility but cannot invent revenue
for consumption. Battery wear uses physical charge plus discharge throughput and a transparent
linear reserve of replacement EUR/kWh divided by twice the full-cycle life. It reports
equivalent full cycles separately. Defaults of 300 EUR/kWh and 2,640 cycles are sensitivity
assumptions, not a warranty or chemistry model.

Electrolyser exposure converts exact simulator counters into equivalent hours: operating time,
a 50% premium on low-load time, and declared equivalent-hour coefficients of 2/0.25/2/4 for
cold starts, hot starts, aborted starts, and forced stops. A linear 480 EUR/kW over 80,000
hours reserve then makes that exposure economically visible. Only the capital anchor is tied
directly to the cited JU KPI. The lifetime and event coefficients remain synthetic and must be
ablated. The code makes no degradation, failure-time, maintenance, or stack-replacement claim.
It intentionally does not assign DAC or methanator wear because those components lack the
necessary operating and thermal states.

The sustainable terminal condition is implemented as dense but path-independent shaping.
CH4-equivalent inventory is saleable methane plus the smaller stoichiometric methane potential
of stored H2 and CO2. Any deficit below the episode's initial CH4-equivalent inventory and
battery energy carries a liability; surplus above the start has no credit. Each interval pays
only the liability change, so depletion followed by restoration cancels exactly and the whole
episode telescopes to the final deficit cost. This avoids both free liquidation of sampled
starting resources and terminal overproduction gaming without adding private state to the
policy observation.

Eleven new tests cover exact tariff integration across midnight; malformed price/assumption
rejection; negative-price semantics; liability telescoping; monotonic battery/electrolyser
exposure cost; distinct 57/57/62 observation and 5/4/4 action contracts; physical exogenous
identity; exact appended tariff values; scalar/ledger summation; preservation of the diagnostic
legacy reward; tariff-dependent reward/checkpoint/economic-episode identity with identical
physical transitions; deterministic reset/step; Gymnasium compliance; and cadence rejection.
The first tariff-window test failed because its hand-calculated six-hour cross-midnight mean
omitted the final half-hour. Recalculation showed the implementation's 4.0 EUR/MWh mean was
correct; the erroneous expected value was fixed, with no production change.

The focused economic/common-door selection passes **28 tests in 0.47 seconds**. Strict targeted
MyPy passes with import following disabled and its cache disabled; an earlier import-following
invocation surfaced existing legacy typing debt, and the first cached targeted invocation hit
the already documented mypy unresolved-placeholder internal assertion. Black, compilation,
Git whitespace validation, and the unchanged dependency-lock digest
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` pass. The complete
repository passes **1,151 tests with 2 skips in 40.54 seconds**. Its only warning is the existing
Starlette/httpx test-client deprecation; trailing HiGHS text is unchanged upstream solver output.

No heuristic, MPC, random-policy, or RL episode was compared on the new objective. No price
dataset was downloaded, no coefficient was tuned against an outcome, no protected seed was
opened, and no HEPA or language-layer claim changed. The next bounded step is a short,
predeclared common-door heuristic/MPC smoke comparison to establish whether this task actually
creates nonzero economic separation at comparable delivery and zero hard violations. Only a
positive mechanism check justifies a frozen multi-seed tournament or RL training.

## 2026-09-04 — Economic execution doors hardened and mechanism smoke frozen

Pre-experiment integration review found one remaining route asymmetry. Normalized Gym and
generic `ProcessSetpointCommandV1` actions received the economic overlay, but the inherited
MPC-specific `step_setpoints` method still returned the diagnostic version-3 reward. Simply
adapting MPC output to a generic command was not sufficient for the existing MPC controller:
its next solve is fail-closed until `observe_transition` reconciles the richer
`MPCSetpointIntervalExecutionV1`, including commitment boundaries and planned residuals. A
future comparison could therefore either lose MPC provenance or silently score the wrong
objective.

`PlantEconomicSetpointEnvV2` now captures one immutable pre-step economic boundary and applies
one shared post-step ledger to both the generic process-command door and the rich MPC setpoint
door. Normalized Gym actions continue to reach the generic door dynamically. The inherited
legacy share-action method is explicitly rejected before physical movement, so no public
controller route can silently bypass the economic objective. Checkpoint schema 5 now records
all four execution-door semantics; this intentionally changes the default checkpoint digest
from the earlier `be79106` value to
`49da4e3aa6a97fd07a2c87980d0b7e4138392bf189f9383b9030511774ab6215`.

Two tests were added. A tiny real service-MPC solve executes through the economic environment,
returns the economic scalar, retains its diagnostic legacy reward and rich execution object,
and successfully closes the controller's pending transition. A separate fixture proves the
legacy share door raises before decision counters or telemetry move. The economic suite now
passes 13 cases; the economic plus common-door selection passes **30 tests in 0.94 seconds**.

Before opening a controller trajectory, `ECONOMIC_MECHANISM_SMOKE_PROTOCOL.md` freezes exactly
one six-hour, rate-35, seed-0 development cell. Arm H is the transparent common-door legacy
heuristic. Arm M is the current delivery-oriented service MPC with a six-hour task and planning
horizon; it is explicitly not relabelled as economic MPC and neither arm uses tariff in its
decision. Both receive the same physical/economic episode. E1--E6 require complete execution,
typed authority evidence, zero shutdown/out-of-range/forced-stop events, identical delivered
and unmet methane within `1e-6 kg`, more than 1 EUR net-value separation plus a material raw
driver difference, exact additive/liability accounting, and a one-cell descriptive claim
boundary. A failure is retained without same-revision retry or coefficient tuning.

The current complete repository gate passes **1,153 tests with 2 skips in 45.06 seconds**.
Strict targeted type checking, Black, compilation, dependency-lock identity, and Git whitespace
also remain clean; the warning and trailing HiGHS logging are unchanged from the preceding
entry.

No six-hour action has been requested, no solver outcome beyond the tiny unit fixture has been
used, and no heuristic/MPC ranking has been observed. The next commit must add a hash-bound,
clean-source, exclusive-output runner and its outcome-free tests. Only after that runner is
committed may the single registered cell execute.

## 2026-09-04 — Controller-neutral RL integration completed and exercised end to end

The integration audit separated physical authority from training-stack readiness. The common
setpoint work had already made heuristic, normalized Gym, and MPC targets physically
comparable, and the economic task already exposed a 62-observation/four-action Gym interface.
However, the maintained trainer, checkpoint loader, and policy evaluator still assumed the
historical V1 30-observation/five-action environment. A policy could be trained through the new
Gym class, but there was no repository-supported way to prove which V2 task produced its bytes,
load it against that exact task, reconstruct its structured observation, bind its output to a
common command, or compare it with a non-learning controller through retained physical
evidence. That—not another environment change—was the remaining controller-neutral RL gap.

Checkpoint admission is now environment-explicit. `PolicyEnvironmentContract` carries ordered
observation and action names, interface version, environment ID, interface digest, and optional
full task digest. Historical callers that omit it retain the frozen V1 behavior. Both V2
four-target tasks supply it explicitly and require the full checkpoint contract, so a model is
rejected when plant design, episode distribution, safety constraints, diagnostic objective,
tariff, economic assumptions, observation semantics, or action semantics differ. Non-mock
policy bytes carry their own SHA-256 and are verified before deserialization. Missing or
malformed metadata, a missing digest, changed bytes, or a wrong task fails rather than creating
or substituting a policy. Fixed-action checkpoints remain visibly untrained wiring fixtures.

`NormalizedSetpointPolicyControllerV1` is the learned side of the common door. It reconstructs
the exact float32 observation from `ControllerObservationV2`; the economic variant appends the
same five causal tariff values used by Gym and validates tariff and economic-assumption
identities. Deterministic inference produces four normalized values, which the existing common
adapter clips and decodes into absolute electrolyser, DAC, methanator, and battery-charge
targets. Each decision binds the public physical observation, optional economic observation,
exact policy vector, checkpoint metadata, environment contract, raw/bounded action, decoded
targets, and issued `ProcessSetpointCommandV1`. The controller has no internal heuristic or idle
fallback. Inference failure is retained without physical movement.

The final review found one genuine fail-open seam in that adapter. Construction validated the
checkpoint against an environment containing frozen safety constraints, but `compute_command`
still accepted a constraints argument and did not compare it with the checkpoint-bound value.
A caller could therefore have changed observation normalization and target limits while
retaining the old task identity. The controller now stores and reports the admitted constraints
and rejects runtime drift before inference or state movement. It already enforced plant design,
commitment profile, observation kind, and economic identities.

The new `ControllerNeutralTrainingConfigV1` round-trips the complete physical or economic task,
seed, vectorization, PPO settings, checkpoint cadence, output, resume source, and notes. Nested
unknown keys are rejected. JSON numbers are normalized to their declared scalar types so
semantically equal values do not create different hashes; booleans, numeric strings, non-finite
values, invalid PPO ranges, and incompatible rollout/batch sizes fail early. Dry-run constructs
and resets the exact task without importing the optional trainer and reports 57/4 or 62/4
shapes, interface/full-task identities, episode arithmetic, and Stable-Baselines3 whole-rollout
rounding. Real training is explicit. Resume admission checks the exact task before loading.
Intermediate and final saves are complete checkpoint directories rather than unauthenticated
ZIP files. Existing paths are never overwritten, and missing dependencies or rejected resume
sources no longer leave an empty directory that could be mistaken for a run.

The paired evaluator creates independent heuristic and policy environments for each unique,
non-negative seed and requires their complete manifest identities to match. Both arms issue
`ProcessSetpointCommandV1` and retain every decision, interval execution, and one-minute
application. The learned arm additionally checks on every decision—not only reset—that its
structured encoder produced the exact vector returned by Gym; a two-decision fixture exercises
the stateful solar-trend field. Decisions are retained even if downstream dispatch subsequently
fails. Runtime timings remain in the report but outside the deterministic episode and
evaluation digests. JSON publication uses exclusive creation. Fixed checkpoints are labelled
integration smokes; trained checkpoints remain descriptive until seeds, benchmark arms,
selection rules, and statistics are separately frozen.

A real optional-dependency smoke used Stable-Baselines3 2.7.0, Torch 2.10.0, Gymnasium 1.2.0,
and NumPy 2.0.2 from the local `/opt/anaconda3` runtime. It trained PPO for two transitions on
the economic four-target task, saved and hash-verified the model, resumed it for two additional
transitions, reloaded the four-transition checkpoint, and completed a paired seed-11
heuristic/policy evaluation in **2.30 seconds**. This is intentionally far below one useful
rollout and is evidence only that the real serialization/resume/inference path works. The
project `.venv` does not contain the optional trainer, so the same regression test skips there
instead of pulling Torch into ordinary development or CI. No reward comparison from this smoke
is interpreted or retained as a controller outcome.

The already preregistered economic mechanism runner was also completed without opening its
trajectory. It validates both H and M reset identities before the first action, executes H
through generic command evidence and M through the rich transition-reconciled MPC door, retains
partial failures, applies frozen E1--E6 accounting/claim gates, requires clean unchanged source,
and publishes atomically to a new file outside Git. Review corrected draft controller IDs to the
actual heuristic and service-MPC issuer constants and made incomplete-arm accounting
reportable. The protocol and dependency lock remain exactly
`41c0ce9a1d49bcbfca34d88c0fb20b7fc7eb0ab91d15ed405aded512135cd795` and
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.
The registered six-hour H/M trajectory is still unopened.

The first complete-suite attempt provided a useful immutability check. A narrative clarification
had also been placed in `setpoint_spaces.describe()`. Although it did not change the interface
digest, that machine-readable limitation is embedded in the economic audit manifest and changed
the frozen seed-0 manifest from its preregistered identity. The manifest test failed before any
experiment ran. The code wording was restored and the clarification remains only in narrative
documentation; the frozen manifest and protocol identities are unchanged.

The final focused compatibility selection passes **134 tests with 1 optional-dependency skip in
2.72 seconds**. The disposable real-SB3 test separately passes in **2.30 seconds**. Strict
targeted MyPy passes on all nine touched source modules with import following and cache disabled;
Black, Python compilation, Git whitespace validation, protocol hash, and dependency-lock hash
pass. After restoring the frozen wording, the complete repository passes **1,176 tests with 3
skips in 42.90 seconds**. The only warning is the existing Starlette/httpx deprecation; trailing
HiGHS output remains upstream solver logging.

This increment completes the controller-neutral RL software integration, not the RL research
claim. No research-scale four-target policy, hyperparameter selection, protected multi-seed
evaluation, MPC/RL tournament, HEPA fit, language-layer experiment, field calibration, or
commercial result exists. The immediate empirical step remains the single frozen six-hour
heuristic/MPC economic mechanism cell. If E1--E5 pass, a separately preregistered development
tournament should establish whether meaningful non-saturated control headroom exists. Only
then is a bounded RL baseline scientifically justified; otherwise more environment realism is
required before compute is spent on learning.

## 2026-09-04 — Registered economic task-expression cell passed

The single preregistered six-hour H-versus-M mechanism cell was opened only after the runner
and controller-neutral RL integration were committed and the worktree was clean at
`a46aabf9cd3d8ee7387055c07b1c689e2d53f74f`. The protocol hash remained
`41c0ce9a1d49bcbfca34d88c0fb20b7fc7eb0ab91d15ed405aded512135cd795`, the dependency
lock remained `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`,
and the exclusive external destination did not exist.

The first shell invocation through `uv run --frozen` failed before Python initialized because
the sandbox could not write the global uv cache. It created no output, reset no environment,
and opened no controller trajectory. With source, protocol, lock, and destination unchanged,
the already-installed project `.venv` launched the committed runner. That was the sole
controller execution. H ran first and M second as registered; the paired cell completed in
5.514 seconds without source movement or retry.

All frozen gates passed. E1 retained 24/24 decisions and 360/360 physical minutes for each arm,
normal termination, and no failure. E2 retained 24 interval executions and 360 minute
applications per arm with zero shutdown, out-of-range decisions, or forced stops. M additionally
closed 48/48 solver stages, used no fallback, and reported zero evidence-chain, accounting,
authority, or reachable-tracking violation. Seven plant-refusal/rerouting minutes were fully
attributed. E3 passed because both arms delivered exactly 210 kg and missed zero. E5 recomputed
all interval/episode, component, terminal-liability, and paired-identity equalities.

E4 established the intended task-expression mechanism. H's net operating value was
`-556.3149208402143 EUR`; M's was `-173.7826353839970 EUR`, a descriptive M-minus-H difference
of `+382.5322854562173 EUR` at identical service. M accepted 73.924 kWh more solar and 1,251.861
kWh less process energy, but incurred 1,446.648 kWh more battery throughput and 4.425 more
electrolyser-equivalent hours. H ended 135.754 kg CH4-equivalent below its starting endowment;
M ended with no deficit, a full battery, and 337.967 kg H2. H ended with 511.960 kg CH4 and no
H2. The task is therefore distinguishing intertemporal inventory and operating behavior rather
than awarding another delivery tie.

The exact value decomposition prevents a vague winner story. Service contributed 840 EUR to
both arms. Relative to H, M gained 543.016 EUR by avoiding methane-endowment liability and
1.362 EUR in solar opportunity cost, but paid 82.196 EUR more battery reserve and 79.650 EUR
more electrolyser-life reserve. The liability term alone is 142% of the net difference. Both
arms have negative absolute values under the alpha ledger. The result proves one-cell task
discrimination, not controller superiority, commercial value, or calibration.

Independent post-run verification reopened the 4,635,756-byte artifact, reproduced raw SHA-256
`10711d88d7c22c5601ff4abd4282349d468a4e35c8019229a06445cb95637454` and report SHA-256
`acca646c7b2705bd278d022fd687d34303ba9b9acef5a0ae56d3b8d9503b684b`, recomputed the
heuristic evidence digest, passed the deep MPC run verifier with evidence-graph SHA-256
`8623359f02d7fe2ca375b59cff89bb97b0a927575387dbf8607720eae1e0197e`, and reproduced
E1--E6 exactly from detached arm payloads. The source gate contains identical clean revision and
worktree identities before and after the run.

The passed mechanism gate authorizes a separately frozen economic development tournament, not
an unbounded RL run. Seed 0 is now inspected and cannot become holdout evidence. The next
protocol must declare fresh seeds, at least two demand regimes, paired tail/worst-case metrics,
longer or cyclic endpoint checks, and coefficient ablations that isolate the dominant methane
liability from tariff, battery, and electrolyser terms. A substantive PPO baseline is justified
only if that tournament shows consistent, service-comparable residual headroom that is not an
artifact of one synthetic terminal coefficient. The complete result and artifact custody are
recorded in `results/v2_economic_mechanism_smoke_a46aabf.md`.

## 2026-09-04 — Economic robustness tournament frozen before fresh seeds

The one-cell pass did not directly trigger PPO. Its dominant 543.016 EUR methane-liability
difference and six-hour endpoint make a repeated robustness gate necessary. Repository seed
custody shows that seeds 0--3 were directly inspected during controller development and seeds
0--19 appeared in earlier V2 baseline work. Seeds 20--24 were not present in any recorded V2
trajectory or fixture; they are now consumed as fresh development seeds by the frozen protocol
and may never be relabelled as holdout evidence.

`ECONOMIC_ROBUSTNESS_TOURNAMENT_PROTOCOL.md` freezes 20 paired H/M cells before any of those
seeds is reset: horizons 6 and 24 hours, delivery rates 35 and 42 kg/h, and seeds 20 through 24.
The matrix contains 40 trajectories, 2,400 controller decisions, and 36,000 one-minute
applications. M keeps a six-hour causal planning horizon in both task horizons; neither arm sees
tariff or economic coefficients. The comparison remains a task probe with unequal controller
information, not an algorithm ranking.

R1--R3 require the entire declared matrix, exact identity/accounting, and zero hard safety or
authority failures. R4 requires at least four of five equal-service cells per horizon/rate
stratum. R5 requires those cells to have more than 1 EUR separation plus a material raw driver.
R6 repeats that gate after algebraically removing methane-endowment liability from both arms,
preventing the seed-0 dominant term from carrying readiness. R7 requires both 24-hour strata to
pass independently. R8 reports, without a pass threshold, every ranking flip across frozen
methane, wear, and solar-cost multiplier grids.

The raw-file protocol SHA-256 is
`6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412`.
No seed 20--24 environment has been constructed or reset, no controller outcome has been opened,
and no coefficient has been changed. A complete R1--R7 pass will permit only separately
registered bounded PPO development and a tariff-aware economic MPC implementation. Failure of
service comparability, nonterminal expression, or full-day robustness returns the task for
repair before training.

## 2026-09-04 — Economic robustness runner and deep verifier implemented outcome-free

Implementation began from the frozen tournament rather than from a controller result. The
initial analysis draft correctly encoded the 20-cell order, paired economic components, and
R4--R6 thresholds, but an admission audit found three protocol gaps before it was allowed near a
protected seed. First, the cell execution check did not directly bind the MPC settings identity,
closed pending-transition state, rich evidence counts, or heuristic evidence ledger. Second,
the aggregate did not yet report every coefficient sensitivity, ranking flip, latency, and
six-versus-24-hour distribution by stratum. Third, its deterministic cell projection embedded
the complete analysis—including wall time—and its verifier assumed every arm was complete,
which would make a retained failure unverifiable. These were software findings only; no
environment was reset to discover them.

`economic_robustness.py` now owns the frozen cell and matrix contract. A canonical cell derives
its exact episode arithmetic and one of two frozen service-MPC settings identities. The
heuristic verifier recomputes its aggregate evidence hash, decision/command/execution digest
ledgers, and minute cardinality. The MPC path invokes the existing deep setpoint-run verifier,
then additionally checks its controller, settings, expected/completed decisions, pending state,
solver-stage counts, interval executions, and physical minutes against the cell. Per-arm reward,
component, interval-driver, methane-liability, and battery-liability accounting is recomputed.
Safety requires zero shutdown, out-of-range, forced-stop, accounting-chain, target-overshoot,
reachable-tracking, unattributed-deficit, and support-authority violations.

The per-cell analysis reports every registered physical/economic M-minus-H metric, service
comparability, default task expression, no-methane expression, settlement completion, and all
frozen algebraic sensitivities. The stratum aggregator now retains all five values in seed order,
complete/missing counts, mean, population standard deviation, median, extrema, the raw-minimum
seed convention, and sign counts. R8 includes every 0/0.5/1/2 methane, wear, and solar multiplier,
all three leave-out views, higher-arm labels, and sign flips by seed and stratum. A separate
horizon table shows the 6-hour and 24-hour default/no-methane distributions and change in mean,
without inventing a direction threshold. Controller, solver, dispatcher, cell, and tournament
latencies remain reportable but are removed from deterministic evidence projections.

`economic_robustness_experiment.py` implements the registered one-shot path. It rejects a
relative, in-repository, existing, or parentless output directory. Before directory creation or
reset it requires a clean completely identified revision, the exact protocol bytes, and the
frozen lock. Within each cell it constructs both environments, resets and identity-checks both
before H acts, executes H then M sequentially, computes the cell analysis, and publishes the
cell immediately using exclusive non-overwriting creation. It reopens that artifact and deeply
verifies it before proceeding. A reset, arm-invocation, or analysis exception becomes typed
retained evidence; subsequent cells still run and missingness cannot be replaced by a
surviving-subset aggregate. After the matrix, the source identity is rechecked and `report.json`
references each raw, full, and deterministic cell hash.

The report verifier independently reopens all cell files; rejects missing, extra, duplicate,
reordered, renamed, path-escaping, or byte-changed artifacts; recomputes each cell analysis;
rebuilds all strata and R1--R8; and verifies the matrix's deterministic and full hashes. The
common heuristic executor now records decision-compute latency separately from its dispatcher
latency. This timing augments the full evidence hash but is deliberately absent from the
deterministic outcome hash.

Twelve new outcome-blind tests use fabricated controller evidence under non-protected
identifiers 901 and 902, plus one 15-minute generic-door wiring reset at non-protected seed 903.
They cover the protocol/lock identities, 2,400-decision and 36,000-minute matrix arithmetic,
strict cell serialization, settings hashes, sensitivity algebra, cell gates, four-stratum
aggregation, deep tamper rejection, timing exclusion, output refusal, paired
reset-before-action order, preflight-failure retention and continuation, full-directory
verification, and registered seed-substitution refusal. Together with the existing economic
smoke tests, the focused selection currently passes **19 tests in 0.89 seconds**. A read-only
compatibility check reopened the already-published seed-0 mechanism artifact; the new verifier
accepted its heuristic/MPC evidence and all seven per-cell conditions, including nonterminal
expression. Building a detached cell wrapper around those retained arms exposed one final
identity edge: integer-valued constructor inputs such as `6` and `35` validated but could derive
JSON numbers different from a reconstructed `6.0`/`35.0` cell. The frozen cell now normalizes
both values and its seed at construction, and the complete detached cell verifier passes. That
check performed no simulator reset and no controller action.

No seed 20, 21, 22, 23, or 24 environment has been constructed or reset. No registered
controller trajectory, value, inventory, weather realization, or latency has been observed.
The broader controller/economic compatibility selection passes **73 tests with one optional
training-dependency skip in 6.41 seconds**. The complete repository passes **1,188 tests with
3 skips in 43.76 seconds**. Its one warning is the existing Starlette/httpx deprecation;
trailing HiGHS messages remain unchanged solver output. Black, Python compilation, strict
targeted MyPy with imported legacy modules skipped and cache disabled, Git whitespace,
Makefile command expansion, the frozen protocol hash, and the unchanged dependency-lock hash
all pass. This runner and admission record are ready for a clean commit; the tournament remains
forbidden until that commit exists and the worktree is clean.

## 2026-09-05 — Economic robustness execution registered before outcome

The runner, deep verifier, tests, and admission record were committed cleanly as
`7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e`. The frozen runner contract is
`6f789629393188f406c929305721aa257e33e329fc3b5b85da9e5d62d77d7d38`; protocol and lock
identities remain `6fb5eb11933ba5ee601849210b4e61278e4cec2ddb97c8e31b05817ee7141412`
and `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.

Before creating a worktree, output, or environment, the exact execution was registered in
`ECONOMIC_ROBUSTNESS_TOURNAMENT_REGISTRATION.md`. The executable revision is the runner commit,
not this later narrative commit. It will be checked out detached at
`/private/tmp/plantanagent-economic-robustness-7b1b656`. The new exclusive output directory is
`/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_economic_robustness_7b1b656`.
Both paths were absent when registered.

The registered execution contains all 20 cells and no selection surface: horizons 6/24 hours,
rates 35/42 kg/h, seeds 20--24, H then M, sequentially. The first execution is final regardless
of gate outcome. A failed controller or cell remains evidence; no alternate directory, retry,
seed, coefficient, controller edit, or parallel cell is authorized. At registration time no
seed 20--24 environment had been constructed or reset and no tournament outcome had been opened.

## 2026-09-05 — Economic robustness tournament completed; full-day task gate rejected

The exact registered execution was launched once from the clean detached
`7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` worktree with its detached `backend/src` and
`simulator/src` ahead of the installed runtime. It wrote only to the predeclared external
directory. The process was allowed to run without partial artifact inspection or concurrent
analysis. It completed the 20 sequential H/M cells in 288.832 seconds and returned the expected
nonzero gate-failure status only after writing and self-verifying the complete report. A second
deep whole-directory verification from the same exact detached source reproduced the report,
cell ordering, raw/full/deterministic identities, and R1--R8 observations. No cell was rerun.

The completed artifact set contains 20 cell JSON files plus `report.json`, occupying 220,620
KiB on disk. The raw report SHA-256 is
`26c9ab4a85693dab05d93f7a1b65e86564c74bfe667ee54813901a466d1e0755`; its domain report
identity is `e0fe451861cda01f0a788adafed9bf2767f778d0141d7c76147b6eb0f3831f7f`, and its timing-free
deterministic evidence identity is
`cff97bfc678f0b7ca47c516f3f7a80e39434aae83a6597576b1539eba5110b96`. The source gate records
identical clean worktree identity
`eb34c2f8415f865ca6626a5e37e276441f3401e6d2c8cfdda3d4a13b82c5bbd6` before and after the
run. Seeds 20--24 are now inspected development evidence and can never be reused as held-out or
formal validation seeds.

R1, R2, and R3 pass. All 40 arms reach their exact termination, decision, minute, settlement,
and evidence counts. Every MPC transition and both solve stages close without fallback. Paired
manifests, source, components, interval drivers, reward traces, endowment liabilities, and
artifact graphs verify. All arms have zero shutdown, out-of-range decisions, and forced stops;
M additionally has zero accounting-chain, target-overshoot, reachable-tracking,
support-authority, or unattributed-deficit violations. The largest M decision latency is
557.907 ms, largest solver-stage latency 307.866 ms, and largest dispatcher latency 0.594 ms,
so the failure is not a timeout or execution-integrity artifact.

R4 fails exactly at the horizon boundary. Both six-hour strata are service-comparable in 5/5
cells; both 24-hour strata are comparable in 0/5. Consequently R5 and R6 also fail in the
full-day strata because their definitions require equal service, and R7 fails because six-hour
success cannot substitute for full-day robustness. This chained result must not be paraphrased
as absence of economic or physical separation: the full-day arms differ strongly, but their
value difference is confounded by product delivery.

At six hours, H and M both deliver exactly 210 kg at rate 35 and 252 kg at rate 42 in all fresh
seeds. Default M-minus-H net value is negative in all ten cells: mean -149.833 EUR at rate 35
and -133.001 EUR at rate 42. Removing methane liability makes those means -228.949 and
-223.751 EUR. This reverses the earlier seed-0 direction and shows that its +382.532 EUR M
advantage was not stable across development weather/inventory draws. The new H direction is
not itself commercially stable: removing declared wear changes mean M-minus-H value to +87.259
and +100.694 EUR, while the service-plus-solar view yields +8.144 and +9.944 EUR. Doubling
methane liability flips two of five cells in each stratum to M. R8 therefore confirms that
uncalibrated synthetic coefficients still control the six-hour ranking.

At 24 hours, H averages 100.00% fill at rate 35 and 97.97% at rate 42; M averages only 73.66%
and 70.82%. Mean M-minus-H delivered mass is -221.262 and -273.686 kg. Mean raw net differences
are -4,557.844 and -5,447.800 EUR, but they are not eligible operating-cost comparisons:
M-minus-H unmet penalties alone contribute -4,425.239 and -5,473.728 EUR, with lower service
value contributing another -885.048 and -1,094.746 EUR. Favorable methane-liability changes
partly offset the service loss. Removing that term widens the gaps to -5,427.167 and
-6,686.385 EUR.

The retained decision graphs make the full-day mechanism deterministic and interpretable. In
every M arm the first two six-hour commitments are essentially complete, the third is partial,
and the fourth is zero. Immediately after the third settlement every M arm has zero CH4 but
still holds 7.122--8.000 MWh battery energy, 177.575--470.290 kg H2, and
976.664--2,543.948 kg CO2. The frozen planning horizon is six hours, exactly one commitment
period, so the model sees only the next obligation. Its terminal tie-breaker values CH4 and
convertible H2/CO2 as one methane-equivalent quantity. The authoritative action model permits
battery support only for the electrolyser; DAC and methanator require contemporaneous solar.
The locally optimal plan therefore retains convertible feedstock rather than producing enough
deliverable CH4 before sunset, then cannot convert it during the dark final window.

This is an informative negative result rather than evidence that Environment V2 remains
trivially solved. Extending the episode from one to four settlements reveals intertemporal
inventory and renewable-window pressure. It is also not evidence that H is intrinsically
superior to optimization or RL: H and M have different logic and information, M was deliberately
frozen with the short horizon, and the economic coefficients are synthetic.

The protocol decision is now enforced: do not begin substantive PPO training and do not tune
MPC weights on the opened matrix. The next versioned work package must protect service
lexicographically, span the next renewable window and all intervening commitments, distinguish
deliverable methane from feedstock requiring future conversion, and retain the common setpoint,
dispatcher, guard, and evidence path. Any repair may use seeds 20--24 only as contaminated
development cases. Formal evaluation will require new seeds plus unseen weather, demand,
tariff, plant, and equipment regimes. The complete tables, interpretation, claim boundary, and
artifact custody are recorded in
`results/v2_economic_robustness_7b1b656.md`.

## 2026-09-05 — Full-day horizon-only repair frozen before implementation or M24 outcome

The robustness result identified a concrete full-day mechanism but did not distinguish whether
the six-hour lookahead alone caused the failure or whether the unchanged methane-equivalent
terminal value also requires repair. I therefore froze
`FULL_DAY_HORIZON_REPAIR_PROTOCOL.md` as a single-variable causal probe before implementing its
runner, resetting a protected environment, or requesting an M24 action on seeds 20--24. The raw
protocol SHA-256 is
`e203d1bb12246bc43278b686779afdfde310f8b56e487dac7f9d05947079e0e1`.

M24 is the existing `ServiceMPCSetpointControllerV1` with only
`planning_horizon_h` changed from 6 to 24 hours. The task horizon remains 24 hours and the
control interval remains 900 seconds. Every terminal, delivery, startup, curtailment, battery,
solver, plant, constraint, forecast, tariff, economic, command, dispatcher, guard, and evidence
setting remains fixed. The resulting M24 settings identity is
`67877c94438fc1bf1bd3ad564f17d860f77cea10b82f07bba0d52b6a6f6b1778`; the rejected M6
full-day identity is
`28dd0dbbd732f4defcc700e3fcb91e59b2d14d6df7bac728963090e37ba48b0f`.

The frozen matrix contains only the ten already contaminated 24-hour development cells: rates
35 and 42 kg/h crossed with seeds 20--24. It creates ten new M24 trajectories, 960 controller
decisions, 1,920 solver stages, and 14,400 physical minutes. The old H and M6 arms are not rerun;
the future runner must load and deeply verify their immutable artifacts from the registered
robustness directory at executed source `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e`. It must
reject any changed reference byte, semantic identity, source state, task identity, or output
destination before producing an action.

The acceptance logic is intentionally stronger than “more product.” F1--F3 require exact
source/reference custody, all ten complete executions, closed evidence, accounting, safety, and
authority. F4 proves that the intervention actually exposes 96 planning intervals and exactly
four public targets at controller boundaries 24, 48, 72, and 96 while leaving every other
identity fixed. F5 requires no cell to regress against M6, positive fourth-settlement delivery
in every cell, and full request plus H-comparable service in at least four of five cells at each
rate. F6 requires at least a 90% mean unmet-methane reduction and enough post-third-settlement
CH4 for the complete fourth request in at least four of five cells per rate. F7 reuses the
registered equal-service economic-expression rules; F8 retains latency and the explicit
development-only claim boundary.

This order prevents outcome-driven weight tuning. If horizon-only repair fails, the next
versioned intervention must explicitly protect deliverable methane across renewable outages
and/or change the physical energy-supply architecture. If it repairs service but not economic
expression, calibrated operational pressure comes before RL. Only a passing, separately
verified repair can authorize a separately frozen bounded PPO baseline, tariff-aware economic
MPC, and cyclic Environment V2 beta design. No M24 protected action, policy training, held-out
evaluation, or controller/commercial claim was opened by this protocol freeze.

## 2026-09-05 — Horizon-repair pre-code audit found and froze evidence clarifications

Three independent read-only reviews mapped the frozen horizon probe onto the implemented
controller, run ledger, old artifacts, and verifier before new code was written. They did not
reset an environment or open an M24 action. The review found two literal protocol ambiguities
and one verification weakness that had to be resolved before implementation could be admitted.

First, F4's prose says later delivery targets are labelled “inferred,” but
`MPCDeliveryTargetV1` accepts the exact canonical value `inferred_recurring`. The implemented
target inference already yields boundaries 24, 48, 72, and 96 with source sequence `announced`,
`inferred_recurring`, `inferred_recurring`, `inferred_recurring`. Changing the schema would be
a second intervention, so clarification 001 binds the prose to the existing exact value.

Second, existing MPC run evidence stores public-observation, forecast, and projection hashes
but not the complete public issue or projection bodies. Existing economic arm execution also
retains scalar rewards and episode totals but discards each interval's delivery and economic-
component row. The M24 runner must now capture a full first-boundary public observation, public
issue, 96-interval projection, and complete ordered interval economic ledger before and during
the new trajectory. All are separately hashed, cross-bound to event zero or its public
transition, and excluded from controller inputs beyond what the existing MPC already receives.

Third, `verify_mpc_setpoint_run_digests` correctly checks the supplied outer and evidence-graph
hashes but does not deserialize each nested event and rebuild the semantic run summary. A
coherently rehashed nested mutation could therefore evade that routine alone. The new admission
now requires a detached typed codec that reconstructs every target, solve stage, selected plan
step, setpoint, dispatch, guarded action, commitment state, minute application, execution, and
residual; requires exact typed round trips; and rebuilds counts, energy, attribution, chains,
latencies, and identities through `build_mpc_setpoint_run_evidence`. This stronger check must
verify the old M6 references as well as M24.

The audit also inspected what can truthfully be recovered from the immutable reference bytes.
M6 contains its complete boundary records and residuals, so its settlement deliveries and
post-settlement inventories are reconstructable. H retains only boundary/application hashes,
reward scalars, and episode aggregate metrics; it does not retain intermediate boundary bodies,
per-settlement delivery rows, or per-interval economic components. Rerunning H would violate the
frozen reference rule. The clarification therefore requires explicit `not_retained` and
missingness for those H fields, `not_applicable` for MPC-only H fields, and keeps F5/F7 on the
available H aggregate service/economic evidence. No threshold or denominator changes.

These interpretations are frozen in
`FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001.md`, raw SHA-256
`8dc341b142a41214b7845b0f5debab29b7f23751ff8f7cd71ab68ac30f6b7ce3`. It also defines exact
citations for report, cell, embedded H, embedded M6, and new M24 identities; requires strict
canonical JSON, duplicate-key, symlink, regular-entry, and path-custody checks; and recognizes
the reference report's expected `complete_gate_failed` research result as structurally valid
evidence. The original protocol bytes, M24 intervention, F1--F8 gates, contaminated matrix, and
claim boundary remain unchanged. No protected reset, solve, rollout, or policy training was
performed.

## 2026-09-05 — Numerical and cross-revision horizon-repair semantics frozen

The completed protocol audit identified remaining ambiguities that could otherwise let outcome
interpretation or an unrelated runtime edit enter the horizon comparison. They are now frozen
in `FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002.md`, raw SHA-256
`e2ed43908242a228669859fb7111e4b4f24513d23f467daf60b8c9b1f7692966`, before implementation
or any protected M24 action.

The clarification makes fourth delivery material only above `1e-6 kg`, defines the inclusive
M24/M6 no-regression inequalities, binds full service and H comparability at `1e-6 kg`, and
defines 90% unmet repair as
`mean(M24) <= 0.10 * mean(M6) + 1e-6 kg` with a materially positive M6 denominator. It fixes
post-third adequacy at request minus `1e-6 kg` and records the production-aware final settlement
identity: starting CH4 plus interval production minus post-settlement CH4 must equal retained
delivery. The existing strict 1 EUR and raw-driver thresholds remain unchanged.

F1--F7 are explicitly chained while their intrinsic observations remain visible; F8 retains an
independent latency observation and exact claim text. Every per-rate series must include ordered
seed values, expected/observed/missing counts, missing seed IDs, and all tied extrema. Sign counts
apply only to paired contrasts. Global custody and gate counts may span rates, but comparative
performance may not.

Reference custody now means the root report plus all 20 cells, exactly 21 canonical regular
nonsymlink entries. Only cells 11--20 enter M24 analysis. The identity-availability table
distinguishes files, embedded H/M6 arms, deterministic citations, and evidence graphs, and
clarifies that the final root raw hash is returned out of band because a report cannot contain
its own byte hash. Compact derived old metrics are permitted; complete old arms are not copied.

Most importantly, admission must diff the registered M24 execution source against historical
revision `7b1b656964dcac4a25c6f5d4d89fed8abf4d4e0e` and reject any modified or deleted
pre-existing simulator, controller, safety, or V2 runtime file. Newly added experiment/verifier
modules are allowed only when they do not alter or enter the historical runtime path. Every M24
stage must also report the historical SciPy 1.18.1 and HiGHS 1.12.0 versions. Event zero must
match M6's public observation, public issue, and full projection identities; only
`planning_horizon_h` may differ in the settings body.

This second clarification changes no task, controller input, objective, plant, dispatcher,
guard, matrix, threshold intent, or claim boundary. No environment was reset, no solver was
called, and no controller outcome or policy training was opened.

## 2026-09-05 — Controller-neutral V2 RL integration closure audit and hardening

The controller-neutral RL work was made the immediate milestone and the M24 runner work was
paused before implementation or any protected action. The existing `a46aabf` increment passed
its narrowly stated offline gate, but two independent read-only reviews were asked to test the
stronger word “finished” across both supported tasks, checkpoint custody, resume behavior,
evaluation ordering, and the repository's live controller surface. The review found real gaps;
the prior completion label was therefore not accepted without repair.

The resulting implementation and regression matrix were frozen in commit `4f3e5b4`
(`fix(rl): close controller-neutral admission gaps`).

First, JSON constraint parsing delegated to the permissive live-operator merge helper. That
helper intentionally treats junk as “no cap,” so a training file such as
`{"electrolyser_max_kw": "junk"}` could silently remove authority while the run remained
apparently valid. The V2 training boundary now separately requires exact numeric types: optional
caps accept only `null` or finite nonnegative real values, battery reserve accepts only a finite
real in `[0, 0.95]`, and booleans, numeric strings, junk, NaN, infinity, overflow, unknown fields,
and invalid direct Python construction fail. The operator-facing V1 merge semantics are
unchanged.

Second, resume metadata could overstate what was run. Stable-Baselines3 restores PPO's policy and
optimization settings from the parent artifact; the old resume path supplied only environment,
device, and seed, yet saved any newly requested PPO settings as provenance. Resume admission now
parses and cross-checks the parent's stored training task and rejects any changed embedded policy
or PPO setting. Device remains the only PPO-field override because it is explicitly applied by
`PPO.load`; run duration, output, vector count, seed, cadence, and notes remain new-leg
provenance. Each child checkpoint records the absolute parent path, raw parent-metadata SHA-256,
policy-artifact SHA-256, task digest, and parent timestep count. Parent metadata and artifact
bytes are rechecked before deserialization and before every intermediate or final save, closing
the admission/use race and making multi-leg lineage auditable.

Third, metadata identity alone did not prove that a real SB3 ZIP embedded the same concrete Gym
spaces. Saving and V2 loading now compare model and declared observation/action space types,
shapes, dtypes, lows, and highs. A mismatched artifact is refused before a checkpoint directory
is claimed on save and before a loaded model can infer on evaluation. Artifact bytes retain their
existing content hash; the new check closes accidental copying or coherent mislabelling across
the 57/4 and 62/4 tasks.

Fourth, the paired evaluator previously ran the heuristic before constructing the policy
adapter, so a caller that handed it an already loaded checkpoint plus the wrong task could move
one arm before discovering incompatibility. It now reopens and verifies the persisted checkpoint,
matches it to the selected task, and compares the persisted and in-memory metadata before either
arm resets. Runtime refusals are converted into the same explicit CLI refusal status, and JSON
publication continues to use exclusive creation.

Coverage is now symmetric. Outcome-free fixtures prove exact normalized-Gym versus checkpoint-
controller command parity for the physical 57/4 task and economic 62/4 task, including telemetry,
control state, reward/accounting evidence, and the common dispatcher/guard/plant path. Both tasks
complete paired heuristic/policy smokes. Bidirectional physical/economic substitution is rejected
for checkpoint load, evaluation preflight, and resume before SB3 artifact loading. New CLI tests
cover stored-task recovery for both tasks, exclusive JSON creation, overwrite refusal, and clean
runtime-failure handling. Strict constraint tests exercise every cap and reserve boundary, and
adversarial tests reject in-memory and serialized model-space mismatches.

The ordinary environment omits the optional trainer by design, so its focused gate reports
**104 passed, 2 skipped in 0.78 seconds**. The optional real runtime separately executes both
tasks: each trains exactly two transitions, saves, resumes for two more, binds the parent,
reloads with internal-space verification, and completes a paired evaluation. Both pass in
**12.91 seconds**. These four transitions per task are serializer/wiring checks, not useful
rollouts and not controller outcomes. The complete repository passes **1,276 tests with 4 skips
in 44.44 seconds**; the warning remains the existing Starlette/httpx deprecation and trailing
HiGHS text remains upstream solver logging. Targeted strict typing passes on all six changed
production modules with import following and cache writes disabled; the cached host invocation
first reproduced the already documented mypy unresolved-placeholder internal assertion.

The phrase “controller-neutral V2 RL integration” remains deliberately scoped to the research
task. The selectable `/api/controller/rl` control-room slot is the historical one-minute 30/5
V1 runtime. It lacks V2's causal forecast issue, commitment context, 15-minute cadence, economic
tariff observation, and held absolute-setpoint execution. Making a V2 checkpoint selectable there
by deleting fields, inventing context, or converting targets back into instantaneous shares would
break the trained contract and create an unsafe second physical meaning. A separately versioned
live V2 runtime is therefore a later product-integration milestone, after the task itself passes
the controller-readiness gate.

No research-scale policy, protected RL seed, hyperparameter search, M24 trajectory, HEPA fit,
language-layer experiment, controller ranking, or commercial claim was opened. With this software
closure complete, the scientific blocker is unchanged: the six-hour MPC fails the fourth full-day
commitment, so substantive PPO remains unauthorized. The next research step is the already frozen
horizon-only M24 mechanism probe; live V2 orchestration follows only after a credible task and
controller have earned deployment work.

## 2026-09-05 — M24 implementation admitted and exact one-shot execution registered

The frozen full-day horizon-only repair is now implemented without changing any pre-existing
simulator, controller, safety, or V2 runtime file. Commit `90788cc` on the main research branch
contains four additive production modules plus tests and Makefile entry points. The same logical
implementation was replayed onto the clean post-protocol base as `f7ee304`; the outcome-free
admission record was then committed as executable revision
`d18082594b7fea692c9fff146c54d9b0f2f31733` on `codex/m24-horizon-execution`.

The admission layer requires the exact historical report plus all 20 cell files, canonical JSON,
regular nonsymlink custody, immutable registered hashes, an unchanged dependency lock, a clean
authenticated source diff against `7b1b656`, and a new external output directory. It rejects any
modified or deleted existing runtime file and any undeclared new runtime/package surface. The
typed codec rebuilds every nested M6/M24 target, solver stage, setpoint, dispatch, guard,
commitment boundary, minute application, interval execution, residual, event, aggregate, and
digest rather than trusting coherent outer rehashes.

The analysis contract retains the complete event-zero observation/issue/96-interval projection,
all 96 M24 interval rows, delivery and physical-boundary chains, execution energy, commitment-
derived wear exposure, production-aware final settlement, tariff, terminal liability, and all
seven additive objective components. Historical H/M6 values use explicit `available`,
`not_retained`, and `not_applicable` states. All five seeds remain in each rate denominator, and
F1--F7 are chained while F8 reports latency and enforces the exact claim boundary.

The final outcome-free suite passes 96 focused tests. Its unmocked synthetic ten-cell
run/publication/verifier test found two real JSON-boundary defects: tuple-valued episode fields
were being compared directly with reopened JSON arrays in both task and complete-cell
reconstruction. Canonical JSON equality now closes those boundaries without weakening any value
or digest requirement. Other attacks cover duplicate keys, path escape, symlinks, TOCTOU, extra
files, reordered coherently rehashed citations, noncanonical bytes, canonical body tampering,
source drift, missing evidence, threshold edges, and failed-cell retention without environment
construction.

Two production-path integration runs used only non-protected seeds 905 and 906. Each completed
96 decisions, 1,440 physical steps, 96 executions, 1,440 minute applications, and four
settlements. They exposed and fixed the 06:00 episode time-origin assumption and the exact
retained H allocation paths. A clean read-only admission then reopened the 215 MB historical
directory, deeply verified all 21 artifacts, typed-rebuilt all 20 M6 runs, and selected only
full-day cells 11--20 in canonical rate/seed order. No H/M6 controller was rerun.

The repository passed 1,372 tests with 4 skips; the four production modules pass strict typing,
all eight new source/test files pass Black and compilation, Git whitespace checks pass, and the
frontend production build passes. The existing large 3D frontend chunk warning is unrelated.

`FULL_DAY_HORIZON_REPAIR_REGISTRATION.md` now binds the exact executable revision above, the
fresh detached worktree `/private/tmp/plantanagent-m24-d180825`, and the unused external
destination
`/Users/filippogombac/.codex/visualizations/2026/08/31/01a05842-8e4b-73b1-8810-42296fdaca5b/v2_full_day_horizon_repair_d180825`.
It authorizes one sequential ten-cell M24 run only. No protected action or outcome had occurred
when the registration was written, and no RL, controller-superiority, held-out, economic,
commercial, commission, HEPA, or language-layer claim is opened by this software gate.

## 2026-09-05 — Registered M24 run passed F1--F8 and repaired full-day service

The one authorized protected execution ran sequentially from detached clean source
`d18082594b7fea692c9fff146c54d9b0f2f31733` into the previously absent registered destination.
It did not rerun H or M6. The runner first reopened and typed-rebuilt the complete immutable
historical evidence directory, then exclusively published and immediately verified each of ten
M24 cells, repeated source/reference admission, published `report.json`, and invoked the whole-
directory verifier. A separate detached invocation subsequently repeated verification from the
same exact revision. Both returned exit code zero; there was no retry or parameter change.

The report status is `complete_all_gates_passed`. All ten cells terminated normally with exactly
96 decisions, 1,440 one-minute physical applications, 192 successful solver stages, four
settlements, closed transitions, and no fallback: 960 decisions, 14,400 physical minutes, and
1,920 solver stages in total. Every first decision bound the unchanged public forecast issue to
96 consecutive 15-minute intervals and exactly four delivery targets at boundaries 24, 48, 72,
and 96. Settings differed from M6 only in the registered 6-to-24-hour planning horizon.

Service repair is complete on these development cells. At 35 kg/h, M6 delivered a five-seed mean
618.738 of 840 kg and left 221.262 kg unmet; M24 delivered 840 kg and met all four commitments in
every seed. At 42 kg/h, M6 delivered 713.840 of 1,008 kg and left 294.160 kg unmet; M24 delivered
1,008 kg and met all four in every seed. The fourth settlement delivered exactly 210 or 252 kg
in all ten cells. Each post-third-settlement CH4 state covered the complete fourth request, and
the production-aware final material balance closed against both residual and interval evidence.

M24 matched H aggregate service in all five 35 kg/h cells and four of five 42 kg/h cells. In the
remaining r42/s22 cell M24 safely exceeded H by 102.367 kg, so the frozen equal-service rule kept
the cell visible but excluded it from F7. The eligible economic-expression rules passed 5/5 and
4/5. This is task discrimination, not an economic winner: default M24-minus-H value averaged
+137.884 EUR at 35 kg/h and +125.973 EUR across eligible 42 kg/h cells, but removing the synthetic
methane-endowment liability made every eligible difference negative, with means -171.095 EUR and
-163.522 EUR. Doubling declared wear also made the mean difference negative at both rates. The
reward coefficients must therefore be calibrated or uncertainty-bounded before policy selection.

The physical contrasts explain the ranking sensitivity. Against equal-service H cells, M24 used
about 4.3--4.4 MWh less accepted process energy but 2.4--2.5 MWh more battery throughput and about
2.9 additional equivalent electrolyser operating hours, while reducing terminal methane-
equivalent deficit by roughly 72--77 kg. Batteries still ended near 7.98 MWh. Thus the original
failure was caused by temporal/product-form planning, not exhaustion of the stored-energy asset.

All prohibited safety, accounting, chain, range, overshoot, tracking, support-authority,
shutdown, forced-stop, and unattributed-deficit counts were zero. The common safety path remained
active: cells contained 3--7 guard corrections and 23--46 plant refusals/target deficits, all
independently attributed without loss of service. Worst M24 controller p99/max was
965.081/965.081 ms and worst dispatcher p99/max was 0.587/2.955 ms, so F8 passed comfortably.

The 11-file external artifact set occupies about 140 MB. Raw `report.json` SHA-256 is
`44a11d071e3f4820757c2174c35119fe99aa4016a0cfdc73a2e6196e53c32974`; report-domain identity is
`eb3a68232ddef74d96e391c4979ab7aa2c15238b1ca08a4a0a4162fc04a664c4`; deterministic evidence is
`1d12982aed4f9e32f8868393a1b226525eb268023bdf77b7b286e3f872307c12`. The full gate table,
seed-level service/economic values, physical mechanism, latency, cell byte hashes, limitations,
and architecture decision are recorded in
`results/v2_full_day_horizon_repair_d180825.md`.

The task-readiness blocker is closed, but the benchmark is now harder: a horizon-aware model-based
controller already solves deterministic service. The next experiment must ask whether the
controller-neutral PPO path improves calibrated economic or risk outcomes under genuinely new
weather, tariff, demand, inventory, and plant regimes, against H, M24, and a new tariff-aware
economic MPC. In parallel, Environment V2 beta should add cyclic multi-day operation, forecast
revision, correlated uncertainty, sensor imperfection, faults/degradation, constrained backup
energy, and product logistics. HEPA and the language supervisor remain later perception and
human-governance layers, not substitutions for the physical controller.

The exact mandatory boundary remains: contaminated development mechanism only; no RL,
controller-superiority, calibrated-economic, held-out, site, commercial, savings, commission, or
generalization claim.

## 2026-09-05 — Controller-neutral four-arm build completed

The immediate objective was narrowed to finishing the controller-neutral RL integration as a
real executable build, not another architecture review. No PPO training, protected seed,
research-scale rollout, HEPA fit, or language-layer experiment ran during this increment. Work
was split into independent economic-MPC and benchmark tracks, followed by a separate read-only
integration audit. The starting controller/RL/MPC regression set passed 89 tests with 2 optional
training-runtime skips before implementation.

Commit `656d66862634827945585e6d5e469b18f80ee693` adds the missing tariff-aware economic
comparator. It is a separately versioned controller, not the existing service MPC under a new
label. The model composes the validated service MILP and appends two non-negative terminal
deficit hinges: methane-equivalent endowment and battery-energy endowment. Surplus receives no
credit. Solve stage one maximizes feasible delivered methane. Solve stage two locks that service
optimum within the declared numerical tolerance and minimizes a EUR objective comprising the
positive-price opportunity cost of accepted solar, bus-side battery charge/support throughput,
modeled electrolyser running and cold-start reserve, and the two terminal liabilities. The
controller verifies that the observation's five tariff summaries agree with its complete
checkpoint-bound day-ahead tariff, projects only the current public issued-mean forecast, emits
the same `ProcessSetpointCommandV1` used by H and PPO, and propagates solver/validation failure
without fallback.

The new economic evidence contracts cross-bind observation, forecast projection, settings,
assumptions, sparse model, plan, first-step targets, and issued command identities. Plan
construction independently checks typed/contiguous delivery and step sequences, both sides of
the service optimum, the sum of step delivery, request caps, and reconstruction of the stage-two
objective from named cost components. The optimizer does not claim exact minute-scale wear:
low-load classification, ramping, hot starts, aborted starts, forced stops, forecast error,
guard intervention, and nonlinear plant response remain realized simulator measurements and are
declared model mismatch.

The independent audit found a genuine numerical boundary defect before completion. A supported
24-hour task at 100 kg/h and seed 905 solved optimally in HiGHS, but its raw vector contained
`-4.55e-13` kW at a non-negative startup-power bound. The original canonicalizer could snap
valid small interior values because the tolerance scaled with a large upper bound; when that
disturbed a coupled row, it returned the complete raw vector, and the strict physical plan schema
rejected the negative epsilon. The repair now projects only actual lower/upper violations within
the absolute solver tolerance, rounds admissible binary noise, independently audits the complete
canonical vector with the bounded projection budget, and fails closed instead of leaking a raw
out-of-domain command. Reported solver objectives are checked against the raw vector while plan
cost evidence is recomputed from the audited canonical vector. The exact 24-hour case is now a
permanent regression and issues a non-negative controller-neutral command.

Commit `317b094402e025af887a8b7a57945164d63bcbbc` adds the unified evaluator. Each seed is
instantiated as four independent copies of the checkpoint-recovered economic task. Reset
preflight compares physical episode, economic episode, assumptions, manifest, and checkpoint
contract identities before an arm moves. It constructs H, canonical rolling M24, the production
economic MPC, and the checkpoint-bound deterministic PPO adapter, then runs every arm through
its own safety guard, one-minute simulator, settlement path, and realized EUR ledger. H, E, and
PPO retain every typed decision, common command, interval ledger, and minute application. M24
retains its richer closed transition/evidence graph. Controller and dispatcher latency remain in
the raw report. A failed arm is never replaced, every failure remains visible, typed economic
solver stages survive the error boundary, and a cell enters numeric aggregation only if all four
arms complete.

The production qualification cannot be obtained with a fixed-action fixture, injected economic
controller, or noncanonical M24 tuning. It requires a trained Stable-Baselines3 PPO checkpoint,
reopens and validates the checkpoint/task contract and embedded Gym spaces before movement, and
requires clean committed source. Reports retain the full Git revision, worktree-status digest,
`uv.lock` digest, Python/numerical runtime versions, checkpoint metadata bytes, policy artifact
digest, and task identities. Deterministic evidence excludes wall-clock timing and checkpoint
filesystem location; relocating identical checkpoint bytes now reproduces the same evidence
identity. The maintained command requests both the `train` and `optimisation` dependency groups,
accepts at most 32 unique non-negative seeds, and publishes only to an absolute new destination
outside the source repository without overwriting an owner artifact.

Outcome-free coverage now includes real one-decision and four-decision four-arm runs, common
manifest equality, complete minute evidence, rolling economic-controller state, fixed-policy
qualification refusal, custom-arm/tuning refusal, zero-movement preflight failure, typed solver
failure retention, no-fallback incomplete-pair exclusion, seed bounds, exclusive/absolute CLI
publication, timing-independent identity, checkpoint-relocation identity, strict economic-plan
cross-binding, and the high-demand numerical regression. Verification at the completed source
state was:

| gate | result |
|---|---:|
| new economic MPC + four-arm tests | 19 passed |
| adjacent V2 RL/economic/MPC regression | 121 passed, 2 skipped |
| complete Python repository | 1,391 passed, 4 skipped |
| strict typing, five new production modules | passed |
| Black, compileall, Git whitespace | passed |
| frontend production build | passed |

The complete repository run emitted the existing Starlette/httpx deprecation warning and the
known trailing HiGHS diagnostic text; neither is introduced behavior. The optional SB3 stack is
not hydrated in the ordinary local environment, so the four skipped tests remain intentional.
An attempted grouped CLI dependency check could not download a transitive `mpmath` wheel because
network access was unavailable; the already installed runtime still imported the CLI, and the
maintained grouped command is syntactically verified. No checkpoint or result was fabricated to
hide that external dependency state.

This closes the software dependency requested for controller-neutral evaluation. It does not
open a controller result: no research-scale PPO checkpoint exists in repository custody, and the
colleague's artifact was not supplied to this run. It also does not make the current alpha task a
commercial benchmark. E sees the complete static public tariff schedule while alpha PPO sees
five tariff summary fields, and deterministic M24 already solves service. The next scientific
increment is therefore to admit the colleague checkpoint for a bounded, explicitly development-
only four-arm run, calibrate or uncertainty-bound the economic coefficients, then freeze a
prospective multi-policy-seed protocol and detached semantic verifier. Environment V2 beta must
subsequently introduce cyclic multi-day operation, common versioned forecast information,
forecast revisions and correlated errors, demand/logistics uncertainty, sensor imperfections,
degradation/faults, and constrained backup energy. HEPA remains a separately ablated predictive
representation/safety layer, and natural language remains an operator-governance interface; they
must not be credited with controller value until factorial evidence isolates their contribution.

## 2026-09-05 — Cyclic stochastic-market beta implemented without opening an outcome

The next task was deliberately changed from running the finished alpha comparison software to
removing the task ceiling that made the colleague's PPO/heuristic parity unsurprising. No long
experiment was started. The implementation contract was frozen in
`CYCLIC_MARKET_BETA_PROTOCOL.md` before any beta controller result: a separately identified task,
unchanged alpha physical sampling, stochastic realized price plus one causal forecast vintage per
decision, exact daily resource returns against one non-ratcheting reference, and finite private
DAC outages. The contract explicitly separates scalar EUR value from hard cycle feasibility and
forbids a protected comparison until a new outcome-blind split and finite policy-selection rule
exist.

The task choice followed the repository's own negative evidence. M24 already repairs deterministic
full-day service, while the alpha economic ranking is sensitive to synthetic liabilities and gives
the economic MPC a complete repeating tariff that PPO sees only as a small summary. A more useful
task must make future information uncertain and revisable, prevent the controller from spending
starting inventory once, and require recovery from unavailable process capacity. Literature on
the Industrial Benchmark, safe constrained control, and coupled power-to-gas dynamics was used to
justify those classes of difficulty, not to calibrate coefficients or claim field fidelity. The
market and outage processes remain declared synthetic engineering stressors.

Implementation was split into independent market and disturbance contracts, then composed locally
with cycle state, Gym fields, economic accounting, policy admission, and training. This preserved
clear ownership between latent exogenous data and controller-visible data.

The new `market.py` generator uses the existing 24-hour tariff only as a civil-time baseline. A
regime-dependent AR(1) residual supplies serial correlation and rare signed arrivals create
decaying excursions. `calm`, `volatile`, and `scarcity` are immutable version-bound profiles;
`mixed` privately samples one concrete profile. Every realized price is finite and clipped to the
existing -300 to 300 EUR/MWh observation domain. The default issue generator uses 64 ensemble
members, but that number and its innovation scale are task configuration rather than hidden
constants.

One `PublicPriceForecastIssueV1` is generated for every controller decision. It contains the
current realized price and 24 ordered hourly bins of mean, p10, and p90, giving a fixed 73-field
public vector. An issue conditions on current price and its own spawned forecast stream but never
reads future realized truth. The explicit external-truth seam made this causal property directly
testable: changing only later truth preserves the current issue byte-for-byte and changes a later
issue after its new current price becomes available. Hourly means can be projected identically to
any whole subdivision of an hour, including the 15-minute MPC grid. The public serializer omits
selected regime, seed, stream state, future truth, and future vintages; the private audit serializer
retains the generator version, exact profile, baseline, initial RNG identities, truth/forecast
digests, and replay requirements.

The new `beta_disturbances.py` generator builds only finite `dac_outage` faults supported by the
existing simulator. Each exact 24-hour cycle owns an independent deterministic stream and may
contain at most one outage. Defaults are enabled, probability 0.5 per cycle, and a one-to-four-hour
duration range. Drawn onsets and durations are on the physics lattice. Extending an episode leaves
existing cycle draws unchanged, and disabling the feature produces an exactly empty schedule.

Disturbance integration exposed a subtle simulator-boundary issue during outcome-free review. The
fault registry refreshes immediately before a dispatch while boundary telemetry describes the
interval just completed. Allowing an outage to end exactly at the cycle boundary could therefore
leave a stale offline indication at that boundary and suppress the next cycle's first behavior.
The generator now requires one healthy physics dispatch before every cycle end. Generated faults
are validated through normalized `FaultSpec` semantics and the simulator fault registry remains
the authority. Only ordinary health and telemetry effects are public; schedule and stream data are
private audit material.

`beta_config.py` freezes exact 24-hour cycle geometry, four independent return fractions and
tolerances, the opening-resource reference, and one immutable settlement record. The reference is
read from authoritative simulator state at episode reset. Battery energy, H2, CO2, and saleable
CH4 targets all default to 100% of that same reference. Each boundary reports target, actual,
deficit, tolerance, feasibility, and a dimensionless violation cost. Targets never ratchet after a
bad cycle, and resources are not collapsed into methane-equivalent inventory because process form
and conversion time remain operationally important.

The composed `PlantCyclicMarketEnvV2Beta` has the distinct environment ID
`plant-env-v2-cyclic-market-beta-v1`. Its observation concatenates the unchanged 57-field physical
vector, the exact 73-field public market issue, and five fields for cycle progress and signed
battery/H2/CO2/CH4 return margins. The result is a declared finite 135-field space. Its action is
the existing four normalized absolute process targets, so all commands continue through
`ProcessSetpointCommandV1`, the common reserve-aware minute dispatcher, independent guard, and
simulator.

Two narrow protected seams were added to inherited environments. `PlantEnvV2._sample_episode`
delegates to the historical alpha sampler by default; beta overrides it before simulator loading.
`PlantEconomicSetpointEnvV2` now asks protected methods for the interval opportunity price and its
public context; their default implementation is the unchanged alpha tariff. Beta supplies the
current realized market price and exact current issue. This avoids duplicating the authoritative
economic ledger while keeping alpha stream order and price behavior unchanged.

Beta root children are fixed as inventories, weather, contracts, market, and disturbances. The
first three retain the established alpha indices 0--2; beta takes only children 3 and 4. The audit
manifest preserves the original physical exogenous digest under an explicit alpha name and adds a
beta exogenous digest over physical, market, disturbance, and cycle identities. The checkpoint
contract separately binds the 135/4 interface, generator versions and settings, economic
assumptions, cycle semantics, stream allocation, disturbance visibility, and execution doors.
Source revision and dependency-lock hashes remain an experiment-runner responsibility rather than
being fabricated by the environment.

The live observation at each beta decision is a typed
`CyclicMarketControllerObservationV1`: physical state, the current market issue, cycle state,
economic-assumption digest, and beta-task digest. Generic economic access returns this richer type
rather than silently downgrading to the alpha deterministic-tariff view. An alpha-only economic
MPC therefore fails its input contract if used directly against beta instead of receiving the
wrong forecast or privileged price information.

The strict controller-neutral PPO path now accepts beta only when a
`CyclicMarketTaskConfigV1` is present and valid. JSON parsing rejects unknown cycle, market, and
disturbance settings. Direct CLI selection binds the version-one default beta task when no beta
configuration was loaded and removes stale beta settings when switching back to alpha. The normal
environment factory constructs beta for training, and the policy contract binds its environment
ID, field order, interface digest, and full checkpoint digest.
At inference, the normalized policy adapter verifies the economic and beta-task identities,
encodes exactly 57 + 73 + 5 fields, checks shape against checkpoint metadata, performs deterministic
policy inference, and issues the same observation-bound four-target command as the two alpha tasks.
Alpha/beta substitution is therefore rejected before relying on model output.

The existing paired heuristic/policy evaluator now understands beta manifests and retains beta
exogenous identity, economic accounting, every cycle settlement, aggregate feasibility, cumulative
constraint cost, typed decisions, and minute physical executions. The heuristic remains the
transparent physical baseline and does not consume the additional market/cycle fields; the policy
receives exactly its checkpoint-declared vector. Failures are retained and no fallback controller
is invented. A normally terminated beta trajectory that violates any 24-hour return constraint is
now recorded with failed task status and a typed `CycleConstraintViolation`, while termination and
completed-decision evidence remain intact. A future scientific runner must still enforce its
preregistered complete-pair and eligibility rules rather than dropping failed arms.

Outcome-free tests were added for deterministic and immutable market/disturbance replay, strict
hash identity, caller seed preservation, public/private field separation, complete finite forecast
shape, quantile ordering, causal future-truth invariance, later-vintage revision, hourly forecast
projection, explicit/mixed regimes, serial correlation, empty outage ablation, horizon-prefix
invariance, per-cycle outage cardinality, lattice alignment, finite lifecycle, and fail-closed
configuration/geometry validation. Environment, checkpoint, policy, training, and alpha-regression
coverage add 135/4 prefix parity, realized-price attribution, public/private manifest separation,
exact two-cycle settlement, Gym/checkpoint common-door parity, bidirectional checkpoint rejection,
and training-config reconstruction. No test fixture is being reinterpreted as a
controller-performance result. The post-composition focused command covered the beta environment,
controller-neutral matrix, policy integration, and strict training configuration; it completed
with **105 passed and 2 skipped**. No training or performance-bearing rollout was part of that
gate.

The build now supports a real 135/4 beta PPO training and common-authority inference path, but it
does not yet support a scientifically fair beta controller tournament. The existing tariff-aware
economic MPC and H/M24/E/PPO runner are bound to the alpha economic task. The next production
increment is a separately versioned beta economic MPC consuming only the current public issue and
explicitly modelling the four return constraints. The current trainer is ordinary scalar-reward
PPO: cycle margins are observable and settlement cost is retained, but the learner does not yet
enforce that cost. A constrained-RL contract with a declared feasibility budget is therefore a
separate next milestone, with standard PPO retained as an ablation. Four-arm runner/verifier
support follows those controller contracts. Economic coefficients must then be calibrated or
uncertainty-bounded, and the new train, development, and held-out regime/seed split plus finite
training-seed/checkpoint-selection rule must be frozen without inspecting outcomes.

No beta training, protected seed comparison, controller-superiority result, HEPA fit,
language-supervisor experiment, site validation, savings estimate, ROI, or commission claim was
opened. The implemented contribution is the harder causal experimental substrate: recurring
renewable-powered operation under revised electricity value, resource sustainability, and
unannounced process loss, with enough identity and evidence separation to test controller value
later without repeating the alpha information asymmetry.

Final hardening closed three additional software gaps before commit. Direct CLI selection of the
beta now constructs and binds the complete default `CyclicMarketTaskConfigV1` rather than failing
with a missing extension; switching back to an alpha task removes stale beta settings. Disturbance
duration/grid compatibility is validated while the task and checkpoint contract are constructed,
before reset or physical movement. Finally, a normally terminated policy episode that violates any
24-hour resource return is recorded with failed task status and a typed
`CycleConstraintViolation`; its normal termination, completed decisions, settlements, and cost are
all retained, so the evaluator cannot call an infeasible trajectory successful or erase its
physical execution.

The real training CLI was then exercised in `--dry-run` mode against the beta identifier. It
constructed the default 72-hour task, reported 135 observations, four actions, 672 decisions,
10,080 physics steps, distinct beta and retained physical exogenous digests, and the beta-specific
output directory. It explicitly printed `nothing was trained`; no optimizer or gradient step was
started.

The final release gate ran no training and no protected rollout. The complete Python repository
finished with **1,461 passed and 4 skipped** in 64.98 seconds. The one warning is the existing
Starlette/httpx deprecation; the known trailing HiGHS diagnostics also remain. Strict typing passed
for the 11 touched/new production modules after excluding only third-party missing-stub/import
diagnostics and inherited unused override-ignore markers. Black passed for all 17 touched/new Python
files, and `compileall`, Git whitespace validation, the frontend production TypeScript/Vite build,
and frontend `oxlint` all passed. Vite retained its existing large-chunk advisory. These are release
checks only; they do not alter the no-outcome claim boundary above.

## 2026-09-05 — Beta controller-neutral comparison released after semantic hardening

Work continued without a research-scale training run or protected evaluation. The immediate
objective was to finish the controller-neutral beta integration rather than infer controller value
from the earlier environment-only gate. The release-validated integration consists of three linked
pieces: a beta-native economic MPC, a four-arm H/M24/E-beta/PPO runner, and a detached semantic
verifier. Its own final combined release gate is recorded below rather than reusing the earlier
1,461-test beta-environment result as indirect validation of these new files.

The beta-native economic MPC preserves the existing two-stage service-then-cost formulation while
changing its information and terminal contracts for the cyclic task. It reads only the current
public physical boundary, issued solar and price projections, observed current price, public cycle
position and return margins, and declared plant/economic contracts. It does not receive future
price truth, future forecast vintages, the private market regime, or the private DAC outage
schedule. Stage one maximizes service; stage two locks that result and minimizes the declared
opportunity-price, throughput, wear, startup, and terminal-deficit terms. Both stages contain hard
next-boundary return rows for battery, H2, CO2, and saleable CH4. Alpha and beta plan/evidence types
remain separate so the new assumptions cannot pass through an older public API under an alpha
identity.

The runner creates four independent beta environments for each seed and requires exact preflight
agreement on their physical, economic, market, disturbance, cycle, interface, task, and exogenous
identities before allowing movement. H receives the physical view, M24 additionally receives the
public solar issue, and E-beta and PPO receive the complete market/cycle surface appropriate to
their typed interfaces. These are recorded information-profile ablations rather than falsely
described equal-information competitors. All four still issue the same four absolute plant targets
through the common minute dispatcher, independent safety guard, and simulator. Failure is retained;
there is no controller substitution or surviving-arm average. A cell is value-eligible only when
all arms complete the exact execution lattice, every daily resource return is feasible, hard safety
passes, and delivered service is comparable within the registered tolerance.

Safety classification was materially strengthened during adversarial review. Summary fields such as
minimum SOC and shutdown count were initially too easy to rewrite coherently with the outer report
hashes. The environment now retains the complete sequence of audit `PlantState` boundaries from
step zero through every executed physics minute. The audit trace is exposed through a protected
environment seam and deliberately omitted from Gym observations and `info`; controllers therefore
gain no additional state. Generic H/E-beta/PPO execution cross-checks every trace boundary against
the existing minute-application boundary hashes, while M24 cross-checks against its already
retained full transition records. The runner derives minimum SOC and critical/offline minutes from
that run-wide ledger rather than relying solely on accumulated summaries.

Expected DAC unavailability also required exact semantics. A reported private schedule is not an
authority because an attacker could replace it and recompute downstream hashes. The verifier now
recreates the disturbance episode from the cell seed using the registered beta root-stream
allocation and requires exact equality with the private audit manifest. It converts each finite DAC
fault into its executed half-open minute lattice. A critical/offline minute is exempt only if that
exact executed step is scheduled, DAC is offline, no non-DAC health record is offline, and battery
SOC remains above the configured reserve. A missing DAC indication, simultaneous battery or other
component outage, reserve breach, off-lattice declaration, or critical/offline minute outside the
regenerated schedule remains unexpected. Component health status and aggregate plant status are
also derived from the recorded condition, closing an attack that relabelled `critical` as `offline`
while preserving local hashes. Forced-stop deltas are reconstructed from the public commitment
counter and out-of-range decisions from each controller's retained evidence before the verifier
rebuilds arm, cell, and aggregate eligibility.

Economic-plan verification was strengthened for a different reason. A plan reconstructed from
published step values may not recover a solver's exact raw vector when several auxiliary variables
are degenerate, so comparing its digest to the solver-stage digest can reject an honest result or
leave an opaque provenance claim. The beta plan now retains the selected stage-two solution vector
explicitly. The verifier checks its exact cardinality and finite values, reconstructs the primary
service-lock row, binds the retained vector to the published stage-two solution digest, audits
bounds, integrality, all model rows and the added lock, and recomputes named costs and the economic
objective from the same witness. It also checks consistency between that witness and the typed plan
projection and first issued command. Adversarial cases now include a changed value, missing value,
non-finite value, changed stage digest, and a coherently rehashed vector/digest combination.

This change has a deliberate limit. The stage-one raw solution vector is not retained, so its
solution digest remains provenance emitted by the solve rather than an independently reconstructible
optimality witness. The detached verifier does not rerun HiGHS or either controller. It proves that
the retained secondary witness is model-feasible, satisfies the reported service lock, and agrees
with its digest, costs, objective, plan projection, and command chain. It does not independently
prove that the primary service value or secondary cost is globally optimal. A future stronger
certificate would have to retain and audit the primary vector and, for optimality rather than
feasibility, either rerun the frozen solver or carry an independently checkable primal/dual or MIP
certificate. The current documentation now states this boundary instead of calling both solver
stage identities independently verified.

The resulting evidence file is intentionally much larger than operational telemetry and contains a
private disturbance manifest, exhaustive plant-state history, controller evidence, and solver
witnesses. The runner consequently requires a new external absolute output and refuses overwrite.
Protected artifacts should remain access-controlled until their registered opening; they should not
be streamed into routine operator logs or committed merely because the verifier can read them. A
commercial runtime will need a smaller operator-safe telemetry projection while retaining this
full artifact only for governed audits.

The only full-path runtime exercise in this increment remains an outcome-free fixed-action smoke at
seed 34. An explicitly untrained policy fixture and the three nonlearned arms each executed 1,440
physical minutes. All four passed the current hard-safety rule, all four failed at least one strict
cycle return, and delivered service was not comparable. The runner consequently produced zero
eligible value comparisons while retaining all four trajectories. This is evidence that execution,
failure retention, and exclusion mechanics operate on a complete day. It is not evidence for or
against PPO, H, M24, or E-beta and did not consume a protected seed, train an optimizer, select a
checkpoint, or justify a commercial ranking.

Focused verification now passes. The four-arm runner's dedicated suite finished **17/17**, and the
combined runner, beta MPC schema/solver, and environment minute-truth selection finished **37/37**.
The expanded detached-verifier suite finished **35/35 in 142.40 seconds**. Its first execution
exposed two real production defects rather than being weakened to preserve a green result. The
minute-state fold paired the complete before/after arrays with `zip(..., strict=True)`, whose unequal
lengths rejected every honest transition; it now pairs the two one-step-offset slices and covers
each executed transition once. After that cascade was removed, E-beta classification incorrectly
required the economic decision schema to contain the generic heuristic/PPO `invalid` field. E-beta
now contributes its contractually fixed zero out-of-range count, while H and PPO still require and
fold their Boolean invalid evidence. The passing suite includes the coherent state/hash/status,
disturbance-seed substitution, and retained-vector/digest attacks described above.

Verifier Black, Python compilation, and focused typing pass. Verifier-test Black, compilation, and
Git whitespace validation also pass. After the explicit metadata and seed annotations, the final
complete Python repository rerun finished with **1,535 passed, 4 optional skips, and one existing
Starlette/httpx deprecation warning in 198.61 seconds**. Strict scoped MyPy passed for the five
touched production modules. Black passed all 10 changed Python files. `compileall`, Git whitespace
validation, and both beta command-line `--help` imports passed. The frontend production build and
lint passed while retaining the existing warning for a bundle larger than 500 kB. No training or
protected rollout occurred during these gates.

The controller-neutral integration dependency is therefore closed without opening a controller
outcome. The following scientific increment remains prospective constrained RL: freeze an
outcome-blind train/development/held-out regime and seed split, declare a finite number of training
seeds and a checkpoint-selection rule, define a constraint budget and admission semantics for the
separate cycle-cost signal, and retain ordinary scalar PPO as an ablation. Only after those rules
and economic coefficient uncertainty are frozen should development training begin; the held-out
comparison must then be opened once. HEPA and natural-language supervision remain separate future
factorial layers and are not credited to this controller integration.

## 2026-09-05 — Outcome-blind constrained-RL protocol frozen before implementation

No beta environment was reset, no training dependency was invoked, and no learned-controller
outcome was inspected in this increment. The work started from the now-clean controller-neutral
beta release and addressed the next scientific dependency identified there: ordinary PPO can
observe the cycle margins, but its optimizer never consumes the separate cycle-constraint signal.
Observing a constraint is not the same as optimizing or satisfying it.

The code audit located the exact authoritative signal. At every 24-hour boundary,
`CycleSettlementV1` retains battery, H2, CO2, and CH4 targets, actual values, deficits, tolerances,
feasibility, and the sum of four normalized non-negative violations. `PlantCyclicMarketEnvV2Beta`
places that settlement under `cycle_constraint.settled_this_step`, leaves it absent at all other
transitions, and deliberately keeps it out of the EUR reward. The cumulative-cost field is only an
episode summary; using it as a transition cost would charge every past failure repeatedly. Stock
SB3 PPO ignores both fields. Its current vector factory also uses the training seed as the first
environment seed and then permits implicit auto-reset draws, so it cannot yet prove which episodes
were consumed or keep optimizer and episode randomness independent.

The review also found that the released beta runner is the wrong final statistical surface even
though its execution evidence remains useful. It requires all four H/M24/E-beta/PPO arms to be
cycle-feasible before any value is aggregated. H and M24 intentionally lack the complete market
and return objective, so either can legitimately fail a cycle and thereby suppress the primary
learned-policy versus E-beta comparison. The held-out campaign will retain those ablations but use
pairwise admission for its primary estimand: learned policy and E-beta must both complete, remain
safe, satisfy every return, and provide comparable service. H and M24 cannot disappear, but they
will not be allowed to veto that named pair.

Primary literature was reviewed before choosing the algorithm. PID Lagrangian methods were
introduced to damp the oscillation and overshoot of ordinary multiplier ascent while retaining a
PPO-family implementation; the newer nu-PI work provides a possible later multiplier refinement.
CPO and FOCOPS remain useful cross-family references, but introducing second-order or another
policy optimizer before validating the first cost-aware rollout path would confound controller
architecture with optimizer replacement. Research on zero-violation RL also warns that a zero
cost limit does not cause empirical violations to disappear automatically, especially with a
sparse cost. The protocol therefore distinguishes expected-cost training from exact simulator
admission and explicitly forbids calling the learner itself a hard-safety mechanism.

`CONSTRAINED_RL_BETA_PROTOCOL.md` now freezes the primary learner as
`plant-beta-ppo-pi-lagrangian-v1`. It retains the unchanged EUR reward, adds one separate four-head
cost critic, one independent non-negative multiplier per resource, undiscounted cost returns, and
PI multiplier control with derivative gain fixed to zero. Plain integral-only PPO-Lagrangian and
ordinary scalar PPO are mandatory matched controls. All share the existing 135/4 task, policy and
reward-critic architecture, PPO hyperparameters, synchronous four-environment rollout, stochastic
training actions, deterministic evaluation, and common dispatcher/guard/simulator authority.

The canonical transition contract is four-dimensional rather than only the existing scalar sum.
At a settlement, each resource cost is the excess deficit beyond tolerance divided by the larger
of one or its target; otherwise it is zero. Training divides each boundary value by the seven
cycles per episode, so the undiscounted episode sum is mean per-cycle normalized violation. Four
zero budgets preserve the fact that no resource may compensate for another. Cost GAE uses gamma
one and lambda 0.95. The PI update uses `K_P=0.25`, `K_I=0.01`, no derivative term, 0.95
proportional smoothing, initial multipliers one, and independent projection to `[0,100]`. Actor
advantages use the declared `(reward - weighted costs) / (1 + multiplier sum)` rescaling. These
numbers are hypotheses fixed before outcomes, not tuned plant constants.

All episode seeds below one million are now conservatively treated as development-contaminated.
Training owns the range beginning at 1,000,000; development is exactly 2,000,000--2,000,031;
held-out is exactly 3,000,000--3,000,031; and new implementation fixtures are restricted to the
9,000,000 range. Merely recording a seed does not authorize its environment construction. The
training allocator maps each vector index and local episode ordinal to a deterministic explicit
training seed, records every reset and manifest, and must restore exact cursors. Optimizer/network
seeds are separately frozen as 101, 211, 307, 401, and 503. This makes all algorithm families see
the same ordered episode stream without conflating it with policy initialization.

Each of the three families receives exactly 1,048,576 decisions per optimizer seed: 128 complete
8,192-transition rollouts and no early stopping or outcome-dependent extension. Only the final
checkpoint is eligible, closing checkpoint-time selection. The registered core budget is therefore
15,728,640 decisions across fifteen runs. A run is consumed after its first gradient update; it
cannot be retried, extended, or replaced on the same revision after failure. Checkpoints must bind
the algorithm, four cost channels and critics, multiplier history, allocator and reset ledger,
optimizer seed, task/source/dependency identities, and artifact bytes while remaining loadable for
ordinary deterministic PPO inference.

Before development, fixture-only tests must prove terminal cost capture, vector auto-reset
behavior, hand-computable reward/cost GAE, channel isolation, PI versus integral-only equivalence,
scalar-gradient independence, allocator/resume identity, checkpoint substitution failure, and a
genuine tiny train-save-load-resume-inference path. No development seed is available to make those
tests pass. Only after a clean implementation commit may all fifteen final policies be evaluated
on the 32 development seeds.

A policy is development-admissible only if every development episode verifies, completes, remains
hard-safe, has no forced stop or invalid command, satisfies all four returns at all seven cycle
boundaries, and matches E-beta service. Family champions are selected lexicographically by paired
value IQM, lower-20% CVaR, median, positive fraction, then lower optimizer seed. Inference
latency is reported against its independent deployment gate but is not a development-selection
variable. At
least four of five PI policies plus a positive-IQM/non-negative-tail PI champion are required before
held-out opening. Failure is retained as a negative development result; it does not consume or
quietly replace a held-out seed.

The held-out protocol lists 32 exact seeds but keeps their manifests and outcomes sealed. Its
future campaign must retain H, M24, E-beta, all five PI policies, all five integral-only policies,
and all five scalar policies, sharing baseline evidence rather than rerunning it for each pair.
Uncertainty uses a 10,000-replicate paired bootstrap at frozen seed 20260905 and a hierarchical
policy-seed/episode-seed bootstrap for family claims. The success rule requires complete verified
primary pairs, zero learned hard-safety/forced-stop/invalid events, every return, exact service,
a positive lower 95% IQM-delta bound, at least 24/32 positive deltas, non-negative lower-tail CVaR,
p99 inference below 100 ms, and positive held-out median value for at least four of five PI seeds.

Even that gate would establish only a synthetic-beta controller result under declared economic
assumptions. The protocol carries forward the existing algebraic coefficient sensitivities and
requires every ranking reversal. Site savings, ROI, field safety, commission value, and deployment
remain gated on calibration or defensible uncertainty bounds, hardware-in-the-loop operation, and
site replay. HEPA and natural language remain later factorial predictive and supervisory layers;
neither is credited to this controller result.

An independent pre-freeze adversarial review then found four blocking ambiguities and several
outcome-relevant omissions before any code or protected reset. The protocol had accidentally
forbidden the bounded optimizer smoke required by its own implementation gate; policy admission
did not require the E-beta side of every pair to pass the same execution/safety/cycle gates; the
registered no-resume rule conflicted with the fixture resume test; and advantage normalization and
multiplier-update ordering were not exact. All were corrected before commit. Only fixture-domain
optimizer smokes are authorized, every primary statistic now requires a complete 32-pair vector,
and resume is explicitly a fixture-tested capability disabled for all 15 registered runs.

The same review froze shared actor/reward initialization independently from cost-network RNG,
sample-standard-deviation advantage normalization with epsilon `1e-8`, and the exact sequence
collect/return/multiplier-update/current-rollout optimization. It also fixed the registered
allocator endpoint: each environment receives 262,144 decisions, completes 390 episodes, and ends
64 decisions into episode ordinal 390, consuming exactly seeds 1,000,000--1,001,563 across the four
workers. Economic sensitivity is now explicitly policy-minus-E-beta and scales methane endowment
liability without conflating unmet-delivery penalty. PI/scalar family contrasts use matched
optimizer seeds, failed policies remain in the denominator, outage strata are cycle-level, and
latency cannot select a development champion. The remaining statistical implementation and host
measurement contract must be hash-bound before the first development reset. No outcome was opened
while making these corrections.

## 2026-09-05 — Controller-neutral constrained-RL lifecycle closed under fixture evidence

This increment completed the software prerequisite frozen in the constrained-RL beta protocol.
It did not open a registered training seed, a development seed, a held-out seed, or a controller
comparison. The only optimizer executions were bounded fixture-domain integrity tests. Therefore
this entry establishes that the three learner families can train, checkpoint, reload, infer, and
continue through the real beta plant interface; it does not establish that any learned policy has
mastered the task or outperforms the heuristic or economic MPC.

The new learner core is a deliberately narrow Stable-Baselines3 PPO specialization rather than a
second environment or controller authority path. `plant-beta-ppo-pi-lagrangian-v1` and
`plant-beta-ppo-i-lagrangian-v1` retain the normal reward actor and value critic while adding one
four-head cost critic for battery, H2, CO2, and CH4 cycle-return deficits. The primary family uses
the preregistered proportional-integral multiplier update; the integral-only family fixes the
proportional term to zero. `plant-beta-sb3-ppo-scalar-v1` uses the same actor/reward architecture,
rollout schedule, optimizer seed, and production plumbing but disables the cost critic and
multipliers, making it the matched scalar-reward ablation. All three act stochastically only while
training and use the same deterministic inference adapter afterward.

The core now treats every optimizer update as auditable state. Each completed rollout records its
transition count, reward/cost return identities, cycle settlements, multiplier state, gradient and
optimizer-step accounting, actor/reward parameters, cost-critic parameters where applicable, and
the previous rollout digest. Live Adam step counts must reconcile exactly with cumulative ledger
counts. Policy topology and module classes are checked exactly, including the Tanh architecture;
an apparently shape-compatible activation substitution is rejected before any environment reset.
Reward and each cost advantage are normalized independently, the frozen actor combination is
applied, and PI/I/scalar family isolation is tested. Once an optimizer step has occurred, an
exception poisons that run permanently rather than permitting an outcome-dependent retry. Exact
Python, NumPy, Torch, policy, optimizer, multiplier, episode-accumulator, and observation state are
retained for continuation.

Checkpoint evidence is no longer a self-vouching JSON envelope. `ConstrainedPPO.load` alone can
mint a non-serializable validation attestation after reopening a filesystem artifact and checking
its embedded state against live policy parameters, optimizer state, settings, architecture,
rollout ledgers, timestep cut, and checkpoint observation. The attestation binds the artifact
SHA-256, algorithm, timestep count, full constraint-state digest, and observation digest. Every
checkpoint admission, including an ineligible fixture checkpoint, requires this load-issued
capability. Registered-evaluation eligibility is derived only when the run kind is registered and
the artifact attestation exists; callers cannot select it with a Boolean or authenticate a copied
mapping by echoing it back.

Exact fixture continuation adds a receipt rather than trusting matching parent labels. The child
binds the parent artifact, full constraint envelope, policy and optimizer digests, learner RNG,
multiplier, episode accumulator, external allocator pin, checkpoint observation, timestep count,
and rollout-ledger tip. The receipt is finalized by the first resumed rollout index and digest, and
a resumed model cannot be saved before that first rollout is complete. Parent and child allocator
ledgers must form an exact prefix, and only an episode-boundary fixture checkpoint is physically
reconstructible. Registered runs remain non-resumable as frozen by the protocol.

The production training path now selects the exact constrained family from the beta run contract,
uses the deterministic four-environment seed allocator, and binds a 135-value beta observation
space to the four normalized process targets. Fresh runs require pristine allocator custody.
Resume first admits the complete recursive parent lineage, restores the allocator, reconstructs
the physical vector environments at an episode boundary without inventing ledger entries, checks
the resulting observation byte-for-byte, loads with `force_reset=False`, and only then restores
learner RNG and optimization state. The common controller adapter remains the sole actuation path:
policy output becomes an absolute setpoint command, crosses the dispatcher and safety guard, and
is applied by the unchanged plant simulator.

Saving and loading now bind the live model to more than internally consistent evidence. The exact
algorithm settings, optimizer seed, four-environment count, PPO hyperparameters, policy topology,
Gym spaces, allocator mode/split/base seed/episode horizon, runtime dependencies, source content,
and cumulative SB3 rollout cut must all agree with the serialized training config. A resumed cut
must equal the parent cut plus the whole-rollout ceiling of the requested increment. Source and
dependency provenance are checked before training, before publication, and again after artifact
serialization for both fixture and registered runs. Metadata is reread and compared byte-for-byte
after recursive admission or inference loading, while artifact hashes are rechecked, closing the
observed metadata/artifact time-of-check/time-of-use substitutions. A constrained artifact cannot
be relabelled as generic PPO or fixed beta provenance. Missing training dependencies retain an
operator-facing installation error rather than being misreported as checkpoint corruption.

The benchmark and general evaluation loaders now make the claim boundary explicit. A qualified
beta comparison requires registered-evaluation eligibility and recursively re-admits the trained
artifact immediately before use. Fixture policies remain loadable for engineering inference and
common-door tests but are rejected by the qualified benchmark. This distinction prevents a tiny
smoke-trained model from being presented as commission evidence while preserving a fast build
gate.

Three rounds of adversarial review materially changed the implementation. Early evidence drafts
accepted stale allocator progress, minimal fabricated rollout records, caller-echoed state,
runtime relabelling, reordered reset ledgers, and coordinated but unproven parent/child mappings.
Those paths now fail through nominal load-issued attestations and continuation receipts. The final
production audit found additional gaps in space/config binding, allocator identity, cumulative
timestep interpretation, post-save fixture provenance, optional-dependency behavior, and metadata
replacement; all were closed and covered at the public save/load/resume boundary. A final audit
also found that an arbitrary unignored repository-local fixture output would alter its own
worktree identity during save. Fixture outputs are now either external or beneath the ignored
`runs/` artifact root; registered outputs remain absolute and external.

The genuine resume gate then exposed a separate common-control defect that unit fixtures had not
reached. When the guard removed an offline DAC share and returned it to the battery, ordered
IEEE-754 addition could put the four-share sum one ULP above one. The simulator correctly
sanitized that action again, producing differences near `3e-17`; exact execution evidence then
correctly rejected the guarded and applied actions as different. The evidence checks were not
relaxed. Instead, the guard now canonicalizes its final action through the same sanitizer used by
the simulator, so `ValidatedAction.action` again means byte-for-byte what the plant records. The
exact failing witness is retained as a regression. An independent review confirmed the layer
choice and found no mismatch over five million randomized guard-redistribution cases.

The implementation is recorded as four auditable commits: `46e74c4` implements and tests the
constrained learner core; `720101d` implements artifact attestations and resume receipts;
`fc7c403` fixes exact guarded-action canonicalization; and `0ecc09b` completes production
train/save/load/resume/inference plus adversarial boundary coverage. The focused core/evidence gate
passed 89 tests in 14.71 seconds, the historical policy/protocol/benchmark regression gate passed
51 in 25.80 seconds, the new adversarial production-boundary gate passed 8 in 14.46 seconds, the
output-provenance contract gate passed 12 in 19.31 seconds, and the common-control gate passed 57
in 1.09 seconds. All three real learner families completed a 384-transition 135/4 train-save-load-
inference/common-door fixture. The initial exact scalar resume run found the one-ULP guard defect;
after the fix, a fresh parent/child rebuild passed in 105.54 seconds. Independent reviews passed
the core authenticity/continuation contract and the final production integration with no remaining
P0/P1 finding.

The final clean release gate ran the entire Python repository after all fixes: **1,760 tests passed
in 414.83 seconds** with only the existing Starlette/httpx deprecation warning. Strict scoped MyPy,
Black, Python compilation, and Git whitespace checks passed for every touched production/test
surface. Frontend TypeScript/Vite production build and `oxlint` passed; the existing greater-than-
500-kB `Plant3DPage` chunk advisory remains. No protected reset or registered optimizer step was
performed by any of these gates.

The research conclusion is intentionally limited but important: controller-neutral constrained RL
is now an executable and falsifiable part of the architecture rather than a placeholder. The next
step is not to complicate the environment blindly. First, a short fixture-only throughput and
artifact-size benchmark must estimate the wall-clock and storage cost of the already frozen
15-policy registered campaign. That benchmark does not authorize the campaign. After the estimate
is reported and registered training is explicitly authorized, the fixed five-seed PI/I/scalar
matrix can be run once, followed by the already frozen development selection and sealed held-out
comparison against E-beta, H, and M24. If the learned families again tie the baselines, that is
direct evidence that the current beta remains too easy or its decision pressure is misplaced; the
next environment increment should then add one separately ablatable commercial mechanism such as
degradation/maintenance state, forecast error, or a shared balance-of-plant bottleneck. HEPA and
natural-language supervision remain downstream factorial layers and receive no value credit until
the controller task itself demonstrates a meaningful, reproducible operating tradeoff.

## 2026-09-05 — Fixture-only constrained-RL capacity runner implemented and audited

No capacity cell, protected seed, registered optimizer update, development reset, held-out reset,
or controller outcome was opened in this increment. The work implemented the outcome-blind
infrastructure measurement required before the registered 15-policy campaign can be responsibly
scheduled. The capacity measurement remains a one-shot execution gate after this implementation is
committed and the source tree is clean; the tests below use dependency-injected filesystem fixtures
and do not train a policy.

`CONSTRAINED_RL_CAPACITY_PROTOCOL.md` first froze exactly three fresh sequential processes in PI,
integral-only, and scalar order. Every process uses optimizer seed 101, fixture base 9,880,000, the
unchanged default beta task and `PPOConfig`, four synchronous environments, one complete
8,192-transition rollout, 320 minibatch optimizer steps, 16 allocator records, final-only saving,
and no resume. The report is forbidden from publishing rewards, costs, actions, multipliers,
losses, policy outputs, market regimes, outages, inventories, or any controller ranking. It may
support only a host-specific timing/storage projection and cannot qualify its fixture checkpoints
for registered evaluation.

The new `training/capacity_benchmark.py` implements that contract as a production runner and a
detached verifier. Before output creation it reopens the protocol through a stable regular-file
reader, matches its frozen SHA-256, requires clean beta runtime provenance, records source,
dependency, interpreter, simulator, CPU, memory, Torch-thread, BLAS/OpenMP, platform, and monotonic
clock identity, and requires separate operator attestations for exclusivity and first-attempt
custody. The public execute and verify APIs expose no injectable measurement hooks; private seams
exist only for deterministic adversarial tests. Output must be a new absolute directory outside
the repository, and the three production-training child commands execute sequentially with no
retry or selection path.

Each cell retains exclusively published stdout and stderr bytes while the report contains only
their paths, sizes, and hashes. Successful output is reopened through the production policy loader
and load-issued checkpoint attestation. Admission checks the exact external and internal training
config bytes, algorithm, optimizer seed, task contract, 8,192-transition cut, fixture-only and
registered-ineligible flags, absence of a resume parent, runtime provenance, one rollout, 320
optimizer steps, allocator mode/split/base, and the exact set of seeds 9,880,000--9,880,015. It
inventories exactly `policy.zip`, `metadata.json`, and `training_config.json`, including logical and
allocated bytes and file SHA-256 values, and measures the maximum canonical rollout and allocator
entry sizes needed by the structural storage allowance.

End-to-end wall time uses the parent monotonic nanosecond clock around the complete child; the
rollout ledger separately supplies collection-plus-optimization time. Fixed overhead is the
non-negative difference. Per-family projections use one fixed overhead plus 128 rollout terms for
each registered policy and five policies per family. Sensitivity factors 1.00, 1.25, and 1.50 apply
only to the 128-rollout term. Storage adds the frozen conservative allowance for 127 future rollout
entries and 1,548 future allocator entries before calculating five-policy family totals and the
greater of two-times projected storage or projected storage plus one GiB. A failed cell removes the
campaign projection and result claim but does not prevent the remaining frozen attempts from being
retained.

The runner publishes one exclusive, fsynced, canonical `report.json`. Its full domain digest binds
every field. A separate timing-free identity now binds the protocol, clean runtime, host, formula
contract, cell order, exact config and output layout, logs, checkpoint identities, failure class,
and campaign arithmetic. Verification checks the outer digest before expensive checkpoint loading,
requires exact schemas and arithmetic, rejects duplicate or non-finite JSON, rejects path escape
and every intermediate or final symlink, rehashes all referenced regular files, independently
recaptures both runtime and host identity, and re-admits every successful checkpoint rather than
trusting report booleans.

The first independent adversarial review found substantive pre-execution defects: public test hooks
could forge host/timing fields; the report asserted no retry without a corresponding operator
attestation; intermediate symlinks were followed; Mac architecture text could be guessed as a CPU
brand; the internal checkpoint config was inventoried but not compared; signal return codes could
not self-verify; timeout output was lost; arbitrary exception text could leak an outcome; the
sensitivity formula rounded before rather than after multiplying 128 rollouts; formula constants
and the one-hour timeout were not bound; and the timing-free identity omitted material artifact
evidence. All were corrected before any capacity execution. Failure messages are now categorical
and sanitized, unrestricted diagnostics remain only in hash-identified files, signed return codes
are valid, partial timeout streams are retained, intermediate path components are inspected without
following symlinks, host identity is immediately recaptured, and public production functions fix
their real clock/process/admission implementations.

Because the one-hour timeout and explicit first-attempt assertion can affect whether the one-shot
measurement succeeds, the protocol was amended outcome-blind before execution. It now freezes an
exact 3,600-second child timeout, treats timeout as a retained non-retriable failure, requires both
operator attestations, requires immediate host recapture, and specifies exact rational sensitivity
rounding. Its new raw SHA-256 is
`223947ccd8fb0b8c909ea89ee4019818081f4bf2f884301019450551f7b18b88`.

The dedicated capacity suite passes 29 tests in 0.55 seconds. It covers all three exact configs,
formula binding and integer arithmetic, synthetic sequential success and reconstruction, public API
closure, both operator assertions, dirty-runtime refusal, external/new output custody, internal
config substitution, self-consistent report tampering, early outer-digest rejection, current-host
substitution, intermediate/final symlinks, ambiguous/noncanonical JSON, negative signal exits,
timeout partial streams, checkpoint-admission failure, failure-message redaction, and absence of
controller-outcome fields. The expanded learner/checkpoint/provenance regression gate passes 185
tests in 207.97 seconds. Black, Python compilation, strict source MyPy, scoped test MyPy, Makefile
command expansion, and Git whitespace checks pass. The complete Python repository gate then passes
**1,789 tests in 408.41 seconds**, with only the existing Starlette/httpx deprecation warning.
Frontend TypeScript/Vite production build and `oxlint` also pass; the existing greater-than-500-kB
`Plant3DPage` advisory remains. A second production-compatibility audit found no remaining P0/P1
defect and confirmed byte-level compatibility with the real trainer and checkpoint lifecycle.

The implementation is not itself capacity evidence. The protocol clarification is committed
separately as `4e0a169`; the next action is to commit the runner, tests, operator commands, index
entry, and journal as one auditable pre-outcome release, confirm a clean revision, then execute once
into a new external root after the two operator attestations are true.
While those three fixture cells run, no polling or unrelated token-consuming work is warranted.

## 2026-09-05 — First capacity attempt retained as a three-cell engineering failure

The implementation release was committed as `fab4a55` after the 1,789-test gate, and the worktree
was confirmed clean. The external destination
`/private/tmp/plantanagent-beta-rl-capacity-fab4a55` did not exist; no prior real capacity attempt
existed at that source revision; and no other repository benchmark had been intentionally launched.
An initial Make invocation was blocked before the Python runner or output root existed because the
restricted execution sandbox could not access uv's existing cache. It consumed no cell. The same
command was then admitted with the required cache access and executed once with explicit
exclusivity and first-attempt assertions.

The runner retained all three frozen fixture cells and published a canonical self-verified failed
report. Its raw file SHA-256 is
`9b1bd155d70d8356dadc4a1faa86f9bb721c163af4f5e51004a3509362a97134`, report-domain SHA-256 is
`f3d2ab32dbb86c7c4268d3b114e2ee44aff5d29b02f2c4e4e725e162acd20895`, timing-free evidence SHA-256
is `dd41dac90b9fffae63b639d27fe07d31baf225ec8a0025d3a40e8dfd81d4491d`, and runtime-provenance
SHA-256 is `0f6fd3b9ba8dc5c0a40f37034a480cd44557f03ee8b099acc8971d90bdd5ffab`.
The report status is `failed`; result claim and campaign projection are both null.

PI-Lagrangian ran for 127.084657500 wall seconds, integral-only for 133.561394000 seconds, and
scalar PPO for 130.659857167 seconds. Each child exited with status two, retained its distinct
stdout hash, and retained the same 71-byte stderr hash
`af0b82782468004843c654611226c87347c7daa5051fc3aaf9ad73ab9098d00c`. No checkpoint was
published. The failed timings are diagnostic only and cannot be extrapolated under the frozen
protocol. Same-host verification with the original host-observation privileges reproduced the
report identity and failed status. A restricted verifier correctly refused host equality because
the sandbox could not observe the CPU brand or physical-core count available to the admitted run.

The categorical report deliberately omitted the raw error text, so the first stderr log was opened
to diagnose the common failure: checkpoint-state construction rejected `rollout cost scaling is
incoherent`. The first stdout log was then opened and unexpectedly contained SB3's ordinary
rollout table, including fixture reward and cost-related summaries. That inspection breached the
operator side of the capacity protocol's no-outcome boundary. No value is reproduced, analyzed,
compared, or used to tune the learner; the other two stdout logs were not opened. All affected
episodes are fixture seeds 9,880,000--9,880,015 and permanently ineligible for registered evidence,
but the breach is recorded explicitly rather than hidden.

Static diagnosis found a family-neutral evidence-representation defect. Every transition already
binds exact `training_cost = raw_cost / 7`. The rollout collector independently sums raw values and
the already-divided values; the checkpoint verifier then compared the two aggregate arithmetic
paths at absolute tolerance `1e-15`. Binary64 addition is not associative, and 84 settlement
records can create several ULPs of harmless representation drift. The optimizer rollout completed,
but checkpoint serialization failed before any artifact became admissible. Scalar PPO encountered
the same invariant because it retains cost evidence for audit while excluding that evidence from
its gradients.

`CONSTRAINED_RL_CAPACITY_REPAIR_001.md` now freezes the repair before code. The collector must keep
both independent totals, require consistency at `rtol=atol=1e-12`, and canonicalize only the
rollout-ledger representation as raw total divided once by seven. Per-transition buffers, GAE,
completed-episode multiplier inputs, model parameters, PPO settings, algorithms, seeds, environment,
and campaign remain unchanged. A pure 84-contribution floating-point witness, material-mismatch
failure, channel isolation, complete focused/full regression gates, repaired protocol hash binding,
and a report/domain schema version bump are mandatory. Revision `fab4a55` cannot be retried. Only
one new attempt on the future clean repair revision may produce a capacity projection.

## 2026-09-05 — Capacity Repair 001 implemented without reopening an outcome

No real 8,192-transition capacity cell, registered optimizer step, development reset, held-out
reset, controller comparison, HEPA fit, or natural-language evaluation ran in this increment. The
failed `fab4a55` artifacts remain immutable, their timings remain ineligible for projection, and
the two unopened fixture stdout logs remain unopened. This increment implements the sole numerical
repair authorized by `CONSTRAINED_RL_CAPACITY_REPAIR_001.md` and the evidence boundaries frozen in
clarifications 001 and 002. A repaired capacity execution remains forbidden until this work is
committed, the worktree is clean, and the complete repository/static/frontend release gates pass
on that committed revision.

The learner continues to extract and independently accumulate both transition representations:
raw four-resource settlement cost and the existing transition cost divided by seven cycles. At
rollout finalization, a new pure canonicalizer validates that both four-vectors are finite and
non-negative, computes the stored training vector exactly once as elementwise raw total divided by
seven in binary64, and requires the independently accumulated divided path to agree at frozen
`rtol=atol=1e-12`. The pending rollout and final ledger now store the raw total and that canonical
vector. Checkpoint verification reconstructs the same elementwise division and requires exact
tuple equality. Diagnostic logging publishes both `constraint/raw_rollout_cost_{resource}` and the
canonical `constraint/rollout_cost_{resource}`. The float32 rollout buffer, reward and cost GAE,
completed-episode multiplier input, PI/I updates, scalar-family gradient isolation, network,
optimizer, PPO settings, transition count, batch order, seeds, and environment are unchanged.

The adversarial numerical witness uses 84 repetitions of `(3.7, 0.0, 1.1, 0.0)`. It proves that
independently summing values already divided by seven differs from summing raw values and dividing
once by more than `1e-15`, while the canonical ledger relation is exact and the two zero channels
remain zero. Separate tests reject materially inconsistent independent totals, negative entries,
NaN, and infinity, and assert the exact raw/canonical logger values without cross-channel leakage.

Independent audit of the first implementation found two evidence-system issues that did not alter
the learner but had to be closed before one-shot re-execution. First, private dependency-injected
helpers could be composed into a production-shaped report and generic verifier path. Clarification
001 froze the correction before those source changes. The production `report.json` is now built
only inside the public fixed-dependency execution entry point; production execution and
verification accept no process, clock, host, runtime, or checkpoint hook. Hook-bearing execution
can issue only a separate `synthetic-report.json`, synthetic schema/domain/execution class, null
capacity claim, and null campaign projection. Production verification rejects synthetic schema at
envelope load before runtime, host, or checkpoint access. Checkpoint admission is also split:
shared code only parses inert paths/configurations; production admission invokes the fixed real
loader and refuses synthetic identity, while the hook-bearing admission helper first requires the
synthetic schema and execution class.

Second, the earlier deterministic digest included post-run cells and therefore timing-transitive
artifact fields, even though it was described as timing-free, while the host thread environment
accepted arbitrary strings. The V2 deterministic identity is now an exact allowlist containing
only the bound protocol/repair chain, claim boundary, clean runtime digest, non-clock host capacity,
operator assertions, frozen cell contract, planned algorithm order and seeds, normalized config-
template digest, and campaign arithmetic. Status, exit/failure, wall/user/system CPU time, stdout,
stderr, absolute-root-dependent config bytes, checkpoint presence/hashes/sizes/ledger maxima,
measurements, and projections remain bound by the full report digest but cannot enter the pre-run
identity. Adversarial tests mutate every one of those direct and transitive classes and require the
deterministic digest to remain fixed while the full digest changes; protocol, runtime, non-clock
host, contract, algorithm order, optimizer seed, fixture seed, assertions, and arithmetic changes
must alter it. Two distinct external roots must share the deterministic digest while retaining
different full/config digests.

Clarification 002 freezes the host-thread representation as either JSON null or an integer parsed
from a canonical positive ASCII base-10 string in `1..2,147,483,647`. Whitespace, signs, zero,
leading zeros, non-ASCII digits, lists, annotations, timestamp-like text, outcome-like text, and
overflow fail before creation of the output root. Verification applies the same integer/null
schema and independently recaptures it. The runner records the operator-selected environment but
does not inject or rewrite thread variables in child processes. Failure evidence is likewise
closed: reports contain only one of `training_timeout`, `training_nonzero_exit`,
`training_launch_error`, or `checkpoint_admission_error` with fixed phase/message pairs. Arbitrary
exception classes and messages remain absent from canonical evidence.

The frozen raw document digests now bound by the report protocol identity are:

- Repair 001: `55a14887379c48232abc75913950a02de83db3cd5e9144f9ecf323a0a26d5c53`;
- Clarification 001: `2175c56d85ceb2b363b85316a9e90d437f1249d00a869d86e99676525e22beda`;
- Clarification 002: `f4db393d54d48aa5deca69b5b1c220bccd94b5fd87bac4fc6b4b40416feca8ea`.

The final focused learner/capacity suite passes **89 tests in 12.97 seconds**. The authoritative
locked-environment learner, capacity, checkpoint, production-integration, production-integrity,
training-protocol, runtime-provenance, and strict-config gate passes **285 tests in 209.70
seconds**. An initial system-Python invocation passed 270 tests but could not run 15
distribution-provenance-dependent cases because source-path imports do not install `plant-sim`
package metadata; the same complete selection passed through the locked `uv` environment above.
Black, Python compilation, and Git whitespace checks pass for all four changed Python files.
Strict scoped MyPy passes the capacity module directly and the constrained learner with only its
three pre-existing redundant-cast diagnostics disabled; cache writes and imported legacy modules
are disabled to avoid the already documented host-MyPy serializer defect. Two independent
read-only reviews report no remaining P0/P1 evidence-boundary issue. The full repository and
frontend gates remain the next clean-revision admission checks, not a result claimed by this entry.

## 2026-09-05 — Repair-001 capacity attempt retained; metadata canonicalization failure frozen

Repair 001 and both evidence clarifications were committed as clean revision
`27a47d0604ffb9624a40683a610f2f8f4bb6c58a`. On that exact committed tree, the complete Python
repository passed **1,820 tests in 416.74 seconds** with only the existing Starlette/httpx warning
and known trailing HiGHS diagnostics. The focused 89-test and locked 285-test gates, strict scoped
typing, Black, compilation, Git whitespace, frozen-document hashes, frontend production build, and
frontend `oxlint` also passed. Vite retained only the existing large 3D chunk advisory.

The worktree was then confirmed empty; the revision had no prior attempt; the new external root
`/private/tmp/plantanagent-beta-rl-capacity-27a47d0` did not exist; no training, pytest, capacity,
or other repository benchmark process was present; and all five bound thread environment variables
were absent. The single authorized Repair-001 attempt ran with explicit exclusivity and
first-attempt assertions. No other work or analysis ran concurrently.

The attempt completed once and failed closed. Its raw report SHA-256 is
`b7b4a9e3be62f62c7f82d55584ec5d29a7df57e8566051ace62c7f4699b31abc`, report-domain digest is
`e34facf8760e25c74adc1aff4cb4e35e7327e36bed0f23f341e28982f9a0e1f6`, deterministic pre-run
digest is `ef1344f59687d92f537985cbf61251f6a743e178f9ddd768a8965a7fc00a3c91`, and runtime digest is
`085ac94fc9b6caf259f7d2c644844d96a25d6d1f4d2936f3d5f88bba1834211e`. Immediate and detached
same-host production verification reproduced the authenticated failed status, null result claim,
and null campaign projection.

All three frozen children exited zero. PI-Lagrangian ran for 129.781126917 wall seconds,
integral-only for 129.247395542 seconds, and scalar PPO for 128.619359792 seconds. Each published
the exact three checkpoint files, but the parent retained the fixed
`checkpoint_admission_error` category and therefore admitted no checkpoint, measurement, or
projection. All stderr logs are empty. Unlike the preceding failure, none of the three stdout logs
was opened; their bytes remain available only through report-bound size/hash records. These failed
timings cannot enter a campaign estimate.

Read-only diagnosis invoked the existing checkpoint admission function on each retained artifact
without retraining or adding it to a report. Every family produced the same error: `checkpoint
training config differs from capacity cell`. The exact `training_config.json` byte comparison
immediately before that check had passed. A second diagnostic compared only the static
`training_config` mapping and did not print or inspect checkpoint evidence, rewards, costs, actions,
multipliers, losses, policy outputs, markets, outages, inventories, or rankings. The only unequal
top-level keys were `episode` and `beta_task`: nested bounds, duration ranges, forecast windows, and
anchors were tuples in `config.to_dict()` and lists after the metadata JSON round trip. Both sides
produce the same canonical JSON bytes, but raw Python object equality distinguishes tuple and list.

This is a family-neutral checkpoint-admission representation defect after successful child
training and serialization, not a learner or controller failure. It was masked in the V1 attempt
because the earlier rollout-cost invariant prevented checkpoint publication. The complete result,
including report/log/config/unadmitted-artifact identities, is retained in
`results/v2_constrained_rl_capacity_27a47d0.md`. Revision `27a47d0` and its output root are closed;
there will be no same-revision retry, projection, partial checkpoint reuse, or controller claim.

`CONSTRAINED_RL_CAPACITY_REPAIR_002.md` is now frozen before source changes. Its sole repair retains
the exact standalone `training_config.json` byte check and compares the complete metadata mapping
by serializing it with the same canonical JSON, UTF-8, and trailing newline as the expected config.
It forbids field-specific coercion, subset comparison, ignored keys, or tolerance and leaves every
subsequent metadata, runtime, task, evidence, artifact, rollout, allocator, optimizer, and policy
load check unchanged. Tests must accept only the representation-equivalent tuple/list round trip
and reject changed nested values, ordering, missing/extra keys, and non-finite values. The next
production schema is V3 and synthetic schema is V2, both bound to the exact Repair-002 document
hash `1809e88942a0d075409e80c12219a1352bb2abfe4210a73580446e2a82012b06`. Only after
implementation, independent audit, a clean commit, and complete release gates may one new revision
receive one new three-cell capacity attempt.

## 2026-09-05 — Capacity Repair 002 implemented; no third attempt opened

No capacity cell, registered optimizer step, development/held-out reset, controller comparison,
HEPA fit, or supervisor evaluation ran during this implementation. The failed V2 report and all
external `27a47d0` artifacts remain unchanged. Repair 002 was already frozen and committed in
`82a616c` with raw SHA-256
`1809e88942a0d075409e80c12219a1352bb2abfe4210a73580446e2a82012b06` before either production or
test source changed.

The capacity module now extracts the existing exact training-config serialization into one shared
`_training_config_mapping_bytes` function. It retains the prior sorted, two-space-indented,
`allow_nan=False`, UTF-8, trailing-newline encoding byte-for-byte; it does not substitute the
compact MPC evidence serializer. `_training_config_bytes(config)` passes the dataclass mapping
through that function, while the new metadata verifier first requires the loaded metadata value to
be a complete mapping and then passes the whole mapping through the same function. Exact serialized
bytes must match. There is no field-specific list/tuple coercion, subset comparison, ignored key,
or numerical tolerance.

Checkpoint admission still reads and exactly compares the independent `training_config.json` file
before loading metadata. Only after that byte check does the new whole-object metadata comparison
run; algorithm/seed/timestep, constrained evidence, environment/task identity, runtime, allocator,
rollout, optimizer-step, artifact, and production policy-load checks remain in their original order.
The learner and every optimizer-affecting module are untouched by Repair 002.

The production report, deterministic-evidence domain, and verifier now require V3. Synthetic
fixtures use a separately versioned synthetic V2 report and deterministic domain. The protocol
identity retains Repair 001 with both clarifications and appends Repair 002 with its exact path and
raw digest. Production verification rejects production V1/V2 and synthetic V2 before runtime,
host, or checkpoint access; synthetic verification symmetrically rejects synthetic V1 and
production V3 before invoking any injected hook.

The new tests prove that the raw dataclass mapping and JSON-round-tripped mapping are unequal as
Python objects while producing identical frozen config bytes. A production `_admit_checkpoint`
test writes the exact config file, supplies correctly identified JSON-round-tripped metadata, and
proves execution advances beyond the repaired comparison to the deliberately failing next campaign
identity gate. Changed scalar and nested sequence values, missing/extra keys, reordered sequences,
and non-finite values all fail. A separate ordering test supplies internally consistent metadata
but a changed standalone config file and proves metadata loading is never reached.

The final capacity suite passes **69 tests in 0.96 seconds**; capacity plus the unchanged constrained
learner passes **100 tests in 15.04 seconds**. The authoritative locked learner/capacity/checkpoint/
production/provenance/config selection passes **297 tests in 213.49 seconds**. Strict scoped MyPy,
Black, Python compilation, and Git whitespace checks pass for the changed production/test files.
An independent read-only audit found no P0/P1 implementation issue. Its initial P2/P3 test-evidence
requests are the production-wiring, self-consistent-file-ordering, symmetric cross-version, and
explicit-domain tests described above and are now closed.

The audit also recorded a pre-existing checkpoint-format limitation outside Repair 002: the generic
metadata loader uses ordinary `json.loads` and converts `training_config` with `dict(...)`, so raw
duplicate keys or a JSON array of key/value pairs are normalized before this semantic comparison.
The capacity verifier binds the raw metadata file hash and later reloads the artifact, but this
repair is not described as authenticating a globally canonical raw `metadata.json` encoding. That
broader checkpoint-format hardening must be separately scoped rather than smuggled into the frozen
admission repair.

The Repair-002 implementation is not yet capacity evidence. It must be committed, confirmed clean,
and pass the complete Python repository plus frontend release gates on that exact revision before
the one permitted V3 attempt can begin. While that future three-cell attempt runs, no parallel work
or token-consuming analysis will occur.

## 2026-09-05 — Repair-002 V3 capacity attempt passed and campaign budget authenticated

Repair 002 was committed as clean revision
`4f505b89c36dbb5542fd255e469e3c729791cee2`. The staged diff contained only the capacity module,
its tests, and four status documents; an independent final read-only review found no actionable
P0/P1/P2 issue. Immediately before the commit, the capacity suite passed 69 tests in 1.01 seconds,
the capacity/constrained selection passed 101 tests in 3.68 seconds, and Black, compilation, scoped
strict typing, and Git whitespace checks passed. The frozen Repair-002 document still matched raw
SHA-256 `1809e88942a0d075409e80c12219a1352bb2abfe4210a73580446e2a82012b06`.

On the exact committed tree, the complete repository passed **1,832 tests in 415.48 seconds** with
only the existing Starlette/httpx warning and trailing HiGHS diagnostics. The frontend TypeScript/
Vite production build and `oxlint` passed; Vite retained only the existing large 3D chunk advisory.
The tracked and untracked worktree was empty after those gates.

The pre-execution custody check confirmed that the new external root
`/private/tmp/plantanagent-beta-rl-capacity-4f505b8` did not exist, no repository record referenced
`4f505b8` as an earlier attempt, no capacity/training/controller-benchmark/test process was running,
and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS` were all absent. The host had 9,286,464 KiB free on the target filesystem,
well above the frozen checkpoint allowance. The Make invocation supplied both required exclusivity
and first-attempt assertions. All other agents were stopped or complete before execution. During
the attempt, no parallel repository work, edit, test, subagent task, or other experiment ran; the
root agent only polled the parent process for completion.

The only attempt at `4f505b8` completed once. All three children exited zero, their checkpoint
directories passed the complete production admission path, and the runner issued and immediately
verified a V3 report. A separate same-host Make invocation then reopened the report, recaptured the
runtime and host, re-admitted all checkpoint artifacts, recalculated the evidence and projections,
and returned the same completed status. The authenticated identities are:

- raw `report.json` SHA-256:
  `3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa`;
- report-domain SHA-256:
  `78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e`;
- deterministic pre-run evidence SHA-256:
  `4676d4862b3d783be2d4990034bbcde31be66ef7a8d591331b007ace6651b27c`;
- runtime-provenance SHA-256:
  `970041ae08346527aa9e5fd31743fdf7be4d82ac805c96d6152bdb3881974a5d`; and
- production schema/domain: `plant-beta-constrained-rl-capacity-report-v3`.

The bound host was an Apple M4 Pro with 12 physical/logical cores and 24 GiB memory, using eight
Torch intra-op and twelve inter-op threads. The runtime was CPython 3.12.7, NumPy 2.5.2,
Gymnasium 1.3.0, Stable-Baselines3 2.9.0, Torch 2.13.0, and SciPy 1.18.1, with exact lock,
dependency-entrypoint, simulator-source, worktree-content, and worktree-status identities.

Each cell proved 8,192 controller transitions, 122,880 physical minute steps, one rollout, 320
optimizer steps, 16 fixture allocator entries, and one final checkpoint:

| Family | Wall s | Rollout s | Fixed overhead s | Controller transitions/s | Checkpoint logical bytes |
|---|---:|---:|---:|---:|---:|
| PI-Lagrangian | 129.550079666 | 117.208728250 | 12.341351416 | 69.892405816 | 1,245,824 |
| integral-only | 130.401020125 | 117.977248541 | 12.423771584 | 69.437116913 | 1,245,805 |
| scalar PPO | 128.532934000 | 116.023039708 | 12.509894292 | 70.606665888 | 578,575 |

All three stderr records are empty. None of the three stdout logs was opened; their byte counts and
digests remain bound in the report. No observed reward, settlement-cost, action, multiplier, loss,
policy-behavior, market, outage, inventory, safety, service, cycle-return, or controller-ranking
value was operator-inspected or published.

The frozen registered arithmetic is 15 policies, 1,920 rollouts, 15,728,640 controller transitions,
235,929,600 physical minute steps, 614,400 minibatch optimizer steps, and 23,460 reset manifests.
The authenticated sequential wall projections on this host are:

| Rollout factor | Nanoseconds | Hours | Days |
|---:|---:|---:|---:|
| 1.00x | 224,960,145,645,820 | 62.4889 | 2.6037 |
| 1.25x | 281,153,588,285,660 | 78.0982 | 3.2541 |
| 1.50x | 337,347,030,925,500 | 93.7075 | 3.9045 |

The structural checkpoint allowance totals 182,818,220 bytes (174.35 MiB) and the frozen rule
recommends 1,256,560,044 bytes (1.17 GiB) free for checkpoint growth. The duration multipliers are
mechanical planning scenarios, not confidence intervals, guarantees, or parallel speed claims;
the storage calculation deliberately ignores ZIP compression and is a structural allowance rather
than a prediction of exact ZIP encoding. Per-family timing differences are operational capacity
measurements, not a statistical comparison of algorithms. No timing or checkpoint from either
failed capacity attempt enters the V3 projection.

The report's only admissible result claim is: “On the recorded host and clean source revision,
three fixture-domain, registered-shaped first rollouts imply the reported sequential capacity
scenarios. No registered seed or controller-performance outcome was opened.” As a protocol-process
decision rather than an experimental claim, this valid report closes the infrastructure measurement
required before requesting registered execution. No registered optimizer seed, development/
held-out reset, policy selection, controller comparison, or controller-performance outcome was
opened. The detailed result is retained in `results/v2_constrained_rl_capacity_4f505b8.md`. The
next step is an authorization packet for the already frozen 15-policy protocol; registered
execution remains closed until explicit approval.

## 2026-09-05 — Post-capacity registered-study readiness audit and custody design frozen

No registered, development, or held-out environment was constructed or reset during this audit.
No training process, optimizer update, controller comparison, or HEPA operation ran. Three
independent static reviews compared the current implementation with the complete frozen
constrained-RL protocol.

The learner foundation is substantive: the single-run path enforces the three family identifiers,
five optimizer seeds, 1,048,576-transition budget, four synchronous environments, final-only
checkpoint, no resume, registered allocator domain, exact 1,564-reset endpoint, and production
checkpoint eligibility. It captures clean runtime provenance, the beta checkpoint retains all 128
rollouts and allocator custody, and the beta benchmark can independently load a registered final
policy.

The audit nevertheless returned a training **no-go**. Registered validation allowed arbitrary
otherwise-valid plant, episode, reward, economic, market, cycle, and disturbance settings, so the
`registered` label alone did not prove the frozen beta task. The repository also had no committed
3-by-5 registry, exact execution order, preclaimed output map, durable consumed-attempt ledger,
sealed process-log custody, complete failure accounting, or detached 15-checkpoint campaign
verifier. The generic one-run CLI permits an output override and cannot safely substitute for that
missing campaign boundary.

Development and held-out readiness also returned **no-go**. The existing beta runner is a strong
four-arm integration instrument, but it makes H and M24 blockers for PPO/E-beta eligibility,
aggregates surviving subsets, accepts arbitrary one-to-32 seed ranges, and contains only one policy.
There is no complete 15-policy campaign, all-32-or-no-score vector, frozen IQM/lower-CVaR/median/
strict-positive selector, family-denominator rule, generic policy-minus-E-beta sensitivity
rescorer, hierarchical bootstrap, Wilson interval, performance profile, registered 1/1-thread
latency procedure, held-out report, or executable ten-clause success rule. A detached verifier gap
also permits coherent mutation of the policy controller's fallback descriptor; that must be closed
before its evidence is reused.

`CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md` is now frozen before source changes. It
binds the successful capacity report, exact M4 host planning context, the full registered task
template, family/seed matrix, deterministic paths, tracked user-authorization record, conservative
pre-launch attempt claims, sequential child/log handling, checkpoint admission, production versus
synthetic evidence, detached verification, and fixture-only adversarial gates. Removing only seed,
output path, and algorithm from the complete default registered config produces task-template
domain SHA-256 `89abbe6288d512c254d8186f56c6d269b9f273fb49bbb13d24d7b02425bfa77b`.

The custody protocol is a build authorization only. It does not create the later user authorization
record and permits no registered optimizer step or protected reset. After this layer is implemented
and release-validated, the development analysis and campaign schemas must be implemented and
hash-bound before requesting the multi-day experiment.

## 2026-09-10 — V2 visibility preview scoped before implementation checks

The user prioritized making the existing V2 backend visible to understand the
current product. A read-only frontend/backend audit found that the existing
control room and state-driven 3D scene still receive V1 telemetry, while the V2
public forecasts, commitments, cycle margins and economic/command ledgers have no
operator-facing API.

The bounded product hypothesis is that one connected V2 inspection view will
expose the current capabilities and integration gaps. The implementation scope is
recorded in `V2_VISIBILITY_PREVIEW.md`: a separate `/v2` page and lazy API session,
unchanged default beta task, fixed disposable fixture seed `9,000,034`, explicit
one/four-decision requests, H and optional experimental E-beta, and public-only
presentation data. No registered training, development or held-out execution is
authorized by this increment. No benchmark, training or protected report has been
opened during the scoping audit.

Work is split between the V2 API/fixture checks and frontend presentation, with
shared DTO semantics and parent integration review. All inherited user document
changes and the untracked registered-training custody protocol are preserved.
Validation and any fixture failures will be appended after implementation.

## 2026-09-10 — V2 Lab connected and fixture verification completed

Implemented an additive `/api/v2/demo` router and `/v2` control-room view without
changing V1 control, the simulator, any research task/controller, or any frozen
training/evaluation protocol. The canonical UI contract projects current public
V2 data: actual physical state, solar mean/p10 windows, market mean/p10/p90 bins,
recurring methane obligations, realized economic contributions, daily resource
margins and settlements, and requested/applied power with command/execution
identities. H is explicitly forecast-blind. Optional E-beta exposes its actual
retained schedule. Neither PPO nor HEPA is represented as available or trained.

The preview has one lazy in-memory session, only fixture seed `9,000,034`, explicit
one/four-decision requests, strict input rejection and run/revision preconditions.
It accepts no checkpoint, seed, task or artifact path. Synchronous FastAPI worker
routes serialize access independently of the V1 event loop. There is no automatic
clock, fallback, supervisor write authority or registered experiment. A fixed
168-hour task is available to inspect, but no full-week benchmark was run.

Independent source review found one P1 presentation defect: a failure during an
interval could expose partially advanced telemetry with the prior ledger/counters,
or prevent snapshot generation entirely. The fix commits a deep public snapshot
after each fully projected decision, including intermediate successes in a batch.
A failed run shows only that boundary, explicitly warns that a partial interval
may have executed, refuses further advancement and serves subsequent reads without
accessing damaged environment state. Tests inject real partial minute movement
and a post-physics projection failure. The defect is retained here rather than
misrepresented as a successful first implementation.

Validation evidence:

- The original nine API tests passed in 22.11 seconds, including one fixture day
  crossing delivery and resource-return settlements. After the failure fix and
  real E-beta projection test, 11 tests passed in 28.29 seconds with the already
  exercised daily test excluded; all 12 passed in the subsequent full run.
- The complete Python repository run returned **1,843 passed, 1 failed** in
  475.91 seconds. The failure was the scalar case of
  `test_real_beta_family_trains_saves_loads_and_crosses_common_physical_door`:
  `RuntimeProvenanceChangedError` identified only `worktree_content_sha256` during
  resume admission. The root agent had edited frontend/documentation files while
  this fixture test was running. The custody check correctly rejected that
  change. No production or test code was weakened. With all source edits stopped,
  the exact failing case passed in **22.50 seconds**. This is full-suite coverage
  plus a focused rerun, not a claim of one clean-revision all-pass execution.
- Scoped MyPy, Black, compilation and whitespace checks passed for the new backend
  files. The final frontend TypeScript/Vite build and oxlint passed. The existing
  large lazy 3D chunk advisory, Starlette/httpx deprecation and inherited Three.js
  deprecation warnings remain; no browser application error was observed.
- Browser checks exercised a real H hour, real E-beta decisions and retained plans,
  controller reset, 2D/3D switching, exact-value disclosures, desktop and 360px layout,
  and two-tab stale-command refusal followed by explicit refresh with no automatic
  retry. Chart labels retain 11px screen size at the narrow width and the page has
  no document-wide horizontal overflow. Controls were moved above the plant and
  the delivery table gained an explicit all-commitments expansion after visual QA.

The existing servers on ports 8000/5173 did not respond to the initial HTTP check
and were left running. An isolated preview was started on 8001/5174 with the V1
supervisor disabled. The frontend proxy now accepts optional `PLANT_API_TARGET`,
retaining port 8000 by default. Locked optional dependencies were installed with
`uv sync --frozen --all-groups --inexact`; `uv.lock` did not change. The V2 browser
preview is left available for the user with the clock paused.

Existing repository tests include disposable optimizer smokes, which ran only in
their predefined fixture domains. No registered training, development or held-out
seed was opened, no protected policy was selected, and no comparative performance
or commercial claim follows. Verified artifact replay, full-frame historical
scrubbing and governed live intervention remain future product increments.

## 2026-09-10 — Unified commission demo inventory and product proposal

The user requested a complete V1-to-V2 presentation inventory and a plan for one
cohesive demo product. Source inspection at `ed5a24e`, research-document review and
read-only browser inspection confirmed the central integration gap: the V1 provider
still supplies Overview/3D/Agent/Events/Scenarios, while V2 uses a page-local session.
Current V2 telemetry exposes the latest public snapshot, not historical plant frames,
commitment timers, detailed exposure or the original forecast attached to every plan.

`UNIFIED_COMMISSION_DEMO_PLAN.md` records the proposed Plant/Planning/Review workspace,
17 capability mappings, reuse of existing 2D/3D and design tokens, a shared cursor and
contextual inspector, public-record dependencies, staged acceptance criteria and the
boundary between an illustrative fixture and admitted research evidence. The product
hypothesis is that connecting obligations, equipment, decisions and outcomes will make
the implemented research understandable within a five-minute walkthrough. This is a
proposal for discussion, not a new frozen research protocol or an implementation claim.

Validation for this documentation-only increment is source/DTO traceability, inspection
of the current browser view, local-link existence and whitespace review. No simulation
step/reset, optimizer, registered/development/held-out execution, checkpoint selection
or performance comparison was run. Application code and all pre-existing user edits,
including the untracked registered-training custody protocol, remain unchanged.

## 2026-09-10 — Weather visuals and independent feature panels accepted into the plan

The user requested 3D clouds and a satellite-like overhead weather view to make
stochastic solar reduction and forecasts immediately understandable. They also
requested a separate Overview and independent feature panels, especially Economics,
instead of scrolling through multipurpose pages. The plan now supersedes its initial
three-view grouping with shared-session Overview, Plant, Weather, Planning, Deliveries
& resources, Economics, Events and Evidence destinations.

Source inspection found a scalar site-wide weather factor, no spatial cloud/wind
model, and an intentional night-time mask in public V2 telemetry. Solar output also
combines resource and equipment-health effects; curtailment is a separate quantity.
The proposed visual adapter therefore preserves observation/forecast/unknown status,
uses public sample-time-aligned data, distinguishes equipment losses from weather,
and labels the overhead rendering as synthetic. It cannot infer night-time clouds,
forecast cloud positions, or read private future truth. Weather graphics are a
presentation layer and leave the frozen physics/135-observation contract unchanged.

`UNIFIED_COMMISSION_DEMO_PLAN.md` now records the revised navigation, weather treatment,
public-data dependencies, per-panel responsibilities, migration and acceptance checks.
Validation is documentation/source consistency, local-link existence and whitespace
review. No application code, simulator execution or research protocol changed, and
all pre-existing user documentation edits remain separate.

## 2026-09-10 — Unified workspace implementation and bounded rehearsal declaration

The user authorized completion of the agreed demo and removal of separate V1/V2
product entries. One V2 mission now supplies eight panels and the shared history
cursor. Legacy URLs redirect within that workspace; V1 source/API remain preserved.
The product hypothesis is that observed weather, exact decision inputs, equipment
execution and delivery/resource outcomes become inspectable as one coherent story.

Before running presentation recordings, declare exactly one H attempt and one E-beta
attempt on existing disposable seed 9,000,034. Each targets 168 hours and has a
20-minute wall limit; no retry, alternative seed, winner selection or hidden fallback.
The recording CLI writes an exclusive attempt marker, source digest and public
record checksum outside Git in runs/commission-demo. Preserve stopped/failed arms.
The page initially opens saved E-beta at decision 32 (8 elapsed hours), or H if no
E-beta record is available; this is a fixed presentation cursor, not outcome selection.
These runs exercise the demo only: no training, admitted research, development
selection or held-out evaluation is authorized by this declaration.

## 2026-09-10 — Unified workspace completed; presentation outcomes retained

Implemented one V2-backed product with Overview, Plant, Weather, Planning,
Deliveries & resources, Economics, Events and Evidence. Shared state retains the
run, controller, selected equipment and completed-decision cursor across panels.
Legacy frontend URLs redirect into this product; V1 source and APIs remain intact.
The user explicitly retained the original visual identity and rejected tutorial
copy. The workspace therefore uses the existing logo, palette, fonts and pill
controls, concise page names and operational labels. Weather and uncertainty
qualifications stay factual, with research/methodology detail in Evidence.

The public API now retains complete detached frames, original decision forecast
issues, commitment state/timers/exposure, planned inventories, exact minute applied
quantities, observed boundary events and per-resource cycle settlements. Indexed
history reads, bounded previous-session archives and checksum-sealed public JSON
recordings support playback without executing a controller. Failure exports preserve
only the last successfully projected boundary, including the terminal failure status.
Night-time cloud state remains unobserved; solar equipment losses are not painted as
weather. Synthetic cloud/shading geometry reads only public attenuation. Existing
3D units and flow colors are reused with an overhead weather view and 2D alternative.

The two declared rehearsal attempts ran exactly once. H completed 168 hours, delivered
5,564.720180 kg, left 1,155.279820 kg unmet, failed all seven daily resource returns,
and accumulated EUR -28,834.280382 simulated value. E-beta stopped at 9.25 hours with
BetaMPCSolveError, after 240 kg delivered, zero settled shortfall and no daily cycle
settlement; cumulative value was EUR -938.238778. The failure remains visible through
its earlier frames, and no fallback or replacement attempt was used. These unequal
records are not a controller ranking or an admitted experiment. Raw records are
outside Git; UNIFIED_WORKSPACE_REHEARSAL_MANIFEST.json retains both identities,
checksums, dirty-source disclosure and the common Python-source digest.

Validation: 21 focused API/replay/weather tests passed. The full frozen-source Python
suite then passed **1,853 tests** in 482.52 seconds, with one existing Starlette/httpx
deprecation warning. compileall passed. The final frontend build, TypeScript and
oxlint passed after the subsequent copy/layout cleanup; Git whitespace validation
passed. Browser checks covered all panels, fixed-record loading, minute execution,
historical night masking, original forecast issues, full-week failed settlements,
keyboard focus containment, redirects and 390/1280-pixel layouts. Overview's weather
readout was moved out of its clipped mission column. Fresh-load browser error logs
were empty. Existing Three.js deprecation and lazy 3D bundle-size warnings remain.

Routine development failures were resolved: formatter downloads needed network
permission; the initial black invocation found no installed module, so formatting
used isolated ruff/Prettier tools. A temporary hot-refresh context error during the
provider extraction cleared on reload and did not recur on fresh loads. A browser
heading assertion initially reached the cold API load before the page was ready;
redirect destinations and the loaded page were subsequently verified. No failure
was represented as a passing check.

UNIFIED_WORKSPACE_IMPLEMENTATION.md records startup instructions, retained outcomes,
contract semantics and scope limits. M24, admitted beta PPO, HEPA/NLP, governed live
intervention, site/economic calibration and HIL integration remain pending. No
registered training, selection, held-out evaluation or protocol change occurred.
All pre-existing user documentation and the untracked custody protocol are preserved
outside this implementation commit.

## 2026-09-11 — Registered custody admission foundation

Hypothesis: the frozen task digest, path-derived config classification and explicit campaign
authority can close the generic learner's audited bypasses without changing plant physics,
learning settings, seed domains, comparator behavior or the unified product UI.

The inherited custody protocol and its four status-document edits were preserved unchanged and
committed as `8c3d535` before source edits. Registered task validation now requires the exact
frozen digest; JSON decoding rejects duplicates and non-finite constants/numbers before run-kind
classification. Registered files require complete canonical bytes and regular nonsymlink,
singly linked custody. Reserved campaign config names remain registered even if their plan is
removed. All registered CLI overrides, including output destinations, are rejected. The generic
registered learner remains closed until an authenticated campaign claim can admit a child.

New outcome-free infrastructure provides the exact 15-cell matrix, exclusive file and directory
persistence barriers, complete retained-artifact inventories, bounded sealed subprocess logs,
process-group cleanup, read-only tracked authorization and commit-lineage admission, a fixed
production checkpoint-loader adapter, exact plan/claim/receipt/report schemas, and the closed
failure precedence. These are components of the custodian; they do not yet constitute a complete
campaign runner or detached verifier. REGISTERED_TRAINING_CUSTODY_BUILD.md tracks each remaining
admission obligation and records representation choices before registered outcomes exist.

The initial config-focused suite passed 226 tests. Process fixtures first exposed three failures
when querying an exiting process group before reaping its leader; rerunning outside the sandbox
did not resolve them. Cleanup now reaps before probing and treats only ESRCH as disappearance,
with bounded retry for Darwin's transient EPERM. The filesystem/process suite then passed 15
tests; adding handled parent SIGTERM/SIGHUP fixtures raised that to 17 passing tests. One
authorization test exposed an exception-class mismatch from the reused strict host validator;
the custody boundary now normalizes that categorical schema rejection, and all 20 authorization
tests passed. Existing loosely typed dataclass parsing was made generic, and an obsolete callback
type suppression removed, to pass strict typing of the touched modules.

The combined config, authorization, inventory/process and negative checkpoint suite passed 270
tests before the final two interruption fixtures. Inert document tests then passed 129 tests,
including every pair of failure-precedence collisions and the complete sequential claim chain.
Strict typing passed the eight source modules checked at that point. These are incremental
fixture results, not the complete clean release gate. No registered, development or held-out
reset, optimizer update, policy comparison, selection or production authorization occurred.

The final foundation regression run passed **605 tests in 39.40 seconds**, including the existing
constrained learner, beta checkpoint, runtime provenance and capacity suites. Strict typing passed
all nine touched source modules; Black checked all 15 touched source/test files, Git whitespace
checks passed, and the ordinary training CLI help imported successfully. The dependency lock and
frontend are unchanged. The complete campaign, detached verifier and downstream analysis gates
remain pending; this commit is an auditable intermediate foundation, not a release admission.

## 2026-09-11 — Detached beta policy descriptor custody closed

Hypothesis: a coherently rehashed benchmark could relabel the PPO controller's fallback or
information contract because the detached verifier validated only selected descriptor fields.
Static inspection confirmed the missing checks. The verifier now requires the complete exact
controller descriptor, the exact loaded-checkpoint fields, the canonical label/description,
forecast and information declarations, and `raise; no controller-internal fallback`. The
controller, dispatcher, solver, plant and economic behavior remain unchanged.

Eight new adversarial descriptor tests reject changed or missing fallback declarations,
misleading information/forecast text and undeclared controller/checkpoint fields. They explicitly
forbid environment reset and admit the genuine fixed-policy descriptor before mutation. The full
existing detached beta verifier suite passed **43 tests in 131.33 seconds**. A subsequent
annotation-only cleanup separated branch-local MPC/process execution variable names and replaced
an untyped file-identity lambda with the same typed projection; the eight new tests also passed
after the first cleanup. Complete release validation remains a later gate.

The host's older mypy initially produced 17 errors, including NumPy/source-resolution issues and
existing function-wide variable collisions. An isolated mypy 2.3.1 installation under
`/private/tmp`, explicit `MYPYPATH=backend/src:simulator/src`, and the annotation cleanup now pass
strict checking of the verifier. The same current tool also passes all nine custody source
modules. Black and Git whitespace checks pass. No dependency lock, protocol, evaluation seed
domain, UI identity or training authorization changed; no registered or held-out run occurred.

## 2026-09-11 — Registered campaign execution and detached custody reconstruction

Hypothesis: the fixed 15-cell plan can be executed and closed without outcome-dependent operator
choices by separating durable receipt collection from production reporting and independently
reconstructing the report from retained evidence. The registered runner now prepublishes the
authorization copy, plan and all 15 canonical configs, claims each cell durably, executes the
fixed sequential command with sealed logs, applies the frozen continuation/abort precedence,
and retains complete failed artifacts. A single-use anonymous parent pipe binds the current
claim/config to the registered learner; it reopens the complete registry and prior receipt chain
before environment/model construction. Standalone registered training still has no authority.

The production report is available only through fixed production dependencies. Synthetic
collection and verification use a separate filename/schema and always carry a null production
claim. Detached verification reopens every config, receipt, log and inventory, independently
re-admits completed checkpoints through the existing production loader, reconstructs all counts
and statuses, and checks the evidence again after loading. The runner unconditionally invokes
a fresh-process verifier after durable report publication, including when local recapture fails.
Read-only forensic inspection classifies interrupted or malformed claims without authorizing
retry, reconstructing a terminal receipt or issuing a verified claim. CLI errors use fixed
categorical text so exception payloads cannot reveal sealed outcomes.

The first integrated campaign/process suite passed 38 tests; the combined campaign, verifier,
launch, registered-config and process run passed 230 tests. Eight subsequent read-only launch
admission cases and publication/forensic checks passed in a 25-test targeted run. Seven new
after-load corruption cases exposed six missed mutations in the synthetic verifier (report,
plan, authorization, claim, log and earlier policy bytes); the extra-root case was already
rejected. Production and synthetic verification now share the final complete evidence recheck,
including root identity and exact directory contents. All 40 verifier tests then passed in
31.50 seconds. The expanded 27-test campaign suite passed in 11.16 seconds, including retained
partial/forbidden artifacts, prior receipt/log/artifact mutation, physical root replacement,
and config corruption before the first claim. Strict typing passed all 16 custody/config/learner
source modules, and Black checked all 23 custody source/test files.

REGISTERED_TRAINING_CUSTODY_BUILD.md records the parent-channel representation, independently
reconstructible process-time boundary, storage-abort object and non-authorizing forensic schema.
The frozen protocols, task, optimizer matrix, dependency lock, V1 and unified UI remain unchanged.
No production authorization, registered reset/update, development/held-out reset, checkpoint
selection or policy-performance outcome was opened. Complete regression and clean release
validation follow; the downstream analysis build remains pending.

The combined custody, exact-config, constrained learner, checkpoint, runtime and capacity
regression passed **697 tests in 84.40 seconds**. Final static review then found that unexpected
files in the open campaign namespace could reach the child gate and unnecessarily consume an
attempt before rejection. The parent now checks every root/config/log/attempt/artifact directory
before claiming, after the process and after admission. Twelve new cases cover unexpected root,
log, future-claim and future-artifact entries at all three boundaries; the expanded campaign
suite passed **39 tests in 13.54 seconds**. The protocol and learner command remain unchanged.
CLI import/help, the Makefile's nonexecuting command expansion, dependency-lock cleanliness,
both frozen protocol hashes and Git whitespace checks passed. The next complete repository
gate will run on the clean committed campaign implementation, without opening study seeds.

## 2026-09-11 — Complete custody release gate at `ad6ad1e`

Clean revision `ad6ad1e85a0269f592d42b7c9edcd6c64f2c0579` passed **2,285 Python tests in
536.22 seconds**, with successful process exit. The existing Starlette/httpx deprecation
warning remained; native HiGHS cleanup diagnostics appeared after the passing test summary.
Compilation, CLI help/import, Git whitespace, unchanged dependency lock, Black checks on
all 29 Python files changed since the frozen custody protocol, strict mypy 2.3.1 checks on
all 17 changed source modules, frontend production build and oxlint also passed. Git status
was empty before and after the gate. The existing large 3D frontend chunk warning remains.

An optional type-check expansion into the unchanged `training/ppo.py` reported a redundant
cast at line 150. That module is outside this implementation's changed-source set; no source
was modified to suppress the diagnostic. The required 17-module scope passed on the clean
revision. Test collection found the same 2,285 tests and performed no environment resets.

The custody runner/verifier implementation is now validated; this status update records its
tested revision and does not claim that a future code or authorization revision has passed.
All genuine model training in this gate was disposable fixture-domain test work. No registered
training, development/held-out reset, outcome selection or production authorization occurred.
The fixed three-family/five-seed study and both protocol hashes remain unchanged.

Read-only downstream inspection confirmed two prerequisites: the current four-arm aggregator
filters on H/M24 feasibility and therefore cannot supply the registered PPO/E-beta pairwise
selector, and registered checkpoint loading requires exact training source/runtime provenance.
Before requesting the multi-day study, the frozen source still needs all-32 pairwise eligibility,
five-policy family retention, the lexicographic champion selector, generic policy-minus-E-beta
sensitivity, registered uncertainty/latency/scorecards and detached development/held-out
campaign verification. Later stage permission and evidence publication must preserve the
training source identity. These findings required no manifest, protected seed, checkpoint
outcome, sealed log or controller evaluation to be opened.

## 2026-09-11 — Outcome-blind study analysis representations

The previous turn completed the custody implementation and its clean release evidence, which
is substantive progress toward the unchanged study goal. Current HEAD `b0ee589` is clean.
The next hypothesis is that pure numerical rules can be specified and verified independently
before connecting them to protected evaluation execution. CONSTRAINED_RL_STUDY_ANALYSIS_BUILD.md
now records the complete remaining layer checklist and the numerical representations for
32-pair summaries, exact selection, conditional family missingness, matched optimizer contrasts,
paired/hierarchical resampling, fixed absolute-EUR profile thresholds, Wilson intervals and
additive policy-minus-E-beta sensitivity. The common episode bootstrap block preserves shared
exogenous/baseline pairing; family summaries explicitly retain a denominator of five policies.

These choices are recorded before implementation or any study outcome. They do not alter
the existing task, optimizer matrix, seed domains, primary contrast or success rule, and do not
authorize training or evaluation. The numerical layer alone will grant no verified eligibility
or production claim. The operational pairwise, campaign, scorecard/latency and detached-stage
verification layers remain required before a training request.

## 2026-09-11 — Study numerical kernels and independent arithmetic checks

The analysis representations were committed as `ea081f6` before source implementation.
New isolated study modules now provide the exact source contract, 32-pair statistics,
10,000-replicate paired and two-stage family bootstrap, Wilson intervals, fixed performance
profiles, exact optimizer matching, all-five family selection and generic additive policy-minus-
E-beta sensitivity. Missing/ineligible pair slots prevent a policy score; every optimizer seed
remains visible. Source-ledger admission is deliberately not inferred by the numerical layer:
selection/sensitivity reconstruction returns `source_evidence_verified: false` and null claim.
The full detached stage verifier and physical evidence binding remain necessary.

The initial statistics suite passed 32 tests. Independent expected values use closed-form
arithmetic and sort/fsum/manual linear interpolation rather than calling the implementation's
aggregate helpers. They check repeated bootstrap observations, common episode resampling across
optimizer rows, between-policy variance and the absence of Cartesian optimizer comparisons.
Selection tests cover each lexicographic priority, exact zero, lower-seed ties, failed training,
unverified reports and missing pairs despite arbitrarily large surviving values. Sensitivity
tests independently rescore weighted input ledgers, preserve unmet-service penalty when changing
methane liability, reverse every sign when swapping inputs and retain exact-zero/tiny reversals.

All three suites passed 77 tests; numeric sequence/float64-overflow rejection cases increased
this to **82 passing tests in 0.42 seconds**. Autouse guards prohibit plant construction/reset,
learner construction and policy loading in these tests. Initial strict typing identified seven
annotation/export/narrowing issues; explicit re-exports and checked local narrowing resolved
them, and all five study source modules passed. No plant/controller behavior, optimizer setting,
frozen upstream protocol, dependency lock or UI changed. No protected seed, manifest, admitted
checkpoint, sealed log, selection outcome or study execution was opened.

The numerical contract digest is
`94c4ce4b665ab31228b124f9fe056c98fb37ab5b349a12f785c2368b4d095d72`.
The next integration must derive pairwise eligibility from independently verified complete
arm evidence, bind numerical results to those source ledgers, and implement the remaining
scorecard/latency, development/held-out custody and detached verification requirements.

The combined numerical, custody, config, constrained learner, checkpoint, runtime and capacity
regression passed **791 tests in 93.11 seconds**. Strict typing passed all 22 custody/verifier/
study source modules; Black checked the nine new numerical source/test files, compilation and
Git whitespace checks passed, and the dependency lock is unchanged. This is an intermediate
implementation regression, not the final clean study release gate. The latter remains required
after verified input binding and stage execution/verification are complete.

## 2026-09-11 — Standalone arm and pairwise evidence build

The previous turn is classified as progress: commits `ea081f6` and `0051a57` bind and
implement the numerical core, with 791 passing regression tests. Current HEAD `0051a57`
is clean. The next hypothesis is that individually sealed, independently verified arm
evidence can remove the legacy all-four eligibility coupling while retaining its complete
minute/decision/safety/cycle/accounting checks. The study build document now specifies a new
inner arm format that removes only the legacy pair-eligibility field/reason; old benchmark
semantics remain unchanged. Synthetic fixtures will use only the 9,000,000–9,999,999 domain.
No production seed or policy is opened, and production evaluation admission remains a later
explicit stage-custody connection, not a consequence of arithmetic or synthetic verification.

The new standalone arm and pair modules now admit only explicitly synthetic,
external canonical files. Scope checks precede manifest reconstruction; protected
seeds and registered configurations are rejected. Each arm reuses the complete
minute/decision/forecast/cycle/accounting verifier. The pair independently reopens
PPO and E-beta, checks their episode/source/public-information identities and every
completion, safety, return and service condition, and computes value/sensitivity
only for an eligible pair. H/M24 remain separately retained context. There is no
production execution, checkpoint admission or stage-authorization entry point in
these modules.

Inspection found an existing zero-interval accounting gap: a normal failed arm's
metrics were not checked when its first controller decision failed. The shared
verifier now reconstructs exact initial economic and physical metrics for that
case. This preserves the failure record while rejecting fabricated profit,
service, terminal inventory or wear in unfiltered scorecards. Historical four-arm
eligibility semantics remain unchanged; the new inner format/domain omits only
the historical all-four flag and paired-cell exclusion.

The first new fixture invocation failed before evaluation because its setup used
an unsupported heuristic-factory argument. The fixture now explicitly intercepts
the canonical heuristic's compute method within its disposable test context; the
fixed checkpoint and relaxed-return positive-pair geometry are solely synthetic.
The corrected suite passed **48 tests in 178.09 seconds**. It proves a valid
PPO/E-beta pair survives a failed H context arm, and rejects rehashed physical,
controller, accounting, scope, file and pair-report substitutions. Strict mypy
passed the three changed source modules. Subsequent cases add every independent
pair gate, inclusive service tolerance, preflight failure semantics and identical
file replacement during graph verification; these await the combined regression.
No study seed, trained/registered policy, sealed log, production outcome or
registered authorization has been opened or created.

The expanded combined regression passed **204 tests in 331.05 seconds**, with
successful process exit. This comprises 79 standalone/pair integration/adversarial
cases, 82 numerical cases and 43 historical benchmark-verifier cases. Native
HiGHS cleanup text followed the passing summary. The identical-content replacement
cases prove that inode/stat identity is retained through both arm admission and
the detached verifier's final graph check. Every independent pair condition and
all three inclusive 1e-6 kg service comparisons are tested. Strict mypy on all
three changed source modules, Black on nine source/test files, compilation,
whitespace and unchanged dependency-lock checks passed. Both frozen upstream
protocol SHA-256 values remain unchanged. This is an intermediate gate, not a
new complete clean release claim.

Read-only follow-up inspection identified the next custody connection explicitly:
`beta_controller_benchmark` and `policy_evaluation` currently admit checkpoint
eligibility but their public execution functions do not require development/
held-out stage authority. Before any registered study opening, those routes must
be closed or connected to the fixed admitted stage runner, including direct
`execute_policy_episode` and `execute_beta_controller_benchmark_cell` calls. No
protected reset was used to establish this finding. Remaining work also includes
all-32/all-15 registry binding, latency/scorecards, source-bound external stage
authority and detached campaign reconstruction. Training authorization is absent.

## 2026-09-11 — Registered evaluation route closure

Commit `dd6f18b` completed the standalone synthetic arm/pair milestone and leaves
the overall study goal active. The next hypothesis is that historical public
beta evaluation entry points can be closed before any reset without changing
V1, unregistered fixture behavior, checkpoint loading used by training custody,
or historical evidence verification. A shared rejection-only boundary now refuses
registered task/checkpoint declarations and the frozen training, development and
held-out seed domains in historical beta evaluation routes. It does not grant
stage permission, add a bypass flag, or authorize any protected execution. Both
paired front doors and direct policy/heuristic/cell execution paths are covered;
the admitted study runner will use fixed private arm execution instead. This
change is being tested with forbidden construction/reset/load sentinels only.

All **73** new rejection tests passed in 0.19 seconds. They cover all three
families and five optimizer identifiers, each protected range boundary, every
public execution route, both legacy loaders/readmission paths, fixture-config
substitution for a registered checkpoint, beta-environment relabelling, ambiguous
seed types and empty seed batches. No plant constructor/reset or policy/metadata
loader is permitted in these tests; registered configurations are inspected only
as source-defined values. The combined route-closure, historical benchmark,
policy-integration and CLI regression passed **119 tests in 7.74 seconds**, with
successful exit and existing native HiGHS cleanup diagnostics. Strict mypy passed
all three changed source modules. The registered training loader remains separate
and unchanged. No registered run or protected evaluation was executed.

## 2026-09-11 — Complete study latency evidence build

The previous goal turn made progress through commits `dd6f18b` and `0e3a27d`:
standalone/pair evidence and historical evaluation route closure, with 204 and
119 passing targeted regression cases. Current HEAD is clean. The next hypothesis
is that explicit inference/warm-up/failure and end-to-end timing can be added to
the fixed private arm runners without changing their deterministic scientific
outcomes. The study build document now freezes timing boundaries, all-call
nearest-rank statistics, warm-up input/count, host requirements and source/authority
separation. Existing timing excludes failed generic controller calls and has no
separate inference or complete warm-up record, so it cannot support the registered
p99 gate alone. No protected environment or trained policy is opened for this build.

The first numerical/call-custody pass reached 45 successful cases, then the
fresh-child host check rejected missing CPU identity. The instrumented four-arm
fixture likewise rejected at host admission after confirming identical timed/
untimed PPO deterministic evidence. A read-only probe established the cause:
sandboxed `sysctl` omitted the CPU brand/core count, while a permitted hardware
read reported Apple M4 Pro and 12 cores. The hardware requirement is retained;
fixtures will rerun with access to that metadata. No registered arm, stage or
study seed was involved. Initial strict typing found one redundant cast, which
was removed; complete typing will be rerun with the fixture gate.

With permitted hardware reads, the timing arithmetic, host and four-arm physical
integration suites passed **71 tests in 103.12 seconds**. The fixture performs
1,000 warm-ups, then actual timed policy calls; its original untimed policy has
exactly the same deterministic physical/economic digest. Detached checks bind
every attempted decision, inference input, physical minute and retained solver
stage, and reject rehashed timing/host/reference substitutions and identical-file
replacement. A separate **4-test** extension passed in 58.10 seconds: real H/PPO
pre-movement failures retain attempted-call durations (and the failed policy
inference), and synthetic-clock arithmetic accepts exactly 100,000,000 ns while
rejecting 100,000,001 ns. Fresh-child configuration verifies Torch 1/1 without a
plant reset or policy training. Failed solver-stage inclusion cases are added to
the upcoming regression. Strict typing passed all three changed source modules.

The complete latency/evaluation/legacy-verifier regression passed **239 tests in
239.80 seconds**, with successful exit and existing native HiGHS cleanup text.
It comprises 77 new timing/host/instrumented cases, 119 route/evaluation/CLI cases
and 43 historical verifier cases. Failed solver stages are retained for both M24
and E-beta. The source contract digest is
`71678ee3f3578c6fc2a8fef115790498bbba64584029a5acfc6c8c3d3753d558`.
Black, compilation, strict mypy for all three changed source modules, Git
whitespace and dependency-lock checks passed. No registered checkpoint, training
campaign, protected reset or production authorization was opened. The optional
private-runner instrumentation preserves legacy serialized arm fields and leaves
V1 and the product interface unchanged. Production stage authority, full
operational scorecards, final clean release and explicit later authorization
remain required; the unchanged overall goal is not complete.

### Operational scorecards and training diagnostics — outcome-blind implementation

Hypothesis: complete physical witnesses can support the mandatory operational
scorecards without filtering failed arms through economic pair eligibility or
treating correlated cycles as independent observations. The separate operational
representation is fixed in `CONSTRAINED_RL_STUDY_ANALYSIS_BUILD.md` before any
registered training, development opening or held-out opening.

Implemented complete episode/cycle projections from the independently verified
standalone arm: original physical/economic maps, all additive components, returns
and deficits, event/exposure rates, exact outage schedules and censored capture
recovery. An all-32 arithmetic summary retains every missing slot, per-resource
per-cycle Wilson denominators, all observed deficit identities, observed-prefix
versus full-episode metric distributions, and episode-clustered outage/market
strata. The detached synthetic report reopens every referenced physical arm and
checks actor/task/checkpoint/source consistency and file stability before closing.
It cannot grant a registered registry, checkpoint or result claim.

Training diagnostics retain the exact 3-family x 5-seed slots, separate completion
and checkpoint admission, and every admitted endpoint rollout. Cost exposures,
completed-episode means, multiplier/proportional/integral trajectories, saturation,
optimizer diagnostics and failures remain explicit. The chain parser checks each
rollout but does not replace full artifact/state/custody admission. Production
stage custody must provide those admitted inputs; this arithmetic layer says so.

Initial arithmetic fixtures passed **51 tests in 1.78 seconds**. Strict typing
found nullable/mixed-dictionary annotations in the new implementation; those
were corrected, and all four changed source modules passed strict mypy. The
first new physical/arithmetic plus existing arm/pair regression passed **156
tests in 336.13 seconds**. It includes real H/M24/E-beta/PPO witnesses, failed H
before movement, adversarial rehashed reports, source tampering, exact-slot scope
checks before manifest access, and identical-byte file replacement detection.

Review then separated complete-episode safety from observed-prefix safety and
changed unobserved cycle values from empty-sum zero to null. An initialized,
independently verified zero economic prefix remains observed; a preflight failure
has no economic or safety observation. Two explicit cases cover that distinction;
the expanded arithmetic suite passed **53 tests in 1.70 seconds**. Final regression
and typing are pending after these changes. The operational contract digest is
`97c67d7bbae8fed259ece7ca0bed5710078fcb26a49b77e66c14bbae99f64959`.

No protected seed, registered checkpoint, training campaign or sealed training
log was opened. V1 and the product interface are unchanged. Both frozen upstream
protocols are unchanged. Stage authority/registry/execution, full checkpoint
binding, detached development/held-out closure, final clean release and explicit
later authorization remain required for the unchanged overall goal.

The final operational/arithmetic/existing arm-pair regression passed **158 tests
in 332.90 seconds** with successful process exit. A final mypy check found a
local-name collision between a flattened metric dictionary and a later resource
list; the local was renamed after the provenance-sensitive tests completed,
without changing report semantics. Strict mypy now passes all four changed source
modules. The 53 reset-free arithmetic cases pass again in 1.83 seconds; Black
checks all six changed Python files, compilation and whitespace checks pass, and
`uv.lock` is unchanged. The registered training authorization remains absent.
These are intermediate implementation gates, not a complete clean release or a
registered controller result.

### Fixed study registry and complete aggregate calculation

Hypothesis: the preregistered study can be composed from the independently
verified arm, timing and training layers while retaining every one of its 576
stage slots and preserving the distinct primary and matched-secondary estimands.
The build contract now fixes episode-major H/M24/E-beta/15-policy order and
deterministic development/held-out sibling roots; the read-only registry does
not open a stage or alter the training root. Future stage storage/time budgets
remain distinct from the training-only 5 GiB floor.

The audit identified a real integration risk: subtracting two E-beta-relative
vectors would suppress a valid policy/scalar comparison whenever E-beta failed.
The existing physical comparison kernel now supports the explicit matched-scalar
comparator with its own labels and gate; it cannot emit E-beta-labelled
sensitivity for that secondary pair. Historical default pair semantics remain
unchanged. Complete primary and secondary vectors separately feed the frozen
equal-policy family estimator and hierarchical bootstrap.

Implemented aggregate composition for all 18 actors, 15 final policy identities,
all 32 episode slots, original operational scorecards, every policy/family
coefficient setting and sign reversal, fixed development selection, frozen
held-out choices, five opening conditions and the ten held-out clauses. Missing
or unadmitted policies have no value vector, and all skipped slots retain their
training custody reason. Complete source verification, registered hardware and
publication facts remain explicit inputs from the future detached stage layer;
this calculation has a null result claim and never authorizes execution.

The timing projection now retains all raw channel samples plus the independently
verified arm-envelope and host digests. Campaign latency pools actual samples
across the 32 slots, including failed calls, rather than averaging episode
percentiles. Raw samples and projected identities remain reconstructed from the
sealed timing and physical files, with new rehashed-substitution cases.

The initial **46 reset-free campaign/registry tests passed in 26.99 seconds**,
including complete synthetic composition, independent secondary pairs, frozen
choices, all ten negative clauses, exact identities and unequal-size latency
pools. These are explicitly arithmetic fixtures, not physical controller results.
Five additional cases cover complete training failure/missingness and malformed
held-out choices. Initial mypy errors were heterogeneous JSON container
annotations, corrected without altering calculations; strict typing now passes
all five changed source modules. The combined physical/latency/campaign regression
is pending. No registered checkpoint, study reset, sealed log or authorization
was opened, and the product interface is unchanged.

The combined campaign, standalone arm/pair and latency regression passed **210
tests in 345.49 seconds**, with successful process exit and permitted CPU metadata
reads for the existing timing fixtures. The three added timing cases reject
substituted raw samples (even with recomputed percentiles), projected host and
projected arm digests. The unchanged primary pair behavior and the independent
matched-scalar gate both pass their regressions.

Final review made the aggregate's already-required source prerequisites explicit:
held-out success requires frozen identities; its five PI inputs must have the
exact distinct optimizer identities, and its champion must be their unchanged
member. The development identity condition also requires registered hardware.
These arithmetic-only changes and the added identity case pass the final **52
campaign tests in 33.44 seconds**. Formatting of all seven changed Python files,
compilation, whitespace and unchanged-lock checks pass. Final strict typing
passes all five changed source modules. Production stage execution, external authority,
registered checkpoint binding and full detached stage reconstruction remain the
next build work. The overall readiness goal remains active; no study result or
training authorization is claimed by this milestone.

### Lossless study storage and production training-input binding

The previous registry/analysis milestone is committed at `b7476c7`. Hypothesis:
the existing complete training verifier can supply all-15 study inputs without
trusting a cached verdict or weakening its training host contract. This requires
reopening admitted checkpoint metadata for the full rollout scorecard and
retaining process failures, checkpoint rejections and unstarted slots explicitly.
No stage permission is conferred by that projection.

A read-only filesystem check found approximately 28 GiB available on this host.
The training-only 5 GiB floor is not an evaluation-evidence capacity estimate.
Added a bounded, exclusive, lossless canonical-JSON gzip writer/reader as stage
storage infrastructure. Its **42 byte-level tests passed in 0.35 seconds**,
including independently encoded members, exact inclusive limits, multi-megabyte
expansion, invalid JSON, corruption, extra members, retained failed publications,
indirect paths, concurrent growth and identical-byte replacement. No protected
episode or checkpoint was opened, and no study-capacity claim follows from this
storage implementation.

The training verifier now exposes its complete reconstructed report while the
existing compact CLI verdict delegates to the same fixed production path.
The new study bridge reopens admitted metadata, retains all 15 slots and checks
the whole evidence tree and original source/runtime/host after projection.
Strict typing passes all three changed source modules. Reset-free integration
and adversarial bridge fixtures are the next gate; no authorization or sealed
registered evidence is available or used.

The first bridge run found an implementation error: a byte equality check used
the JSON-value comparison helper, which cannot serialize bytes. Three cases
failed at that boundary; it now compares the raw canonical bytes directly.
The resulting **32 bridge tests passed in 30.26 seconds**. They exercise complete
receipt/report reconstruction with explicit synthetic provenance/model/parser
dependencies, all-15 admission ordering, full 128-row hash chains, process and
checkpoint failures, storage-aborted missing slots, wrong entry-point rejection,
graph substitution and original training-thread identity. These are not real
registered checkpoint admissions. Review corrected a negative test to use the
actual scalar family identifier, ensuring it tests a valid cross-family ledger
substitution rather than an unknown identifier. Archive descriptor rereads now
enforce the same stored-byte cap as initial reads, including concurrent growth.
The combined regression follows these corrections.

The combined archive/bridge/training-custody/operational/campaign regression
passed **291 tests in 120.40 seconds**, with successful process exit. Strict
typing passes the three implementation modules. The fixed stage registry is now
aligned with the selected lossless storage format: its physical and timing
destinations end in `.json.gz`, and it binds the archive schema. This changes no
historical synthetic evidence path and opens no stage. Registry-specific checks
and the final static/whitespace/lock gate follow. The training and beta protocols,
V1 and product interface are unchanged; stage authority/capacity/execution and
full detached stage closure remain required for the active overall goal.

The aligned registry passed **16 focused cases in 0.76 seconds**, including two
new rehashed archive/schema-path substitutions. Final review found that the
metadata-only tree snapshot was too restrictive for failed checkpoint artifacts:
a retained forbidden node would reject the entire bridge even when the complete
training verifier correctly recorded that arm as failed. The snapshot now records
such nodes without following or opening them, leaving location/status admission
to the complete verifier. An added dangling-artifact fixture preserves the three
failed optimizer slots and the other twelve admitted policies. The final bridge
suite passed **33 tests in 33.32 seconds**; original host and identical-replacement
checks remain intact.

Strict typing passes all four changed source modules (with the final bridge
checked again), Black checks seven Python files, and compilation/whitespace/lock
checks pass. Both frozen protocol SHA-256 values are unchanged, and the tracked
training authorization remains absent. No real policy was loaded, protected seed
reset, sealed registered log inspected or experiment started. The latest complete
clean repository gate remains `ad6ad1e`; this milestone is an intermediate
implementation gate. Production stage authorization, storage capacity, single-arm
execution and detached development/held-out reconstruction remain next.

### Independent single-arm study execution

The previous goal turn made verified progress at clean `a0d246c`: the complete
training-input bridge, compressed archive storage, final 33-case bridge and
16-case registry checks. Hypothesis: the existing private physics/timing runners
can execute one environment and one reset per actor without the old helper's
four simultaneous resets, preserving deterministic evidence exactly. This is
necessary to retain baselines once and give all 576 stage slots separate custody.
The build contract now states this execution boundary before implementation.
Only non-registered tasks and fixture seeds may exercise the new public fixture
entry point; production stage authority remains unopened.

Implemented a one-environment execution engine using the unchanged private
physics/timing runners, and a fixture-only exclusive producer plus complete
compressed physical/timing verifier. The public fixture producer also checks
checkpoint-declared scope before environment construction, closing registered
checkpoint substitution through an otherwise fixture-labelled task. Initial
strict typing identified two imports accessed through a module that did not
export them; direct controller imports fixed this without behavioral changes.
All three initial implementation modules pass strict typing.

The first single-arm/compressed integration gate passed **29 tests in 187.74
seconds**, with permitted CPU metadata reads and successful exit. Every arm's
physical/economic digest matches the unchanged four-arm reference. Each actor
has one environment and one reset; PPO has exactly 1,000 pre-reset warm-ups.
Reconstruction, retained warm-up/controller failures, protected task/seed scope,
rehashed archive references and identical-byte replacement are covered. Existing
native HiGHS cleanup text remains. No trained policy or protected seed was used.

The next authority boundary now has fixed external input/copy and compressed
report destinations. Added exact budget arithmetic for all 576 slots, a single
PPO profile shared across the 15 policies, both stages' initial storage reserve,
logs and control-document allowances. Budget values must be explicit and frozen
before development; the arithmetic does not measure capacity or authorize a
stage. Evidence-limit types reside in the pure budget module so reading a
registry does not import the physical execution layer. The initial combined
budget/campaign arithmetic checks completed before adding the external authority
input itself to the conservative five-document storage allowance. Final combined
physical, historical latency, budget and registry regression follows.

The initial budget/campaign arithmetic gate passed **97 tests in 33.87 seconds**;
after including the external authority input in the storage bound, all **43
budget cases passed in 0.45 seconds**. Added three rejection cases for invalid
actor/checkpoint combinations before destination consumption or execution.

The final combined new single-arm, historical latency, archive, budget and
campaign regression passed **251 tests in 331.79 seconds**, with successful exit
and the existing native HiGHS cleanup text. All five changed source modules pass
strict typing; Black checks seven Python files, and compilation/whitespace/lock
checks pass. Both upstream protocol hashes remain unchanged and the tracked
registered training authorization remains absent. The single-arm fixture uses
only 9,000,000-series seeds and an untrained fixed policy; its 24-hour/hourly
geometry is not a capacity measurement for the 168-hour/15-minute study. No
registered checkpoint, protected reset or long experiment was opened.

The next production work is the external stage-authority admission, durable
opening and per-cell claims, fixed sequential child orchestration, and complete
detached reconstruction through the trained-checkpoint bridge. The private
single-arm engine is ready for that enclosure; it does not itself authorize a
registered reset. Stage capacity, the final complete clean release gate and
later explicit authorization remain required. V1 and the product interface are
unchanged, and the overall goal remains active.

The execution-contract digest is
`617689e17b528adc73addb4f67ac85f40c60cfd5b190bcbb3de6c780c9594371`.

### Connecting production study authority and custody

The previous turn made verified progress at clean `a445e56`: one-environment
execution, compressed witness reconstruction and explicit budgets, with 251
passing combined cases. Hypothesis: those components can be enclosed by fresh
training/development admission, external one-stage authorization, live-parent
launch permissions, a serial host lock and durable per-cell records without
weakening either seed custody or the training/evaluation host distinction.

The build contract now records that enclosure. Production data/control byte
allowances are mechanical planning bounds, not exact filesystem allocation
measurements. A production authority must retain at least 1 GiB additional
filesystem reserve; capacity review must include rounding and metadata overhead.
The private physics verifier body is shared after separate fixture/registered
scope admission. No registered authorization is being created and no protected
seed, trained checkpoint or experiment is being opened for this build.

The stage parent, child and detached CLI are now connected. Initial strict typing
found two implementation issues: the nullable failure variable was inferred as a
plain string, and a timestamp validator was imported indirectly. Both are fixed;
all 12 changed source modules pass strict typing. Review added input-tree and
live-parent recapture immediately before child execution/publication, nonsymlink
cell admission and detection of an identical-byte claim replacement before the
parent may advance. The stage budget now includes the detached process receipt
and both of its bounded logs. Its current execution-contract digest is
`16d683ffc9a9f30076acd86efccc7c2c7f8a9acdb5e89daa7d27c729276da6e8`.

The first reset-free integration run is still completing. Its full 576-slot
successful-fixture test has passed, including fresh aggregate reconstruction,
cached-verdict checks and rejection of rehashed report/choice/physical changes.
One earlier launch test failed because its fixture supplied `None` instead of a
different valid runtime; the corrected test changes a real runtime record's
revision. The focused corrections/additions pass **7 tests in 3.74 seconds**,
including complete production training-receipt/ledger reconstruction with explicit
synthetic model and host dependencies, all 128 rollout rows, closed CLI paths,
and bounded detached process receipt retention. These results confer no real
training or protected-episode admission. Full failure-retention and physical
wrapper regression remain underway; no capacity or release claim follows yet.

The initial stage/authority/launch/budget suite completed with **112 passing tests
and the one fixture-type failure in 665.60 seconds**. The corrected valid-runtime
case is included in the already passing seven-case follow-up. Every full-stage
case passed: all 576 sequential claims, 480 skipped failed-training slots plus 96
failed baseline processes, full aggregate reconstruction, all retained storage/
operator/integrity abort slots, forbidden layouts, consumed claims and report
tampering. These reset-free checks use explicit process/physical substitutes and
therefore do not assert real registered physics or capacity. The physical wrapper
and historical arm regression now follow with source held unchanged during runs.

The final physical-wrapper, standalone-arm and historical-admission regression
passed **196 tests in 498.39 seconds**, with successful process exit and the existing
native HiGHS cleanup text. All four new wrappers preserve the corresponding
fixture physical/economic digest; detached reconstruction makes no plant/model
execution call. Rehashed claim/source/actor/seed/checkpoint substitutions are
rejected before physical reconstruction, and physical/timing/host changes fail
their complete witness checks. The new execution fixtures use only 9,000,000-series
seeds and an untrained fixed policy; they are not registered evaluation or the
168-hour/15-minute storage-capacity measurement.

All 76 stage cases and 43 budget cases are covered by the initial 112-pass run
and the seven-case correction/addition run; the initial test-only failure remains
recorded above. Strict typing passes the 12 changed source modules, Black checks
18 Python files, and compilation passes. The final whitespace, unchanged-lock,
upstream protocol hashes, absent training authority and CLI-help checks pass.
This closes the production stage implementation milestone, not the overall
training-readiness goal. Measured stage capacity, concrete frozen budgets, the
complete clean repository release gate and explicit later authorization remain.
The latest complete clean repository gate is still `ad6ad1e`; no registered
training, protected evaluation, checkpoint selection or deployment has occurred.

### Full-geometry study capacity fixture

The previous goal turn made verified progress at clean `c0d1dbc`: production
stage authorization, isolated sequential children and complete detached custody.
Hypothesis: the existing producer/verifier can provide honest memory, storage and
execution observations at 168 hours and 15-minute cadence before any registered
policy is trained. The earlier 24-hour/hourly fixtures cannot establish that:
M24's lookahead grows from 24 to 96 intervals, and a retained E-beta prefix is not
a completed week. A read-only check found 29,659,987,968 bytes available; this is
available space at that moment, not a reserved study allocation.

Frozen `CONSTRAINED_RL_STUDY_CAPACITY_FIXTURE.md` and its thin one-shot tool retain
all four actors on fixture seed 9,100,001. The entire physical task is unchanged;
PPO uses an untrained zero-action checkpoint and no optimizer runs. Limits are
600 seconds per child, 1 MiB per log, 1 GiB/4 GiB stored/decoded physical evidence,
8 MiB/64 MiB timing, an 8 GiB opening floor and a sampled 12 GiB child RSS stop.
The source revision determines the sole external destination. These are probe
bounds, not frozen production stage budgets or trained-inference measurements.

All **25 reset-free launcher/configuration cases passed in 2.42 seconds**. They
cover the full task identity, rehashed plan/seed/budget/host substitution, consumed
roots, retained process failures and incomplete prefixes, no continuation after
interruption, and an actual disposable child which durably records its lowered
RSS cap and exits with code 73 without allocating a large buffer. Initial strict
typing found a NumPy sequence annotation mismatch, an inferred heterogeneous row
list and an untyped source revision used to construct a Path; these are corrected
with a literal fixed action, explicit row type and source-revision validation.
No capacity fixture or registered experiment has run yet. The frozen protocol,
tool and fixture checks will be committed before the bounded capacity run.

The final probe gate passes **26 tests in 0.88 seconds**, including the added
one-destination/full-object-ID validation. Strict typing, Black, compilation,
CLI help, whitespace and unchanged-lock checks pass. No real model or plant was
executed by this gate; the memory-limit check used a separate disposable process
with a one-byte test cap. The next action is the fixed full-geometry fixture
measurement at the clean committed revision, within the recorded bounds.

The fixture protocol/launcher was committed at clean `17331fe`. An initial command
mistyped the destination with 13 revision digits. The fixed-path guard rejected
it before root creation or child launch; terminal exit and root absence were
verified before correcting the path. The one four-arm fixture then ran at
`/private/tmp/plantanagent-study-capacity-17331fe5b89a/` and closed normally.
Its report-domain SHA is
`4c10df0a87beb258d7dc6625229d259641158e63020dc39361e7b07ca796c56a`;
raw SHA is `2a9fc73241ad4b4568938558b86bbbe182dea95fa611f27ff3ed342b96e81986`.

H and the untrained zero-action proxy completed 672 decisions / 10,080 minutes,
with 1.486/1.526 GiB peak RSS and 11,409,255/11,628,605 stored physical bytes.
Both were cycle-infeasible; this is no controller-performance success. M24 stopped
after 36 decisions at its explicit unsupported degraded/offline DAC boundary.
E-beta stopped after 52 decisions because its selected plan failed independent
hard battery-return validation. All four child exits were zero because the
producer correctly retained and verified those controller-failure prefixes; no
timeout/log/RSS cap or fallback occurred. Prefixes do not establish full-week
capacity, so the complete study capacity gate remains open. Compact evidence and
the full failure account are retained in `results/v2_study_capacity_17331fe.*`.

The next read-only diagnosis inspects E-beta's retained failing fixture boundary
and its existing model/plan validation contracts. Hypothesis: the rejection may
be a solver/validator numerical-boundary mismatch rather than an execution
resource limit. No controller, cycle tolerance or solver setting will be changed
to pass this capacity fixture, and no actor or protected seed is being rerun.

### E-beta terminal-return numerical diagnosis

At documentation-only `9f16e3f`, one saved-public-boundary model reconstruction
used the unchanged implementation and two existing solver stages; no environment
or controller was constructed. Both stages were optimal. The economic battery
terminal value was two ULPs (`1.8189894035458565e-12` kWh) below its hard minimum,
reproducing the strict plan rejection. The original failed candidate was not
archived; this is a retained reproduction, not its exact retrospective residual.
The external diagnosis report raw SHA is
`4af1b7e0dfd06af838b8aae39774295a2bf13bd09f3e477f748510296b1f8d2b`.

The diagnosis document now records a separate pre-training implementation scope:
duplicate the four existing singleton return rows as explicit terminal variable
lower bounds, intersect stronger existing bounds, retain all rows and reuse the
unchanged economic canonicalizer/residual audits. The beta model version changes
to bind this representation. No task/solver tolerance, objective or feasible
region changes. Fixed saved-vector tests and existing regressions will validate
the repair; no capacity actor, plant trajectory or protected seed is rerun.

The repair is implemented in the beta model only: four explicit terminal lower
bounds intersect their original physical bounds, and the beta model identity is
now `cyclic-return-economic-mpc-v2`. The shared solver and strict plan validator
are unchanged. A 47,887-byte public-input/fixed-vector fixture retains the
reproduction and its source-report digest. All **five new reset-free cases pass
in 0.54 seconds**: correction of the original deficit, interior-value preservation,
material-infeasibility rejection, all-four-bound equivalence and preservation of
a stronger physical reserve.

The combined beta/alpha MPC, benchmark, detached verifier and standalone-arm gate
passes **219 tests in 637.47 seconds**, with successful process exit and the
existing native HiGHS cleanup text. Strict typing passes both source modules;
Black checks all three changed Python files. All 15 registered task identities,
both upstream protocol hashes, dependency lock, simulator and frontend are
unchanged. Training authorization is still absent. The original negative capacity
result is not superseded; no full-week actor or protected seed was rerun. A new
separately scoped capacity check, concrete stage budgets and complete clean
repository release gate remain before the training-authorization packet.

### Separately scoped capacity check after the numerical repair

The numerical repair is committed at `9c76d9a`. Before execution, a separate
capacity scope now fixes the same fixture seed, all four actors and all original
limits. Its distinct schema and both raw protocol hashes are bound by the thin
launcher. Hypothesis: exposing equivalent terminal bounds removes the observed
numerical rejection and either permits a full E-beta capacity observation or
reveals another retained failure. M24's known modeling limitation is unchanged;
its actor will not be removed or replaced. The original negative report remains
authoritative for its own attempt. This scope opens no registered experiment,
policy optimization or protected seed and does not authorize training.

All **27 reset-free launcher checks pass in 0.76 seconds**, including explicit
binding of the original and repair scopes. Strict typing and Black pass; the
original capacity protocol, task/learning protocols and dependency lock remain
unchanged. The new source/protocol will be committed before the bounded run.

The follow-up fixture ran once at clean `cfa16a6` and closed normally. Its report
raw SHA is `819e4b499560679c7726abbe805a69511ed10b724bd015d374419802675807ef`;
domain SHA is `bb83ac37a12146d5834ad2e275a6d5188fe6425667ed6a315e4d9972e012c039`.
All four child exits were zero with no timeout/log/RSS cap or fallback. H and the
untrained fixed proxy completed 672 decisions/10,080 minutes and were again
cycle-infeasible; the proxy also retained forced-stop safety failure. M24 stopped
after 36 decisions at the same unsupported DAC boundary. E-beta passed decision
52 but stopped after 54 decisions when the primary service model was infeasible.
The numerical repair therefore removes the demonstrated roundoff rejection but
does not establish controller or capacity readiness. Full original and follow-up
failures remain visible in the compact `results/v2_study_capacity_cfa16a6.*` evidence.

A solver-free reconstruction of E-beta's saved public boundary shows 5012.459604808251
kWh in the battery against a 5013.628497150464 kWh required minimum and zero issued
solar energy across the remaining 42 intervals. Even lossless storage of all
remaining forecast solar leaves a 1.16889234221253 kWh deficit. Its preceding plan
had placed the terminal battery exactly at the minimum; actual minute operation
fell short. This explains infeasibility from that boundary, not infeasibility of
the opening task. No new plant reset, optimizer, solver call or control repair
was used for this algebraic diagnosis. Its domain SHA is
`f219e66c576240fa99e78845688259ad182fd2225d9f7a7f9e2574603322f5ea`.

The remaining primary-comparator gap is preservation of feasibility under forecast
and model error before the production window closes. It needs an explicit future
controller-design scope, not another tolerance patch or a silent capacity retry.
No stage budget, complete clean release claim or registered training authority
follows. All 15 registered tasks, protected seeds, V1 and the product interface
remain unchanged; the overall training-readiness goal is still active.

### Complete implementation gate and recovered capacity report

Hypothesis: the missing historical capacity report can be restored without a
new experiment if existing evidence reproduces its exact frozen bytes. Scoped
inspection found the original temporary capacity root present but containing
zero regular files; the cause of removal is unknown. A historical tool response
retained the empirical report fields but projected only part of the static cell
contract. The first reconstructed candidate failed both frozen hashes and was
not admitted. Restoring the six omitted static fields through the unchanged
deterministic `_cell_contract()` and canonical serialization produced the exact
17,380-byte original: raw SHA
`3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa`,
domain SHA `78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e`.
No empirical values were substituted, no sealed outcome log was inspected and
no environment/reset/model/optimizer was constructed for recovery.

The exact report and a recovery record were published exclusively outside Git
under `/Users/filippogombac/Developer/PlantAnAgentResearchArtifacts/capacity-4f505b8/`.
Production regular-file reading, both hashes, the complete static contract and
campaign-projection arithmetic pass. Fresh host identity exactly matches the
recorded capacity host. Original checkpoints and logs remain absent, so there
is no claim of fresh detached verification of that complete historical artifact
set. The report-bound training admission prerequisite is recoverable; no new
capacity run is needed merely to replace the missing report.

The complete Python gate ran with source held clean at
`12936c7627940f2ede04405bbbba28603bb8b10f` and finished with **3,002 passing
tests, one existing FastAPI/Starlette deprecation warning, in 2,037.91 seconds**.
Process exit was zero; the existing native HiGHS cleanup text was retained.
Strict mypy passes all 36 source modules changed since `ad6ad1e`; Black checks
all 57 changed Python files. Compilation, CLI help/imports, whitespace, all 15
registered task identities, unchanged lock and frontend production build/lint
pass. The frontend retains its existing large-bundle warning. No repository
file changed until every running gate process had closed.

`TRAINING_READINESS_REVIEW.md` now separates that completed software gate from
unresolved scientific and execution readiness. The exact recovered report,
proposed new persistent training root and frozen 15-policy time/log/storage
limits are explicit. One unfrozen stage-budget scenario reconstructs through
the production budget parser: 22.40 GiB per stage, 46.80 GiB required at
development opening. The selected volume had 25.17 GiB free at 05:38 UTC on
September 11, leaving 21.64 GiB to obtain before further training allocation.
Neither a storage reservation nor a stage capacity claim is inferred. Full-week
MPC measurements remain incomplete, and all prior negative attempts are retained.

The review recommends settling E-beta's forecast/model-error reserve behavior
before final source freeze. Preserving the baseline is a valid alternative with
the all-32 eligibility risk accepted explicitly; fixture failure does not prove
failure on unopened development seeds. No strategy change, protocol amendment,
capacity retry, source-freeze approval or production authority is created here.
The compact review evidence raw SHA is
`6a846d800613434126b654f5e36d93965492494c11bfabb42092cd9be437166f`.
This documentation commit records the gate at `12936c7`, not a new full-suite
run at its own revision. Final comparator/capacity/storage decisions and explicit
training approval remain; the overall goal is active and this turn made progress.

The final documentation check passes whitespace, review-link existence, compact
evidence raw digest and exact budget reconstruction. Both frozen upstream protocol
hashes and `uv.lock` remain unchanged. The proposed training root and tracked
training authorization are still absent. No second full-suite run was warranted
for this documentation-only record.

### Authorized E-beta uncertainty repair

The user explicitly authorized implementation of forecast/model discrepancy
diagnosis, conservative scheduling and early reserves before PPO training. The
prospective scope in `BETA_MPC_UNCERTAINTY_REPAIR.md` fixes a public-band-p10 policy,
1% capacity buffers and a two-hour daylight lead, with spendable observed buffers
after the restoration deadline. These values are declared before the revised
controller is run; no parameter search is authorized. Existing mean-policy
behavior and negative evidence remain reproducible. Focused tests and three
bounded fixture seeds are specified; no registered execution is opened.

The retained minute ledger corrects the earlier weather-error interpretation:
available PV was 24.7389 kWh versus 20.7066 kWh forecast, but 7.9821 kWh was
curtailed. The held charging ceiling discarded surplus early, then falling
sunlight starved charging late. Accepted PV is not available PV. Applied battery
flows reconcile within 2.9e-12 kWh. The compact 16-boundary fixture preserves
the original archive digest and proves this decomposition without a reset.
Before running the revised controller, the scope was extended to constrain
constant solar allocations by the minimum public physics-sampled daylight
envelope within each interval, scaled by the current band-p10 scenario.

The implementation now retains mean-policy defaults for explicit historical
construction and selects `public-p10-early-reserves-v1` in the product and study
factory. Public issue hashes, separate p10 projection identity, fixed buffers,
daylight deadline, spendable observed surpluses and constant-power constraints
are reconstructed by the detached verifier. Original daily return rows and
task tolerances remain unchanged; no fallback or slack was added.

The initial 25-case mean-policy/model/projection gate passed. A new buffer test
initially used exact equality for a floating-point difference; 15 other cases
passed, and its assertion now uses a 1e-12 arithmetic tolerance without altering
controller or settlement tolerances. The broader gate had 115 passes, one stale
mean-projection test-helper failure, and 32 failures/errors due to sandbox-denied
CPU identity. The helper now reconstructs the declared policy. The 45-case
targeted rerun with real CPU identity passes in 331.62 seconds, including every
affected case, complete physical/latency reconstruction and tamper rejection.
Strict mypy passes all ten changed source/tool modules. Product API tests,
compilation, unchanged simulator/frontend/lock and upstream protocol checks pass.
No full-week revised controller or registered policy has run at this point.

The final 17-case uncertainty/schema/roundoff regression passes in 3.24 seconds.
The revised descriptor explicitly identifies band-p10 projection and the sampled
constant-power assumption. The bounded fixture launcher binds this scope, clean
source, fixed three-seed order and exclusive external destinations, and reopens
the physical/timing archives in the parent. Final typing and whitespace checks
pass. The implementation is committed before the three authorized full-week
fixtures; their outcomes remain unknown at this checkpoint.

The three authorized fixtures ran once at clean `c5b049b`. All processes exited
zero and their complete retained prefixes passed detached reconstruction, with
no timeout/log/memory cap or fallback. They stopped at 72/144/25 decisions;
seed 9,100,002 also failed its first CH4 settlement. No complete-week or positive
controller result follows. Report SHA is
`a2b4bbfd7b53c2b934ea2f4057bb7e789399295603a7954af53279c0eabb3938`.
Compact results and exact delivery comparisons are retained in
`results/v2_beta_uncertainty_c5b049b.json`.

The solar/reserve repair passed the original sunset boundary, but the old model
can withhold methane deliveries to protect predicted reserves. The physical
settler has no such choice: it always ships min(available, request). For example,
at decision 71 on the first seed the model expected 41.933 kg while actual shipment
was 240 kg. The next prospective scope, `BETA_MPC_DELIVERY_MODEL_REPAIR.md`, fixes
this independently demonstrable model mismatch with exact mandatory settlement,
preserving every failed attempt and all uncertainty-policy parameters. This is
part of the user-authorized model repair, not an outcome-based parameter search.

The exact-settlement implementation adds one binary per delivery to the revised
policy only. Existing methane balance and request bounds plus the disjunction
(full shipment or empty post-shipment tank) enforce the simulator's minimum rule
independently of objective weights. The legacy mean/M24/alpha formulation remains
unchanged. The policy identity advances to `public-p10-early-reserves-v2`; the
complete solution witness and detached model audit include the added variables.

The first focused model/roundoff gate passed 17 cases in 3.50 seconds; strict
mypy passed five changed modules. The broader model/solver/verifier/product gate
passed 92 cases and failed one previous assumption that the default product
fixture always yields an E-beta plan. Direct inspection at fixture 9,000,034
confirmed primary-stage infeasibility (HiGHS status 2), before any physical
advance. Public-boundary structural diagnostics also found infeasibility with
p10 without extra buffers, mean with buffers/constant-power rows, and mean
without either added row set, all retaining mandatory settlement. Those modified
model objects are diagnostic relaxations, not admissible controller plans or
performance comparisons; no protected seed was opened. This does not establish
infeasibility of the nonlinear physical plant.

The product keeps its fixed scenario and records the opening failure without
advancing or falling back. A separate low-load unit fixture tests projection of
a feasible real plan; the fixed product scenario has its own explicit failure
regression. The API now derives task horizon/cadence from its actual environment,
and its existing information line accurately describes the revised E-beta policy.
No frontend or visual styling changed. The final API/uncertainty gate completed
with 21 passes and one existing deprecation warning in 36.55 seconds; its exact
terminal result was recovered from this task's command record after compaction,
without duplicating the process. This commit precedes the second, distinct
three-seed fixture; no complete-week success or training readiness is claimed.

The second fixed fixture ran once at clean `a82f11c44e36`. All three seeds
stopped on the opening controller call with primary-stage infeasibility; each
child exited zero and its retained zero-step failure passed detached physical
and timing reconstruction. Child wall times were 4.90/4.03/4.03 seconds, with no
cap or fallback. Report-domain SHA is
`ae2c31f2044f41bfcfb7d0aeb54594cbc385dd9865c9bdc50a35178827748814`.
These are successfully retained failures, not successful controller executions,
safety-operation evidence or full-week capacity observations.

A saved-public-record necessary energy bound explains this result without any
new reset, solver call or future-weather inspection. Positive final CH4 forces
both nighttime shipments to be full. A partial 18:00 shipment would empty the
tank; at most 225 kg can be produced before sunset, insufficient for the 480 kg
night deliveries plus final stock. Hence 18:00 must also ship fully; noon must
ship at least min(initial CH4, 240). Original resource returns imply at least
916.17/960/960 kg production and 32.82/34.39/34.39 MWh process electricity under
the model. Issued mean PV is only 28.40/27.83/29.43 MWh, before buffers, firm-power
restrictions, startup energy or losses. Battery-return tolerance contributes
less than one micro-kWh. Thus solver infeasibility is consistent with a simple
public energy certificate; the former shipment-withholding relaxation hid it.
This is not proof that the actual weather/physical task or unopened seeds are
infeasible. `tools/beta_uncertainty_result.py` reproduces the compact artifact
from saved records, checking compressed and decoded archive digests; it invokes
no environment or optimizer. Its initial interactive inspection used an incorrect
list index on the boundary-ledger dictionary, raising KeyError without mutation;
the corrected decoder reads its explicit decision-boundaries field.

The compact result raw SHA is
`d7710664d243d49c3dd356584148c954a8981d00495de6184af2ec0a3bd8258a`.
The report and readiness/roadmap now state the negative result, including the
opening product E-beta failure. No fallback, task relaxation or reserve parameter
search is added. The repair implementation is distinct from readiness: future
work must define explicit supervisory behavior for an infeasible issued scenario
and investigate task/model feasibility before comparator freeze. Registered
training, development and held-out evaluation remain closed.

The compact report reconstructs byte-for-byte, and frozen custody/lock, unchanged
frontend/simulator and absent training-root/authorization checks pass. Strict
mypy initially found an unguarded optional forecast in the new result-only
summarizer; it now explicitly rejects a missing public forecast. This changes
neither the already recorded controller executions nor the compact artifact.

At clean `3b6ac67`, the complete affected-controller/product/evidence regression
passed **291 tests**, one existing Starlette warning, in **798.26 seconds**,
exit zero. It covers mean and p10 projection, physical/economic/beta models,
schema/controller/solver, roundoff, benchmark and detached verifier, standalone
arm/evidence/latency, and product API behavior. Source stayed fixed throughout.
All 13 changed source/tool modules pass strict mypy; all 16 changed Python files
pass Black and compilation. This is an affected-surface gate, not a new complete
repository/training release gate.

Final inspection found an independent model-identity edge case: with optional
reserve rows removed, power rows were omitted from the model hash. A regression
first failed with two different constraint sets sharing a digest. Serializing
each optional row set independently fixes it. Production revised models contain
both sets and legacy mean models contain neither, so their serialized bodies,
plans and recorded outcomes are unchanged. The subsequent schema/model/uncertainty/
roundoff gate passes **24 tests in 4.41 seconds**. No full-week fixture was retried
for this identity-only edge case.

Both completed attempts were also preserved byte-for-byte under the existing
external research artifact root. All 40 files match their original sizes and
SHA-256 values. Preservation record SHA is
`33035df329b87554c6ea7d78d52cf8eb0a8fe42a98aa6e161d956f802346a941`.
Original paths/source identities stay intact; the result summarizer reproduces
the same compact bytes from either original or preserved location. Final typing,
formatting, whitespace, compact reconstruction and preservation checks pass.
Training root and authorization remain absent; both frozen upstream protocols,
lock, simulator and frontend are unchanged. The repair is implemented with
negative fixture evidence; comparator reliability and final training admission
remain unresolved.

### Commission operating controller

The user authorized the simplified commission scope: physical correctness and
safety remain hard, extra conservative forecasting is deferred, daily inventory
restoration becomes a measured operating target, and an infeasible forecast must
not end the product run. `COMMISSION_OPERATING_SCOPE.md` freezes this prospective
implementation and its single bounded fixture rehearsal. Product MPC receives a
new identity; the strict E-beta and all registered study contracts stay unchanged.
The hypothesis is usable supervisory operation with explicit shortfall/recovery
and fallback records, not reliable fulfillment or an economic performance win.

Implementation adds `operational-economic-mpc-v1` outside the research controller
package. It composes the existing mean/public-price economic model with exact
mandatory delivery and held-power rows, a rolling 24-hour horizon, hard physical
reserve and commitment constraints, and fixed opening-endowment depletion costs.
It adds no p10 or early-reserve policy and no hard daily-return constraint. Typed
solver/plan rejection alone selects the existing heuristic for one interval; the
next boundary attempts planning again. Source/command hashes bind each public
observation, model, selected plan or fallback, targets and operating outlook.
Missing predictions stay null; stale data, reference ratcheting and unexpected
software exceptions stop rather than inventing commands.

The product API now offers H/MPC. Research E-beta and all `rl/v2` source remain
unchanged. Public frames carry the distinct operating policy and mode, forecast
shipment shortfalls and per-resource outlook; events and recording summaries
retain fallback/recovery outcomes. Planning, delivery/resources and Evidence use
these fields in the existing visual design. Old recordings keep their E-beta
identity, strict daily-return meaning and stopped outcomes. Saved MPC is preferred
when available. V1, simulator, CSS, 3D assets, lock and both frozen protocols are
unchanged; protected runs and training authorization remain unopened/absent.

The real opening probe on fixture 9,000,034 solves in Recovery mode, forecasts
roughly 960 kg delivery over 24 hours and depleted gas stocks while preserving
the opening battery. This is a prediction, not demonstrated restoration. One
interactive browser interval uses the actual API/physics and displays that
outlook, shipment predictions and opening references. Browser review confirms
the existing visual identity, new controller choices and separate research
availability. The original stopped E-beta recording remains loadable and labelled.

Validation: the initial API gate passed 14 tests in 34.81 seconds. New controller
tests initially had four assertion-path errors (metadata nesting and event
location), then one incorrect equality for the fixed opening battery variable;
these were test mistakes, corrected against the existing model/API schema.
The complete affected product gate subsequently passed **31 tests in 71.82
seconds**, with one existing Starlette warning. It covers real mean plans, exact
shipments, hard safety-floor and held-power rows, model/evidence identity, three
typed fallback reasons, no fabricated predictions, return to planning, stale
input rejection, immutable endowments, 97 guarded intervals across a daily
deficit, API boundaries, public recordings and historical daily-label semantics.
All five changed backend modules pass strict mypy; frontend TypeScript/build and
oxlint pass, retaining the existing Three.js bundle-size warning.

Before the rehearsal, review found that the old TimeoutError alarm could be
converted into a recoverable solver failure and swallowed by the new fallback.
The recording runner now uses a dedicated BaseException deadline and preserves
only the last completed public prefix outside the execution timer. A fixture
forces the deadline through the planner/session exception boundaries and verifies
the retained wall-limit failure with zero fallback decisions. It passes separately
in 2.61 seconds after the final prefix-access correction; it does not run a week
or create an MPC attempt outside its temporary test directory. Black passes for
all nine changed Python files, and whitespace checks pass. This is targeted
product validation, not a complete repository or training release gate. The
single full MPC rehearsal has not yet started at this implementation commit.

At clean `ccc86903cd7dbfcad0c2ebc2f93b7cc6c0f6ecaa`, the exclusive MPC rehearsal
ran once and completed all 168 hours, 672 decisions and 10,080 physics minutes in
309.71 seconds, exit zero. No source changed during execution. It delivered
5,938.0718 of 6,720 kg, with 781.9282 kg shortfall and 22/28 fully met commitments.
There were 667 Recovery decisions, five Fallback decisions and deficits at every
daily inventory review. Net simulated value was −18,929.62 EUR. Every fallback
was status-2 infeasibility: four second-stage economic solves and one primary
delivery solve. All returned to a valid plan on the next decision. Their specific
causes remain undiagnosed; no retry/tuning/search followed. HiGHS also printed
internal transformNewIntegerFeasibleSolution notices after the manifest; process
exit remained zero and the notices are not treated as additional solver success.

The exact public record is retained outside Git, with content SHA
`a42cf8729324454fe785b575db584b97ebc5f8fb9537e987bcf24d347b9021bb`
and raw file SHA `bc3a5e7381c9f0a45870b6e3d9200163d08856df256110f21cb6f6de8897114b`.
The compact result/report preserve all daily deficits, fallback stages, physical
metrics, final inventory depletion and economics. A result-only summarizer checks
recording integrity and manifest agreement and invokes no optimizer or reset.
Its initial result SHA is
`323e0b34d07a75ed366b2f2111d175e328c153596b6c76c48459f151de148999`.
The historical H/E-beta recordings are preserved. Browser playback now opens the
complete MPC recording; boundary 127 shows the actual heuristic fallback with
null predictions, and boundary 128 restores its economic plan with 82.4 kg
forecast shortfall. This meets the scoped continuity/visibility objective while
explicitly retaining unmet deliveries, inventory depletion and unresolved planner
failures. It does not establish sustainable operation, comparative savings,
plant readiness or admission of the registered research study.

Final saved-result checks reproduce the compact JSON byte-for-byte; its summarizer
passes strict mypy and Black. Both historical attempt manifests exactly match
their committed copies, and their public content digests still validate. Backend,
simulator, frontend, lock and frozen protocols remain byte-unchanged from the
rehearsal source revision. Browser review also confirms all seven daily Deficit
rows at the final boundary; the workspace is left on the complete MPC recording
at its existing eight-hour Overview cursor. The final overview and Planning panel
retain the original palette, typography, logo and 3D scene.

### PPO becomes a commission submission requirement

The user explicitly rejected deferring PPO training beyond the commission:
demonstrating the environment's RL capability is essential to its value.
`COMMISSION_SUBMISSION_SCOPE.md` makes executed training, gated evaluation and
inspectable learned-policy evidence required alongside the unified product.
This corrects the prior recommendation to keep PPO off the submission's critical
path. Useful learning and comparative superiority remain empirical questions;
a training smoke is not the requested deliverable.

Read-only review confirms that scalar/PI/I learners, custody and evaluation
software exist and the canonical campaign has not started. The 15-run matrix,
final-checkpoint-only rule and protected domains are unchanged. E-beta's failures
obstruct strict selection, not the ability to perform PPO updates; product MPC's
relaxed operating targets do not silently replace that research contract. The
plan closes the task/comparator decision and product inference/evidence interfaces
before source freeze, then capacity/storage, the release gate and concrete
execution authority. Held-out remains conditional on the frozen development gate.

The existing recorded training projection is 62.49 hours on its M4 Pro host;
evaluation is additional. A current read-only filesystem check finds 26.22 GiB
free versus the earlier unfrozen 46.80-GiB evaluation reservation scenario.
No files were deleted or provisioned. The proposed registered root and tracked
authorization remain absent. No optimizer update, plant reset, protected outcome
inspection or experiment occurred. The exact commission rubric and deadline
still need user input. An initial combined documentation patch was rejected for
a journal-context mismatch and made no changes; a later readiness-review patch
also required matching its actual line wrapping before application. The corrected
patches preserve the journal and clarify the historical E-beta product statement.
No software behavior changed; validation checks the diff
and frozen protocol/dependency hashes rather than rerunning unrelated tests.

### PPO launch readiness: prospective implementation scope (2026-09-11)

The user requests a correctly working system ready to train PPO. The bounded
[PPO readiness build](PPO_READINESS_BUILD.md) fixes the remaining public evidence
and recording interfaces before checkpoint provenance freezes the whole source.
Hypothesis: the existing admitted training ledgers and four-target policy adapter
can populate the unified product without exposing live campaign outcomes or
changing the learner, task or comparator. The future presentation policy is fixed
prospectively to PI-Lagrangian seed 101 on fixture 9,000,034; no replacement or
winner claim is allowed. Execution of that rehearsal is not part of this build.

A read-only compression probe may use already-saved capacity archives to assess
storage savings; it creates no new controller outcome. Research E-beta's opening
infeasibility and the product MPC's shortfalls/deficits remain retained. The final
checks must establish a new clean implementation gate. Registered seeds, campaign
root and canonical training execution authorization remain unopened/absent.

### PPO readiness interfaces, negative outcomes and Lexar storage (2026-09-11)

Implemented the prospective build without changing any frozen task, family,
optimizer seed, reward, comparator, selection gate, protocol bytes or lockfile.
The Learning panel uses the established product styles and separates training
from operating panels. It displays all 15 admitted/failed slots, rollout objective
sums, resource costs, multiplier and cost-value diagnostics. Future stage exports
reconstruct the complete production study and retain every actor's 32 slots,
execution/safety/cycle counts, ineligible pairs, failures and frozen selection.
Public file checksums are explicitly distinguished from the exporter performing
fresh source/custody admission. The HTTP runtime never reads training logs or
loads policies; import also succeeds with Torch and SB3 unavailable.

The offline producer fixes PI-Lagrangian optimizer seed 101 and fixture 9,000,034
before outcomes. It uses the existing normalized four-target adapter, minute
physics and guard; no explicit forward plan is invented. Policy/task mismatch
fails before reset. The producer requires closed production training admission,
retains one exclusive attempt, stopped prefixes and no fallback/replacement, and
quarantines a trajectory if post-run custody rechecking fails. Its execution is
not part of this readiness build. H/MPC remain the HTTP controller choices.
An integration defect was found and fixed: attempt-marker JSON files consumed
the recording catalog's eight-entry cap before filtering and could hide a newly
saved PPO recording. Filtering now precedes the cap.

Initial public-export tests: 12 passed, 3 failed. Two failures exposed an omitted
valid training campaign status (`failed`); that status is now accepted and retained.
The third was a test assumption: the existing synthetic process-failure fixture
fails a later slot, not the fixed first policy. The no-substitution test now
explicitly supplies a failed first slot. Initial typing found two nullable-map
annotation errors, subsequently repaired. The combined affected-surface gate
then passed **54 tests in 105.89 seconds**, with the existing Starlette/httpx
deprecation warning. The expanded recording tests passed **6 tests in 12.60
seconds**; expanded evaluation projections passed **9 tests in 26.32 seconds**.
Frontend production build/lint passed; the pre-existing 3D bundle-size warning is
retained. Browser inspection shows the Learning pending and held-out sealed
states in the unchanged visual identity. The stopped local preview servers were
restarted for this check. No synthetic curves were published as product evidence.

A read-only recompression probe authenticated and decoded all four saved
`cfa16a6` capacity archives. Gzip level 6 reduced the full H/fixed-PPO archive
sizes from 11,406,113/11,635,665 to 7,844,757/7,965,963 bytes; level 9 produced
7,271,437/7,386,834 bytes. Both decoded byte streams match exactly. The incomplete
M24/E-beta archives are retained too. No controller or optimizer ran, no saved
archive was rewritten, and the production archive format remains unchanged.

The user explicitly authorized use of `/Volumes/Lexar`. Read-only inspection
found ExFAT, 256 KiB allocation units and 350,689,951,744 free bytes before the
probe. The production custody/archive functions passed exclusive mkdir and
file/directory fsync, regular readback, duplicate rejection, identical-byte
replacement identity change and compressed archive roundtrip. Probe artifacts
remain in the task-owned `PlantAnAgentResearch/readiness-storage-probe-5874d4c9937d`
directory. No existing user data was changed. The compact
[storage evidence](results/v2_ppo_readiness_storage_20260911.json) retains all
measurements. New proposed root:
`/Volumes/Lexar/PlantAnAgentResearch/registered-beta-v1`, still absent.
The evaluation proposal retains existing caps and raises its filesystem margin
from 2 to 4 GiB, for a 48.80 GiB two-stage opening reserve plus training artifacts.
This is sufficient storage, not a new full-stage capacity measurement.

The [launch plan](PPO_LAUNCH_PLAN.md) makes the scientific decision explicit:
retain research E-beta's three opening infeasibilities and accept the risk that
its failure prevents every primary development score; product MPC is no
substitute. Negative development leaves held-out sealed. The next gate is the
complete repository/static/frontend check on the clean implementation and a
read-only packet binding exact source, host, protocols, all 15 configs and root.
No canonical authorization, campaign root, registered reset or policy rehearsal
has been created or executed. Final clean-gate evidence is retained outside
tracked source so the checked revision can remain the later authorization parent.


### PPO clean release gate completed (2026-09-11)

Clean implementation b5ea55e9c12571574f3452a6d145c6b8d51c2d87, tree
3d7a128e00fdd7ac4c5c0d9592edaffdb2da768e: 3,052 Python tests passed
in 2,227.36 seconds. Strict typing passed for 20 changed source/tool files;
Black passed for 32 Python files. Compilation, registered/study/export CLI
checks, frontend production build/lint and whitespace checks passed. The
worktree stayed clean and unchanged throughout. Existing Starlette/httpx,
frontend bundle-size and HiGHS diagnostic output are retained, not suppressed.

The exact gate, command arguments, log hashes and launch proposal are in
/Volumes/Lexar/PlantAnAgentResearch/readiness-b5ea55e/.
Release-gate SHA-256:
e45a08c92c306df8d11d1cf4ef2306e9679097d017495759ed84c921438e07b2.
Launch-proposal SHA-256:
2f0aeda94ecc7f37b31f9b46fbae69f343343b00a007044f1f86cfe834cf3be9.
The fixed 15-policy matrix, protocol/task/lock identities, host and sufficient
Lexar reserve have been checked. Research E-beta's negative fixture results and
risk of ineligible development remain explicit. No registered training,
protected evaluation, canonical authorization or campaign root was created.

This entry is retained alongside the checked implementation's external evidence
so the source stays unchanged. It can be appended to the tracked journal in the
later permitted authorization/status-only commit, whose sole parent must be the
explicitly approved implementation. Readiness is complete; multi-day training
requires the separately scoped execution approval in the frozen custody protocol.


## 2026-09-11 — Separate one-policy pilot before the full registered matrix

The user explicitly requested: “Let's start by training one policy on one seed
and see what happens.” This prospectively changes the immediate execution scope
to one exploratory policy. The original 15-cell protocol and all protected seed
domains remain intact. No canonical registered authorization is created.

Hypothesis: the existing PI-Lagrangian learner may improve delivery/economic
behavior or resource-return costs from its initialization within the existing
1,048,576-transition budget. This is uncertain; an unchanged or worse policy,
physical diagnostics, missed deliveries and every resource deficit remain results.
One seed cannot establish training robustness or superiority to MPC.

The [pilot specification](PPO_SINGLE_POLICY_PILOT.md) fixes optimizer seed 101,
four production beta environments, training episode seeds 9,500,000–9,501,563,
the original learner/task settings, an external Lexar root, a 10-hour training
cap, and fixed post-training comparisons with initialization and H on four
other nonprotected scenarios. A separate runner composes the existing learner
and allocator without modifying any production fixture/registered validator,
checkpoint format, simulator, guard, reward, configuration or frontend. The
long pilot has its own format; it is neither a bounded fixture checkpoint nor
registered evidence. A one-use root and child attempt markers prevent retries;
source/runtime/host rechecks, bounded subprocess logs, completed rollout records,
reopened final model evidence and explicit failure reports retain custody.

Initial pilot tests: 8 passed, 1 failed. The new untrained-policy export included
the attached cost critic, which the generic SB3 policy constructor cannot restore.
The runner now saves initial tensor weights and restores them strictly into a
copy of the final model's identical policy architecture. The trained constrained
checkpoint had already reopened correctly. The repaired focused suite passed
9 tests. Expanded export and real full-week H fixtures then found a JSON boundary
issue: serialized tuples become lists, so direct Python object comparison rejected
an equivalent task. Pilot specification/task admission now compares canonical
JSON bytes. Two accompanying roundtrip test assertions were corrected to compare
canonical bytes too. Initial strict typing also caught callback and beta-environment
type narrowing issues, subsequently addressed. No production pilot reset or
optimizer step has occurred during these tests; real-learning fixtures use
9,990,000-series seeds and tiny budgets.

The final pilot gate is deliberately scoped to the isolated runner and its
production constrained-training, allocator, configuration and provenance
dependencies. The prior complete 3,052-test gate is inherited evidence, not a
new full-suite result. The exact clean-source gate, subsequent launch receipt
and final outcome will be retained outside Git while the pilot is active.

The repaired expanded pilot suite passed **10 tests in 28.63 seconds**. Strict
typing passed for the isolated runner. Two additional admission/attempt tests
passed in 1.75 seconds: a real provenance/plan JSON roundtrip rejects mutated
gate bytes, and a repeated child refuses its consumed attempt before training.
A full 168-hour H integration fixture completed and serialized all seven daily
settlements; its seed is separate from the pilot scenarios and no comparative
performance result was inspected. Final clean-commit regression evidence follows
outside tracked source.

## 2026-09-12 — Completed pilot failure and adaptive learning diagnosis

The one-policy pilot at clean source `912d9e6` completed all 1,048,576 transitions,
128 rollouts and 40,960 minibatch updates. Training took 15,437.93 seconds;
evaluation took 131.74 seconds. Both children exited zero with empty stderr and
the supervisor closed with no failure. The external root remains
`/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1`. Its exact source gate
passed 263 scoped Python tests plus static/CLI checks; the earlier complete
3,052-test repository gate is inherited, not a full test count at `912d9e6`.

On all four fixed descriptive stochastic weeks, final PPO is worse than its
initialization and H. Mean delivery fill is 30.623% for final PPO, 84.307% for
initial PPO and 89.506% for H; mean operating value is -105,701.56, -24,941.47 and
-9,846.52 EUR respectively. Final PPO produces 1,835.50 kg of methane per week
against initial PPO's 5,443.06 kg and H's 5,803.74 kg. Final hydrogen inventory is
almost full while methane inventory is empty. All three controllers fail at
least one resource return in every daily cycle across the four episodes. The
pilot did not perform the strict minute-level safety gate or detached replay;
guard authority alone is not evidence of those passes. No registered or held-out
evaluation has been opened, and no RL benefit is established.

Read-only diagnosis inspected saved trajectories, rollout ledgers and tensor
weights; it performed zero new environment resets and optimizer steps. The
final reward critic's second tanh layer is fully saturated, with effectively
constant prediction near -806 EUR despite discounted return means around
-22,000 to -27,000 EUR. The policy commands nearly maximum electrolysis on most
decisions and much less methanation: mean targets change from approximately
1,509/598/90/1,251 kW to 2,984/410/32/1,159 kW in electrolyser/DAC/methanator/
battery-charge order. Methanator applied energy matches its dispatched energy,
pointing to the upstream command/allocation behavior rather than downstream
guard refusal. Electrolyser hot starts rise from 18.5 to 261.25 per week and
nonproductive energy from 5,248.75 to 18,594.25 kWh. Manual tensor inference
reproduces recorded actions within 4.2e-7. Source review found no demonstrated
action-order, sign or pilot-wiring defect. This identifies poor learning and
critic fit, not a uniquely proven causal mechanism.

The [compact diagnosis](results/ppo_pilot_912d9e6_diagnosis.json) retains all
controller means, network/dispatch diagnostics and input/output digests. Two
saved-data reproduction scripts are committed under `tools/`. Original analysis
scripts and outputs were archived without changing the pilot root under
`/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1-analysis-20260912`;
manifest SHA-256 is
`59338429618b473dbe118a943f60803006b4c43b09b339b412ba5c3e101223b7`.

The user subsequently granted full research autonomy to diagnose and iterate
toward verified operational benefit. The new
[adaptive protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) preserves the
original registered study and protected seeds. Five fresh, equal-budget
131,072-transition cells compare PI/scalar learning with raw/scaled optimizer
reward and one lower-exploration PI cell. Hypotheses are poor reward conditioning,
competition from independently standardized sparse resource-cost advantages,
and excessive exploration. The physical task, raw EUR accounting, resource
signals, dispatcher and guard remain unchanged. No claim is manufactured by
increasing demand or weakening forecasts/baselines. Following the user's
service-ceiling clarification, preserved delivery is a requirement and economic
efficiency is the main opportunity when existing service is already high.

New isolated research modules provide the batch supervisor, read-only rollout
diagnostics and native PPO/H/OperatingMPC/M24/E-beta evaluation with complete
minute truth chains. Saved commands are replayed in a separate process, with
strict safety, individual resource settlements and incomplete/failure records.
The replay verifies physical consequences; it does not independently establish
the authenticity of a reported controller exception. Checkpoint identities,
source/host/gate checks and closed process receipts are separately bound by the
parent. All eight development seeds and all ten actor identities are fixed
before any reset. Local failures remain explicit slots; operator interruption
or source/integrity loss aborts remaining execution. Future practical-effect
thresholds, new confirmation data and optimizer-seed replication remain required
before an operational-benefit or algorithm-level claim.

Initial diagnostics and batch integration checks passed 15 tests in 39.14 seconds.
An independent launch review found three admission gaps: failed training could
still supply a checkpoint, initial weights lacked digest verification, and an
operator interruption could be treated as a local failure and launch the next
child. All three were repaired with regression tests; the expanded batch suite
passed 11 tests in 39.56 seconds. A real serialized replay regression exposed
the tuple-to-list JSON mismatch at the evaluator boundary. Comparison now uses
canonical JSON values, including scalar type distinctions. A combined typing
check also exposed mypy 1.11.2's inability to resolve the venv NumPy 2.5.2
`NDArray` alias. Expanding only the annotations repaired the exact-environment
check; diagnostics still pass all seven state-preservation/gradient tests. The
complete evaluator suite passed 29 tests in 151.10 seconds, including real
compressed-file replay and tamper checks. A final cross-module audit found that
an integrity mismatch could otherwise become an ordinary failed child and let
the batch continue. A reserved integrity exit now stops the parent after receipt
publication; expected unavailable policies remain local failed slots.
The six focused failure/admission/interruption checks passed after that repair,
and strict typing of all three research modules plus Black for all eight new
Python files passed. No production task, existing controller, frozen learner,
registered protocol or frontend file was changed.
No adaptive production reset or training step has occurred at
this implementation entry; only disjoint 9,990,000-series fixtures ran. Final
clean-source gate and launch evidence will be stored outside tracked source so
the experiment's source identity remains stable while it runs.

## 2026-09-12 — Adaptive batch 001 closed: conditioning improved, benefit unproved

Executed the frozen five-arm development screen at clean revision
`238fe7ca763254b70af63a4ef0690a9d4523aa60` after a 379-test scoped gate passed in
404.67 seconds, plus Black, strict mypy, compilation, CLI and whitespace checks.
The full repository and frontend gates were not repeated for this launch; the
inherited complete gate remains 3,052 Python tests at `b5ea55e`. The source stayed
clean and unchanged while the experiment ran. The five arms crossed PI/scalar PPO
with raw/0.001-scaled rewards and added a 0.2-initial-standard-deviation PI arm.
Each trained 131,072 transitions, sixteen rollouts and 5,120 minibatch optimizer
steps using optimizer seed 101. The hypotheses concerned reward conditioning,
resource-cost signal competition and exploration, with production requirements,
accounting, dispatcher and guard unchanged.

The supervisor closed at `2026-09-12T14:23:56.520417+00:00`. All 165 closed process
receipts had empty failure codes. Independent full artifact audit passed for
five trained endpoints, eighty evaluation slots and eighty physical replays:
64 complete episodes and sixteen retained controller-failure prefixes. M24
rejected degraded/offline DAC in all eight scenarios; E-beta's service LP was
infeasible in all eight. Their weekly aggregates remain null. A clean child exit
or valid physical replay does not imply controller completion or independently
authenticate the reported controller exception. Full audit and endpoint/ledger
checks were performed only after authoritative batch closure.

The [closed results](results/ADAPTIVE_RL_BATCH_001_RESULTS.md) and
[compact digest index](results/adaptive_rl_batch_001_results.json) retain all ten
actors and all eight seed slots per actor. Mean service / operating EUR were:
initial 85.37% / −20,475.35; H 87.98% / −9,713.02; OperatingMPC 90.20% / −4,012.23;
A PI-raw 76.45% / −32,070.27; B PI-scaled 53.15% / −70,549.01; C scalar-raw
57.55% / −62,371.54; D scalar-scaled 73.55% / −37,592.42; E PI-scaled-lowstd
90.53% / −11,637.03. H passed strict safety in 8/8 cases and OperatingMPC in 6/8;
the initial and every trained policy passed in 0/8. No actor satisfied all daily
resource returns. All forty trained-policy strict failures involved forced
stops, while the battery safety threshold of 0.05 was respected. The dispatch
floor near 0.15 must not be mislabeled as a battery safety failure.

All sixteen rollout curves support a conditioning improvement: the scaled
reward critics avoided full tanh saturation and reached pre-final-update
explained variance 0.929, 0.938 and 0.971 for B, D and E, while A/C remained near
zero. Scaled diagnostics implied less common gradient clipping, but all eighty
diagnostic minibatches still implied clipping. Operating counterevidence is
retained: scaling PI lost service and EUR in all eight pairs, while scaling
scalar improved both in all eight. Aggregate reward/cost gradient opposition
was intermittent and multipliers did not explode; this does not establish PI
signal conflict as the sole mechanism. MSE retains scaled-unit labels, null
explained variance is preserved, and deterministic pre-update diagnostic
minibatches are not presented as actual SGD/Adam updates.

E improved every development case relative to the initial policy and B, but its
mean value was EUR 1,924.01 below H and EUR 7,624.79 below OperatingMPC. Its worst
paired scenario, 9,700,005, had both less delivery and worse economics than H and
OperatingMPC. An independently archive-bound physical diagnosis retained all
24 E/H/OperatingMPC episodes and all 56 daily boundaries per actor. E had 88
forced stops, 69 in intervals aborting a startup and 86 in intervals reaching
SOC ≤ 0.151; persistent positive targets and dawn/evening timing suggest a
startup-energy planning limitation. H had zero stops; OperatingMPC had two after
reducing its command during unfinished startup. E requested DAC during every
one of 4,503 declared outage minutes, with the guard removing allocations;
learned fault adaptation is not demonstrated. No counterfactual repair was run
in this batch. These findings motivate prospective engineering tests rather
than a causal benefit claim.

The external evidence root is
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001-analysis`. Its raw audit
SHA-256 is `88048a9e5e5d1bb8bfd3d823824691daf82746fd488bbab5a1856ff57ab739a9`;
the compact index includes separate canonical audit digest, exact evidence
file digests and all per-slot summary/archive/replay digests. Full curves,
minute evidence and checkpoints remain outside Git. No registered training,
formal selection, held-out or confirmation evaluation, protected-seed access,
operational benefit, RL superiority or commercial savings is claimed. This
entry records the closed batch only and does not revise its frozen protocol.

The outcome-blind audit/report tools and three fabricated-artifact test files
are archived under `tools/adaptive_batch_001/`. Their original external copies
were frozen while source `238fe7c` was still running. After archival, all 31
fixture checks passed in 6.64 seconds without reopening production outcomes or
resetting a production environment. The auditor intentionally requires the
original exact clean source and gate; a later checkout cannot silently reissue
its audit. The operational diagnosis and its two reproduction scripts were
archived unchanged under the external analysis root's `operation/` directory.

## 2026-09-12 — Operating MPC respects unfinished startup

The closed batch exposed two OperatingMPC forced stops after its fallback
reduced an unfinished startup's target to the legacy 450 kW floor. The active
beta startup requires 600 kW. OperatingMPC now preserves that startup draw when
current public solar and reserve-, health- and operator-aware battery support
can supply it. Decision evidence records the original/issued targets, available
power, startup state and adjustment status. The fallback identity and original
typed solver failure stay visible. This is an incumbent controller correction,
not learned-policy benefit; the next experiment must compare against it.

Synthetic cold/hot startup fixtures reproduce the old forced abort and show
completion after the correction at 650 kW solar, or 550 kW with authorized
battery support, through the unchanged minute dispatcher, guard and simulator.
Current shortage and later weather loss still expose forced stops. A direct
operator equipment cap below startup draw preserves the existing fail-closed
model-input rejection. Off/running modes and sufficient existing commands retain
their behavior. These tests establish the local correction, not an eight-seed
operating result or safety guarantee under future weather uncertainty.

All 20 focused test cases were covered: 19 passed in 51.62 seconds and the added
future-weather-shortage case passed in 3.25 seconds. Strict mypy, Black and
whitespace checks passed. During test construction, eight synthetic-clock versus
forecast fixture failures and one mistaken expected operator-cap behavior were
resolved by correcting the fixtures/expectation; no plant or forecast code was
changed. No development/protected seed or production training reset was run.
V1, the registered M24/E-beta implementations and the frontend remain unchanged.

## 2026-09-12 — Prospective bounded operating-effort PPO screen

Batch 001 recovered average delivery under lower exploration, but persistent
approximately 1.5 MW commands caused repeated starts and expensive battery/
electrolyser use. The new [batch 002 protocol](ADAPTIVE_RL_BATCH_002_PROTOCOL.md)
tests a restricted operating prior: PPO adjusts H's higher-load electrolysis
and downstream effort instead of choosing all absolute targets from scratch.
Four fresh 131,072-transition cells compare direct/residual representations
under PI/scalar learning, all with optimizer reward scale 0.001, initial std
0.2 and exactly zero actor output heads. This includes the previously missing
scalar comparison under lower exploration. It is a comparison of complete
method packages, not an isolated causal estimate of the added four features.
Production obligations, public information, simulator, dispatcher, guard and
resource settlements remain unchanged. The repaired OperatingMPC is the required
current-source incumbent. Fresh training seeds start at 9,610,000 and all ten
actor identities use the eight fixed development seeds 9,700,100–9,700,107.

The separate residual wrapper returns the actual 135 base features plus four
cached H target fractions. It computes H once per boundary without another
observation-encoder call and sits inside allocator custody. Electrolyser targets
at/below the 1,200 kW stable-running floor remain H's; adjustments above the
floor and proportional downstream corrections are bounded. In particular,
H's 450 kW night target is not multiplied above the 600 kW startup threshold.
An exactly zero residual executes the original float64 H command and invalid
flag. This is an action parameterization, not a feasibility or safety certificate.
Its separately hashed 139-feature checkpoint/interface includes the mapping and
H implementation sources; existing production beta loaders cannot accept it.

The residual evaluator stores both physical/public observations and actual
policy inputs. Detached replay executes saved absolute commands, separately
recomputing H to verify input features and residual mapping; it never runs the
learned policy. Review exposed four evidence gaps before admission: unchecked
controller headers, inconsistent failed-decision fields, an unenforced fixture
stopping budget, and a possible mixed archive after physical/evidence failure.
The corrected evaluator binds those fields/budgets and raises an integrity
error for physical/evidence exceptions instead of publishing a replayable-prefix
claim. Nineteen serialized replay, tamper, prefix and admission checks passed
in 73.42 seconds; strict mypy and Black passed for the source.

The wrapper's 37-test suite passed in 91.40 seconds, including complete 168-hour
zero-residual equivalence for commands, all minute states, raw EUR, returned
observations, seven settlements and constraint transitions. Mapping, clipping,
malformed actions, float32/float64 publication and four-stream allocator custody
also passed. Strict mypy and Black passed. A fixture initially used six-hour
decisions against the inherited hourly tariff cadence; correcting that test to
one hour resolved its rejection. No production/protected seeds were consumed.
Batch 002 remains unopened at this implementation entry; final runner checks,
clean-source gate and launch evidence follow before any production reset.

The batch runner's 32 checks passed in 56.37 seconds, including actual tiny
PI/scalar learning and checkpoint reopen for both 135/139 interfaces. Read-only
admission review prompted explicit command-bound training receipts, cross-slot
episode/replay bindings, and fatal integrity classification for post-learning
attestation failures and allocator seed/manifest/cursor custody loss during
learning. Ordinary numerical training failures remain retained failed arms.
Constructor-only checks patch the base reset to fail, establishing that plan and
learner construction do not consume unwrapped episodes. Black, strict mypy,
compilation, CLI and whitespace checks passed. Earlier fixtures correctly caught
a concurrently changed wrapper source and nondeterministic dispatcher timing;
the final suite uses stable source and excludes only wall-clock timing from
physical equality. The original batch runner, registered learner/protocol and
production physical task were not changed. Final source freeze and exact-revision
readiness evidence are stored externally to avoid changing the running checkout.

## 2026-09-12 — Batch 002 closed; residual learning remains behind OperatingMPC

The exact clean `a442a4a9f5b028568d36bbbddf21a34c3609a7cc` readiness gate passed
511 scoped Python tests in 656.33 seconds, Black, strict mypy, compilation, CLI
and whitespace checks. The inherited full repository/frontend gate was not
repeated. Gate SHA-256 is
`60c35ffd2b81dac615aec2d3c77f06c08c1cc14ce50bf84a92f1925c6dc40bcd`.
The supervisor launched at `2026-09-12T15:27:54.454314Z` and closed normally at
`2026-09-12T19:34:57.336127Z`. Four training, 80 evaluation and 80 replay receipts
closed with empty failure codes; all processes were subsequently verified gone.
No outcomes were opened or source changed while experiments ran. Quiet waits
and process-only checks preserved the outcome-blind execution boundary.

The auditor/reporter and their fabricated tests were frozen externally during
execution. After closure, the audit verified 632 hashed artifacts, complete
training/checkpoint/allocator bindings, 64 full episodes and 16 stopped comparator
prefixes against completed detached physical replays. The initial residual and
native H had identical physical/economic projections in all eight scenarios.
The audit's raw SHA-256 is
`a65ba53de363032caced6435a91543b782fcf993dbb93e242a6992f3d40406ea`;
the descriptive reporter's canonical audit digest is separately
`182d49901c00386aa3adbfb562b67b6d199c7705d88a010cf3efddc6938af426`.
Both tools completed successfully. Their byte-identical Git archives under
`tools/adaptive_batch_002/` passed all 88 fabricated checks in 31.07 seconds.

The [closed results](results/ADAPTIVE_RL_BATCH_002_RESULTS.md) and
[compact evidence](results/adaptive_rl_batch_002_results.json) retain every actor,
all 80 slots, daily deficit vectors, failure text and digests. Residual scalar D
improved mean value by EUR 595.39/week over its exact-H initialization; residual
PI C improved by EUR 442.40. Both remained behind OperatingMPC (EUR 7,112.54 and
EUR 7,265.53 respectively), inherited H's five forced stops, lost delivery in
individual scenarios and worsened individual daily CH4 deficits (maximum increases
23.65 kg and 16.84 kg). B improved over A in all eight service/economic comparisons,
but both absolute policies failed strict safety throughout. No policy qualified
for confirmation. M24/E-beta stopped in all eight scenarios and provide no full
weekly means. No actor satisfied every daily resource return.

The task still requires 6,720 kg methane across 28 deliveries over 168 hours.
OperatingMPC achieved 93.94% mean fulfillment with 8/8 strict safety passes; H
achieved 90.20%, D 90.65%. High commitments and high fulfillment remain distinct.
Demand is not increased to create an RL opportunity. OperatingMPC's corrected
startup behavior belongs to the incumbent, and differences between old/new
scenario panels cannot establish its correction's causal effect. Further work
must improve economics at preserved service/resource performance and pass hard
safety. This is adaptive development, not held-out evidence, commercial savings,
formal policy selection, live deployment or an operational-benefit claim.
Original registered families, budgets and protected seeds remain untouched.

Independent saved-data diagnosis bound 194 learning inputs and all 40 operating
archives before deriving mechanisms; no new reset or policy inference occurred.
All sixteen training curves and 21,504 trained-policy action rows remain available.
Critics fit the sampled returns (final pre-update EV approximately 0.973/0.975/
0.949/0.948), exploration remained active, and policy updates continued. Marginal
action correlations do not prove use of forecast quantiles or faults. D had no
direct electrolysis adjustment authority in 3,472/5,376 decisions and reproduced
all four current-H targets in 2,395 decisions. Nonzero standardized PI gradients
on some zero-observed-cost rollouts motivate a normalization hypothesis, not a
causal explanation or permission to remove resource costs.

The physical diagnosis identified all five inherited stops: H lowered its command
below 600 kW during unfinished cold startup with 2,500 kW battery support available;
C/D issued the same E commands at those boundaries. MPC was already running at
four corresponding boundaries and maintained 600 kW in the remaining startup.
C/D's small mean EUR gain over H came from additional delivery and reduced
shortfall penalties, with higher battery throughput, wear and opportunity cost;
equal-service cost savings are not established. C/D worsened four/three daily
methane deficits respectively. The diagnosis scripts, full JSON and Markdown
were archived byte-identically under the external analysis root's `learning/`
and `operation/` directories, with manifests and references in compact evidence.
