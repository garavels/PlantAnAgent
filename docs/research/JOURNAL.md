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
