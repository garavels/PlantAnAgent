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
