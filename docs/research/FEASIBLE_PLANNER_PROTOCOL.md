# Feasible future-information planner protocol

Date: **2026-09-02**

Protocol version: **best-known-feasible-future-information-v1**

Status: **frozen before implementation and before any new full-horizon outcome**

## Purpose and claim boundary

Part I of the planning benchmark provides an optimistic delivery ceiling for a
continuous relaxation. It does not show that the exact five-field action interface can
realise a better trajectory. This protocol freezes the complementary constructive
experiment: a deterministic, budgeted search that advances the real guarded
`PlantEnvV2` and returns only a schedule that passes an exact fresh replay.

The result is a **best-known feasible future-information trajectory under this frozen
library and budget**. It is:

- a constructive lower bound on the best delivery attainable through the current exact
  action interface;
- a diagnostic of realizable scheduling headroom above the legacy controller;
- deliberately allowed to inspect the complete private solar tape and delivery schedule;
  and
- neither a deployable controller, an oracle optimum, MPC, nor evidence that RL has
  learned anything.

The planner is a separate privileged research type. It must not implement the
`ControllerV2` protocol, enter the causal controller evaluator, receive a controller ID,
or be reported in the same information-set contrast as an online policy.

No failure to improve A0 may be interpreted as proof that a better exact schedule does
not exist. It means only that this versioned search failed to demonstrate one. An
improvement demonstrates attainable privileged scheduling headroom; the later causal
MPC/RL question is how much of that headroom can be captured without future truth.

## Outcome and split boundary

The already disclosed 72-hour cells at delivery rates 35 and 42 kg CH4/h for seeds 0--2
are contaminated development cells. Rate 40 kg CH4/h, seed 0, at 24 and 72 hours and
seed 3 are also contaminated as recorded in `JOURNAL.md`. This protocol is frozen before
the planner implementation and before opening any further full-horizon outcome.

Implementation may advance the exact simulator only in explicitly noncanonical unit
fixtures of at most eight controller decisions. Pure algorithmic test doubles that
produce no plant outcome may
exercise arbitrary synthetic sequence lengths, including the 24-decision coordinate-
block boundary. Once this document and the implementation are separately committed on
clean source, the first integrated executions may use only the six declared 35/42
development cells. No uninspected seed, weather regime, delivery rate, or fault regime
may be opened until a prospective amendment freezes it.

## Registered evidence scope

Version 1's registered development matrix is exactly delivery rates `{35.0, 42.0}` kg/h
crossed with seeds `{0, 1, 2}`. It accepts only a canonical `sample_episode` episode
satisfying all of the following:

- `PlantConfig.dt_s == 60` seconds;
- `EpisodeConfigV2.decision_interval_s == 900` seconds;
- `EpisodeConfigV2.delivery_period_h == 6` hours;
- `EpisodeConfigV2.horizon_h == 72` hours;
- the commitment profile is exactly `ael_pressurised_fixed_bop_alpha_v1` at SHA-256
  `c6123ad4c30ef29445fa68e37d7f3d55bdeec05129d8535ab9dca2bc4715c4f9`, including
  `cold_start_s=3600`, `min_stable_load_frac=0.40`, `start_load_frac=0.20`, and
  `hot_standby_load_frac=0.02`;
- the sampled scenario contains no fault specification and the live registry is empty;
- reset-time manifest regeneration, exogenous hash, and checkpoint-contract identity
  all match exactly; and
- the unconstrained Part-I relaxation model and result are both supplied, the result
  succeeds with a witness for this same canonical episode, and their identities bind as
  specified below, because one frozen proposal template depends on them.

All other `PlantConfig`, `Constraints`, `RewardWeightsV2`, and `EpisodeConfigV2` fields
are the repository defaults. Using the canonical-JSON rule in this document, the frozen
configuration identities are:

| Payload | SHA-256 |
|---|---|
| `PlantConfig().to_dict()` | `a0b374d24368c15402b2cfca2d1ce1ecc50e9c931fc2fef9fbd407c3ba95140a` |
| `Constraints().to_dict()` | `ed16cc2355f037b870030e42f743c6aed7b9221da6b5b43ebdcfffc086e59161` |
| `RewardWeightsV2().to_dict()` | `1fee0632e10730014cdf5f4e7d883d8e01d12fd6c8a527b74b30c922cd256ee3` |
| default commitment profile | `c6123ad4c30ef29445fa68e37d7f3d55bdeec05129d8535ab9dca2bc4715c4f9` |
| 35 kg/h `EpisodeConfigV2` | `115612216fc1ad749e6505cc0ea567265222695243078690b19095b3d750b7d7` |
| 42 kg/h `EpisodeConfigV2` | `82838f8f36af09ab09d705292b1ecbe203220fbd7d6ebe0790214912d6d07649` |

The expected checkpoint-contract digests are respectively
`4ef8f7c92c26d0ea5cd0b6a1b4dc7a0384c9776fe265838b081887f760b91a0a` and
`5605c2bf5e99bb1bbec584e29f23773d5233b5eadcdcd11f269bdf03d12adfb0`.
Any different payload or digest is noncanonical and cannot enter the six-cell run.

Tests may explicitly override horizon/cadence and canonical-evidence checks, but such a
result must carry `evidence_scope="noncanonical_unit_fixture"` and can never be written
as benchmark evidence. There is no automatic omission of an unavailable relaxation
template and no substitution of the endowment-preserving diagnostic witness.

## Deterministic inputs and arithmetic

One run is fully identified by the episode seed, `PlantConfig`, `EpisodeConfigV2`,
`Constraints`, `RewardWeightsV2`, checkpoint contract, exact sampled episode, successful
unconstrained relaxation result, and the settings below. All finite real comparisons are
direct comparisons of stored IEEE-754 binary64 values unless an exact rational rule is
specified. No comparison epsilon is used by search ranking, incumbent selection, or
replay equality. Any non-finite value fails the run.

The search has no random number generator, stochastic restart, unordered iteration, or
wall-clock-selected incumbent. Parent order, template order, protected-lane order,
deduplication, sorting, and polishing order are deterministic conditional on completing
the declared algorithm. Emergency wall/RSS validity can make identical inputs complete
on one platform and fail on another; hardware, OS, architecture, and runtime identities
are therefore part of runner evidence, and no best-so-far result survives such a failure.
Canonical JSON means UTF-8, sorted keys, compact separators, `ensure_ascii=False`, and
`allow_nan=False`.

## Exact incumbent A0

Before search, construct a fresh canonical environment and reset it with the registered
seed. Reset `LegacyHeuristicControllerV2`, request one decision from
`env.controller_observation.without_forecast()`, and advance through `step_action` with
the returned `controller_invalid` flag. Validate the descriptor against the frozen A0
payload after reset and after every computation, require a `ControlDecision`, and pass the
exact commitment profile alongside constraints and plant configuration as in the causal
evaluator. Continue through true termination and retain every actual five-field request
and invalid flag.

Replay that detached schedule from a second fresh reset. The generated and replayed
outcome fingerprints must match exactly. The replay, not the generation run, is the A0
evidence record. A0 remains outside all beam pruning, deduplication, and search failure
paths. Its exact prefix state is injected into every nonterminal beam cut, and its
complete terminal score remains a separate candidate during final selection.

The A0 generation and evidence replay compute and compare a complete state-signature
payload after every four decisions. Search creates one third fresh-reset rolling A0
cursor, advances it by the next four stored A0 actions at all 72 depths, and requires its
full payload to equal the stored evidence-replay prefix payload. At each of the first 71
nonterminal cuts, a clone of that cursor, never the cursor itself, enters the candidate
pool and occupies one of the 24 beam slots. At terminal depth 72 the verified A0 result
stays external. The cursor is kept outside beam mutation; 72 prefix environments are not
retained.

## Ordered action-template library

The action library has the following immutable order and zero-based IDs:

| ID | Name | Semantic target |
|---:|---|---|
| 0 | `legacy_live` | live legacy action on the current branch state |
| 1 | `relaxation_projection` | decision-average process targets from the Part-I witness |
| 2 | `idle` | no solar allocation and no battery support |
| 3 | `charge` | all available solar to the battery |
| 4 | `warm_hold` | stack target at the profile hot-standby load |
| 5 | `cold_start_hold` | stack target at the profile start load |
| 6 | `h2_stable` | stack at profile minimum stable load |
| 7 | `h2_full` | stack at its active cap |
| 8 | `dac_full` | DAC at its active cap |
| 9 | `feedstocks_stable` | stable-load H2 and its stoichiometric DAC target |
| 10 | `feedstocks_full` | full-load H2 and its stoichiometric DAC target |
| 11 | `methanate_full` | methanator at its active cap using stored feedstocks |
| 12 | `clipped_stoich_40` | nominal stoichiometric mix at stack fraction 0.40 |
| 13 | `clipped_stoich_70` | nominal stoichiometric mix at stack fraction 0.70 |
| 14 | `clipped_stoich_100` | nominal stoichiometric mix at stack fraction 1.00 |

The serialized descriptors include these IDs, names, formulas, formula version, and all
numeric fractions. The exact canonical JSON is:

```json
{"formula_version":"future-truth-router-binary64-v1","schema_version":1,"templates":[{"generator":"legacy-live-v1","id":0,"name":"legacy_live","stack_fraction":null},{"generator":"relaxation-projection-v1","id":1,"name":"relaxation_projection","stack_fraction":null},{"generator":"constant-idle-v1","id":2,"name":"idle","stack_fraction":null},{"generator":"constant-charge-v1","id":3,"name":"charge","stack_fraction":null},{"generator":"stack-only-router-v1","id":4,"name":"warm_hold","stack_fraction":0.02},{"generator":"stack-only-router-v1","id":5,"name":"cold_start_hold","stack_fraction":0.2},{"generator":"stack-only-router-v1","id":6,"name":"h2_stable","stack_fraction":0.4},{"generator":"stack-only-router-v1","id":7,"name":"h2_full","stack_fraction":1.0},{"generator":"dac-only-active-cap-router-v1","id":8,"name":"dac_full","stack_fraction":null},{"generator":"feedstocks-router-v1","id":9,"name":"feedstocks_stable","stack_fraction":0.4},{"generator":"feedstocks-router-v1","id":10,"name":"feedstocks_full","stack_fraction":1.0},{"generator":"methanator-only-active-cap-router-v1","id":11,"name":"methanate_full","stack_fraction":null},{"generator":"clipped-stoich-router-v1","id":12,"name":"clipped_stoich_40","stack_fraction":0.4},{"generator":"clipped-stoich-router-v1","id":13,"name":"clipped_stoich_70","stack_fraction":0.7},{"generator":"clipped-stoich-router-v1","id":14,"name":"clipped_stoich_100","stack_fraction":1.0}]}
```

Its `planner-action-library-v1` domain digest is
`805bc4a6203026783a0d2d6514d25c83188d9ca4476b7b9c6c7763b6718456f2` and is the
only registered action-library identity. At one branch state, all 15 template IDs are
still attempted: formulas that emit the same first action may diverge after the branch
evolves within a four-decision macro. Only completed macro candidates with an identical
resulting state-signature payload deduplicate. Their complete source sequence and
concrete-request bytes participate in the frozen survivor tie-break; action equality is
not a prerequisite for state deduplication.

A concrete `ActionRecordV1` has exactly these fields: zero-based `decision_index`,
`source_phase`, `source_code`, `source_binding_sha256`, `controller_invalid`,
`battery_share`, `electrolyser_share`, `dac_share`, `methanator_share`, and
`battery_support_kw`. `source_code=-1` and `source_phase="a0_capture"` mean a request
captured from exact A0; integers 0--14 identify the template and use phase `beam`,
`coarse_polish`, or `fine_polish`. An unchanged polishing prefix or suffix retains its
existing phase/code. Every decision in a replaced block receives the replacing phase and
template ID. The source binding is the A0 controller-descriptor hash for A0/legacy-live,
the relaxation-guidance hash for template 1, and the action-library hash for templates
2--14.

Source sequences are compared as per-decision `(phase_order, source_code)` tuples, with
phase order `a0_capture=0`, `beam=1`, `coarse_polish=2`, `fine_polish=3`. Action-record
bytes are the canonical JSON bytes of the complete ordered records. Concrete-request
bytes are the canonical JSON bytes of decision index, invalidity, and five action values
only; they exclude all source fields.

Only a trace whose complete concrete-request bytes equal captured A0 and whose
source sequence is entirely `-1` has incumbent tie priority. An A0-origin trace loses
that priority as soon as a polishing replacement is accepted, even if some or all later
concrete actions remain unchanged.

The A0 controller payload is exactly the canonical JSON returned by the current
`LegacyHeuristicControllerV2.to_dict()`:

```json
{"adapter":"legacy_v1_boundary_telemetry","description":"V1 operator heuristic adapted without behavior changes; it receives only the latest one-minute boundary telemetry and active safety constraints.","id":"legacy_heuristic","information_set":"latest one-minute V2 telemetry plus safety constraints; no preceding-interval aggregate, delivery context, or issued forecast","label":"Legacy forecast-blind heuristic","uses_forecast":false}
```

Its `planner-a0-controller-v1` domain digest is
`d5405d5661b847b67a2999159391fd89fe0543f3459627e7a3a8ab25865a93d5`. A descriptor
mismatch fails input validation rather than silently redefining A0.

### Target powers

Let `Ce`, `Cd`, and `Cm` be the active constraint-resolved electrolyser, DAC, and
methanator caps. Let `Re` be electrolyser rated power. Let `q` be the named stack fraction
and define:

```text
Eq   = min(Ce, q * Re)
SEq  = max(configured stack specific energy,
           profile nominal specific energy)
       * profile.specific_energy_multiplier(Eq / Re)
Hq   = Eq / SEq
CH4q = Hq / h2_per_ch4
Dq   = min(Cd, CH4q * co2_per_ch4 * DAC specific energy)
Mq   = min(Cm, CH4q * methanator specific energy)
```

`h2_stable` and `h2_full` use `(E,D,M)=(Eq,0,0)` at `q=0.40` and
`q=1.00`. `feedstocks_stable` and `feedstocks_full` use `(Eq,Dq,0)` at
those fractions. `clipped_stoich_40`, `clipped_stoich_70`, and
`clipped_stoich_100` use `(Eq,Dq,Mq)`.
`dac_full` uses `(0,Cd,0)`, and `methanate_full` uses `(0,0,Cm)`.
`warm_hold` uses
`(min(Ce, profile.hot_standby_load_frac * Re),0,0)` and `cold_start_hold`
uses `(min(Ce, profile.start_load_frac * Re),0,0)` regardless of the current
commitment mode; the exact simulator decides what those offers mean in that mode.

“Clipped stoichiometric” describes the nominal uncapped calculation. Independent active
caps can clip one route, so these are proposal mixes, not a claim that realised flows or
inventories remain balanced.

For `relaxation_projection`, take the arithmetic mean over the exact Part-I witness
physics intervals covered by the current controller decision. For each expression below,
initialize a binary64 accumulator to `0.0`, iterate physics indices from earliest to
latest using one Python `+=` per term, and divide once by the integer interval count; do
not use `sum`, `math.fsum`, NumPy, or a reordered reduction:

```text
E = mean(solar_to_electrolyser_kw + battery_to_electrolyser_kw)
D = mean(solar_to_dac_kw)
M = mean(solar_to_methanator_kw)
```

The covered half-open interval is
`[env.physics_steps_taken, min(env.physics_steps_taken + episode.decision_stride,
model.physics_steps))`; an empty interval is an action-generation failure. The common
future-truth router takes `S = max(model.solar_available_kw[start:stop])` over exactly
that same interval, scanning from `start` to `stop - 1`.

Clamp only to the corresponding non-negative active cap before applying the common
router below. The supplied `DeliveryRelaxationModel` must have the same `model_sha256`,
checkpoint digest, exogenous episode hash, physics-step count, and canonical-verification
status as the successful, unconstrained, task-upper-bound `DeliveryRelaxationResult`.
The result must have `success=True`, `status=0`,
`preserve_initial_endowment=False`, `is_task_upper_bound=True`, and non-null witness,
certificate, and dual-certified ceiling fields. Witness arrays must match that model's
block lengths. The planner computes and stores a
canonical witness SHA-256 and a relaxation-guidance SHA-256 over the model hash, witness
hash, result status/scope/solver settings, checkpoint digest, and episode hash; elapsed
solver time is excluded. The model supplies the exact solar vector. Any mismatch fails
before A0 generation or search. The relaxation remains merely a proposal generator:
projected actions gain feasible status only by exact simulation and fresh replay.

The evidence runner constructs the unconstrained model once and invokes
`solve_delivery_relaxation` exactly once per cell after the start-source sample, with
feasibility tolerance `1e-7`, physical tolerance `1e-6`, optimality tolerance `1e-5` kg,
marginal-sign tolerance `1e-9`, `solver_options={}`, and `method="highs"` as implemented
by Part I. A failed invocation fails planner input; it is not retried or replaced. No
choice among degenerate witnesses or repeated solver runs may use planner performance.
The endowment-preserving diagnostic is a separate later call and never supplies guidance.
The runner computes `relaxation_result_sha256` over the successful result excluding only
`elapsed_s`, and binds model, result, and witness hashes into settings, every projection
record, and the core result.

These three payloads are exact. The witness payload is
`DeliveryRelaxationWitness.to_dict()` under domain `planner-relaxation-witness-v1`. The
result payload is `DeliveryRelaxationResult.to_dict()` with only its top-level
`elapsed_s` member removed, under domain `planner-relaxation-result-v1`. The guidance
payload under `planner-relaxation-guidance-v1` is exactly:

```text
schema_version=1
model_sha256
result_sha256
witness_sha256
status
canonical_episode_verified
preserve_initial_endowment
is_task_upper_bound
solver
solver_version
solve_settings = DeliveryRelaxationResult.solve_settings.to_dict()
checkpoint_contract_digest
exogenous_episode_sha256
solar_truth_sha256
```

The canonical JSON object uses those field names; `result_sha256` means the domain digest
defined immediately above, and `solar_truth_sha256` comes from the bound model. No other
result, witness, timing, or host field enters guidance identity.

### Common future-truth router

All nontrivial target triples use the same router. At decision `k`, let `S` be the
maximum exact private solar bus value over the next controller hold, using the Part-I
solar vector and truncating only at the episode end. Let:

```text
reserve_energy = constraints.reserve_soc(config) * boundary battery capacity
usable_energy  = max(0, boundary battery energy - reserve_energy)
held_h         = number of remaining physics intervals in this decision * dt_h
Umax           = min(constraints.support_limit_kw(config),
                     Ce,
                     nextafter(usable_energy * discharge_efficiency / held_h, 0))
U              = min(E, Umax)
R              = max(S, (E + D) + M)
```

If `R == 0`, emit `idle`. Otherwise emit:

```text
electrolyser_share = (E - U) / R
dac_share          = D / R
methanator_share   = M / R
battery_share      = max(0, 1 - ((electrolyser_share + dac_share)
                                 + methanator_share))
battery_support_kw = U
```

Each target and raw share must already be finite and non-negative. Compute the raw
four-share tuple in field order `(battery, electrolyser, DAC, methanator)`, with
`battery=max(0, 1-(electrolyser+DAC+methanator))`. If its binary64 sum exceeds one,
multiply all four shares by `nextafter(1/sum, 0)`. If the new sum still exceeds one,
move that common scale once per iteration with `nextafter(scale, 0)`, recompute all four
products, and stop after at most eight iterations. Failure to close by then is a generator
failure. Apply `Action.sanitised` and require bit-identical equality with the closed
action. This frozen proportional closure handles a zero battery share without an
unbounded loop and is part of generation, not a post-outcome repair.

Every formula in this protocol is evaluated top-to-bottom with Python binary64 operators
and the parentheses shown. “Bit-identical action” means equality of the concatenated
`struct.pack(">5d", battery, electrolyser, DAC, methanator, support)` bytes, not Python
dataclass equality. The trace's canonical JSON representation is separately hashed for
portable evidence.

Two frozen reference vectors guard the operation order. With default configurations,
`q=0.40`, boundary battery energy/capacity `4000/8000` kWh, `S=5000`, and a 0.25-hour
hold, the `(battery,electrolyser,DAC,methanator,support)` hex values are exactly:

```text
(0x1.e3ba3ba3ba3bap-1, 0x0.0p+0, 0x1.7357357357356p-5,
 0x1.4414414414414p-7, 0x1.2c00000000000p+10)
```

Their `>5d` bytes are
`3fee3ba3ba3ba3ba00000000000000003fa73573573573563f844144144144144092c00000000000`.
The ordered projection terms `(1+0.5, 2+0.25, 4+0.125)` produce mean
`2.625 == 0x1.5000000000000p+1` under the declared accumulator.
For the raw four-share tuple `(0.0, 0.03491121386362534, 0.7860655356893718,
0.179023250447003)`, whose sum is `0x1.0000000000001p+0`, proportional closure uses
scale `0x1.ffffffffffffdp-1` without another iteration and produces total
`0x1.ffffffffffffep-1`. Unit tests require these exact values.

The support calculation is deliberately conservative over the complete 15-minute hold.
The residual battery share lets dispatch substitute realised solar for contingent
support before drawing storage. The guard is nevertheless authoritative and is rerun at
every one-minute physics step.

`legacy_live` is the one exception to the target router. At each invocation it constructs
and resets a new `LegacyHeuristicControllerV2`, verifies its descriptor against the frozen
A0 payload, then computes once from the branch's current
`env.controller_observation.without_forecast()`, supplying the exact commitment profile.
This prevents controller-instance history from becoming hidden search state. `idle` and
`charge` use their exact constant actions.

## Primary template-beam search

### Search unit and budget

The primary search has a beam width of **24** and advances in non-overlapping
**four-decision (one-hour) macros**. At each macro expansion, one template ID is selected
and that same semantic template is regenerated from the evolving branch state before
each of the four constituent 15-minute decisions. The final macro is shortened only if
the episode has fewer than four decisions remaining. The concrete action and invalid
flag (`False` for every non-A0 planner template) are stored for every decision.

For the registered 72-hour episode there are 72 macro depths. Every retained parent is
expanded by all 15 templates in ID order. The deterministic attempted-edge ceiling is:

```text
15 + 71 * 24 * 15 = 25,575 parent/template macro edges
```

The separately injected A0 prefix is not charged as a template edge. No partial macro is
retained after an exception. Search completes only after every registered macro depth is
expanded and at least one complete searched terminal path survives; exact A0 remains the
separate external incumbent.

Four decisions are long enough for the active one-hour cold-start precursor while
reducing clone pressure relative to a 288-depth tree. This is a search restriction, not
a statement that optimal actions are hourly constant: the chosen template is regenerated
every 15 minutes, and the later polishing stage may replace different one-hour blocks.

### Structural-sharing clone contract

Every branch must still execute the exact `PlantEnvV2.step_action` path. To make the
fixed search practical, version 1 may use a custom clone that deep-copies all operational
state while sharing only validated immutable episode/configuration data.

The implementation is exactly `copy.deepcopy(env, memo)`. Before that call, `memo` maps
the current outer `EpisodeV2` identity to `dataclasses.replace(episode,
stream_spawn_keys=dict(episode.stream_spawn_keys))`. It maps an object to itself only for
the following canonical object paths:

```text
env.config
env.config.solar
env.config.battery
env.config.electrolyser
env.config.h2_tank
env.config.dac
env.config.co2_tank
env.config.methanator
env.config.ch4_tank
env.episode_config
env.episode_config.weather
env.episode_config.electrolyser_commitment
env.constraints
env.weights
env._episode.scenario
env._episode.scenario.initial
env._episode.scenario.faults[*]
env._episode.commitments[*]
env._episode.weather
env.simulator._solar_factor_tape
env.simulator.delivery_ledger._commitments[*]
env.simulator.delivery_ledger._settlements[*]
env.simulator.delivery_ledger._current
env.simulator.electrolyser._commitment_state
```

The compact canonical path-list SHA-256 is
`694c1b8259270bc6be719fc65e93af006fb01c2f10f6a6907734e309f49fcbba` and is stored as
`clone_allowlist_sha256`. Paths that are null or empty simply add no memo entry. The
listed objects are frozen records whose reachable content is scalars, enums, tuples, or
other listed frozen records in the registered fault-free scope.

The clone detaches the simulator components and their internal state,
delivery ledger, event log, fault registry, metrics and nested dictionaries, control
state including nested health mappings, observation encoder state, environment RNG,
action/observation spaces where stateful sampling could matter, and every mutable
container owned by the environment; no other mutable repository-owned identity may share. It may
share frozen plant, episode-configuration,
constraint, and weight objects; frozen weather values and solar-factor tape; frozen
commitments; and immutable scalar/string checkpoint leaves. The outer `EpisodeV2` and
every checkpoint mapping, list, and tuple container are copied recursively rather than
memo-shared. `EpisodeV2` is recreated with a detached `stream_spawn_keys` dictionary,
because that dictionary remains mutable despite the frozen dataclass, while its audited
immutable children may share.

Before planner use, tests must prove:

1. all declared mutable roots have distinct identities;
2. stepping a child leaves the parent and a sibling's complete canonical checkpoint
   payload unchanged;
3. two equal clones receiving equal actions remain byte-identical after every step;
4. the optimized clone and ordinary `deepcopy` remain byte-identical under the same
   short action sequence; and
5. mutating every known nested mutable collection in a child cannot alter its parent.

A post-clone graph audit rejects shared identity for every reachable dictionary, list,
set, deque, NumPy array, NumPy generator/bit-generator, Gym space, environment, simulator,
component, registry, ledger, event log, encoder, adapter, metrics, `PlantState`, and
`ControlStateV2`. Shared Python scalar, enum, function, class, immutable tuple, and the
explicit path-list identities above are allowed. It also requires each cloned
`simulator.components[name]` to be the same cloned object as the corresponding named
simulator attribute, never the source component.

If these checks cannot be met, the optimization is rejected rather than silently sharing
operational state. The clone-contract version and exact allowlist identity are included
in planner settings; clean source provenance binds the implementation itself.

### Exact same-depth deduplication

Version 1 uses **no approximate bins or quantization**. Candidates are deduplicated only
when their complete canonical same-depth state-signature payload is byte-identical.

The signature serializer recursively maps `None`, booleans, integers, and strings to
themselves; maps each finite float to `{"$float64": value.hex()}`; maps enum values to
their declared string; maps tuples/lists in order; maps string-keyed mappings in sorted-
key order; and maps dataclasses by declared field name. It rejects every other type and
all non-finite floats. The resulting object is encoded with the frozen canonical-JSON
rule. This preserves signed zero and every finite binary64 value.

The state-signature object has exactly these top-level fields:

```text
schema_version = 1
signature_scope = "planner-step-action-future-v1"
manifest_sha256
exogenous_episode_sha256
checkpoint_contract_digest
settings_sha256
action_library_sha256
macro_depth
physics_steps_taken
decisions_taken
episode_physics_steps
episode_decisions
simulator
control_state
metrics_fields
encoder_previous_solar_kw
battery_throughput_kwh
```

`simulator` has exactly:

```text
scenario_id
status_memory
step_index
time_s
energy_produced_kwh
energy_curtailed_kwh
h2_produced_total_kg
co2_captured_total_kg
ch4_produced_total_kg
truth_state
telemetry_state
component_snapshots
commitment_snapshot
fault_scope = "canonical_fault_free_v1"
fault_registry
event_log
delivery_ledger
```

`scenario_id` reads `env.simulator.scenario`; `status_memory` reads the private
`env.simulator._status`; `truth_state` reads `env.simulator.state`; and
`telemetry_state` reads `env.telemetry`. `component_snapshots` is exactly an array of
`[name, component.snapshot()]` pairs in this order: `solar`, `battery`, `electrolyser`,
`h2_tank`, `dac`, `co2_tank`, `methanator`, `ch4_tank`. `commitment_snapshot` reads
`env.simulator.electrolyser_commitment`.

The remaining nested objects use these exact keys and array shapes before recursive exact
serialization:

```text
fault_registry = {
  "faults": tuple(env.simulator.faults._faults),
  "sequence": env.simulator.faults._sequence
}
event_log = {
  "events": tuple(env.simulator.events._events),
  "conditions": tuple(sorted(env.simulator.events._conditions.items())),
  "sequence": env.simulator.events._sequence,
  "time_s": env.simulator.events.time_s,
  "clock": env.simulator.events.clock,
  "maxlen": env.simulator.events._events.maxlen
}
delivery_ledger = {
  "frozen": ledger._frozen,
  "schedule": ledger._commitments,
  "pending": tuple((item.id, ledger._pending[item.id])
                   for item in ledger._commitments if item.id in ledger._pending),
  "by_step": tuple((step, ledger._by_step[step])
                   for step in sorted(ledger._by_step)),
  "order": tuple(sorted(ledger._order.items(), key=lambda pair: pair[1])),
  "settlements": tuple(ledger._settlements),
  "current": ledger._current,
  "cumulative_requested_kg": ledger._cumulative_requested_kg,
  "cumulative_delivered_kg": ledger._cumulative_delivered_kg,
  "cumulative_unmet_kg": ledger._cumulative_unmet_kg
}
```

The registered fault-free scope requires `fault_registry` to serialize as exactly
`{"faults":[],"sequence":0}`; any other value is ineligible. `metrics_fields` is an
object containing exactly every `EpisodeMetricsV2` dataclass field by declared field
name, not derived properties. `control_state`, truth/telemetry states, components,
events, commitments, and metrics are all passed through the exact recursive serializer
above; tuple/list values consequently become ordered JSON arrays and dataclasses become
objects keyed by declared field name.

The encoder field is exactly the nested V1 encoder's previous-solar value. Environment,
space, and adapter RNG/bound arrays are copied for clone isolation but omitted from this
planner-only signature because the search permits only `step_action` and never calls
`reset`, `step`, `seed`, or `sample`; the manifest/checkpoint binds their static contract.

The signature's SHA-256 is retained for audit, but equality uses the canonical bytes, not
the digest alone. Private future truth need not be duplicated in each signature because
the manifest and current step bind it. Failure to build this exact payload fails the run;
deduplication may not be disabled and may never fall back to a partial `PlantState` key.

When duplicate candidates are found, retain the higher intermediate rank defined below;
if that rank is equal, retain the exact A0 prefix when present, otherwise the
lexicographically smaller source-token sequence, then the lexicographically smaller
concrete-request bytes. The outcome may not depend on mapping, set, object-identity, or hash
iteration order.

### Intermediate rank

All candidates compared at a beam cut have the same decision depth. Let the next pending
commitment request be `Q`, `T` be exact hours until its due step, and `Cm` the active
methanator cap. When no commitment remains, return `deadline_readiness=0.0` immediately
without evaluating `T` or conversion capacity. Otherwise define:

```text
next_pending = simulator.delivery_ledger.next_pending
Q = next_pending.requested_kg
remaining_steps = next_pending.due_step - simulator.step_index
T = float(remaining_steps) * config.dt_h
H2 = simulator.state.h2_level_kg
CO2 = simulator.state.co2_level_kg
CH4 = simulator.state.ch4_level_kg
feedstock_potential = min(H2 / h2_per_ch4, CO2 / co2_per_ch4)
conversion_capacity = Cm * T / methanator_specific_energy
deadline_readiness = min(Q,
                         CH4 + min(feedstock_potential,
                                   conversion_capacity))
```

Stored methane needs no future conversion. This readiness ignores future process energy
and is only a pruning heuristic.

Every other intermediate field has one authoritative source: cumulative delivery,
service counts, worst fill, startup counters, and curtailed energy come from the named
`EpisodeMetricsV2` fields; inventories, battery energy, commitment mode, and timers come
from the simulator truth state and exact electrolyser commitment snapshot; throughput
comes from the planner-owned accumulator. `remaining_steps` must be positive. Any
non-finite value or disagreement between environment/simulator step counters fails state
signature construction.

Rank candidates lexicographically by:

1. higher `cumulative delivered kg + deadline_readiness`;
2. higher cumulative delivered kg;
3. higher exact service fraction `commitments_met / commitments_settled`, treating no
   settlements as `1/1` and comparing integer cross-products;
4. higher worst commitment fill;
5. higher on-hand methane;
6. higher feedstock potential;
7. higher battery energy;
8. commitment-mode readiness in this fixed order:
   `running`, `hot_standby`, `hot_starting`, `cold_starting`, `off`, `cooldown`;
9. lower startup remaining, cooldown remaining, and shortfall elapsed time, in that
   order;
10. fewer forced stops, aborted starts, and cold starts, in that order;
11. lower curtailed energy;
12. lower battery throughput; and
13. the deterministic path tie-break described above.

This rank is only a pruning heuristic. It is not the task score, a value estimate, or a
bound. The shaped RL reward is deliberately absent.

### Commitment-mode protection and frontier selection

At each nonterminal depth, add the exact A0 prefix to the expanded-candidate pool before
exact deduplication; it consumes no attempted-edge count and wins a same-rank duplicate
as specified above. After deduplication, identify the highest-ranked candidate in each mode
in the fixed commitment order `off`, `cold_starting`, `running`, `hot_standby`,
`hot_starting`, `cooldown`. Build the next beam as follows:

1. insert the surviving exact A0 prefix candidate;
2. insert each mode winner whose state signature is not already represented;
3. fill remaining slots by global intermediate rank; and
4. stop at width 24.

If A0 already represents a mode winner, it satisfies both protections. Mode lanes do not
increase beam width. This prevents every delayed-start precursor from being discarded
only because it has no immediate production while preserving the incumbent guarantee.

The terminal macro depth is different: it does not apply intermediate rank, mode lanes,
or a width-24 terminal cut. Every completed terminal child from every retained parent is
stream-compared with the full final comparator after exact state deduplication. Retain the
single best searched terminal trace and the separate exact A0 incumbent. This prevents a
pruning heuristic whose purpose has ended from discarding the best generated final-score
trace.

## Frozen block-coordinate polishing

Beam search yields one best complete non-A0 trace under the final score when such a trace
survives; otherwise only A0 is polished. Polishing uses exact full-horizon simulation and
concrete action traces; it never edits simulator state directly.

1. Start independently from A0 and the best beam trace. If their concrete-request bytes
   are identical, evaluate the trace once.
2. Make one left-to-right coarse sweep over non-overlapping 24-decision (six-hour)
   blocks `[start, min(start + 24, decision_count))`. For each block, try all 15 template
   IDs in order against the same immutable block-start trace. The template is regenerated
   at each decision inside the replaced block from that trial's evolving state. Before
   and after the block, apply that block-start trace's stored concrete actions unchanged.
3. After all 15 trials complete, choose the highest ten-field final score; searched ties
   use the frozen source/action tie-break. Accept that one trial only when its ten-field
   score is strictly higher than the block-start trace. Equal scores retain the block-
   start trace regardless of tie-break. Continue the sweep from the accepted concrete
   trace.
4. Select the better of the two coarse-polished traces. Only an unchanged exact A0 trace
   wins an equal-score incumbent tie; an A0-origin trace with an accepted replacement is
   searched and uses the global searched-candidate tie-break.
5. Make one left-to-right fine sweep over non-overlapping blocks
   `[start, min(start + 4, decision_count))` using the same 15-template order,
   regeneration, common block-start snapshot, unchanged-suffix, best-trial, and strict-
   improvement rule.

For a registered 72-hour episode this schedules at most
`2 * 12 * 15 + 72 * 15 = 1,440` coordinate trials before exact-duplicate starting-trace
elision. There is no second sweep, reverse sweep, adaptive block, outcome-dependent
neighborhood, or early stopping after improvement.

## Final score and selection

Every complete trace is ordered by this exact lexicographic task score:

1. higher cumulative delivered CH4 kg;
2. higher commitment service fraction, compared exactly by integer cross-products;
3. higher worst commitment fill;
4. higher terminal
   `CH4 + min(H2 / h2_per_ch4, CO2 / co2_per_ch4)`;
5. higher terminal battery energy;
6. fewer electrolyser forced stops;
7. fewer aborted starts;
8. fewer cold starts;
9. lower curtailed energy; and
10. lower battery throughput.

Battery throughput is accumulated from each `step_action` result as:

```text
interval_power.applied_kwh["battery"]
    + interval_power.applied_kwh["battery_support"]
```

This counts bus-side charge plus discharge. `battery_discharged_kwh` alone is not used.
Starting from binary64 `0.0`, the implementation performs one left-to-right `+=` of that
decision's battery term and then its battery-support term after every successful
`step_action`; it does not use a reordered or compensated sum. Search and replay call the
same routine.
Reward, energy intensity, guard intervention, runtime, and LP objective do not enter the
score.

The final-score extraction is exact and uses no interchangeable fallback source:

```text
delivered_kg = metrics.delivered_kg
service_numerator = metrics.commitments_met
service_denominator = metrics.commitments_settled
worst_fill = metrics.worst_commitment_fill_rate
terminal_ch4_equivalent_inventory_kg =
    state.ch4_level_kg
    + min(state.h2_level_kg / config.methanator.h2_kg_per_kg_ch4,
          state.co2_level_kg / config.methanator.co2_kg_per_kg_ch4)
terminal_battery_energy_kwh = state.battery_energy_kwh
forced_stops = metrics.electrolyser_forced_stops
aborted_starts = metrics.electrolyser_aborted_starts
cold_starts = metrics.electrolyser_cold_starts
curtailed_kwh = metrics.curtailed_kwh
battery_throughput_kwh = planner accumulator
```

Here `metrics=env.metrics` and `state=env.simulator.state`. Before scoring, require
binary64 equality between each metric terminal-inventory/battery field and simulator
truth, and between `metrics.delivered_kg` and the delivery ledger's
`cumulative_delivered_kg`; disagreement fails the candidate rather than selecting a
different source.

If all ten fields tie exactly, A0 wins over every searched trace. Between searched
traces, the lexicographically smaller complete source-token sequence wins, followed
by the lexicographically smaller canonical concrete-request bytes. A selected replay with
a score below freshly replayed A0 is an `incumbent_regression` failure, never a fallback.

## Fresh replay and evidence record

Search states are not evidence. The finalists are exactly the unmodified A0 trace and the
one final post-fine searched trace, if distinct. Replay each from a brand-new environment
reset with the registered seed and verify the immediate manifest, exogenous hash, and
checkpoint digest before applying any action. Reuse the already verified A0 replay when
A0 is selected; do not create a third evidentiary A0 result. Apply the full serialized
five-field schedule through `step_action`, preserving the recorded invalid flag. Reject
early termination, truncation, missing or extra decisions, an action after termination,
or incomplete physics coverage.

Strict canonical bytes must match between the terminal search evaluation and fresh
replay for:

- complete `EpisodeMetricsV2.to_dict()`;
- terminal simulator truth state and telemetry;
- complete delivery audit and ordered settlements;
- battery throughput;
- decision/physics counts and termination flags; and
- action-value/invalidity trace.

Fresh replay cannot infer why a concrete action was originally selected. The source-code
trace is therefore construction-validated lineage metadata, not independently replay-
verified behavior. Its length/range grammar and hash are revalidated, and it is bound
into the action-record and result digests, while feasibility replay checks only the exact
action values and invalid flags actually applied.

The outcome fingerprint covers all of those payloads plus the manifest, checkpoint, and
action-trace identities. Both bytes and hashes are checked; hashes alone do not establish
replay equality. Elapsed time, process ID, host telemetry, and source-audit timestamps are
excluded from deterministic result identity.

After both fresh replays, require selected final score not below A0. Compare both A0 and
selected delivered kg directly against the same result's non-null
`dual_certified_delivery_upper_bound_kg`. If neither exceeds it, comparison status is
`valid` and the core result enforces `A0 <= F <= U`. If either exceeds it, retain the
independently replayed feasible schedule and its valid `F-A0` gain, set comparison status
to `bound_bridge_violation`, suppress residual-gap/fraction-closed fields, and make the
runner artifact ineligible for `evidence_ready=True`. Never clamp `F` or `U`. This keeps
a real exact-simulator trajectory while explicitly invalidating the two-sided bracket
under Part I's numerical simulator-bridge qualification.

All persisted records are frozen, slotted dataclasses containing only scalars, tuples,
other declared frozen/slotted evidence records, or verified canonical-payload objects. No
live environment, `PlantState`, mutable metrics object, dictionary, list, or NumPy array
may be retained. Constructors defensively tuple-copy iterable input, validate every
invariant, and `to_dict()` creates detached JSON-ready containers. Version 1 records
admit no undeclared extension fields.

### Canonical payload and hash domains

`CanonicalPayloadV1` has exactly `domain: str`, `canonical_json: str`, and
`sha256: str`. Its hash preimage is the result of this exact Python expression:

```python
b"plantanagent:" + domain.encode("utf-8") + b"\0" + canonical_json.encode("utf-8")
```

Construction reparses the JSON, requires the canonical bytes to reproduce exactly, and
recomputes the digest. The allowed domains are exactly:

```text
planner-action-library-v1       planner-settings-v1
planner-a0-controller-v1
planner-relaxation-result-v1    planner-relaxation-witness-v1
planner-relaxation-guidance-v1
planner-request-trace-v1        planner-source-trace-v1
planner-action-record-trace-v1  planner-state-v1
planner-metrics-v1              planner-terminal-truth-v1
planner-terminal-telemetry-v1   planner-delivery-audit-v1
planner-outcome-v1              planner-replay-binding-v1
planner-deterministic-counters-v1
planner-result-v1               planner-source-snapshot-v1
planner-runtime-v1              planner-runner-artifact-v1
```

A record's own digest field is always omitted from its hash preimage. There is no generic
recursive key removal. The `planner-result-v1` preimage starts from
`FeasiblePlannerResultV1.to_dict()` and removes exactly these paths when present:

```text
deterministic_result_sha256
search.elapsed_ns
search.peak_rss_bytes
failure.message
failure.elapsed_ns
failure.observed_resource_value
a0_incumbent.replay_elapsed_ns
searched_finalist.replay_elapsed_ns
```

It retains stable failure phase/code, every other identity, settings, actions, outcomes,
and deterministic counter payload. Wall/RSS observations and replay timings therefore do
not enter the core digest directly; as already disclosed, the completed-edge position of
an emergency resource failure can still be platform-dependent. The embedded relaxation
result payload already excludes only its top-level `elapsed_s` by the earlier rule.

### Exact settings schema

`PlannerSettingsV1` has exactly:

```text
schema_version=1
planner_kind="best-known-feasible-future-information-v1"
action_library: CanonicalPayloadV1
beam_width=24
macro_decisions=4
max_attempted_edges=25575
mode_protection_order=(off,cold_starting,running,hot_standby,hot_starting,cooldown)
intermediate_rank_version="deadline-readiness-lexicographic-v1"
final_score_version="delivery-service-endowment-lexicographic-v1"
coarse_block_decisions=24
coarse_sweeps=1
coarse_origins=(a0_incumbent,beam_winner)
fine_block_decisions=4
fine_sweeps=1
random_policy="none"
clone_contract_version="deepcopy-immutable-memo-v1"
clone_allowlist_sha256="694c1b8259270bc6be719fc65e93af006fb01c2f10f6a6907734e309f49fcbba"
state_signature_version="planner-step-action-future-v1"
wall_limit_ns=5400000000000
rss_limit_bytes=4294967296
clock_method="time.perf_counter_ns"
rss_method="resource.RUSAGE_SELF.ru_maxrss:darwin-bytes:other-kib"
max_live_environments=57
expected_decisions
expected_physics_steps
protocol_path="docs/research/FEASIBLE_PLANNER_PROTOCOL.md"
protocol_parent_commit="2a3c99a73d6e3f071d9721f72691291f307a6125"
protocol_anchor_commit
protocol_blob_sha256
registered_rates=(35.0,42.0)
registered_seeds=(0,1,2)
plant_config_sha256
constraints_sha256
reward_weights_sha256
episode_config_sha256
checkpoint_contract_digest
episode_config_sha256_by_rate
checkpoint_contract_sha256_by_rate
commitment_profile_sha256
a0_controller: CanonicalPayloadV1
relaxation_model_sha256
relaxation_result_sha256
relaxation_witness_sha256
relaxation_guidance_sha256
settings_sha256
```

`settings_sha256` uses the `planner-settings-v1` domain over every preceding field,
representing each nested canonical payload (`action_library` and `a0_controller`) by an
object containing exactly its `domain` and `sha256`. The registered payloads must match
the two frozen digests above.
For privileged development, `expected_decisions=288`, `expected_physics_steps=4320`,
the two `*_by_rate` fields are ordered `(35.0, 42.0)` pairs containing the digests in the
registered-scope table, and the singular episode/checkpoint fields select the pair for
the current rate. Noncanonical fixtures store their actual singular digests/counts but
retain the registered maps; their core scope and writer rejection prevent promotion.

Both map fields have Python type `tuple[tuple[float, str], ...]` and are encoded in the
settings hash as these exact JSON arrays (never objects keyed by a formatted float):

```json
{"checkpoint_contract_sha256_by_rate":[[35.0,"4ef8f7c92c26d0ea5cd0b6a1b4dc7a0384c9776fe265838b081887f760b91a0a"],[42.0,"5605c2bf5e99bb1bbec584e29f23773d5233b5eadcdcd11f269bdf03d12adfb0"]],"episode_config_sha256_by_rate":[[35.0,"115612216fc1ad749e6505cc0ea567265222695243078690b19095b3d750b7d7"],[42.0,"82838f8f36af09ab09d705292b1ecbe203220fbd7d6ebe0790214912d6d07649"]]}
```

That fragment documents the representation; the settings preimage still contains all
fields in the exact schema. `PlannerSettingsV1.to_dict()` embeds the full three-field
canonical-payload objects for `action_library` and `a0_controller`, while only the
settings-hash preimage replaces each with its exact `{domain, sha256}` pair as stated.

The protocol-only commit immediately following the frozen
`protocol_parent_commit` becomes `protocol_anchor_commit`. Before the first planner code
is committed, that full 40-hex commit ID and the raw SHA-256 of the protocol file bytes at
that commit are installed as immutable implementation constants. Registered validation
requires all of the following: the anchor's sole parent is the frozen parent above; its
changed paths are exactly `docs/research/FEASIBLE_PLANNER_PROTOCOL.md`,
`docs/research/JOURNAL.md`, `docs/research/PLANNING_BENCHMARK_PROTOCOL.md`, and
`docs/research/README.md`; the anchor is an ancestor of the clean execution revision; and
the working protocol bytes, anchor protocol bytes, implementation constant, and settings
`protocol_blob_sha256` are identical. `protocol_anchor_commit` and
`protocol_blob_sha256` enter the settings hash. A later clean edit therefore requires a
new protocol version rather than silently passing under version 1.

### Exact nested evidence schemas

`ActionRecordV1` has the ten fields and grammar declared in the action-library section.
`FinalScoreV1` has exactly `delivered_kg`, `service_numerator`, `service_denominator`,
`worst_fill`, `terminal_ch4_equivalent_inventory_kg`, `terminal_battery_energy_kwh`,
`forced_stops`, `aborted_starts`, `cold_starts`, `curtailed_kwh`, and
`battery_throughput_kwh`. It validates the service fraction and supplies the frozen
comparison rather than collapsing it to a scalar tuple.

`ReplayEvidenceV1` has exactly:

```text
role: a0_incumbent | searched_finalist
manifest_sha256
exogenous_episode_sha256
checkpoint_contract_digest
action_records: tuple[ActionRecordV1]
request_trace: CanonicalPayloadV1
source_trace: CanonicalPayloadV1
action_record_trace: CanonicalPayloadV1
decision_count
physics_steps
terminated
truncated
battery_throughput_kwh
metrics: CanonicalPayloadV1
terminal_truth: CanonicalPayloadV1
terminal_telemetry: CanonicalPayloadV1
delivery_audit: CanonicalPayloadV1
generation_outcome: CanonicalPayloadV1
replay_outcome: CanonicalPayloadV1
replay_binding: CanonicalPayloadV1
final_score: FinalScoreV1
replay_elapsed_ns
```

The trace payload shapes are exact. `planner-request-trace-v1` contains an object with
`schema_version=1` and `records`, where `records` is an array of objects containing
exactly `decision_index`, `controller_invalid`, `battery_share`, `electrolyser_share`,
`dac_share`, `methanator_share`, and `battery_support_kw`. `planner-source-trace-v1`
uses the same wrapper and rows containing exactly `decision_index`, `source_phase`,
`source_code`, and `source_binding_sha256`. `planner-action-record-trace-v1` uses the
same wrapper and the complete ten-field row. Each is constructed directly from the same
ordered `ActionRecordV1` tuple; no source field enters the request trace.

`planner-metrics-v1` contains exactly `EpisodeMetricsV2.to_dict()`;
`planner-terminal-truth-v1` and `planner-terminal-telemetry-v1` contain exactly the
corresponding `PlantState.to_dict()`; and `planner-delivery-audit-v1` contains exactly
`env.simulator.delivery_ledger.audit()`. Their canonical JSON has no extra wrapper.
`planner-outcome-v1` contains exactly:

```text
schema_version=1
manifest_sha256
exogenous_episode_sha256
checkpoint_contract_digest
request_trace_sha256
metrics_sha256
terminal_truth_sha256
terminal_telemetry_sha256
delivery_audit_sha256
battery_throughput_kwh
decision_count
physics_steps
terminated
truncated
```

Generation and replay outcome canonical JSON must be byte-identical.
`planner-replay-binding-v1` contains exactly:

```text
schema_version=1
role
settings_sha256
relaxation_guidance_sha256
manifest_sha256
exogenous_episode_sha256
checkpoint_contract_digest
request_trace_sha256
source_trace_sha256
action_record_trace_sha256
outcome_sha256
```

All trace lengths equal `decision_count`; indices are contiguous from zero; success
requires true termination, false truncation, and the settings' expected physics/decision
counts. The privileged development scope additionally requires 4,320 physics steps and
288 decisions.

`SearchDiagnosticsV1` has exactly `completed_macro_depths`, `attempted_edges`,
`completed_edges`, `generated_candidates`, `a0_cursor_validations`, `a0_injections`, `exact_duplicates`,
`frontier_retained_total`, `peak_frontier`, `coordinate_trials_attempted`,
`coordinate_trials_completed`, `coordinate_trials_accepted`,
`a0_generation_decisions`, `a0_evidence_replay_decisions`,
`fresh_finalist_replay_decisions`, `max_live_environments_observed`,
`termination_reason`, `deterministic_counters: CanonicalPayloadV1`, `elapsed_ns`, and
`peak_rss_bytes`. Counter semantics are those frozen in the resource section.
Successful registered execution has exactly 72 `a0_cursor_validations` and 71
`a0_injections`. `termination_reason` is a string equal to exactly `"completed"` on
success and exactly the associated `PlannerFailureV1.code` on core failure; no other
value is valid.

The `planner-deterministic-counters-v1` payload contains `schema_version=1` followed by
exactly every `SearchDiagnosticsV1` field through `termination_reason` in the order named
above, excluding only `deterministic_counters`, `elapsed_ns`, and `peak_rss_bytes`.
Canonical key sorting, not declaration order, determines its bytes.

`PlannerFailureV1` has exactly `phase`, `code`, `message`, `completed_macro_depths`,
`attempted_edges`, `completed_edges`, `coordinate_trials_completed`, `elapsed_ns`, and
`observed_resource_value`. Allowed phases are exactly `input_validation`,
`relaxation_binding`, `a0_generation`, `a0_replay`, `clone_preflight`, `beam_search`,
`coarse_polish`, `fine_polish`, `finalist_replay`, `bracket_validation`, and
`result_serialization`. Allowed codes are exactly `invalid_input`,
`noncanonical_episode`, `fault_scope_violation`, `relaxation_unavailable`,
`relaxation_binding_mismatch`, `nonfinite_value`, `manifest_mismatch`,
`checkpoint_mismatch`, `a0_generation_error`, `a0_replay_mismatch`, `clone_error`,
`clone_contract_violation`, `action_generation_error`, `action_not_closed`,
`simulator_step_error`, `premature_termination`, `unexpected_truncation`,
`incomplete_horizon`, `state_signature_error`, `attempted_edge_limit`,
`live_environment_limit`, `wall_time_limit`, `rss_limit`, `memory_error`,
`search_incomplete`, `coordinate_error`, `replay_mismatch`, `incumbent_regression`,
`result_invariant_error`, and
`serialization_error`. `observed_resource_value` is a non-negative integer or null.

Both core and runner failure messages use the same exact sanitizer: compute
`" ".join(str(value).split())`, replace every character outside ASCII code points
32--126 with `?`, truncate to the first 240 characters, and substitute
`"unspecified failure"` if the result is empty. The stored message must equal that
function's output. Core messages are excluded from deterministic identity as declared;
runner messages remain ordinary content in the non-deterministic artifact identity.

`IncumbentComparisonV1` has exactly `selected_role`, `scores_equal`,
`actions_equal_to_a0`, `a0_delivered_kg`, `selected_delivered_kg`, `gain_kg`,
`upper_bound_kg`, `bracket_status`, `residual_gap_kg`, and `fraction_closed`.
`bracket_status` is exactly `valid` or `bound_bridge_violation`.
`residual_gap_kg` and `fraction_closed` are null on a bridge violation;
`fraction_closed` is also null when the valid Part-I denominator does not exceed its
frozen optimality tolerance.
For a valid bracket, evaluate in this order:

```text
gain_kg = selected_delivered_kg - a0_delivered_kg
residual_gap_kg = upper_bound_kg - selected_delivered_kg
denominator_kg = upper_bound_kg - a0_delivered_kg
fraction_closed = (null if denominator_kg <= optimality_tolerance_kg
                   else gain_kg / denominator_kg)
```

`scores_equal` applies the complete ten-component final comparator and
`actions_equal_to_a0` compares complete canonical request-trace bytes.

### Exact core result schema and invariants

`FeasiblePlannerResultV1` has exactly:

```text
schema_version=1
planner_kind="best-known-feasible-future-information-v1"
evidence_scope: privileged_offline_development | noncanonical_unit_fixture
privileged_information="complete private episode truth and commitments"
optimality_claim="none"
evidence_ready=False
provenance_gaps=("clean start/end source and dependency-lock audit required",)
success
failure: PlannerFailureV1 | null
seed: int | null
canonical_episode_verified
manifest_sha256: str | null
exogenous_episode_sha256: str | null
checkpoint_contract_digest: str | null
relaxation_model_sha256: str | null
relaxation_result_sha256: str | null
relaxation_result: CanonicalPayloadV1 | null
relaxation_witness: CanonicalPayloadV1 | null
relaxation_guidance: CanonicalPayloadV1 | null
settings: PlannerSettingsV1 | null
search: SearchDiagnosticsV1
a0_incumbent: ReplayEvidenceV1 | null
searched_finalist: ReplayEvidenceV1 | null
selected_role: a0_incumbent | searched_finalist | null
comparison: IncumbentComparisonV1 | null
deterministic_result_sha256
```

On success, failure is null; A0, selected role, and comparison are present; the selected
role resolves to a present replay record; every manifest/checkpoint/settings/guidance
identity agrees; every present final replay validates; selected score is not below A0;
and the comparison records either a valid bracket or an explicit bridge violation. A valid
no-improvement result selects A0. `searched_finalist` is present whenever the one final
post-fine searched trace has concrete-request bytes distinct from A0, regardless of
whether it improves A0; it is null only when that finalist is byte-identical to A0 or no
searched terminal trace exists.

On success, the embedded `relaxation_result` has domain
`planner-relaxation-result-v1`, its SHA-256 equals `relaxation_result_sha256`, and its
canonical JSON supplies the exact non-null `dual_certified_delivery_upper_bound_kg` used
by `comparison.upper_bound_kg`. The writer reparses and checks that relationship; a hash
or bound cannot be revalidated from a detached scalar reference alone.

On failure, failure is present and `searched_finalist`, `selected_role`, and `comparison`
are null. Identity/settings fields contain only values successfully validated before the
failure and are null when that phase could not establish them; this makes early
`invalid_input`, `relaxation_unavailable`, and binding failures serializable without
invented identities. A completely verified A0 record may remain diagnostic; a partial A0
record may not. `evidence_ready` is always false in the core object. The deterministic
result digest uses the `planner-result-v1` domain over all deterministic fields above
except itself and the explicitly excluded diagnostics. Construction and any writer
revalidate the complete object graph and all hashes.

### Runner provenance wrapper

The later runner never mutates a core result. `FeasiblePlannerArtifactV1` wraps it with
exact fields `schema_version=1`, `success`, `evidence_ready`, `core_result`,
`core_result_sha256`, `runner_failure`, `start_source`, `end_source`,
`start_lock_sha256`, `end_lock_sha256`, `runtime_versions`, `source_boundary_claim`, and
`artifact_sha256`. `core_result_sha256`, when present, must equal the embedded core's
`deterministic_result_sha256`.
Start provenance is captured before constructing or solving the Part-I relaxation; end
provenance is captured after the final core result has been serialized. Source payloads
are immutable canonical snapshots.

`start_source` and `end_source` are nullable `CanonicalPayloadV1` values with domain
`planner-source-snapshot-v1`. The runner obtains the source data using
`evaluation.collect_provenance` and admits a snapshot only when its `source` member has
`identity_complete=True`. The canonical snapshot JSON contains exactly:

```text
schema_version=1
git_revision
git_dirty
tracked_binary_diff_sha256
tracked_binary_diff_bytes
untracked_files
worktree_identity_format="git-worktree-content-v1"
worktree_identity_sha256
identity_complete=true
```

`untracked_files` is the path-sorted array from that collector; admitted entries contain
exactly `path`, `kind` (`file` or `symlink`), `content_sha256`, and `bytes`. Repository
absolute paths and Git command arguments are excluded. Incomplete source data produces a
runner failure rather than a partial canonical source snapshot. Each lock field is the
raw SHA-256 of repository-root `uv.lock` bytes at the corresponding boundary.

`runtime_versions` is a nullable `CanonicalPayloadV1` under `planner-runtime-v1`. Its
canonical JSON contains exactly `schema_version=1`, `python_version`,
`python_implementation`, `python_compiler`, `numpy_version`, `scipy_version`,
`highs_runtime_version`, `gymnasium_version`, `platform_system`, `platform_release`,
`platform_version`, `platform_machine`, `platform_processor`, `pointer_bits`,
`byteorder`, `clock_method`, and `rss_method`. Only `highs_runtime_version` and
`platform_processor` may be null. `source_boundary_claim` is non-null only when both
source snapshots are complete and their canonical JSON is byte-identical; it is otherwise
null and the artifact is not evidence-ready. When non-null, it is exactly:

```text
matching complete content identities were observed immediately before relaxation construction and immediately after core serialization; no claim is made about undetected change-and-restoration within the interval
```

`PlannerRunnerFailureV1` has exactly `phase`, `code`, `message`, `elapsed_ns`, and
`child_exit_code`. Its phases are `start_source`, `child_execution`, `end_source`, and
`artifact_serialization`; its codes are `source_unavailable`, `wall_time_limit`,
`child_exit`, `child_protocol_error`, and `serialization_error`. `elapsed_ns` is a
non-negative integer, `child_exit_code` is an integer or null, and `message` uses the
shared frozen sanitizer above. When the child returns a
validated core result and no later runner-level failure occurs, `core_result` and its
digest are present and `runner_failure` is null, including when the core itself records
an ordinary planner failure. Artifact `success` is true exactly when `runner_failure` is
null and `core_result.success` is true. On runner failure, `core_result` and its digest
are either both present (when a validated child result was available) or both null,
`runner_failure` is present, `success=False`, and `evidence_ready=False`.
`start_source`, `end_source`, lock hashes, and runtime versions are nullable only when the
stable runner-failure phase proves they could not be completed; in particular
`phase="start_source"` permits every boundary field to be null. The boundary claim is
also null for two complete but unequal source snapshots, without inventing a runner
failure. A killed child produces a valid failed wrapper without inventing an
untrustworthy core result.

An evidence-ready artifact requires a successful canonical core result with
`bracket_status="valid"`, complete and
clean start/end source identities, equal start/end content/worktree hashes, equal
dependency-lock hashes, the committed protocol/settings identity, and complete Python,
NumPy, SciPy, HiGHS-when-exposed, Gymnasium, platform, and architecture versions. The
boundary requires byte-identical start/end source canonical JSON and establishes matching
observations at its two samples; it does not claim that a
file could not have changed and been restored between them. `artifact_sha256` uses the
`planner-runner-artifact-v1` domain over `FeasiblePlannerArtifactV1.to_dict()` with only
the top-level `artifact_sha256` removed; all nested records are embedded in full. Only
this wrapper may set `evidence_ready=True` or be retained as benchmark evidence.

The validating writer requires an absolute destination outside the detected Git
repository, writes one temporary sibling, flushes and `fsync`s it, atomically replaces
the destination, then reopens and fully revalidates the artifact. Registered cells are
all written externally before any selected small result is copied into Git in a later
separate reporting commit. This prevents one cell's output from dirtying the next cell's
start-source boundary.

## Resource limits and failure semantics

The deterministic algorithmic limits are beam width 24, macro length four decisions,
15 templates, 25,575 attempted beam edges, the frozen coordinate-trial schedule, and a
maximum of **57 live `PlantEnvV2` objects** during beam expansion: up to 24 unexpanded
current-frontier parents, 31 next-frontier candidates comprising a global top-24
reservoir plus as many as six distinct mode winners and the distinct injected A0 clone,
one rolling A0 cursor, and one transient child. References shared between these logical
reservoirs count as one live object, but the limit does not rely on such overlap.
Coordinate and replay phases may hold at most two live environments. Candidate
materialization must not retain all expanded environments; both the top-24 reservoir and
each mode lane are maintained by the same frozen comparator under this ceiling, then
their union is passed through the declared frontier-selection order.

Wall time and resident memory are emergency validity checks, not optimization budgets.
For one registered cell, freeze **5,400 seconds** and **4 GiB peak RSS**. Check them at
reset, every completed macro edge, every completed coordinate trial, and before each
fresh replay. Record monotonic-clock and platform RSS measurement methods. If peak RSS
cannot be measured on the execution platform, fail before search. Version 1 measures
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, interpreting the value as bytes on Darwin
and KiB elsewhere before converting to bytes. Elapsed time is
`time.perf_counter_ns() - start_ns`; the limit is crossed when this value is strictly
greater than `5_400_000_000_000` ns. Benchmark execution uses a fresh process, so RSS is
the absolute process-lifetime peak rather than an increment from an arbitrary baseline.
The child records `start_ns` as the first statement of its worker entry point, before
episode sampling, relaxation-model construction, or the one solver call, and passes that
same origin through every core resource check. The parent starts the external watchdog
immediately before spawning that child, so process startup is also covered externally.
The same injected clock/RSS interface may be used by unit tests, but registered execution
must identify and use these production methods. Limits are also checked after every
fresh-replay decision and after final serialization. Crossing either limit,
`MemoryError`, incomplete expansion, clone/generator/step error, manifest or checkpoint
mismatch, non-finite state, unexpected fault, or replay mismatch fails the complete run.

These boundary checks do not claim that the process is forcibly interrupted at exactly
5,400 seconds while one Python/solver operation is in progress. The runner also owns an
external subprocess watchdog at 5,400 seconds; if it fires, it records a
`wall_time_limit` `PlannerRunnerFailureV1` with a null core result because a killed child
cannot supply a trustworthy core result.

Counters have exact event semantics. `attempted_edges` increments immediately before the
clone attempt for one ordered parent/template macro. `completed_edges` and
`generated_candidates` increment only after every decision in that macro succeeds and its
signature is built. `a0_cursor_validations` increments whenever the rolling cursor
matches one stored prefix; `a0_injections` increments only when its nonterminal clone
enters the candidate pool.
`exact_duplicates` increments whenever a
completed candidate loses or replaces one already represented by the same canonical
state bytes. `frontier_retained_total` adds the selected next-beam length after each cut.
`coordinate_trials_attempted` increments before a trial reset,
`coordinate_trials_completed` after a complete terminal evaluation, and
`coordinate_trials_accepted` only on strict score improvement. Replay-decision counters
increment only after successful `step_action`. A partial failed edge is attempted but not
completed/generated; no edge error is skipped.

A failed run has no `selected`, feasible-planner value `F`, comparison, gain, or fraction
closed. It may retain the independently verified A0 replay and bounded diagnostics:
phase, stable failure code, sanitized message, completed depth, attempted/generated/
deduplicated counts, peak frontier, elapsed seconds, and observed resource value. A
timeout-truncated best-so-far trace is not published. There is no hidden heuristic
fallback under the planner label.

A completed search with no strict improvement is valid: `selected` is the exact A0
replay, `F=A0`, and the report says that no gain was demonstrated under version 1. That
is distinct from search failure.

## Required pre-outcome validation

Only one-to-eight-decision exact-simulator fixtures, hand-solvable relaxation fixtures, or
outcome-free algorithmic test doubles may be used before the code freeze. Tests must
establish all of the following:

1. canonical eligibility and fault rejection;
2. stable settings, library, action, source-lineage, state, outcome, and result hashes;
3. template order/formulas, completed-macro duplicate elimination, share closure, support
   sustainability, final partial holds, sunrise routing, and projection binding;
4. optimized-clone isolation, nested alias resistance, ordinary-deepcopy equivalence,
   sibling independence, and deterministic equal-clone evolution;
5. exhaustive enumeration agreement on a tiny tree when beam width covers every path;
6. exact A0 generation/replay parity and A0 survival through every beam cut;
7. survival of delayed cold-start paths through mode protection;
8. state signatures that change independently with every inventory, timer, previous-
   power, ledger, prior-interval, metric-history, event, and throughput field;
9. exact duplicate survivor and frontier ordering independent of insertion order;
10. direct binary64 score ordering including one-ULP differences, exact rational service
    comparison, every lower-priority score field, and all tie rules;
11. coordinate block boundaries, one-sweep behavior, strict acceptance, fixed suffixes,
    and no action after termination;
12. exact terminal replay across startup, settlement, sunrise, and final-partial-block
    fixtures;
13. rejection of changed/reordered/omitted/appended actions, one-ULP terminal changes,
    wrong seed/config/checkpoint/manifest, and altered delivery audit;
14. throughput accounting that includes a charge-then-discharge sequence;
15. phase-specific fail-closed behavior for edge ceiling, timeout, RSS, `MemoryError`,
    clone, generator, step, source, and replay failures;
16. immutable/detached result records and writer revalidation before any file is created;
    and
17. rejection of this privileged object by the causal-controller evaluator.

Focused tests, the complete repository suite, compilation, formatting, lock validation,
and `git diff --check` must pass on clean committed source before the first six
development cells are executed.

## Decision after the six development cells

For each cell report A0, replay-verified `F`, the Part-I delivery ceiling `U`,
`F-A0`, `U-F`, and `(F-A0)/(U-A0)` only when the denominator clears the Part-I frozen
numerical tolerance and bracket status is valid. Report complete terminal balances and
operational side effects.

- Material `F>A0` with residual `U-F` authorizes a causal service-MPC implementation and
  identifies an action-space scheduling target.
- `F=A0<U` requires search/action-interface diagnosis before concluding that a causal
  optimizer has no value.
- `F` near `U` means the current exact environment is close to saturated by privileged
  planning and should be made more discriminative before claiming RL value.
- A `bound_bridge_violation` preserves the replayed feasible result but yields no valid
  upper bracket, residual gap, or fraction closed; investigate Part I before MPC gating.
- A failure yields no bracket and must be repaired or preregistered as an explicit
  benchmark limitation.

Any threshold for “material” or “near” that will select a prospective claim must be
frozen in the later prospective protocol. These six cells remain engineering evidence,
not field, financial, held-out, or commission evidence.
