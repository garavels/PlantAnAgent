# Plant An Agent

An autonomous, intermittently powered industrial plant — starting with the foundation:
**a small working digital twin plus a control room that watches it.**

The plant today (V7 — the full simplified synthetic-methane chain, everything that can go
wrong with it, a backend that runs it, a supervisory agent that watches it, and the RL
foundation a learned controller plugs into):

```text
                 Solar
                   ↓
            Power allocation (DC bus)
             ├──→ Battery ─────────┐ (support)
             ├──→ Electrolyser ──→ H₂ Tank ──┐
             ├──→ DAC ──────────→ CO₂ Tank ──┤
             │                               ↓
             └──────────(power)──────→ Methanator ──→ CH₄ Tank
```

Every arrow is a real dependency in the simulator: starve the DAC and the CO₂ tank
drains until methanation stops; run the stack hard and the H₂ tank fills; methane only
exists when H₂, CO₂ *and* power show up together.

Since V4 the plant can also *misbehave*: faults derate or trip individual units, cloud
cover cuts generation, a sensor can report a level the plant does not actually have, and
the simulator emits events as it notices the consequences. Scenarios replay those
situations on demand.

Since V5 the plant runs itself. The backend owns the clock, a heuristic controller drives
the whole chain unattended, and every action — its own or an operator's — passes one
validation guard before the simulator sees it. The control room asks and watches; it no
longer drives.

Since V6 the plant is also *supervised*. A supervisory agent sits beside the controller,
not inside it: it sleeps through routine operation, wakes on a situation (a warning, a
critical event, a status change, a collapse in output, or an operator asking), investigates
with a small toolset, and retunes the operating envelope — the same lever a human operator
has. It never allocates a kilowatt and never touches the simulator.

Since V7 the plant is also an **RL environment**. `PlantEnv` wraps the simulator with a
normalised observation, a five-number action and a reward built from the plant's own
counters; PPO training, checkpointing and a deterministic heuristic-vs-policy comparison are
wired up end to end. **No policy has been trained** — V7 is the foundation, and the backend
keeps the heuristic controller unless a compatible checkpoint is loaded successfully.

Since V9 the plant can also be *asked about the future*. `backend/src/plant_backend/hepa/`
implements HEPA — a horizon-conditioned event-predictive architecture — end to end:
telemetry collection, self-supervised pretraining, event fine-tuning, evaluation and two
advisory HTTP routes that answer *P(event within Δt)*. **Nothing has been pretrained or
fine-tuned**: this environment has no useful GPU, so V9 ships the infrastructure and the
tests, and the backend runs exactly as V7 did until a checkpoint is configured.

Since V10 the forecast is *part of the supervision*. The supervisor asks HEPA — on its own
cadence, through one service door — what is becoming likely, turns the answer into levels
and trends, and investigates a credible risk **before** the event: it corroborates the
prediction against telemetry and the units involved, and only then retunes the envelope the
controller drives inside. HEPA still only predicts. With no checkpoint it reports
`unavailable`, raises nothing, and the plant behaves exactly as it did in V8.

The control room has three views over the same live session: **Overview** (the concise
operational summary, plus scenario/fault controls and the event feed), **3D Plant**
(a spatial digital twin, where faulted units turn amber or red under a warning beacon) and
**Agent** (what the supervisor concluded, what it changed, and the trace behind it).

---

## Repository structure

```text
/
├── simulator/          the plant. physics and state, pure Python, no I/O
│   ├── src/plant_sim/
│   │   ├── simulator.py      PlantSimulator: reset() / step(action)
│   │   ├── state.py          Action (in) and PlantState (out)
│   │   ├── config.py         plant design: sizing, limits, efficiencies
│   │   ├── allocation.py     dispatch rules: where each kW goes
│   │   ├── faults.py         fault catalogue, specs, registry, component effects
│   │   ├── telemetry.py      truth -> observation: where sensor faults are applied
│   │   ├── events.py         bounded event stream (info / warning / critical)
│   │   ├── scenarios.py      repeatable situations: start state + fault schedule
│   │   ├── components/       one file per piece of equipment
│   │   │   ├── base.py         Component contract (reset + snapshot)
│   │   │   ├── solar.py        SolarArray
│   │   │   ├── battery.py      Battery
│   │   │   ├── electrolyser.py Electrolyser
│   │   │   ├── dac.py          DirectAirCapture
│   │   │   ├── methanator.py   Methanator (fixed Sabatier recipe)
│   │   │   └── storage.py      GasStorage (the H₂, CO₂ and CH₄ tanks)
│   │   └── __main__.py       headless run: python -m plant_sim
│   └── tests/                behaviour, limits, conservation, determinism
│
├── backend/            the plant-control backend: it drives the simulator
│   ├── src/plant_backend/
│   │   ├── app.py            FastAPI routes + the SSE stream
│   │   ├── session.py        simulator lifecycle: reset/step/history/faults
│   │   ├── orchestration/
│   │   │   ├── runtime.py      PlantRuntime: the decide -> validate -> apply loop
│   │   │   ├── loop.py         SimulationClock: wall-clock pacing, play/pause/speed
│   │   │   └── broadcast.py    fan-out of updates to connected control rooms
│   │   ├── controllers/
│   │   │   ├── base.py         Controller contract + Constraints (the envelope)
│   │   │   ├── heuristic.py    the rule-based operator that runs the plant today
│   │   │   └── rl.py           RLController + RLControllerSlot: the policy, if one is loaded
│   │   ├── rl/               the learning side (V7); imports nothing web
│   │   │   ├── spaces.py       the observation/action contract, as data
│   │   │   ├── observation.py  telemetry -> normalised vector
│   │   │   ├── actions.py      policy vector -> Action (no second guard)
│   │   │   ├── reward.py       what "running the plant well" scores
│   │   │   ├── episodes.py     the episode distribution: scenario + start + weather
│   │   │   ├── metrics.py      plant-side numbers a person compares controllers with
│   │   │   ├── env.py          PlantEnv: reset(seed) / step(vector) / step_action(Action)
│   │   │   ├── checkpoints.py  metadata, compatibility, the mock checkpoint writer
│   │   │   ├── policies.py     checkpoint -> something with .act()
│   │   │   ├── training/       PPO config, model wiring, `train` entry point
│   │   │   └── evaluation/     heuristic vs policy on identical episodes (`make rl-eval`)
│   │   ├── hepa/            future event prediction (V9); torch is optional, advisory only
│   │   │   ├── telemetry.py    the feature schema, as data: source, unit, formula
│   │   │   ├── events.py       the event rules, defined once for every stage
│   │   │   ├── horizons.py     hours <-> sample steps (the horizon grid)
│   │   │   ├── checkpoints.py  weights + the metadata needed to refuse them
│   │   │   ├── inference.py    telemetry history -> P(event within Δt)
│   │   │   ├── evaluation.py   AUROC/AUPRC/Brier/calibration, per event and horizon
│   │   │   ├── api.py          GET /api/hepa/status, POST /api/hepa/predict
│   │   │   ├── service.py      V10: the one door the supervisor asks through
│   │   │   ├── fixtures.py     V10: deterministic stand-ins for tests, labelled as such
│   │   │   ├── data/           trajectories, windows, labels, splits, the collector
│   │   │   ├── models/         causal encoder, horizon predictor, survival head
│   │   │   ├── pretraining/    stage 1: self-supervised representation learning
│   │   │   ├── finetuning/     stage 2: event fine-tuning on a frozen encoder
│   │   │   └── configs/        example collect/pretrain/finetune/evaluate configs
│   │   ├── agent/            the supervisory operator (V6)
│   │   │   ├── supervisor.py   SupervisorAgent: gating, one run, failure paths
│   │   │   ├── triggers.py     when waking it up is worth an LLM call
│   │   │   ├── tools.py        AgentToolbox: the only way it can touch the plant
│   │   │   ├── observations.py telemetry -> what a supervisor can actually see
│   │   │   ├── engines.py      LLMEngine (Agents SDK) + RuleEngine (the fallback)
│   │   │   ├── prompts.py      the supervisory brief
│   │   │   ├── sdk.py          the toolbox wrapped as SDK function tools
│   │   │   ├── trace.py        the operational trace and the incident record
│   │   │   ├── risk.py         V10: HEPA probabilities -> levels, trends, attention
│   │   │   ├── configs/risk.json  the thresholds, per event and per horizon
│   │   │   ├── evaluation.py   the five-scenario regression harness (`make eval`)
│   │   │   └── foresight_evaluation.py  V10: controller / +supervisor / +HEPA (`make foresight-eval`)
│   │   └── safety/
│   │       └── validation.py   the guard every action passes through
│   └── tests/                controller, constraints, guard, loop, agent, RL, API contract
│
├── frontend/           the control room
│   └── src/
│       ├── views.tsx         view manifest -> drives router + top nav
│       ├── pages/            Overview + 3D Plant (the twin) + Agent (the supervisor)
│       ├── components/       shell (top nav, page header) + overview widgets
│       │   └── plant-3d/     the twin: scene, units, flow network
│       ├── hooks/            usePlantSession: the SSE link and the state adapter
│       ├── state/            one shared session, read by every view
│       ├── lib/api.ts        the only place that knows how to reach the plant
│       └── types/plant.ts    mirror of PlantState / PlantConfig
│
├── Makefile            setup / dev / test shortcuts
└── pyproject.toml      uv workspace: simulator + backend in one venv
```

Responsibilities stay strictly separated:

| Layer       | Owns                                        | Never does                          |
| ----------- | ------------------------------------------- | ----------------------------------- |
| `simulator` | physical state and how it changes over time | decide what *should* happen; do I/O |
| `backend`   | the clock, who drives, what is allowed      | model physics                       |
| `frontend`  | showing state and collecting operator input | model physics; own plant state; own the clock |

The supervisory agent lives in the backend and is bound by the same rule: it changes what
the plant is *allowed* to do, never what it physically does.

---

## How to run

Requires Python ≥ 3.11 ([uv](https://docs.astral.sh/uv/) recommended) and Node ≥ 20.

```bash
make setup          # uv sync + npm install
make dev            # backend on :8000 and control room on :5173
```

Then open <http://localhost:5173>.

Individually:

```bash
make backend        # uv run uvicorn plant_backend.app:app --reload --port 8000
make frontend       # cd frontend && npm run dev
make test           # simulator + backend test suite (266 tests)
make eval           # the supervisor against five scenarios, deterministically
make foresight-eval # controller vs +supervisor vs +supervisor+HEPA (fixture forecasts)
make sim            # one simulated day, headless, printed as a table
make rl-eval        # the heuristic baseline through PlantEnv (CHECKPOINT=<dir> adds a policy)
make train-dry      # build the RL environment and print the training plan; trains nothing
```

HEPA (V9) is opt-in, because it is the only part that needs torch:

```bash
uv sync --group hepa        # the optional model dependencies
make hepa-collect EPISODES=8 DATASET=datasets/hepa-smoke   # simulator -> dataset
make hepa-pretrain-dry DATASET=datasets/hepa-smoke         # the stage-1 plan; trains nothing
make hepa-finetune-dry DATASET=datasets/hepa-smoke         # the stage-2 plan; trains nothing
make hepa-pretrain   # stage 1: long, GPU-shaped work (configs/pretrain.json)
make hepa-finetune   # stage 2: needs a stage-1 checkpoint
make hepa-eval CHECKPOINT=checkpoints/hepa/stage2          # per-event, per-horizon metrics
```

The supervisor runs on rules unless `OPENAI_API_KEY` is set, in which case the LLM drives
it (`PLANT_AGENT_MODEL` picks the model, `PLANT_AGENT_ENABLED=0` switches it off).
No key is required to run the plant, the tests or the evaluation.

`make sim` is the fastest way to see the model work:

```text
  clock    solar   ->batt   ->elz   ->dac   ->met   batt%  H2 kg/h  H2 kg   CO2 kg/h  CO2 kg   CH4 kg/h  CH4 kg
 D1 00:01        0        0       0       0       0    34.9     11.5   120.2       0.0    800.0       0.0      0.0   <- night: battery support runs the stack
 D1 08:01     2244       49    1566     449     180     5.1     30.1   103.1     224.4    524.6     150.0    183.9   <- reactor burns the overnight H2
 D1 12:01     4924      993    2816     985     130    33.2     54.2     0.0     492.4    515.4     108.3    750.6   <- H2-limited flow-through
 D1 20:01        0        0       0       0       0    69.8     11.5     7.6       0.0   1368.8       0.0   1355.8   <- DAC/reactor off, H2 rebuilds
```

---

## How the simulator works

The interface is deliberately small:

```python
from plant_sim import Action, PlantSimulator

sim = PlantSimulator()
state = sim.reset()
state = sim.step(Action(battery_share=0.25, electrolyser_share=0.45, dac_share=0.2,
                        methanator_share=0.1, battery_support_kw=600))
```

- **`Action`** — how to split the power available *right now*. `battery_share`,
  `electrolyser_share`, `dac_share` and `methanator_share` are fractions of current
  solar, one per bus consumer; `battery_support_kw` discharges the battery to feed the
  electrolyser. Anything unallocated is curtailed.
- **`PlantState`** — everything observable: levels (`battery_soc`, `h2_level_kg`,
  `co2_level_kg`, `ch4_level_kg`) describe the plant now, flows (`*_kw`, `*_kg_h`)
  describe the timestep that just finished. Full field list in `state.py`; the frontend
  mirror is `frontend/src/types/plant.ts`.
- One `step()` advances `dt_s` seconds (default 60 s). Same actions in, same states out —
  the model is fully deterministic.

### Component rules

Simple and readable on purpose — no electrochemistry.

| Component        | Rule                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| **Solar**        | half-sine between sunrise and sunset, peaking at solar noon; identical every day                       |
| **Battery**      | charges/discharges within power limits, never past capacity or `min_soc`; efficiency applied both ways |
| **Electrolyser** | linear kWh → kg; hard rated ceiling; **below its minimum load it shuts off entirely**                  |
| **H₂ tank**      | stores up to capacity and *refuses* the rest, which throttles the stack rather than venting gas        |
| **DAC**          | linear kWh → kg CO₂ (2.0 kWh/kg); no minimum load; a full CO₂ tank throttles it to zero                |
| **CO₂ tank**     | same species-agnostic `GasStorage` as H₂: bounded, never negative, never above capacity                |
| **Methanator**   | fixed recipe per kg CH₄: **0.5 kg H₂ + 2.75 kg CO₂ + 1.2 kWh**; limited by the scarcest input          |
| **CH₄ tank**     | the product buffer; a full tank throttles the reactor                                                   |

The methanator's recipe is the Sabatier reaction (CO₂ + 4 H₂ → CH₄ + 2 H₂O) collapsed
to mass ratios from the molar masses (0.503 / 2.744, rounded to 0.5 / 2.75) plus a flat
electrical intensity for compression and controls. It only ever draws the power its
actual production needs — power offered beyond what the gases allow spills back to the
battery instead of being burned.

The electrolyser's minimum load is still what makes power allocation interesting, and V3
adds the material version of the same tension: methane needs H₂, CO₂ *and* electricity
in the same hour, so a locally good decision (all solar to the stack, none to the DAC)
becomes a CO₂ shortage an hour later.

### Dispatch order (`allocation.py`)

Every timestep, in this order:

1. Split the solar bus by the action's four shares; anything unallocated is curtailed.
2. Resolve battery support, clamped by what the stack can absorb *and* what the battery can
   deliver — never discharge into a wall.
3. Net out simultaneous charge/discharge (that solar goes straight to the stack instead).
4. Run the electrolyser, capped by rated power and by room left in the H₂ tank. Below
   minimum load it stays off, support is retracted, and its solar spills back to the battery.
5. Run the DAC, capped by rated power and by room left in the CO₂ tank; refusal spills.
6. Run the methanator: CH₄ output is the minimum of what power, H₂, CO₂ and CH₄ tank
   headroom each allow; gases are withdrawn in the fixed ratio and unneeded power spills.
7. Charge the battery with its share plus every spill; what it cannot absorb is curtailed.

Conservation holds by construction, and the tests assert it over a full day:

```text
solar_available    = solar_to_battery + solar_to_electrolyser + solar_to_dac
                   + solar_to_methanator + solar_curtailed
electrolyser_power = solar_to_electrolyser + battery_to_electrolyser
H₂ consumed        = 0.5  × CH₄ produced        (and likewise CO₂ at 2.75 ×)
```

Invalid actions are never rejected, only constrained: shares clamp to `[0, 1]`, are
normalised if they oversubscribe the bus, and `NaN`/`inf`/garbage falls back to zero.

---

## How state flows through the system

```text
  ┌──────────┐   requests (REST)   ┌──────────────────────────┐   Action      ┌───────────┐
  │ Frontend │ ──────────────────> │ Backend                  │ ────────────> │ Simulator │
  │ (adapter)│ <────────────────── │ clock + controller + guard│ <──────────── │ (twin)    │
  └──────────┘   updates (SSE)     └──────────────────────────┘   PlantState  └───────────┘
```

The backend holds the clock. Every 200 ms, while playing, `SimulationClock` asks
`PlantRuntime` to advance `speed` timesteps, and each timestep is the same six steps:

1. **Read telemetry** — `session.telemetry`, the observation a sensor fault may distort.
2. **Decide** — AUTO: the controller. MANUAL: the operator's stored setpoint.
3. **Validate** — `safety.validate()` clamps the action into the operating envelope.
4. **Apply** — `PlantSimulator.step(action)` advances the physics.
5. **Collect** — state, telemetry, events, faults, scenario, control status.
6. **Broadcast** — one self-contained payload to every SSE subscriber.

Paused, the loop advances nothing: simulated time stops. Simulated time depends only on
the number of ticks taken, never on how long the work took, so a slow machine produces
the same trace as a fast one.

The frontend is the last link and the thinnest: `usePlantSession` opens `/api/stream`,
appends frames (keeping ~24 simulated hours for the trend charts), and turns operator
gestures into requests. Every number on screen — readouts, bar fills, tank level, conduit
brightness and flow speed — is a field of `PlantState`. It owns no clock and no physics.

API surface (all under `/api`, proxied from Vite):

| Route                 | Method | Purpose                                                        |
| --------------------- | ------ | -------------------------------------------------------------- |
| `/health`             | GET    | liveness, step index, clock state, subscriber count            |
| `/config`             | GET    | plant design (sizing, limits) so the UI can scale its gauges   |
| `/state`              | GET    | current state + recent frames (a page reload redraws instantly) |
| `/stream`             | GET    | **SSE**: a full payload per tick, and on every change          |
| `/simulation/play`    | POST   | start the clock                                                |
| `/simulation/pause`   | POST   | stop it; simulated time stops with it                          |
| `/simulation/speed`   | POST   | `{speed}` — timesteps per tick (1…64)                          |
| `/simulation/step`    | POST   | `{action?, repeat}` — advance by hand while paused              |
| `/simulation/reset`   | POST   | back to the initial condition                                  |
| `/controller`         | GET    | mode, controller, constraints, and the last decision's notes   |
| `/controller/config`  | POST   | `{mode?, controller?, constraints?, action?}`                   |
| `/controller/rl`      | POST   | `{checkpoint}` — load an RL checkpoint (`null` unloads it)      |
| `/events`             | GET    | recent simulator events                                        |
| `/faults`             | GET    | fault catalogue + what is scheduled or running                 |
| `/faults`             | POST   | `{type, severity, start_s?, duration_s?, channel?}` — inject   |
| `/faults`             | DELETE | clear every fault (`/faults/{id}` clears one)                  |
| `/scenarios`          | GET    | the scenario catalogue                                         |
| `/scenarios`          | POST   | `{scenario}` — reset the plant into that situation             |
| `/hepa/status`        | GET    | whether a forecast can be served, and this build's contract    |
| `/hepa/predict`       | POST   | `{horizons?, telemetry?}` — P(event within Δt), advisory only   |

`/api/step` and `/api/reset` still answer, unchanged, for anything built against V1–V4.

Every state-carrying response is also broadcast to the stream, so a client that asks for
something and a client that only watches see exactly the same thing. Each payload carries
`state`, `frames`, `replace`, `faults`, `events`, `scenario` and `control`, and the
`state` it returns is **telemetry**, not the simulator's private truth.

---

## The 3D twin

The `3D Plant` view (`frontend/src/components/plant-3d/`, React Three Fiber) renders the
site as a place: solar field, power hub, battery container, electrolyser skid, DAC air
contactors, reactor tower and three gas vessels on a fenced pad, with cables and pipes
between them. It is a **visualization layer only** — no physics, no plant state of its
own:

```text
PlantState + PlantConfig
        ↓  deriveVisuals()          <- visuals.ts, the only file that knows field names
PlantVisuals (normalised 0..1 intensities, levels, sun position)
        ↓  one prop per unit
SolarField / PowerHub / BatteryUnit / ElectrolyserUnit / DacUnit / MethanatorUnit
GasVessel ×3 (H₂, CO₂, CH₄) / FlowLine
```

State becomes sight like this:

| Simulator field            | Visual                                                               |
| -------------------------- | -------------------------------------------------------------------- |
| `solar_available_kw`       | panel-face glow, sun position and sky (same half-sine as the model)  |
| `battery_soc`              | light-column fill on the container                                   |
| `battery_power_kw`         | roof lamp: sage while charging, mist while discharging, dim when idle |
| `electrolyser_*`           | stack-seam glow ∝ load; dark and cold below min load                 |
| `dac_*`                    | roof fans spin while capturing (speed ∝ load), intake grilles glow   |
| `methanator_*`             | reactor sight-glass glows ∝ load; lamp turns alarm-tinted when starved |
| `*_fill`                   | gas columns inside the translucent vessels, meniscus rings on top    |
| `*_kw` / `*_kg_h`          | packet streams on each route: speed and opacity ∝ intensity          |
| `solar_curtailed_kw`       | a quiet packet drip from the hub into a ground grate                 |

Every motion is eased in the view (`damp()` in `visuals.ts`), so large steps and resets
read as transitions rather than teleports. Pausing the clock freezes the packets with it;
`prefers-reduced-motion` keeps the levels but stops ambient motion. Selection (clicking a
unit to pin its live values) is the only state the scene owns.

Files that matter:

- `plant-3d/visuals.ts` — the adapter: `PlantState` → `PlantVisuals`. Start here.
- `plant-3d/layout.ts` — where units stand and where flows run, in meters.
- `plant-3d/PlantScene.tsx` — the Canvas composition; nothing else touches three.js setup.
- `plant-3d/UnitChrome.tsx` — hover/pin behaviour and the floating value plates.
- `plant-3d/GasVessel.tsx` — the one vessel component, instanced per species.
- `pages/Plant3DPage.tsx` — the view: scene, HUD readouts, transport bar.

Adding a unit to the twin never touches the simulator: one entry in `plant-3d/layout.ts`,
one component, one flow route — V3 added four units that way.

---

## Extension points

Everything below is *designed for* but deliberately **not built**.

**More process units.** Subclass `Component` (`reset()` + `snapshot()`), give it domain
verbs that clamp their own limits, instantiate it in `PlantSimulator.__init__`, and add it
to the `components` registry — reset and telemetry come for free. V3 used exactly this
path for the DAC, the methanator and the CO₂/CH₄ tanks (`GasStorage` is
species-agnostic); a water tank or a second stack would follow it again.

**Richer physics.** Each component names its own seams: efficiency curves and degradation in
`electrolyser.py`, capacity fade in `battery.py`, weather and forecast noise behind
`SolarArray.power_at`, pressure and offtake in `storage.py`.

**Faults and events.** Built in V4 — see below. Components own their limits, so a fault is
just a component reporting degraded headroom, and `allocation.py` handles the refusal.

**Agents and controllers.** Built in V5 (the seam is `Controller.compute_action()`, the
lever is `Constraints`) and used in V6 by the supervisory agent, which steers through that
lever instead of driving. The simulator never learns who is driving it.

**RL environments.** Built in V7 (see below), and beside the simulator rather than inside
it: `backend/src/plant_backend/rl/` owns the observation, the action conversion and the
reward, while `PlantSimulator` still knows nothing about who is driving it.

**More views.** `frontend/src/views.tsx` is the manifest that drives both the router and the
top nav; the greyed-out entries (3D Plant, Agent, Metrics, Events, Scenarios) are
placeholders pointing at it. Write a page, flip `ready: true`, and it appears.

**3D plant.** Built. The `3D Plant` view reads the same `usePlant()` session as the
Overview and maps `PlantState` onto meshes — see "The 3D twin" below. Adding a unit to
the twin never touches the simulator: one entry in `plant-3d/layout.ts`, one component,
one flow route.

---

## V10: scenarios written in words

The Scenarios page gained a prompt box. An operator describes a situation —

> Create a scenario where solar capacity suddenly drops by 50%, two sensors drift, and the
> DAC becomes degraded after 3 hours.

— and `POST /api/scenarios/generate` turns it into a scenario draft, which is shown for
review, saved on request, and then loaded through the same path a predefined scenario is.

```text
prompt ──> Responses API (strict JSON schema) ──> validation ──> Scenario ──> load_scenario()
```

`backend/src/plant_backend/authoring.py` owns the middle. The schema it sends is *generated
from the simulator*: the fault `type` enum is `faults.CATALOGUE`, `channel` is
`telemetry.SENSOR_CHANNELS`, `additionalProperties` is false everywhere, and there is no
field anywhere in it that could carry code — the model fills in a fixed set of numbers and
strings, nothing else. What comes back is then validated a second time in Python, because a
schema cannot express ranges we care about: hours are clamped to a day, inventories to the
real tank capacities, and severities go through `FaultSpec.parse()`, the same constructor
the predefined scenarios use. A request for something the plant cannot do is either mapped
to the nearest supported fault — recorded in `notes`, surfaced as a warning in the UI — or
rejected with a 422 explaining why.

Saving is session-scoped: `PlantSession.saved` holds the drafts, `scenario_catalogue()`
lists them after the simulator's own with `saved: true`, and `load_scenario("ai_…")` finds
them. The simulator is untouched by this feature; it never learns that a scenario was
written by a model.

The key is read from `OPENAI_API_KEY` and the model from `PLANT_AUTHOR_MODEL`
(default `gpt-4.1-mini`); with no key the endpoint returns a 422 saying so, and the rest of
the page works as before.

---

## V9: HEPA — predicting events before they happen

V9 gives the plant a *forecast*. Everything so far reacts to the present: the controller
allocates against the sample in front of it, the guard clamps the action it is handed, the
supervisor wakes when something has already gone wrong. HEPA answers a different question —

> given the last twelve hours of telemetry, what is the probability that the battery eats
> into its reserve, that the methanator starves, or that the plant goes critical, at some
> point in the next 1, 3, 6, 12 or 24 hours?

It is an implementation of **HEPA (Horizon-conditioned Event Predictive Architecture)**
([arXiv:2605.11130](https://arxiv.org/abs/2605.11130)), a JEPA-style architecture that
learns by predicting *representations* of the future rather than the future itself, and is
conditioned on the horizon it is asked about.

**Nothing in this repository has been pretrained or fine-tuned.** V9 is infrastructure plus
tests: the model, the two training stages, the dataset format, the metrics and the serving
path all exist and are exercised on CPU-sized problems, and no accuracy claim is made
anywhere. The backend behaves exactly as V7 did until `PLANT_HEPA_CHECKPOINT` points at a
fine-tuned checkpoint.

```text
Simulator ──> telemetry dataset ──> stage 1: HEPA pretraining ──> pretrained encoder
                                                                       │
                                            stage 2: event fine-tuning ┘
                                                       ↓
                                             P(event within Δt)
```

### The architecture

```text
context x_{<=t}  (C, 30) ──f_θ──> h_t  (D)  ──g_φ(·, Δt)──> ĥ_{(t, t+Δt]}  (D)
                                                                  │  stage 1: compare
future (t, t+Δt] (K, 30) ──f_θ──> attention-pooled target  (D) ────┘  against this

                            h_t ──g_φ(·, anchors)──> (M, D) ──head──> λ per event, per block
                                                        p(t, Δt) = 1 - Π_j (1 - λ_j)
```

| Piece                     | What it is                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `f_θ` (`models/encoder.py`) | causal Transformer over non-overlapping patches of `P = 16` samples; `d_model = 256`, 2 layers, 4 heads, sinusoidal positions, per-context instance normalisation. `h_t` is the last causal token |
| `g_φ` (`models/predictor.py`) | the horizon-conditioned predictor: Δt enters as an input (log-scaled, Fourier-expanded), so the *same* weights answer "in an hour" and "in a day" |
| target path               | the *same* encoder, no EMA teacher and no stop-gradient, attention-pooled over the tokens inside Δt |
| stage-1 loss (`pretraining/objective.py`) | `(1 - α)·L1(ĥ, target) + α·SIGReg(ĥ)`, `α = 0.1`; SIGReg is the sketched Cramér–von Mises penalty that makes collapse unprofitable, which is why the target encoder can keep its gradients |
| horizons                  | drawn **log-uniformly** during pretraining, so the short horizons that dominate at serving time are not drowned by the long tail |
| event head (`models/heads.py`) | predicts a per-block *hazard* per event and composes it into a cumulative probability, which makes `p(t, Δt)` monotone in Δt by construction rather than by hope |
| stage 2 (`finetuning/`)   | loads the stage-1 checkpoint, **freezes `f_θ`** (the paper's protocol), trains the predictor and the head with masked, positive-weighted BCE |

HEPA is deliberately *not* replaced by something simpler: the representation is learned by
predicting future latents, and the event head is a downstream probe on top of it. A
Transformer classifier trained directly on labels would be a different method with the same
inputs, and would not be this paper.

**Deliberate deviations from the paper**, each one recorded next to the code that makes it:

| Deviation                                                          | Why                                                                                     |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| pretraining horizons are drawn in whole patches                    | the target is pooled over patch tokens, so a Δt ending mid-patch has no distinct target. Fine-tuning and inference use exact sample horizons |
| the downstream head predicts hazards, not a direct per-horizon logit | it forces monotonicity in Δt and shares statistical strength across horizons, which matters when the rare events have few positives |
| model and dataset scale are the plant's, not the paper's           | 30 channels of one plant, sampled every 5 minutes; the paper's sizes are the defaults, the sizes here are configurable |
| `detach_target` exists as a switch, defaulting to off              | the paper does not stop the target's gradients; the switch makes that choice explicit and testable rather than implicit |

### What the model sees

**30 features**, all of them *observed telemetry* — the same `PlantState` an operator sees,
sensor faults included, never the simulator's private truth. The schema lives in
`hepa/telemetry.py` **as data** (each feature carries its source, its unit and the formula
used to scale it), so `GET /api/hepa/status` publishes the exact contract a checkpoint was
trained against, and documentation cannot drift from the encoder:

| Group      | Features                                                                          |
| ---------- | --------------------------------------------------------------------------------- |
| time       | `time_of_day_sin`, `time_of_day_cos` (cyclic, so midnight is next to 23:59)       |
| generation | `solar_available`, `solar_potential`, `solar_output_factor`, `solar_curtailed`     |
| storage    | `battery_soc`, `h2_fill`, `co2_fill`, `ch4_fill`                                   |
| power      | `battery_power`, `battery_support`, `electrolyser_power`, `dac_power`, `methanator_power` + a load fraction per unit |
| production | `h2_production`, `co2_capture`, `ch4_production`                                   |
| condition  | `*_online`, `*_health`, `electrolyser_efficiency`, `electrolyser_temperature`, `plant_status` |

Every channel is divided by the thing that bounds it (rated power, tank capacity, SOC), so
the vector is roughly `[0, 1]` before normalisation — and a *drifting sensor may still
exceed it*, which is information, not an error. Dataset statistics are fitted on the
**training split only** and travel with both the manifest and every checkpoint.

### The five events

Defined once, in `hepa/events.py`, and read from there by the collector, the label
generator, the fine-tuning loop, the evaluation code and the API — because a pipeline where
each stage has its own idea of "starvation" reports a number that measures nothing:

| Event                        | Fires when                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| `battery_reserve_violation`  | observed SOC ≤ the operating reserve (0.15 — the envelope, not the pack's hard floor)   |
| `h2_starvation`              | methanator commanded, methanator not running, observed H₂ below `empty_kg`               |
| `co2_starvation`             | the same three conditions on the CO₂ tank                                                |
| `forced_production_shutdown` | electrolyser or methanator commanded while offline                                       |
| `critical_plant_state`       | reported `plant_status` is `critical` or `offline`                                       |

Labels are **multi-label** (a shutdown usually overlaps a starvation), **strictly future**
(the interval `(t, t+Δt]` is half-open on the left: sample `t` is input, never label),
**cumulative** (matching the survival form the head predicts, hence monotone in Δt) and
**never truncated** — a 24 h label needs 24 h of recorded future, and where the episode
ends first the label is *unknown* and masked out of the loss and the metrics, rather than
quietly counted as a negative.

### Dataset and horizons

```bash
make hepa-collect EPISODES=512 DATASET=datasets/hepa
```

The collector drives the ordinary plant: `PlantSimulator` under the heuristic controller,
episodes drawn from the **same** `rl/episodes.py` sampler V7 uses (scenario, start state,
weather, seed), so an episode is reproducible from `seed + index`. It runs the simulator at
its own 60 s timestep and *resamples* to the model's 300 s sample period, accumulating any
event that occurred inside a sample so the resampling cannot hide one.

```text
datasets/hepa/
├── manifest.json           feature + event contract, thresholds, horizons, normalisation
└── episode_0000.npz …      features (T, 30) float32 | events (T, 5) bool | time_s (T,)
```

Horizons are a grid, in hours, converted to samples once (`hepa/horizons.py`):

| Horizon | 1 h | 3 h | 6 h | 12 h | 24 h |
| ------- | --- | --- | --- | ---- | ---- |
| samples | 12  | 36  | 72  | 144  | 288  |

A window is `context (144, 30)` — twelve hours, nine patches — plus `future`, `labels
(5, H)` and `valid (H,)`. `future` exists because stage 1 needs a target representation of
it; it is a separate field, never part of the input, and at serving time `inference.py`
builds a context and nothing else.

### Checkpoints refuse rather than guess

A checkpoint is a directory of `weights.pt` + `metadata.json`, and the metadata is the
point: feature names and schema version, event names and event-spec version, thresholds,
the horizon grid, the context length, the normalisation statistics, the seed, the training
config, the validation report, and whether it was trained at all. Loading verifies all of
it, and a mismatch is an error — a model trained against different statistics, a different
feature order or a different definition of `h2_starvation` is not a slightly worse model,
it is a wrong one.

### Serving: advisory, and nothing more

```text
GET  /api/hepa/status     is a model loaded, for which events and horizons, and the schema
POST /api/hepa/predict    {horizons?, telemetry?} -> P(event within Δt), per event
```

```json
{
  "probabilities": {"battery_reserve_violation": {"1h": 0.02, "6h": 0.31}},
  "horizons": ["1h", "6h"],
  "observed_at_s": 86400.0,
  "checkpoint": {"stage": "stage2", "trained": true, "context_len": 144}
}
```

Both routes are **read-only and advisory**, and so is everything downstream of them. The
controller and the guard never consult HEPA; the supervisor does, since V10, and only to
decide what to look at (see below). No forecast changes an action by itself. The
model is loaded lazily from `PLANT_HEPA_CHECKPOINT` and cached, so a backend with no
checkpoint starts exactly as V7's did. Everything degrades on purpose — no torch or no
checkpoint is a `503` that names the fix, `/status` still answers because metadata needs no
tensors, and too little history or an unrequestable horizon is a `400`. There is no path
that returns a number without a trusted checkpoint behind it.

The live plant runs at 60 s and the model is trained on 300 s samples, so `/predict`
resamples the session history to the checkpoint's sample period, measuring the spacing from
the frames themselves and walking backwards from *now*.

### Evaluation

`hepa/evaluation.py` reports, per event **and** per horizon: AUROC, AUPRC, Brier score,
precision/recall/F₁ at a threshold, a reliability curve, the base rate and the number of
windows scored. Masked (unknown-future) windows are excluded, and a metric with no
positives to rank is `nan` rather than a comforting 0.5.

```bash
make hepa-eval CHECKPOINT=checkpoints/hepa/stage2 DATASET=datasets/hepa
```

### What is tested, and what is not

Tested on CPU, in seconds (`backend/tests/test_hepa_*.py`): the feature schema and its
metadata; encoding from live simulator telemetry and from serialised frames; normalisation;
every event rule at its boundary; horizon conversion; strictly-future, cumulative, masked
labels; window shapes; dataset round-trips and schema rejection; the collector, including
determinism from the seed and training-only statistics; model shapes; attention causality;
horizon conditioning; hazard monotonicity; the stage-1 loss and its SIGReg term; a few real
optimisation steps; encoder freezing; positive weighting; masked BCE; checkpoint metadata,
compatibility and corruption; a whole stage-1 → stage-2 → inference → evaluation pipeline on
a synthetic dataset with a 32-dimensional model; and both HTTP routes, with and without a
checkpoint.

**Not done here, and it needs a GPU:** collecting a real dataset (hundreds of multi-day
episodes), stage-1 pretraining, stage-2 fine-tuning, and therefore any statement about how
well HEPA predicts these events. The configs in `hepa/configs/` are the starting point for
that run, `make hepa-pretrain-dry` prints exactly what it would do, and until it has been
done this repository claims nothing about forecast quality.

---

## Current research path: controller-neutral Environment V2 RL

The beta now has executed exploratory PPO evidence: the
[first adaptive screen](docs/research/results/ADAPTIVE_RL_BATCH_001_RESULTS.md)
trained five policies, and the
[second screen](docs/research/results/ADAPTIVE_RL_BATCH_002_RESULTS.md) trained
four direct/residual policies, retaining eight scenarios for every controller
in each batch. Residual PPO modestly improved over H, but remained behind
OperatingMPC and failed safety and paired service/resource acceptance conditions.
No PPO has demonstrated operational benefit against the strong operating baseline.
Checkpoints and full evidence remain outside Git. The original registered
15-policy study and protected evaluation seeds remain unopened.

The maintained V1 stack below remains available for the live control-room demo. The research
branch now also has a separately identified RL path for the more realistic Environment V2
tasks. It uses four normalized **absolute process targets**, so a learned policy, the
transparent heuristic adapter, and service MPC no longer receive different physical authority.

```text
62 public physical + tariff values -> policy -> 4 normalized targets
    -> checkpoint-bound ProcessSetpointCommandV1
    -> common one-minute dispatcher -> safety guard -> plant -> EUR ledger
```

The checkpoint records both the policy interface and the complete task digest. A policy is
therefore refused if the plant, episode distribution, constraints, objective, tariff,
assumptions, observation semantics, or action semantics differ. Real policy bytes are hashed;
fixed-action checkpoints remain explicitly untrained test fixtures.

Resume is also fail-closed: policy and PPO optimization settings embedded in the parent cannot
be silently relabelled, and every resumed checkpoint records hashes of its parent metadata and
model artifact. Real SB3 artifacts are checked against the declared Gym shape, dtype, and bounds
both when saved and when loaded.

```bash
make v2-train-dry                                      # constructs and identifies; trains nothing
make v2-train CONFIG=/path/to/training.json OUTPUT=runs/new-v2-run
make v2-rl-eval CHECKPOINT=runs/new-v2-run/checkpoint SEEDS=3
```

No four-target policy has been trained or evaluated as a research contender in this
repository. Disposable two-transition PPO wiring tests exercise both the 57/4 and 62/4 tasks,
save, resume to four transitions, reload, and evaluate each checkpoint when the optional
training dependency is available; these are not controller results. The integration is
preparation for a registered experiment, not evidence that RL outperforms the heuristic or
MPC. See
[`docs/research/CONTROLLER_NEUTRAL_RL_INTEGRATION.md`](docs/research/CONTROLLER_NEUTRAL_RL_INTEGRATION.md)
for the exact admission and claim boundary.

The complete controller-neutral comparison path is now implemented as well. For each admitted
economic-task seed it constructs four independent but identity-matched twins, preflights every
controller before movement, and then executes **H**, rolling **M24**, tariff-aware economic
**MPC**, and deterministic **PPO** through the same guarded one-minute physical application
path. The economic MPC is a real two-stage optimizer: it first preserves the best feasible
delivery service, then minimizes the declared tariff, battery-throughput, modeled wear, and
non-rewarding terminal-deficit costs. Failed arms remain failed outcomes; there is no substitute
policy or heuristic fallback, and incomplete four-way cells do not enter numeric aggregation.
Qualified runs require clean committed source, canonical M24/E arms, and a trained SB3 policy;
the report binds the source revision, dependency lock, numerical runtimes, and checkpoint bytes.

```bash
make v2-controller-benchmark \
  CHECKPOINT=/absolute/path/to/trained/checkpoint \
  SEED_START=100 SEEDS=3 \
  OUTPUT=/absolute/path/to/new-report.json
```

This earlier economic-alpha comparison remains an evaluation build; the subsequent
beta pilot and adaptive results are linked above. The full alpha boundary, including the
economic optimizer's model mismatch and the current unequal tariff representation, is recorded
in
[`docs/research/CONTROLLER_NEUTRAL_BENCHMARK_BUILD.md`](docs/research/CONTROLLER_NEUTRAL_BENCHMARK_BUILD.md).

The harder cyclic-market beta now has a release-validated controller-neutral comparison boundary.
Its public state is 135 values: the 57-field physical boundary, a 73-field causal and revisable
electricity-price issue, and five cycle-progress/resource-return values. A beta-native economic MPC
uses the current public solar/price issue and imposes hard battery, H2, CO2, and CH4 return rows at
the next daily boundary. The beta runner compares **H**, **M24**, **E-beta**, and a checkpoint-bound
deterministic **PPO** on four independent identity-matched twins, while every arm still issues the
same four absolute targets through the common dispatcher, safety guard, and simulator.

Value aggregation is fail-closed: all four arms must complete, meet every cycle return, pass hard
safety, and provide comparable delivery service. A detached verifier reconstructs typed boundary,
command, minute-execution, economic, cycle, classification, and digest evidence rather than
trusting the report's summaries. Its exhaustive minute-truth ledger supports exact regeneration
and attribution of isolated DAC outages while preserving simultaneous plant faults as safety
failures. For E-beta, it independently audits the retained stage-two solution vector, digest,
service lock, feasibility, costs, and objective; the unretained primary-stage digest remains
provenance rather than an independently re-solved optimality certificate. Qualified operation
requires a real trained beta PPO checkpoint, clean committed source, and a new absolute report path
outside the repository.

```bash
make v2-beta-controller-benchmark \
  CHECKPOINT=/absolute/path/to/trained-beta-checkpoint \
  SEED_START=100 SEEDS=3 \
  OUTPUT=/absolute/path/to/new-beta-report.json

make v2-beta-controller-benchmark-verify \
  REPORT=/absolute/path/to/new-beta-report.json
```

The complete repository gate passes 1,535 Python tests with 4 optional skips, scoped typing and
static checks, and frontend build/lint. A bounded untrained seed-34 smoke completed all four
1,440-minute arms safely but produced no eligible comparison because cycle returns failed and
service differed; it is not a performance result. No trained beta checkpoint or controller ranking
is claimed here. See
[`docs/research/CYCLIC_MARKET_BETA_CONTROLLER_BUILD.md`](docs/research/CYCLIC_MARKET_BETA_CONTROLLER_BUILD.md)
for the complete information policy, eligibility gates, verifier scope, integration-smoke finding,
and next research steps. The outcome-blind constrained-learning algorithm, episode allocator,
finite training budget, development selection, and sealed held-out rules are frozen separately in
[`docs/research/CONSTRAINED_RL_BETA_PROTOCOL.md`](docs/research/CONSTRAINED_RL_BETA_PROTOCOL.md);
that protocol authorizes only bounded fixture-domain optimizer smokes, not the registered
training campaign or any development/held-out reset.

The first fixture-only campaign-capacity attempt is retained as an engineering failure rather than
silently retried. All three learner families completed optimization, but checkpoint construction
exposed a family-neutral binary64 aggregation defect between independently summed raw and already-
normalized rollout costs. The frozen repair canonicalizes only rollout evidence as raw total divided
once by seven, checks the independent arithmetic path at the declared tolerance, and leaves every
transition, optimizer input, multiplier update, model setting, seed, and task contract unchanged.
The V2 capacity evidence path now also separates synthetic test issuance from the non-injectable
production runner/verifier and binds an exact pre-run-only deterministic identity. The sole
Repair-001 attempt at `27a47d0` then completed all three child trainers but failed their parent-side
admission because JSON-loaded lists were compared directly with tuple-valued in-memory task
configuration. It produced no capacity projection and cannot be retried. Repair 002 compares the
complete metadata copy through the same JSON byte serializer while retaining the exact config-file
byte check and every later artifact/evidence check. Its sole clean-revision attempt at `4f505b8`
passed all three production checkpoint admissions and detached verification. On the recorded M4
Pro host, the frozen 15-policy sequential campaign projects to 62.49 hours at 1.00x, 78.10 hours at
1.25x, and 93.71 hours at 1.50x, with 174.35 MiB of projected logical checkpoint data and 1.17 GiB
recommended free space. These are fixture-derived capacity scenarios, not confidence intervals,
registered training, controller evidence, or a performance claim.
See the retained [`V1 failure`](docs/research/results/v2_constrained_rl_capacity_fab4a55.md),
[`V2 failure`](docs/research/results/v2_constrained_rl_capacity_27a47d0.md), frozen
[`Repair 001`](docs/research/CONSTRAINED_RL_CAPACITY_REPAIR_001.md), and frozen
[`Repair 002`](docs/research/CONSTRAINED_RL_CAPACITY_REPAIR_002.md), plus the authenticated
[`V3 capacity result`](docs/research/results/v2_constrained_rl_capacity_4f505b8.md).
The next outcome-blind implementation boundary is frozen in the
[`registered training custody protocol`](docs/research/CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md):
it must close exact task, 15-run order, durable attempt, sealed-log, and detached-verification gaps
before an explicit user authorization record can open registered training.

The existing control-room `/api/controller/rl` path remains the historical one-minute 30/5 V1
runtime. It intentionally rejects these 15-minute four-target checkpoints: deploying them live
requires a V2 runtime with the same forecast, commitment, tariff, and held-setpoint semantics,
not a lossy compatibility adapter.

The first preregistered economic task-expression cell passed: heuristic and service MPC both
safely delivered 210 kg with zero unmet demand, while their auditable operating values differed
by 382.532 EUR because terminal inventory, cycling, and equipment exposure differed. This
established one-cell discrimination, not general controller or commercial superiority. See
[`docs/research/results/v2_economic_mechanism_smoke_a46aabf.md`](docs/research/results/v2_economic_mechanism_smoke_a46aabf.md).

The subsequent preregistered 20-cell robustness tournament is now complete. All 40 trajectories
and 36,000 physical minutes pass execution, identity, accounting, safety, and authority gates.
Both six-hour strata remain service-comparable in 5/5 cells, but both 24-hour strata are
comparable in 0/5: the six-hour-lookahead MPC fulfills the first two commitments, partially
fulfills the third, and delivers zero at the fourth while retaining a near-full battery and
stranded H2/CO2. That result did **not** admit substantive RL training. The first
repair changed only the service MPC horizon from 6 to 24 hours on the already contaminated
full-day cells, leaving its objective, plant, command path, and economics unchanged. That
registered M24 probe is now complete: all ten trajectories met all four commitments and the full
request, all F1--F8 evidence gates passed, and a second detached verification succeeded. This
establishes the missing-lookahead mechanism on development evidence; it does not establish RL or
commercial value. The retained reports can be verified with:

```bash
make v2-economic-robustness-verify REPORT=/absolute/external/directory/report.json
make v2-full-day-horizon-repair-verify REPORT=/absolute/m24/directory/report.json
```

See [`docs/research/results/v2_economic_robustness_7b1b656.md`](docs/research/results/v2_economic_robustness_7b1b656.md)
for the complete gate result, mechanism diagnosis, coefficient sensitivity, artifact identities,
and next architecture decision. The frozen follow-up is
[`docs/research/FULL_DAY_HORIZON_REPAIR_PROTOCOL.md`](docs/research/FULL_DAY_HORIZON_REPAIR_PROTOCOL.md),
with its pre-code evidence clarifications in
[`clarification 001`](docs/research/FULL_DAY_HORIZON_REPAIR_CLARIFICATION_001.md) and
[`clarification 002`](docs/research/FULL_DAY_HORIZON_REPAIR_CLARIFICATION_002.md). The outcome-free
software gate and exact execution custody are recorded in the
[`admission`](docs/research/FULL_DAY_HORIZON_REPAIR_ADMISSION.md) and
[`registration`](docs/research/FULL_DAY_HORIZON_REPAIR_REGISTRATION.md). The complete result,
including the coefficient-sensitive economic direction and next benchmark decision, is
[`v2_full_day_horizon_repair_d180825.md`](docs/research/results/v2_full_day_horizon_repair_d180825.md).

---

## V7: the historical RL foundation

V7 makes the plant learnable without letting anything unproven drive it. It adds the
environment, the reward, the training and evaluation entry points, checkpoint handling and
the controller slot a policy loads into — and it trains nothing. **There is no policy in
this repository**, so there is no heuristic-vs-RL result to report, and the code says so
wherever a number could be mistaken for one.

```text
                    Supervisory agent (V6)
                            │  constraints
              ┌─────────────┴─────────────┐
       HeuristicController          RLController
        (default, always)      (only with a loaded checkpoint)
              └─────────────┬─────────────┘
                            ↓
                    safety.validate()          unchanged, authoritative
                            ↓
                        Simulator
```

The learning side lives in `backend/src/plant_backend/rl/` and imports no FastAPI: the
environment, training and evaluation run headless, locally or on a remote box, and the
backend only depends on it for the controller.

### The environment

```python
from plant_backend.rl import PlantEnv

env = PlantEnv()                                  # 48 h episodes by default
observation, info = env.reset(seed=7)             # same seed -> same episode
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`PlantEnv` is a translation layer, not a second plant: `PlantSimulator` still owns the
physics, `load_scenario` still puts it into a starting condition, and every action takes the
backend's path — **decide → `safety.validate()` → `simulator.step()`**. Two doors share that
body: `step(vector)` for a policy and `step_action(Action)` for anything implementing
`Controller`, which is what makes the evaluation below a fair comparison rather than two
rollout loops that drifted apart.

Truth and telemetry stay split in both directions. The **observation is telemetry**, sensor
drift included; the **reward is `simulator.state`**, the plant's own production accounting.
A policy that learns to believe a drifting transmitter is penalised the way a real plant
would penalise it — with methane it did not make.

Episodes end by **truncation, never termination**. An industrial plant has no absorbing
state; a critical condition is something to recover from, not a game over. `terminated` is
always `False`, `truncated` becomes `True` at the horizon, and bootstrapping stays correct
at the cut. Severe fault recovery is deliberately *not* the objective: the episode pool is
ordinary weather and ordinary operation, because the thing being learned is routine
allocation.

### What the policy sees

30 floats, mostly in `[0, 1]`. The order is part of the checkpoint contract, so the whole
table lives in `rl/spaces.py` **as data** — each field carrying its source, its unit before
normalisation and the formula used — and `ObservationEncoder` is the only thing that reads
it. Documentation cannot drift from the encoder, tests assert on declared bounds instead of
hard-coded indices, and a checkpoint records the field order it was trained against.

| Group | Fields | Why it is in there |
| ----- | ------ | ------------------ |
| time | `time_of_day_sin`, `time_of_day_cos`, `episode_progress` | midnight is next to 23:59; the solar day is the plant's only exogenous cycle |
| energy | `solar_available`, `solar_potential`, `solar_output_factor`, `solar_trend` | what is on the bus, what a clear sky would give, and which way it is moving — the forecast the model actually has |
| storage | `battery_soc`, `battery_power`, `battery_reserve_margin`, `h2_fill`, `co2_fill`, `ch4_fill` | every inter-temporal decision in the plant is about these |
| units | `{electrolyser,dac,methanator}_{load,online,health}` | a derated unit converts worse; an offline one converts nothing |
| production | `h2_production`, `co2_production`, `ch4_production` | the flows the shares just bought |
| envelope | `electrolyser_cap`, `dac_cap`, `methanator_cap`, `support_cap`, `reserve_soc` | the supervisor retunes these mid-episode; a policy that cannot see the envelope keeps asking for what it cannot have |

Normalisation is by *design* reference (rating, capacity, peak) rather than by running
statistics, so the vector means the same thing on the first step of training as on a
re-sized plant, with no normaliser to save alongside the weights. `solar_trend` is the only
field with memory (one step), which is why `ObservationEncoder.reset()` exists.

### What the policy says

5 floats in `[-1, 1]` — the range Gaussian and tanh policies are comfortable in:

| Slot | Becomes | Conversion |
| ---- | ------- | ---------- |
| `battery_share` | `Action.battery_share` | `(a + 1) / 2` |
| `electrolyser_share` | `Action.electrolyser_share` | `(a + 1) / 2` |
| `dac_share` | `Action.dac_share` | `(a + 1) / 2` |
| `methanator_share` | `Action.methanator_share` | `(a + 1) / 2` |
| `battery_support` | `Action.battery_support_kw` | `(a + 1) / 2 × Constraints.support_limit_kw(config)` |

`ActionAdapter` clips out-of-range output, reports that it had to (`clipped`, which the
reward penalises mildly), maps support into the envelope's kilowatts, and hands the result
to `Action.sanitised` — the simulator's own normalisation for shares that oversubscribe the
bus. It deliberately stops there: **the adapter is a unit conversion, not a second safety
system.** `safety.validate()` remains the only guard, for the policy exactly as for the
operator.

### The reward

The objective is the plant's objective — *make methane efficiently over time, without
operating unsafely or unstably*. Everything except the first term is a cost on a way an
allocation policy can look productive while wasting the resource or damaging the plant:

| Term | Signal (simulator truth) | Sign | Normalised by | Weight | Why |
| ---- | ------------------------ | ---- | ------------- | ------ | --- |
| `methane` | `ch4_produced_total_kg` delta | + | CH₄ at rated methanator power for one step | 1.0 | the product |
| `curtailment` | `energy_curtailed_kwh` delta | − | solar generated this step | 0.25 | the main way a plant with a battery underperforms |
| `battery_drain` | discharge this step | − | one step at rated discharge | 0.05 | discharging to clear the stack's floor is *correct*, just not free |
| `reserve_breach` | SOC below `Constraints.reserve_soc` | − | SOC points | 2.0 | the reserve is what keeps the plant able to restart |
| `storage_saturation` | tanks at ≥ 98% fill | − | 3 tanks | 0.1 | a full tank throttles the unit feeding it |
| `shutdown` | `plant_status` in `critical`/`offline` | − | per step | 0.5 | time unable to produce |
| `correction` | guard adjustments + out-of-range output | − | 3 clampable consumers | 0.1 | mild on purpose: it teaches the policy to stop asking, it does not stand in for the guard |
| `cycling` | on/off transitions of the three units | − | 3 units | 0.05 | equipment wear the physics does not model |

Every term reads a counter the simulator already maintains, so nothing is double-counted,
and every term is order-1 per step, so the weights survive a change of timestep or plant
size. `info["reward"]` carries the signed `components` (they sum to the scalar) and the raw
`signals` — the only way to tell a policy that learned to make methane from one that learned
to dodge the shutdown penalty by idling. **The weights are chosen for scale, not tuned
against training outcomes**, which is why they are `TrainingConfig.reward` rather than
constants.

### Episodes and determinism

One seed draws the whole episode: scenario from the pool (`clear_day`, `cloud_cover`),
starting battery SOC and H₂/CO₂/CH₄ inventories, and — 30% of the time — one extra mild
`solar_drop` at a random hour. The draw becomes a plain `Scenario` and is loaded through
`PlantSimulator.load_scenario`, so the env adds no second way of starting the plant.

```bash
uv run pytest backend/tests/test_rl_env.py -q     # includes the determinism test
```

Same seed and same actions produce identical rewards and identical observations; different
seeds do not. Nothing in the env or the reward calls a global RNG.

### Training (wired, not run)

One maintained baseline — PPO from Stable-Baselines3 — because the environment is what V7 is
about, and a second algorithm would double the surface with nothing to compare.

```bash
uv sync --group train                             # pulls torch; nothing else needs it
make train-dry                                    # builds the env, prints the plan, trains nothing
uv run python -m plant_backend.rl.training.train --config runs/ppo.json --output runs/ppo-1
uv run python -m plant_backend.rl.training.train --resume runs/ppo-1/checkpoint
```

`make train-dry` is the honest version of "is this ready to train": it constructs and resets
a real environment and reports the shapes, the steps per episode, how many simulated days the
budget buys and which scenario seed 0 drew.

```json
{"seed": 0, "total_timesteps": 1000000, "n_envs": 4, "observation_shape": [30],
 "action_shape": [5], "steps_per_episode": 2880, "simulated_days": 694.4,
 "first_episode": "cloud_cover", "output_dir": "runs/ppo"}
```

`TrainingConfig` (JSON, `rl/training/config.py`) keeps the four concerns apart —
`episode`, `reward`, `ppo`, and the run itself (seed, timesteps, `n_envs`,
`checkpoint_every`, `output_dir`, `resume_from`) — and rejects unknown keys rather than
silently ignoring a typo in a learning rate. Torch is imported only inside the training and
loading paths, so the backend, the tests and `make dev` never pay for it.

### Checkpoints

A checkpoint is a directory: `metadata.json` plus the policy artifact (`policy.zip` for
SB3). The metadata records the observation and action field order, `SPEC_VERSION`, the
`plant_sim` version, the seed, the timesteps trained and the full training config — enough
to know whether the weights still fit the plant.

Loading refuses, loudly and specifically, in every case where it cannot be sure:

| Situation | Result |
| --------- | ------ |
| path missing | `MissingCheckpointError` |
| no `metadata.json`, unreadable JSON, missing artifact | `CorruptCheckpointError` |
| observation/action fields or `SPEC_VERSION` differ | `IncompatibleCheckpointError`, naming the difference |
| Stable-Baselines3 not installed | `CheckpointError` telling you to `uv sync --group train` |

There is **no path that initialises a random policy** when a checkpoint cannot be loaded:
the RL controller simply does not exist, and the reason is on the status. Runs live under
`runs/`, which is git-ignored — trained weights are build output, not source.

`checkpoints.write_fixed_checkpoint(path, action)` writes a `fixed`-format mock that repeats
one constant action. It exists so the loading, selection and evaluation paths can be tested
without torch, and it is labelled a mock everywhere it surfaces: in the metadata
(`trained: false`), in the controller's notes, in the control room's status line and in the
evaluation output.

### Evaluation

```bash
make rl-eval                                      # heuristic baseline alone
make rl-eval CHECKPOINT=runs/ppo-1/checkpoint     # both, on identical episodes
uv run python -m plant_backend.rl.evaluation --seeds 1 2 3 --hours 24 --json out.json
```

Both controllers get the same seeds, the same scenarios, the same starting inventories, the
same horizon, the same guard, the same reward and the same `EpisodeMetrics`, because both are
rolled out through `PlantEnv` — the heuristic through `step_action`, the policy through
`step`.

```text
3 seed(s) x 24 h per controller  [SMOKE TEST -- no trained policy]

controller             CH4 kg      solar use  curtailed kWh        min SOC           idle    corrections         cycles         reward
Heuristic              1768.8          98.1%            833          22.3%          48.8%           52.0          141.0          694.2

No policy has been trained yet: the RL row above is either absent or a
constant-action mock. Performance numbers follow the first real training run.
```

The JSON carries every run, the per-controller aggregate, the checkpoint metadata and
`smoke_test`, which is `true` whenever neither side is a trained policy. Alongside the eight
columns above the metrics also track H₂/CO₂ produced, battery energy discharged, feedstock
shortages while methanation was commanded, time in a critical or offline condition, and how
many steps the guard had to correct. **The heuristic figures are a baseline, not a
comparison**: there is nothing to compare them with yet.

### The backend: three modes, one default

| Mode | Who decides | Available when |
| ---- | ----------- | -------------- |
| `AUTO — Heuristic` | `HeuristicController` | always; the default |
| `AUTO — RL` | `RLController` | only once a compatible checkpoint has loaded |
| `MANUAL` | the operator's stored setpoint | always |

```bash
export PLANT_RL_CHECKPOINT=runs/ppo-1/checkpoint   # loaded at start-up, if it loads
curl -X POST localhost:8000/api/controller/rl -d '{"checkpoint": "runs/ppo-1/checkpoint"}'
curl -X POST localhost:8000/api/controller/config -d '{"controller": "rl"}'
curl -X POST localhost:8000/api/controller/rl -d '{"checkpoint": null}'   # unload
```

Loading a checkpoint never switches controller by itself — that stays an explicit operator
choice. A bad checkpoint is a `400` carrying the reason, and the plant keeps running on the
heuristic. Selecting `rl` without one is also a `400`, with the reason instead of a silent
no-op. Unloading, or an inference failure, hands the plant straight back:
`RLController` returns an idle action and records why, `PlantRuntime` reverts to the
heuristic, notes *"RL inference failed … reverted to heuristic"*, and the RL status turns
`failed`. A controller that throws every 60 s is not a controller.

Switching controllers changes **who decides**, never the path: `safety.validate()` runs
identically for the heuristic, the policy and the operator, and the supervisor keeps
steering through `Constraints` — it supervises *the plant*, not a particular controller, and
the envelope it tightens is exactly the action space a policy is trained inside.

### Control room

Minimal, deliberately: the control deck lists the available controllers under the
AUTO/MANUAL toggle, marks RL `unavailable` or `failed` with the backend's own reason on
hover, and disables the option instead of offering something that cannot drive. There is no
ML dashboard, no training progress bar and no score — none of those would be showing
anything real. Telemetry, the trend charts, the event feed and the 3D replay are untouched;
a recorded session replays the same way whichever controller drove it, because frames are
`PlantState` either way.

### Where V8 plugs in

The seam is now proven from both sides: `PlantEnv` is the training door and
`Controller.compute_action()` is the driving door, with the same guard behind both. V8 is
therefore a *training* task, not an integration task — run `make train`, evaluate the
checkpoint against the baseline with `make rl-eval CHECKPOINT=…`, and if it wins, ship the
checkpoint. Until that run exists, this repository makes no claim about how a learned
controller compares with the heuristic.

---

## V4: faults, disturbances, telemetry and scenarios

V4 gives the plant situations worth reasoning about. It adds no controller and no
recovery logic: the simulator creates problems and reports them, nothing solves them yet.

### Faults are effects, not special cases

A fault never branches the dispatch loop. Each active fault resolves into a small
`ComponentEffect` — a derating and/or an outage — which the target component applies to
its own physics through `Component.apply_effect()`:

```python
from plant_sim import FaultSpec, PlantSimulator

sim = PlantSimulator()
sim.inject_fault(FaultSpec(type="electrolyser_degradation", severity=0.4))
sim.step(action)            # same kW into the stack, ~40% less H2 out
```

`FaultRegistry` owns the lifecycle (`start_s`, optional `duration_s`, active/cleared) and
answers two questions per timestep: `effect_for(component)` and `sensor_biases()`. Adding
a fault type means adding one `FaultKind` to `faults.CATALOGUE`.

| Type | Target | What it actually does |
| ---- | ------ | --------------------- |
| `electrolyser_degradation` | electrolyser | specific energy rises by `1/health`: same power, less H₂, hotter stack |
| `dac_degradation` | DAC | more kWh per kg of CO₂ captured |
| `dac_outage` | DAC | draws no power, captures nothing — CO₂ inventory only falls |
| `battery_degradation` | battery | effective capacity `= nominal × health`; stored energy is clamped to it and SOC stays consistent |
| `methanator_outage` | methanator | no CH₄, and **no H₂/CO₂ consumed** while down |
| `solar_drop` | solar | a disturbance, not a fault: generation `= clear-sky × (1 − severity)` |
| `sensor_drift` | sensors | biases one telemetry channel; physics untouched |

Degradation severity is clamped to 0.95, so no component divides by zero.

### True state vs telemetry

The simulator keeps the two apart:

```python
truth    = sim.state             # what is actually happening
observed = sim.get_telemetry()   # what the instruments report
```

They are identical unless a sensor fault is active, in which case `telemetry.observe()`
biases the affected channel and the fields derived from it — a biased `h2_level_kg`
carries its `h2_fill` with it. **The backend, the control room and any future agent read
telemetry only**; `state` exists so tests and scenario authors can check the truth the
agent is supposed to work out for itself.

### Events

`EventLog` is a bounded stream (400 entries) of `info` / `warning` / `critical` lines with
the simulated clock attached. Events come from simulator behaviour — fault activation and
expiry, low battery, a reactor stopped for want of CO₂, a full tank, a plant-status
change — and condition events are edge-triggered, so a six-hour outage is one line, not
360.

### Scenarios

A scenario is data: a start hour, inventory overrides, and faults scheduled relative to
that start. Loading one resets the plant first, so the same scenario plus the same actions
always produces the same trace.

| Scenario | What happens |
| -------- | ------------ |
| `clear_day` | nominal operation, no faults — the baseline |
| `electrolyser_degradation` | three normal hours, then the stack loses 40% of its yield |
| `cloud_cover` | generation drops 60% for three hours around midday |
| `dac_failure` | the DAC trips; CO₂ inventory drains until methanation stops |
| `multiple_problems` | cloud cover on top of an already degraded stack |
| `sensor_drift` | an H₂ level transmitter starts over-reporting; the physics is untouched |

```python
sim.load_scenario("dac_failure")
sim.run(360, action)
sim.state.plant_status      # 'critical'
```

### Plant status

`plant_status` is derived every timestep from components and inventories, never set by
the UI: `normal` → `degraded` (something is derated, or the reactor is starved) →
`critical` (a unit is out, or the battery is on its reserve) → `offline` (nothing can
run). The frontend renders whatever the field says.

### Control room

The Overview gained one bone panel and one card: **Conditions** (scenario selector, one
button per fault type, and the list of what is running, each clearable) and **Events**.
In the twin, a degraded unit's plate turns amber and an offline unit's red, both under a
pulsing beacon; flows thin out on their own because they are drawn from the physics.

---

## V6: the supervisory agent

V6 adds the second decision-maker. The controller still answers *"what should the plant do
right now?"* every 60 simulated seconds. The supervisor answers a slower set of questions —
*what is happening, is something wrong, why, should the strategy change?* — and it answers
them only when something looks abnormal.

```text
Frontend
   │
Backend ──── SupervisorAgent ──── AgentToolbox   (read: telemetry, events, components)
   │                │                            (write: envelope, mode, pause, shutdown)
   │                ↓
   ├──── Controller  ← the envelope it just retuned
   ├──── safety.validate()
   ↓
Simulator
```

Two properties define the design. First, the supervisor is **beside** the fast loop, not
inside it: `SimulationClock` calls `supervisor.observe(frames)` after the controller has
already decided, validated and applied, and `observe` normally does nothing but evaluate
triggers. When it does fire, the run happens in a background task — the plant keeps
stepping while the supervisor thinks. Second, it has **no privileged access**: it reads
telemetry (which can lie) rather than `PlantState`, and it writes only what an operator
could write.

### The toolset

Twelve tools, all methods of `AgentToolbox`, all recorded in the trace:

| Tool | What it answers or does |
| ---- | ----------------------- |
| `get_current_telemetry()` | the latest reported plant state, flat and compact |
| `get_recent_telemetry(minutes)` | a thinned trend plus means over the window |
| `get_recent_events(limit)` | what the plant itself flagged, oldest first |
| `inspect_component(id)` | observed performance of one unit against its design figures |
| `run_diagnostic(id)` | mass balances: do the levels and the flows agree? |
| `get_controller_status()` | mode, controller, last applied action, guard adjustments |
| `get_controller_constraints()` | the envelope in force, the ratings it sits inside |
| `set_controller_constraints(changes, reason)` | retune the envelope — the main lever |
| `set_operating_mode(mode, reason)` | AUTO / MANUAL |
| `pause_plant(reason)` / `resume_plant(reason)` | stop and start simulated time |
| `request_safe_shutdown(reason)` | all process caps to 0 kW, reserve raised, time stopped |

`inspect_component` is where diagnosis actually happens, because it reports the *ratio* of
observed to design performance: a stack drawing its usual kilowatts at 86.7 kWh/kg against
a 52.0 kWh/kg design is degraded, and that is visible without the simulator's hidden
`severity` ever being exposed. `run_diagnostic` is the only tool that cross-checks two
independent readings of the same thing (a tank level against the flows through that tank),
which is what makes a drifting sensor distinguishable from a real process fault.

Writes are capped at six per run and a failed write returns `{"ok": false, "error": ...}`
instead of raising: the supervisor is expected to read the refusal and carry on.

### Triggers

`TriggerWatch.evaluate()` runs every tick, is pure, and returns at most one trigger — the
most urgent:

| Trigger | Fires on | Priority |
| ------- | -------- | -------- |
| `manual` | an operator pressed *Investigate current plant* | 100 |
| `critical_event` | a new `critical` event | 90 |
| `plant_critical` | status changed into `critical` | 80 |
| `warning_event` | a new `warning` event | 60 |
| `plant_degraded` | status changed into `degraded` | 50 |
| `production_drop` | CH₄ output below half its baseline while the sun was up | 40 |

Then the gating, in order: the agent must be enabled, no run may be in flight, `MIN_GAP_S`
(600 simulated seconds) must have passed since the last run, and the trigger's `key` must
not be on cooldown (`COOLDOWN_S`, one simulated hour) or already attached to an open
incident. Only fresh events count, and the supervisor's own `agent.*` events are excluded
so it cannot wake itself. One degraded electrolyser therefore produces one investigation,
not one per timestep.

### Prompt and engines

`RuleEngine` and `LLMEngine` implement one contract — *given a toolbox and a trigger,
investigate and return a `Diagnosis`* — so the plant cannot tell them apart. `RuleEngine`
walks the same tools in a fixed diagnostic order (instruments, then converting units, then
the energy budget, then feedstock) with explicit thresholds; it needs no network and no
credentials, which is what makes it both the fallback and the harness. Selection is
`default_engine()`: the LLM when `OPENAI_API_KEY` is set, the rules otherwise.

The brief (`prompts.SYSTEM_PROMPT`) is short and operational rather than educational — the
model can read telemetry, so what it needs is the division of labour:

- it is the *supervisory* operator; a controller already handles routine allocation
- investigate before acting: telemetry, trend, events, then the components that could
  explain what it sees
- compare observed performance with design figures; `yield_ratio` well below 1.0 is a
  derated unit
- telemetry can be wrong — a level that moves without a flow is an instrument problem
- distinguish "a component is failing" from "there is not enough energy"
- the levers are the envelope and the mode; never allocate power
- safety and equipment protection before production
- **do nothing if nothing is wrong** — an unnecessary intervention is a failure, not a safe
  default
- finish with a short structured report: category, component, confidence, summary, whether
  a human is needed. Conclusions only, no step-by-step reasoning

### The trace

`AgentTrace` is a bounded stream (240 entries) of `trigger` / `observation` / `tool` /
`decision` / `diagnosis` / `action` / `result` / `error` entries, each with the simulated
clock, the run number and a small dict of numbers behind it. It is a shift log, not a
transcript: the model's reasoning is never stored or exposed. `Incident` is the other half
— trigger, category, component, confidence, summary, the actions applied, whether a human
is needed, resolved or not — so the control room can answer "what is wrong and did anyone
do anything" without replaying the trace.

Four real runs, from `make eval` (rule engine, so these are reproducible):

```text
D1 09:10  trigger      warning_event: D1 09:00 Electrolyser degradation started (severity 40%)
D1 09:10  tool         get_current_telemetry()
D1 09:10  tool         get_recent_events(limit=20)
D1 09:10  tool         get_recent_telemetry(minutes=120)
D1 09:10  tool         run_diagnostic(component=plant)
D1 09:10  tool         inspect_component(component=electrolyser)
D1 09:10  observation  Electrical input normal at 1960 kW; H2 yield 60% of design (86.67 vs 52.0 kWh/kg), stack at 80 degC
D1 09:10  action       electrolyser_max_kw: unset -> 1800
D1 09:10  diagnosis    electrolyser degradation (confidence: high)
D1 09:10  result       1 change(s) applied; plant degraded

D1 07:10  trigger      critical_event: D1 07:01 Plant status: critical
D1 07:10  tool         inspect_component(component=dac)
D1 07:10  observation  Capture unit unavailable: 0 kW drawn, 0.0 kg/h captured, 124 kg left in the buffer
D1 07:10  action       methanator_max_kw: unset -> 70
D1 07:10  diagnosis    dac unavailable (confidence: high)
D1 07:10  result       1 change(s) applied; plant critical

D1 08:10  trigger      warning_event: D1 08:00 Solar generation drop started (severity 60%)
D1 08:10  tool         inspect_component(component=solar)
D1 08:10  observation  Generation 959 kW against 2399 kW clear-sky (40%), with every unit converting to specification
D1 08:10  action       battery_min_soc: 0.15 -> 0.3
D1 08:10  action       electrolyser_max_kw: unset -> 1500
D1 08:10  action       dac_max_kw: unset -> 480
D1 08:10  diagnosis    energy shortfall (confidence: high)
D1 08:10  result       3 change(s) applied; plant degraded

D1 08:10  trigger      warning_event: D1 08:00 Sensor drift started (severity 17%)
D1 08:10  tool         run_diagnostic(component=plant)
D1 08:10  observation  h2_level_kg and the flow readings disagree by 340.0 kg at D1 08:01
D1 08:10  diagnosis    sensor inconsistency (confidence: medium)
D1 08:10  result       No change made; plant normal
```

The four cases are the four different *kinds* of answer: retune the unit that is failing;
protect the inventory a failed unit no longer supplies; recognise weather as weather and
hold energy back; and refuse to act on a reading that disagrees with itself, flagging it
for a human instead.

### How a constraint change actually reaches the plant

```text
toolbox.set_controller_constraints({"electrolyser_max_kw": 1800}, reason=...)
    → PlantRuntime.configure_constraints        Constraints.merged() validates the field
    → HeuristicController.compute_action        plans inside the new envelope
    → safety.validate()                        clamps whatever still escapes
    → PlantSimulator.step()
```

Nothing about that path is new in V6, which is the point: the supervisor uses the operator's
door. There is deliberately no tool that builds an `Action`, and none that reaches the
simulator, the fault registry or a component object. A supervisor that could allocate
kilowatts would do the controller's job badly; one that could clear its own faults would be
marking its own homework. `safety.validate()` stays authoritative — V6 also tightened it, so
solar and battery support are now counted *together* against a component's cap rather than
separately.

One asymmetry is worth knowing: `configure_constraints(changes, by=...)` records who asked. The
operator's envelope is configuration and survives a reset; a supervisor's retune belongs to the
incident that caused it, so `reset()` and loading a scenario put back whatever the agent
overwrote. An operator writing the same field afterwards takes ownership of it back. Without
that, a fresh run would start silently capped by a fault that no longer exists.

### API and control room

| Route | Purpose |
| ----- | ------- |
| `GET /api/agent?trace=n` | status, engine, run count, current incident, recent trace |
| `GET /api/agent/trace?limit=n&run=k` | the whole trace, or one run of it |
| `POST /api/agent/investigate` | one run, now — the *Investigate current plant* button |
| `POST /api/agent/config {enabled}` | switch the supervisor off or on |

The snapshot also rides on every SSE payload as `agent`, so the **Agent** page needs no
polling: status, the latest diagnosis with its confidence, whether a human is required, the
actions applied, the trace, and the incident history.

### Failure is a path, not an exception

Anything the engine can do wrong — no key, a network error, a timeout, an unparseable
response — is caught, recorded as an `error` entry, emitted as an `agent.error` plant event,
and then handed to `RuleEngine`, which needs neither network nor credentials. If that fails
too, the incident closes as `unclear` with `human_intervention_required`. The clock never
awaits the supervisor, so the worst case is a plant supervised less well, never a plant that
stops being controlled.

### Evaluation

`make eval` (`plant_backend.agent.evaluation`) runs five scenarios deterministically and
scores what a regression would break: did it trigger, did it inspect the relevant component,
did it reach the right category, did it act at all, did it hold the battery reserve, and did
it leave the plant alone during normal operation.

```text
scenario                   runs  diagnosis                  acted  result
clear_day                     1  normal                         0  pass
electrolyser_degradation      3  electrolyser_degradation       1  pass
dac_failure                   2  dac_unavailable                1  pass
cloud_cover                   2  energy_shortfall               3  pass
sensor_drift                  2  sensor_inconsistency           0  pass
5/5 cases passed
```

`sensor_drift` is new in the simulator's scenario catalogue; the other four already existed.
Deliberately not a benchmark — it is regression coverage, and it runs without an API key.

The same five cases also pass with `LLMEngine` against the real model (`gpt-4.1-mini`), which
is how the brief was tuned: the model's own runs are what showed that it needed telling that a
plant ramping up is `normal`, and that a degradation diagnosis has to be followed by the cap
it implies. Those runs are not part of the suite, because a regression gate that costs money
and answers differently each time is not a gate: `backend/tests/conftest.py` hides the key so
every test takes the deterministic rule path — the same path the plant falls back to in
production.

### Where the RL controller plugs in

Unchanged by V6, and now proven by having a second decision-maker in the building:
`Controller.compute_action(observation, constraints, config)` is the seam, and
`PlantRuntime.controllers` is the registry. An RL policy replaces `HeuristicController`
without the supervisor noticing — it supervises *the plant*, through the envelope, not a
particular controller, and the envelope it tightens is exactly the action space an RL policy
would be trained inside. `safety.validate()` stays the last word for whichever of them is
driving.

---

## V5: the backend takes the plant

V5 moves control out of the browser. The frontend used to hold the clock and post an
action every 200 ms; now it asks the backend to run and watches what happens. Nothing in
the simulator changed.

```text
Frontend  ──asks──>  Backend  ──drives──>  Simulator
   (adapter)      (clock, controller, guard)     (physics)
```

### The controller interface

One method, and it never sees the simulator:

```python
from plant_backend.controllers import Constraints, HeuristicController

controller = HeuristicController()
decision = controller.compute_action(
    simulator.get_telemetry(),          # observation: telemetry, which may lie
    constraints=Constraints(dac_max_kw=400.0),
    config=simulator.config,            # the plant's design: ratings and recipe
)
decision.action                         # an Action the simulator accepts as-is
decision.notes                          # why, in words, for the control room
```

Because a controller only takes telemetry, constraints and the design, a heuristic, an RL
policy and an optimiser are interchangeable: `PlantRuntime.controllers` is a registry, and
`POST /api/controller/config {"controller": "..."}` selects one. Only the heuristic exists
today.

### The heuristic

Deliberately not optimal — it plans in kilowatts, in the order the plant cares about, and
converts the plan into shares at the end:

1. **Methanation first** whenever both feedstocks are in the tanks — it is the point of the
   plant and the smallest load.
2. **Then CO₂**: the DAC runs at its cap while the buffer is below 50%, and trickles at 30%
   of it once comfortable. CO₂ starvation is what stops methanation.
3. **Then hydrogen**: everything left goes to the stack. Below the stack's minimum load it
   borrows from the battery to clear the floor, or gives up and charges instead — half a
   stack is worth nothing.
4. **The rest charges the battery**, and within 15 points of the reserve the battery takes
   half the bus *first* instead of last.

Full storage pauses the unit that fills it. Offline units are never commanded; degraded
units still are — they simply produce less, which is the honest response to a derating.

### Constraints: the envelope, and V6's lever

```python
Constraints(
    electrolyser_max_kw=800.0,   # absolute kW caps; None means "whatever it is rated for"
    dac_max_kw=400.0,
    methanator_max_kw=None,
    battery_support_max_kw=None,
    battery_min_soc=0.20,        # reserve nothing may eat into, on top of the pack's own floor
)
```

Constraints can only ever *tighten* what the equipment can do, and the SOC reserve is
applied on top of the battery's configured floor, never below it. Both the controller and
the guard read the same object, so a manual command and an automatic one live inside the
same envelope.

### AUTO and MANUAL

| Mode | Who decides | What the operator does |
| ---- | ----------- | ---------------------- |
| `auto` | the controller, every timestep | watches; retunes constraints; switches mode |
| `manual` | the operator's stored setpoint | moves the sliders (`POST /controller/config {action}`) |

Either way the action goes through `safety.validate()` — there is no path from the browser
to `PlantSimulator.step()` that skips it. Switching mode does not disturb the plant; the
next timestep simply has a different author. Mode, controller and constraints are
*configuration*, so they survive a reset; the run in progress does not.

### The guard (`safety/validation.py`)

An interface guard, not the future safety system. It clamps rather than rejects — the
plant keeps running under a bad command and the adjustment is reported back:

- shares clamped, oversubscription normalised, `NaN`/`inf`/negatives dropped
- power beyond a component's rating or its constraint clamped to the limit
- allocations to an offline unit dropped
- battery support clamped to what the pack can deliver above the reserve, and dropped
  entirely at it

The simulator still enforces its own physical invariants — the guard exists so obviously
invalid commands never reach it, and so the operator is told what changed.

### Where V6 plugged in

Built in V6 (see above); the seam V5 left is exactly the one it uses.

```text
Supervisor (V6)                 reads: GET /api/state, /api/events  (telemetry only)
      ↓                         writes: POST /api/controller/config
constraints / mode / controller  \_ or in-process: runtime.configure_constraints(...)
      ↓
Controller                       PlantRuntime._decide()
      ↓
safety.validate()                unchanged, for every author
      ↓
Simulator
```

A supervisory agent belongs in `backend/src/plant_backend/agent/`, and it steers the same
way an operator does: it changes the envelope, the mode or the controller. It deliberately
gets **no** step-level hook — a supervisor that has to answer every 60 simulated seconds
is not a supervisor, and keeping it out of the fast loop is what keeps that loop
deterministic and cheap. `safety/` is where its authority gets bounded when it grows teeth.

### Frontend changes

Small, on purpose. `usePlantSession` lost its clock and gained an `EventSource`; the
control deck gained an AUTO/MANUAL toggle, the driver's name, and a line of the
controller's own reasoning. In AUTO the sliders show the controller's allocation and are
read-only. Everything else in the control room was already reading `PlantState` and did
not move.

---

## V3: what changed, and what it sets up

**The chain.** V3 turned the hydrogen plant into the full simplified synthetic-methane
plant: DAC → CO₂ storage → methanator → CH₄ storage, plus a four-way power bus. The
simulator now enforces real material dependencies — methane cannot exist without H₂,
CO₂ and power in the same timestep.

Major files added:

- `simulator/src/plant_sim/components/dac.py` — `DirectAirCapture`: linear kW → kg CO₂,
  throttled by CO₂ tank headroom.
- `simulator/src/plant_sim/components/methanator.py` — `Methanator`: fixed recipe
  (0.5 kg H₂ + 2.75 kg CO₂ + 1.2 kWh per kg CH₄), limited by the scarcest input, draws
  only the power its production needs.
- `frontend/src/components/plant-3d/DacUnit.tsx`, `MethanatorUnit.tsx` — the new units.
- `frontend/src/components/plant-3d/GasVessel.tsx` — one vessel component, instanced for
  H₂/CO₂/CH₄ (replaces `HydrogenTank.tsx`).

Major files changed:

- `simulator/`: `config.py` (DAC/methanator/CO₂/CH₄ configs), `state.py` (4-share
  `Action`, extended `PlantState`), `allocation.py` (7-step dispatch), `simulator.py`
  (wiring + totals), `__main__.py`, tests (21 → 33).
- `backend/src/plant_backend/app.py` — `ActionModel` gained `dac_share` /
  `methanator_share`; nothing else moved.
- `frontend/`: `types/plant.ts`, `usePlantSession.ts` (default action), `tokens.css` +
  `palette.ts` (two new carrier tints), `OverviewPage.tsx` (six metrics), `PlantFlow.tsx`
  (full-chain diagram), `BalanceCard.tsx`, `ControlDeck.tsx` (four share sliders),
  `plant-3d/` (layout, visuals, scene), `Plant3DPage.tsx` (HUD).

**Decisions that matter for V4 (faults) and beyond:**

- *Components own their limits.* A fault is a component reporting degraded headroom —
  `intake_limit_kw()` on the electrolyser/DAC and `max_power_kw()` on the methanator are
  the exact seams where derating plugs in; `dispatch()` already handles refusal.
- *Battery support still feeds only the electrolyser.* Deliberate V3 simplification:
  DAC and methanator run on their solar shares, so the plant idles at night except for
  the supported stack. Generalising support to all loads is a controller question, not
  a physics one — do it in `allocation.py` step 2 when a controller needs it.
- *No derived rates in state.* Feedstock consumption rates are not `PlantState` fields;
  views derive them from `ch4_production_kg_h` × the recipe in `PlantConfig`. Keep it
  that way: state carries levels and primary flows, not arithmetic.
- *The simulator still knows nothing about its drivers.* The frontend clock stepped it via
  HTTP in V3–V4; since V5 the backend's own loop does, and the simulator did not notice.

---

## Design notes

- **Simulator has zero dependencies.** No numpy, no framework, no I/O. It can be imported by
  a notebook, a training loop or a test with no setup.
- **Units are fixed and documented:** kW, kWh, kg, seconds. Efficiency losses live inside
  components; powers quoted at the bus are always bus-side.
- **The backend is one application, not a fleet.** One in-memory session, one clock, no
  database, no auth, no persistence, no message broker. It owns control; it does not model
  physics, and the frontend never imports the simulator.
- **The frontend is a control room, not an admin panel.** A warm olive-charcoal console inset
  on sage, with bone panels for the surfaces an operator touches (balance, controls). Quiet
  sentence-case labels in Instrument Sans, wide numerals in Martian Mono, monochrome
  hairline trend charts, and hue rationed to one desaturated tint per energy carrier: sand
  solar, sage stored energy, mist stack feed, chalk hydrogen.
- **Design tokens live in one file.** `frontend/src/styles/tokens.css` holds the palette,
  type and geometry; `.card`, `.card--bone`, `.pill` and `.chip` in `global.css` are the only
  shared primitives. Retheming the console does not touch a component.
- **The twin speaks the same language.** `plant-3d/palette.ts` mirrors those tokens as hex
  for WebGL: same carriers, same rationing. The 3D view is lazy-loaded, so the Overview
  never pays for three.js.
