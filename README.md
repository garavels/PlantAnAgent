# Plant An Agent

An autonomous, intermittently powered industrial plant — starting with the foundation:
**a small working digital twin plus a control room that watches it.**

The plant today (V5 — the full simplified synthetic-methane chain, everything that can go
wrong with it, and a backend that runs it):

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

The control room has two views over the same live session: **Overview** (the concise
operational summary, plus scenario/fault controls and the event feed) and **3D Plant**
(a spatial digital twin, where faulted units turn amber or red under a warning beacon).
No agents, no RL yet. The point is the architecture those things will plug into.

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
│   │   │   └── heuristic.py    the rule-based operator that runs the plant today
│   │   └── safety/
│   │       └── validation.py   the guard every action passes through
│   └── tests/                controller, constraints, guard, loop, API contract
│
├── frontend/           the control room
│   └── src/
│       ├── views.tsx         view manifest -> drives router + top nav
│       ├── pages/            Overview + 3D Plant (the twin)
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
make test           # simulator + backend test suite (86 tests)
make sim            # one simulated day, headless, printed as a table
```

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
| `/events`             | GET    | recent simulator events                                        |
| `/faults`             | GET    | fault catalogue + what is scheduled or running                 |
| `/faults`             | POST   | `{type, severity, start_s?, duration_s?, channel?}` — inject   |
| `/faults`             | DELETE | clear every fault (`/faults/{id}` clears one)                  |
| `/scenarios`          | GET    | the scenario catalogue                                         |
| `/scenarios`          | POST   | `{scenario}` — reset the plant into that situation             |

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

**Agents and controllers.** Built in V5 — the seam is `Controller.compute_action()` and
the lever is `Constraints`. See "V5" below for exactly where V6 plugs in. The simulator
never learns who is driving it.

**RL environments.** `reset()` / `step(action)` / `PlantState.to_dict()` are already
Gym-shaped. A `PlantEnv` wrapper adding an observation vector and a reward belongs beside
the simulator, not inside it.

**More views.** `frontend/src/views.tsx` is the manifest that drives both the router and the
top nav; the greyed-out entries (3D Plant, Agent, Metrics, Events, Scenarios) are
placeholders pointing at it. Write a page, flip `ready: true`, and it appears.

**3D plant.** Built. The `3D Plant` view reads the same `usePlant()` session as the
Overview and maps `PlantState` onto meshes — see "The 3D twin" below. Adding a unit to
the twin never touches the simulator: one entry in `plant-3d/layout.ts`, one component,
one flow route.

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

### Where V6 plugs in

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
