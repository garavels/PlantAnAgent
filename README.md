# Plant An Agent

An autonomous, intermittently powered industrial plant — starting with the foundation:
**a small working digital twin plus a control room that watches it.**

The plant today (V4 — the full simplified synthetic-methane chain, and everything that
can go wrong with it):

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
├── backend/            the seam. thin dev API today, agent layer later
│   └── src/plant_backend/
│       ├── app.py            FastAPI routes over one plant session
│       └── session.py        holds the simulator + rolling history
│
├── frontend/           the control room
│   └── src/
│       ├── views.tsx         view manifest -> drives router + top nav
│       ├── pages/            Overview + 3D Plant (the twin)
│       ├── components/       shell (top nav, page header) + overview widgets
│       │   └── plant-3d/     the twin: scene, units, flow network
│       ├── hooks/            usePlantSession: the clock and the link
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
| `backend`   | who drives the plant, and how it is exposed | model physics                       |
| `frontend`  | showing state and collecting operator input | model physics; own plant state      |

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
make test           # simulator test suite (56 tests)
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
  ┌──────────┐   action (JSON)     ┌──────────┐   Action      ┌────────────┐
  │ Frontend │ ──────────────────> │ Backend  │ ────────────> │ Simulator  │
  │ (browser │ <────────────────── │ (session)│ <──────────── │ (twin)     │
  │  clock)  │   PlantState[]      └──────────┘   PlantState  └────────────┘
  └──────────┘
```

1. The browser holds the clock. While playing, it ticks every 200 ms and asks the backend to
   advance `speed` timesteps (`POST /api/step`).
2. `PlantSession` passes the action to `PlantSimulator.step()` and appends the resulting
   states to a rolling history.
3. Every frame comes back to the frontend, which keeps the last ~24 simulated hours for the
   trend charts and renders the newest one.
4. Every number on screen — readouts, bar fills, tank level, conduit brightness and flow
   speed — is a field of `PlantState`. The UI adds no physics of its own.

Dev API surface (all under `/api`, proxied from Vite):

| Route     | Method | Purpose                                                       |
| --------- | ------ | ------------------------------------------------------------- |
| `/health` | GET    | liveness + current step index                                 |
| `/config` | GET    | plant design (sizing, limits) so the UI can scale its gauges  |
| `/state`  | GET    | current state + recent frames (a page reload redraws instantly) |
| `/step`   | POST   | `{action, repeat}` → every frame produced                     |
| `/reset`  | POST   | back to the initial condition                                 |
| `/events` | GET    | recent simulator events                                       |
| `/faults` | GET    | fault catalogue + what is scheduled or running                |
| `/faults` | POST   | `{type, severity, start_s?, duration_s?, channel?}` — inject  |
| `/faults` | DELETE | clear every fault (`/faults/{id}` clears one)                 |
| `/scenarios` | GET | the scenario catalogue                                       |
| `/scenarios` | POST | `{scenario}` — reset the plant into that situation          |

Every state-carrying response also carries `faults`, `events` and the loaded `scenario`,
and the `state` it returns is **telemetry**, not the simulator's private truth.

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

**Agents and controllers.** `PlantSession.step()` is the seam:

```python
action = request.action              # manual control (today)
action = agent.decide(self.state)    # autonomous control (later)
```

The simulator never learns who is driving it.

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

### What V5 can build on

- **Observation**: `GET /api/state` (telemetry + per-unit `health` + `plant_status`),
  `GET /api/events`, `GET /api/faults`. A controller needs nothing else to see the plant.
- **Control**: `POST /api/step` with an `Action` — unchanged since V1.
- **Situations**: `POST /api/scenarios` for reproducible episodes, `PlantSimulator.run()`
  for headless ones.
- **The seam**: `PlantSession.step()` still decides who drives. A supervisory agent reads
  telemetry and events and returns an `Action`; the simulator does not change.

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
- *The simulator still knows nothing about its drivers.* The frontend clock steps it via
  HTTP today; a heuristic controller or agent slots into `PlantSession.step()` tomorrow.

---

## Design notes

- **Simulator has zero dependencies.** No numpy, no framework, no I/O. It can be imported by
  a notebook, a training loop or a test with no setup.
- **Units are fixed and documented:** kW, kWh, kg, seconds. Efficiency losses live inside
  components; powers quoted at the bus are always bus-side.
- **The backend is a seam, not an application.** One in-memory session, no database, no auth,
  no persistence. It exists so the frontend never imports the simulator.
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
