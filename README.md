# Plant An Agent

An autonomous, intermittently powered industrial plant — starting with the foundation:
**a small working digital twin plus a control room that watches it.**

This first iteration models one thing only:

```text
Solar
  ↓
Power allocation
  ├──→ Battery
  └──→ Electrolyser
             ↓
          H₂ Tank
```

No agents, no RL, no methanation, no 3D plant yet. The point is the architecture those
things will plug into.

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
│   │   ├── components/       one file per piece of equipment
│   │   │   ├── base.py         Component contract (reset + snapshot)
│   │   │   ├── solar.py        SolarArray
│   │   │   ├── battery.py      Battery
│   │   │   ├── electrolyser.py Electrolyser
│   │   │   └── storage.py      GasStorage (the H₂ tank, and later CH₄/CO₂)
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
│       ├── pages/            Overview (the only page that exists yet)
│       ├── components/       shell (top nav, page header) + overview widgets
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
make test           # simulator test suite (21 tests)
make sim            # one simulated day, headless, printed as a table
```

`make sim` is the fastest way to see the model work:

```text
  clock    solar   ->batt   ->elz   batt%    elz    H2 kg/h   tank kg  state
 D1 09:46     3830      830    3000    13.7   3000       57.7     298.5  RUN
 D1 12:46     5000     2000    3000    71.6   3000       57.7     471.5  RUN
 D1 14:16     4698        0    3000   100.0   3000       57.7     558.1  RUN   <- battery full, solar curtailed
 D1 19:31        0        0       0    92.0    700       13.5     790.0  RUN   <- running on the battery
 D1 03:46        0        0       0     5.1      0        0.0     164.2  off   <- battery at min SOC, stack down
```

---

## How the simulator works

The interface is deliberately small:

```python
from plant_sim import Action, PlantSimulator

sim = PlantSimulator()
state = sim.reset()
state = sim.step(Action(battery_share=0.3, electrolyser_share=0.7, battery_support_kw=600))
```

- **`Action`** — how to split the power available *right now*. `battery_share` and
  `electrolyser_share` are fractions of current solar; `battery_support_kw` discharges the
  battery to feed the electrolyser. Anything unallocated is curtailed.
- **`PlantState`** — everything observable: levels (`battery_soc`, `h2_level_kg`) describe
  the plant now, flows (`*_kw`, `*_kg_h`) describe the timestep that just finished.
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

The minimum load is what makes the problem interesting: a trickle of solar is worthless to
the stack, so something has to choose between buffering it or spending stored energy to
lift the stack above its floor. That choice is exactly what an agent will make later.

### Dispatch order (`allocation.py`)

Every timestep, in this order:

1. Split the solar bus by the action's shares; anything unallocated is curtailed.
2. Resolve battery support, clamped by what the stack can absorb *and* what the battery can
   deliver — never discharge into a wall.
3. Net out simultaneous charge/discharge (that solar goes straight to the stack instead).
4. Run the electrolyser, capped by rated power and by room left in the tank. Below minimum
   load it stays off, support is retracted, and its solar spills back to the battery.
5. Charge the battery with its share plus any spill; what it cannot absorb is curtailed.

Energy is conserved by construction, and the tests assert it:

```text
solar_available   = solar_to_battery + solar_to_electrolyser + solar_curtailed
electrolyser_power = solar_to_electrolyser + battery_to_electrolyser
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

---

## Extension points

Everything below is *designed for* but deliberately **not built**.

**More process units.** Subclass `Component` (`reset()` + `snapshot()`), give it domain
verbs that clamp their own limits, instantiate it in `PlantSimulator.__init__`, and add it
to the `components` registry — reset and telemetry come for free. DAC, methanation and a
CH₄ tank are the intended next three; `GasStorage` is already species-agnostic.

**Richer physics.** Each component names its own seams: efficiency curves and degradation in
`electrolyser.py`, capacity fade in `battery.py`, weather and forecast noise behind
`SolarArray.power_at`, pressure and offtake in `storage.py`.

**Faults and events.** Components own their limits, so a fault is a component that reports
degraded headroom. The dispatch order in `allocation.py` already handles refusal gracefully.

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

**3D plant.** A future `3D Plant` view reads the same `usePlant()` session as the Overview
and maps `PlantState` onto meshes. The simulator stays completely independent of any visual
representation — the single-line diagram already proves the state is enough to drive one.

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
