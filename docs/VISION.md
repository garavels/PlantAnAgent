# Intermittent Abundance — Project Vision

## 1. What we are building

We are building an autonomous control and simulation platform for an industrial plant powered by intermittent renewable energy.

The plant is represented by a digital twin: a software simulator that tracks the state of components such as solar generation, batteries, electrolysers, gas storage, direct-air capture, and methanation. On top of that simulator, we will progressively add controllers, machine-learning policies, predictive risk models, and an LLM-based supervisory agent.

The long-term goal is not simply to simulate a plant. The goal is to explore how an intermittently powered industrial system can operate autonomously, anticipate future problems, recover from faults, and make better decisions under uncertain energy availability.

A useful mental model is:

```text
Simulator = the world
Controller = how the plant operates minute to minute
HEPA = what could go wrong if we follow a proposed plan
Agent = supervisory intelligence for unusual or strategic situations
Frontend = how a human sees and interacts with the system
```

---

## 2. Core problem

Renewable power is cheap when available but intermittent. Industrial equipment cannot always turn on and off arbitrarily, and different components depend on one another.

A locally optimal action can create a future bottleneck. For example, using all available solar power to produce hydrogen now may leave too little battery capacity or too little CO2 production later, forcing the methanation process to stop.

The system therefore needs to reason over:

- changing renewable-energy availability;
- inventories such as battery charge, hydrogen, and CO2;
- component limits and operating constraints;
- future weather;
- component degradation and faults;
- competing production objectives;
- uncertainty about what will happen next.

---

## 3. Repository architecture

The project should remain a monorepo with three main areas:

```text
/
├── frontend/
├── backend/
└── simulator/
```

### `simulator/`

The digital twin. It defines the plant state and the rules governing how that state changes over time.

It should not depend on the frontend, the LLM, or any particular controller.

Conceptually:

```text
current state
    +
action
    +
environment / disturbances
    ↓
simulator.step(...)
    ↓
new state + telemetry + events
```

### `backend/`

The intelligence and orchestration layer.

Eventually this may contain:

```text
backend/
├── api/
├── agent/
├── controllers/
├── hepa/
├── safety/
└── orchestration/
```

Responsibilities include:

- exposing APIs to the frontend;
- running controllers;
- hosting an RL policy;
- running the supervisory agent;
- evaluating plans with HEPA;
- applying safety constraints;
- connecting all intelligent components to the simulator.

### `frontend/`

The human-facing control room.

It visualizes simulator state, provides controls, displays alerts and agent activity, and eventually includes a 3D representation of the plant.

The 3D plant is only a visualization of simulator state. It is not the digital twin itself.

---

## 4. Digital twin

The simulator is the foundation of the project.

The earliest version should remain deliberately simple:

```text
Solar
  ├──→ Battery
  └──→ Electrolyser
             ↓
          H2 Tank
```

Example state:

```python
{
    "time": 12.0,
    "solar_power_kw": 80,
    "battery_soc": 0.60,
    "electrolyser_power_kw": 50,
    "h2_storage_kg": 12.4
}
```

Example interaction:

```python
state = simulator.reset()
state, telemetry, events = simulator.step(action)
```

Later versions can add:

- direct-air capture;
- CO2 storage;
- methanation;
- methane storage;
- component temperatures;
- ramp-rate limits;
- startup and shutdown constraints;
- degradation;
- sensor failures;
- equipment outages;
- weather-driven solar generation;
- stochastic disturbances.

The simulator should be modular enough that each physical component can evolve independently.

### Dispatch rules

Every timestep the power router (`allocation.py`) decides where each available kW goes. The rules are applied in order:

1. **Split solar by the action's shares.** Anything the action leaves unallocated is curtailed.
2. **Resolve battery support.** The battery may discharge to help the electrolyser, but only up to its own power and energy limits and only up to what the electrolyser can still absorb.
3. **Cancel round trips.** Charging and discharging the battery in the same step is wasteful, so the overlap is sent straight to the electrolyser instead.
4. **Run the electrolyser.** The stack only starts if total power is above its minimum load. It is capped by rated power and by the remaining H2 tank headroom. If it cannot run, its solar share spills back to the battery.
5. **Charge the battery.** The battery absorbs its share plus any spill from the electrolyser. What it cannot absorb is curtailed.

These invariants hold by construction:

```text
solar_available = solar_to_battery + solar_to_electrolyser + solar_curtailed
electrolyser_power = solar_to_electrolyser + battery_to_electrolyser
```

The hard minimum load on the electrolyser is what makes dispatch interesting. A trickle of solar may be too small to run the stack, so the agent must choose between buffering it in the battery or discharging the battery to lift the stack above its floor.

---

## 5. Frontend vision

The frontend should feel like a modern industrial control room rather than a generic admin dashboard.

The first useful version only needs an overview page showing:

- solar power;
- battery state of charge;
- hydrogen production;
- hydrogen storage;
- a simple live plant-flow diagram;
- play / pause / reset controls.

The frontend can later expand into several views, but these should not all be built initially.

Possible future sections:

- Overview
- 3D Plant
- Flows
- Metrics
- Agent
- Events & Alerts
- Controls
- Scenarios
- Settings

### 3D plant

A 3D plant visualization is desirable for presentation value, but it is not a prerequisite for the system to work.

It should eventually reflect simulator state. Examples:

- a running component visibly active;
- an offline component darkened;
- a faulted component highlighted;
- flow intensity changing with throughput;
- storage tanks visually filling or emptying.

The same simulator must work even if the 3D view is removed entirely.

---

## 6. Controllers

The controller handles routine plant operation.

At first this should be a simple heuristic controller. For example:

- prioritize the electrolyser when solar is abundant;
- charge the battery with surplus power;
- preserve a minimum battery reserve;
- respect component limits.

This gives us a deterministic baseline before introducing learning.

Later, the same simulator can be exposed as an RL environment.

Conceptually:

```python
observation = env.observe()
action = policy(observation)
next_observation, reward, done, info = env.step(action)
```

A learned RL policy can then handle continuous operational decisions such as power allocation.

The purpose of RL is not merely to create a benchmark. It may become a specialized low-level controller used by the larger autonomous system.

---

## 7. Supervisory agent

The LLM-based agent operates at a higher level than the controller.

It should not micromanage every valve or power setpoint. Its role is closer to a remote plant operator.

Typical responsibilities:

- investigate abnormal telemetry;
- diagnose likely faults;
- inspect component status;
- reason about unexpected situations;
- change operating modes;
- modify controller constraints;
- choose between available controllers or strategies;
- request simulations or diagnostics;
- trigger safe shutdown or degraded operation when necessary.

Example:

```text
Hydrogen output unexpectedly drops
        ↓
Agent inspects telemetry
        ↓
Electrical input normal
Temperature abnormal
        ↓
Probable electrolyser degradation
        ↓
Agent reduces allowed electrolyser capacity
        ↓
Controller continues operating under new constraints
```

This creates a separation of timescales:

```text
Controller / RL policy:
"What should I do right now?"

Agent:
"What is happening, and should our strategy change?"
```

---

## 8. HEPA — predictive risk model

HEPA is a future predictive-risk module inside the backend.

It does not directly control the plant. It evaluates the likely consequences of a proposed operating plan.

Its target is conceptually:

```text
P(negative event within horizon | current state, history, forecast, proposed actions)
```

Possible predictions:

- probability of battery depletion within 6 hours;
- probability of CO2 shortage within 24 hours;
- probability of forced reactor shutdown;
- probability of thermal-limit violation;
- probability of product-quality failure.

A controller may propose multiple plans. HEPA scores their future risk, safety rules reject invalid plans, and the system chooses among the remaining candidates.

Conceptually:

```text
Controller proposes plans
        ↓
HEPA predicts risk
        ↓
Safety filter
        ↓
Best acceptable plan
        ↓
Simulator / real plant
```

HEPA becomes useful only if it improves decisions compared with simpler approaches.

Training data can be generated from the simulator by running many scenarios with different weather, actions, disturbances, and injected faults, then recording what happens over future horizons.

---

## 9. Intended final architecture

The eventual system may look approximately like this:

```text
                         FRONTEND
                            │
                            ↓
                         BACKEND
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ↓               ↓                ↓
      Supervisory Agent  Controllers       HEPA
            │          heuristic / RL   predictive risk
            │               │                │
            └───────────────┼────────────────┘
                            ↓
                       Safety Layer
                            ↓
                         Actions
                            ↓
                     PLANT SIMULATOR
                            ↓
                state / telemetry / events
                            │
                            └────────→ backend + frontend
```

This is a target architecture, not the required architecture for the first implementation.

---

## 10. Development stages

The project should be built incrementally. Every stage should leave behind something that works on its own.

### Stage 0 — Foundation

Create the monorepo and clean boundaries between `frontend`, `backend`, and `simulator`.

### Stage 1 — Minimal digital twin

Build:

- solar model;
- battery model;
- electrolyser model;
- H2 tank;
- discrete simulation loop;
- simple actions and telemetry.

No agent. No RL. No complex backend.

Success condition: the plant can run through a simulated day and produce sensible state transitions.

### Stage 2 — Minimal frontend

Build a polished overview UI connected to the simulator.

Success condition: the user can watch the plant evolve and control play / pause / reset.

### Stage 3 — Complete plant model

Add the rest of the synthetic-fuel chain:

- DAC;
- CO2 storage;
- methanator;
- methane output;
- additional physical constraints.

Success condition: energy and material flow coherently through the full plant.

### Stage 4 — Faults and scenarios

Add:

- degradation;
- component failures;
- sensor faults;
- weather disturbances;
- scenario injection.

Success condition: the simulator can produce meaningful abnormal situations.

### Stage 5 — Backend and simple controller

Introduce the backend properly and move operational decision-making into a controller layer.

Success condition: the frontend no longer needs direct knowledge of simulator internals.

### Stage 6 — Supervisory agent

Add the LLM agent and a controlled tool interface.

Success condition: the agent can identify selected faults and make sensible high-level recovery decisions.

### Stage 7 — RL environment

Expose the simulator through an RL-compatible environment and train a policy, potentially using Prime Intellect.

Success condition: the learned controller performs useful continuous operation and can be compared against the heuristic baseline.

### Stage 8 — Agent + learned controller

Allow the supervisory agent to invoke or constrain the learned controller.

Success condition: routine control and high-level reasoning operate as separate but coordinated layers.

### Stage 9 — HEPA

Generate synthetic trajectories from the simulator, train the predictive-risk model, and integrate it into plan selection.

Success condition: HEPA improves risk-aware operation on held-out scenarios.

### Stage 10 — Presentation layer

Only after the core system works, invest heavily in presentation quality:

- high-quality 3D plant;
- animated flows;
- fault visualization;
- agent trace visualization;
- scenario playback;
- comparison views;
- demo-specific polish.

---

## 11. What not to do early

Avoid premature complexity.

Do not begin with:

- realistic chemical simulation;
- a large microservice architecture;
- databases unless clearly needed;
- authentication;
- deployment infrastructure;
- nine fully developed frontend pages;
- elaborate 3D work;
- RL before the simulator is stable;
- an LLM directly controlling every timestep;
- HEPA before enough simulated data exists.

Each sophisticated layer depends on the simpler layers beneath it.

---

## 12. Main technical thesis

The project explores a layered form of autonomy rather than trying to make one model do everything.

```text
Physics simulator
    ↓
Specialized controller / RL policy
    ↓
Predictive-risk model
    ↓
Safety constraints
    ↓
LLM supervisory reasoning
```

Different components solve different problems.

The controller handles fast repetitive decisions. HEPA estimates future risk. The safety layer enforces hard constraints. The agent handles interpretation, diagnosis, strategy changes, and exceptional situations.

---

## 13. Demo vision

A strong final demo should make the architecture obvious without requiring explanation.

Example sequence:

1. The plant runs normally under changing solar conditions.
2. The UI shows live power and material flows.
3. A fault is injected into the electrolyser.
4. Hydrogen production falls and telemetry becomes abnormal.
5. The supervisory agent investigates the issue.
6. The agent identifies likely degradation and changes operating constraints.
7. The controller adapts while the plant remains operational.
8. In a later version, HEPA can warn about a future shortage before it happens.
9. The 3D plant visually reflects the changing state throughout the sequence.

The audience should be able to understand the core idea in seconds:

> This is a simulated renewable-powered industrial plant that can operate itself, anticipate problems, and recover when things go wrong.

---

## 14. Guiding principles

1. Build the world before building intelligence for the world.
2. Keep simulation, intelligence, and visualization separate.
3. Start with simple models and increase realism only when it changes decisions.
4. Every intelligent component needs a clear job.
5. Prefer a working end-to-end system over an isolated sophisticated model.
6. Optimize presentation after the architecture is credible.
7. Every stage should remain independently demoable.
8. Treat the final architecture as a direction, not a constraint on early implementation.

---

## 15. One-sentence description

**An autonomous digital twin for intermittently powered industrial plants that combines simulation, learned control, predictive risk modeling, and LLM-based supervision to keep production efficient and resilient under changing energy availability and faults.**
