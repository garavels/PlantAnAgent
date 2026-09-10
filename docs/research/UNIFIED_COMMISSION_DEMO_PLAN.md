# Unified commission demo: product and presentation plan

Status: **user-steered product plan, grounded in source revision `ed5a24e`; implementation remains pending. This is not an experiment protocol.**

Date: **2026-09-10**

User refinement: make stochastic weather and forecasts visually legible through
clouds in the 3D plant and a satellite-like overhead view. Keep Overview separate
from dedicated feature panels, especially Economics, so the main navigation replaces
scrolling through a multipurpose page. This supersedes the earlier three-view grouping;
the shared session, evidence boundaries and research sequence remain accepted direction.

## Product objective

Present one virtual-commissioning workspace for a renewable synthetic-methane plant.
An unfamiliar viewer should understand its obligation within 30 seconds and, within
five minutes, follow one decision from the information available at the time to its
physical and economic consequences. Assume a mixed technical/commercial commission;
the exact event rubric, allotted presentation time and deployment requirements still
need to be established before the final rehearsal scope is fixed.

The current beta task supplies a concrete mission: 240 kg CH4 every six hours over
seven days, under uncertain renewable availability and opportunity prices, equipment
constraints, and four independent daily resource-return requirements. Those are
synthetic task parameters, not a site commitment or demonstrated performance.

The product proposition is scheduling, explainable execution and reproducible
commissioning. Controller identity is inspectable. The interface must remain useful
whether the eventual recommended controller is MPC, a heuristic, PPO or a hybrid.

## What the repository currently presents

The V2 Lab now displays genuine V2 public data and executes manual fixture decisions.
However, the top-level Overview, 3D Plant, Agent, Events and Scenarios still consume
the V1 `PlantProvider`; V2 has its own page-local hook. Metrics is an unimplemented
navigation item. Moving between these pages can therefore change the underlying
simulation while leaving the product shell unchanged.

The process diagram and 3D geometry are reusable. Their equipment labels currently
derive from shared V1-shaped `PlantState` and `PlantConfig`, so they do not explain
the separately modelled V2 electrolyser commitment modes and timers. The diagram is
also difficult to read when compressed beside the resource column in a narrow app
panel. The current long V2 page exposes data without connecting selected equipment,
forecast issues, plans, settlements and outcomes through one inspectable moment.

The following audit distinguishes what exists in the research runtime from what the
current public demo API actually supplies. Earlier alpha specifications describe
intermediate stages; the beta controller build and current source define the present
135-observation/four-action boundary.

## V1-to-V2 inventory and graphical treatment

“Visible” means present in the current V2 Lab. “Projection” means the research runtime
already carries relevant records but an operator-facing contract still needs work.
“Later integration” means more than rearranging existing graphics.

| Addition relative to the inherited V1 task | Present state | Proposed presentation and purpose |
| --- | --- | --- |
| Recurring methane offtake, exact deadlines and shortfall settlement | Visible: obligations, delivered/unmet kg and settlement table | Persistent next-delivery strip; deadline marks on the shared timeline; requested/delivered/shortfall bars. Distinguish produced, stored and delivered methane. |
| Seven-day operation, 15-minute decisions and one-minute physical execution | Visible: current clock and bounded manual steps; no historical frame playback | One time cursor shared by every view; day/cycle ribbon; decision ticks with minute detail on demand. Advancing the simulator and playing a recording are separate controls. |
| Seeded causal solar weather and issued forecasts | Visible: current solar and four aggregate mean/p10 forecast windows; no spatial weather model | Cloud/shading illustration in Plant and a dedicated Weather panel with synthetic overhead view, observed reduction and issued forecasts. Preserve issue age, aggregate windows and unknown night-time conditions. No invented solar p90, spatial forecast or smooth hourly precision. |
| Stochastic opportunity prices and revisable 24-hour mean/p10/p90 forecasts | Visible: current price and latest issue; previous issues are not retained for UI use | Price band beneath solar; compare the issue used by a decision with later observed prices and subsequent revisions. Clearly distinguish a price forecast from an observed price. |
| Electrolyser cold/hot startup, standby, cooldown, minimum up/down and ramp limits | Projection: commitment snapshot exists outside the currently served `PlantState` | Equipment inspector and mode lane; show actual mode, startup timer, time before an allowed stop/restart, and ramp headroom. Use the active commitment profile, not V1 load-threshold heuristics. |
| Load-dependent production and measured equipment exposure | Projection: nonlinear response and startup/standby energy, starts, forced/aborted stops, low-load time and load movement exist | Equipment detail shows power versus H2 production, nonproductive energy and exposure counters. These explain why an immediate load change may not yield immediate hydrogen. |
| Common four-target command interface for electrolyser, DAC, methanator and battery charging | Visible: requested and mean applied power | Four aligned target tracks, linked to the corresponding plant units. Make battery charging a ceiling, and distinguish direct charging from recovered spill. |
| A minute dispatcher and unchanged independent guard shared by controllers | Visible: intervention count/notes and digests; projection needed for detailed allocation/guard attribution | Decision inspector follows requested target → feasible allocation → guard change → physical application. Display reasons only when supported by execution records. A dispatch gap alone is not a guard intervention. |
| Causal service MPC, tariff-aware MPC and beta-native economic MPC | E-beta's actual power plan is visible; M24 is implemented offline but absent from the demo selector; planned inventories/deliveries need projection | Planning view stacks equipment schedules and predicted inventories with delivery and return boundaries. Label model predictions. E-beta maximizes service first, then optimizes its declared economic objective within its constraints. |
| Daily battery/H2/CO2/CH4 return constraints against a fixed opening reference | Visible: signed current margins and an aggregate pass/fail per cycle; per-resource settlement detail needs projection | Four separate inventory tracks with fixed return markers; seven-cycle settlement matrix. A temporary negative margin means “to restore by settlement”; it is not an immediate safety breach. |
| Explicit EUR service value, shortfall, renewable opportunity cost, wear reserves and endowment liabilities | Visible: signed cumulative ledger | Dedicated Economics panel with a reconciled value waterfall, contribution history and assumptions. Keep it out of the Overview's main content. Do not turn renewable opportunity cost into invented grid purchases, trading revenue or ROI. |
| Private, seeded, finite DAC outages | Current component condition is visible; incident onset/recovery history is not projected | Highlight the observed unit and affected flows; chronological incident marker; before/after plan and delivery consequences. Future outage occurrence or recovery time must remain unavailable to the decision view. |
| Measurable forecast error, nonlinear/model mismatch and target-realization gaps | Underlying records exist; the UI lacks decision-linked historical context | Select a decision to compare its forecast, predicted production/inventory and realized interval. Separate documented mechanisms from tentative explanations. This is also where future PPO contributions could be diagnosed. |
| Transparent heuristic adapters, feasible future-information planner and LP upper bound | Offline research tools and development results; only H is in the interactive demo | Evidence explains the comparator's information access and task version. Future-information planners and relaxed bounds appear only as labelled research references, never as selectable causal operating controllers or beta results. |
| H/M24/E-beta/PPO benchmark, failure retention and detached semantic verification | Software exists; no admitted trained beta PPO campaign outcome | Evidence offers controller information-access comparison, complete arm status and eligibility reasons. H is forecast-blind; M24 uses solar but not market/cycle objectives; E-beta and PPO share the full public beta surface. Do not rank incomplete/ineligible comparisons. |
| Scalar, PI-Lagrangian and I-Lagrangian PPO, strict checkpoint/seed/source custody | Infrastructure and frozen 3-family × 5-optimizer-seed design; campaign custody/analysis still incomplete | Evidence drawer shows implementation/training/selection/held-out stages only from factual records. No trained-policy selector, training-success animation or performance tile before admission and results exist. |
| Versioned contracts, command/execution digests and experiment provenance | Some identifiers visible; full verification evidence stays outside routine telemetry | Compact evidence status, source/task/controller identities, verifier scope and expandable provenance. An identifier alone must not earn a “verified” badge. |

Resource return is evidence about repeated inventory use within this simulator. It
does not establish water sustainability, lifecycle emissions, certified renewable
fuel, equipment remaining life or site-level profitability; those require additional
models and calibration. Avoid corresponding headline metrics in the present demo.

HEPA, the supervisory rule/LLM agent, natural-language scenario authoring, general
fault injection and operator envelope changes are inherited capabilities or intended
integrations, not completed V2 additions. Preserve their V1 implementation and plan
explicit V2 connections rather than implying the old pages operate on the V2 run.

## One product, independently accessible feature panels

Use a compact sidebar rather than squeezing every destination into a horizontal
tab bar. Overview, Plant, Weather, Planning, Deliveries & resources, Economics,
Events and Evidence are independent destinations, not sections on one long page.
All share the same run identity, controller, cursor, selected equipment/event and
evidence status. Scenario setup is reached from the mission header.

| Panel | Responsibility | Keep elsewhere |
| --- | --- | --- |
| Overview | Compact mission summary, plant status, next delivery, current renewable availability and actionable exceptions; links to the relevant panel | Detailed economics, full forecast charts, equipment tables and event history |
| Plant | Large interactive 3D plant with weather effects, process-diagram alternative and equipment inspector | Full planning or financial analysis |
| Weather | Synthetic overhead site/weather view, observed solar attenuation, forecast windows and revisions | Market-price analysis and invented geographic weather claims |
| Planning | Equipment schedule, predicted inventory and obligations; concise solar/price context tied to the selected decision | A second full Weather or Economics page |
| Deliveries & resources | Contract settlements, production/stored/delivered distinction, inventory trajectories and daily return requirements | Cost accounting |
| Economics | Realized EUR contributions, cost/wear assumptions and operating-value history | General plant telemetry |
| Events | Chronological faults, mode changes, guard actions and delivery/cycle events with links to the relevant context | An undifferentiated stream embedded in Overview |
| Evidence | Controller comparison, eligibility, provenance, research status and verification scope | Routine operational readouts |

The Overview should fit the agreed presentation viewport without vertical scrolling.
Each feature panel presents its primary visual immediately; secondary tables can
expand or scroll locally. This is organized access, not a requirement to compress
all information into a single screen. Unavailable features do not receive empty
placeholder pages; panel delivery follows the implementation increments below.

```text
Plant An Agent       Mission | Controller | Day/cycle | Simulation / Recording
┌────────────────────┬──────────────────────────────────┬─────────────────────┐
│ Overview           │                                  │ Selected context    │
│ Plant              │ One focused feature panel        │ Equipment, decision │
│ Weather            │                                  │ or incident         │
│ Planning           │ Plant: 3D / process diagram      │                     │
│ Deliveries         │ Weather: overhead + forecast     │ Relevant data and   │
│ Economics          │ Economics: value decomposition   │ explanation         │
│ Events             │                                  │                     │
│ Evidence           │                                  │                     │
└────────────────────┴──────────────────────────────────┴─────────────────────┘
Shared time cursor and mode-specific controls; changes persist across panels
```

### Overview and Plant have separate jobs

Overview is the mission landing page. Its visual summary should answer what the
plant is doing, what is due next and what needs attention. A compact site preview
can carry current cloud/solar conditions and open Plant or Weather on selection.
Do not move the existing V2 Lab's long stack of panels into this landing page.

Make the existing 3D plant the desktop presentation focal point, with a one-click
process-diagram alternative. Selected equipment opens the same inspector in either
representation. Keep the mission context visible and move dense forecast tables,
delivery ledgers and economic contributions into their dedicated panels.

Add V2-specific labels through a separate presentation adapter: commitment mode,
timers, actual versus requested power and measured production. Add a delivery/outlet
indication for real offtake settlements, rather than depicting CH4 as an endlessly
filling tank. Animate only observed or recorded events; freeze flows when paused.
Do not infer “on track” from current CH4 storage alone. A published model prediction
may support “planned delivery”; actual delivery receives its own settlement status.

### Weather: make renewable uncertainty physically understandable

The visual sequence is **cloud/shading change → available solar change → plant
response**, followed by inspection of what the forecast had predicted. Weather has
two synchronized presentations:

- **In the 3D plant:** a lightweight procedural cloud layer, softened sunlight and
  field shading make reduced solar availability legible. Keep sun position tied to
  time of day and cloud attenuation separate from sunrise/sunset. Cloud opacity and
  the illustrated shading respond to the observed attenuation at the selected time;
  they do not calculate a second version of plant power. Exact kW stay visible.
- **In Weather:** an overhead view of the same synthetic site and surroundings,
  inspired by satellite presentation, shows the cloud layer and the solar field.
  A focused companion chart shows clear-sky potential, available power and the
  selected issued forecast. Selecting a forecast window previews its aggregate
  expected availability and lower estimate with an explicit forecast label.

The current backend is a time series of a site-wide resource factor multiplying a
clear-sky curve. It has no cloud geometry, wind vector, georeferenced region or satellite
imagery. The initial overhead view therefore uses the existing site geometry and
stylized terrain, with a persistent **Synthetic weather view** label. Spatial cloud
shape/movement is an illustration; it must not imply measured cloud coverage, an
incoming front, direction of travel or a predicted arrival time. Actual satellite
imagery or regional meteorology is a later site-data integration.

The visual adapter should use these rules:

1. Read only the selected public snapshot/recording and explicitly published forecast
   issue. Preserve the physical sample time; boundary-stamped telemetry may describe
   the preceding physical interval. Do not sample the private future weather tape or
   generator state to animate clouds.
2. Use observed available versus clear-sky potential power to illustrate attenuation
   when daylight and known solar health permit attribution. Keep equipment degradation
   separate; a solar fault must not silently create a cloud. If attenuation cannot be
   isolated, say so instead of deriving weather from low output alone.
3. At night or negligible clear-sky potential, cloud attenuation is **unobserved**.
   V2 intentionally masks its latent resource factor then. Display night-time lighting
   and the lack of a current observation; do not turn the public zero into overcast
   conditions or reconstruct private cloud truth. Issued future forecasts remain usable.
4. Forecast means/p10 are aggregate PV availability windows, not cloud percentages,
   geographic rasters or per-minute trajectories. Compare like-for-like interval
   observations when showing forecast error. Use window transitions, not fabricated
   precise cloud movements. No solar upper quantile is currently published.
5. Use one deterministic visual adapter for 3D and overhead presentation. Scrubbing
   reproduces the same visual state; pausing freezes atmospheric movement as well as
   plant flows. Optional interpolation is a display transition, not a measured path.
   Respect reduced motion and provide an inexpensive flat cloud-overlay fallback.

Use “reduced sunlight” or “reduced solar availability” in the UI. This weather model
reduces incoming resource; it does not model a weather-driven loss of panel conversion
efficiency. Keep curtailed power separate as well: unused available energy is not
energy lost to clouds.

The public presentation contract needs sample time, observation/forecast/unknown
status, clear-sky potential, available power, attribution/health information and
forecast issue/window references. These are projections/derived display fields;
they must not expand the frozen controller observation or change weather physics.

### Planning: why is this schedule sensible under the available information?

Use aligned lanes for solar, market price, the four equipment targets, predicted
inventory and delivery/return requirements. Share the same absolute episode time
axis and vertical cursor. Preserve forecast window resolution and units.

Click a decision to see exactly the forecast issued for it, its retained plan, and
the subsequent completed interval. The current Lab can show a newer forecast beside
the previous decision's retained plan; timestamps identify this, but that combination
must not become the explanation for the earlier decision. Retaining original issues
is a backend dependency.

Use “forecast,” “planned” and “observed” labels plus line styles. Do not describe
E-beta as optimizing uncertainty quantiles: it currently uses means. If the heuristic
is selected, show its actual immediate targets and its declared lack of a forward
plan. Do not manufacture a forecast-aware rationale for it.

### Dedicated outcomes, economics and evidence

Deliveries & resources presents service and daily resource returns. Economics owns
the EUR decomposition, assumptions and contribution history. Events provides a
searchable/filterable chronology with links to equipment and decisions. Evidence
owns comparative results and verification. Selecting a delivery, incident or value
contribution preserves the shared cursor and opens the appropriate context; it does
not require finding the matching chart further down a multipurpose page. A failure
remains inspectable and cannot be replaced with a successful reset's history.

A later fixture comparison can place H and E-beta beside one another on the same
predeclared scenario with synchronized cursors. Their differing information access
remains explicit. It is an illustrative comparison, not held-out evidence or an
equal-information claim. Extend to M24 and admitted PPO only through the appropriate
adapters and evidence gates. Earlier alpha results retain their original task labels.

### Reuse and migration

| Existing surface | Destination in the unified product |
| --- | --- |
| Overview | A compact V2 mission landing page, separate from detailed feature panels. |
| V2 Lab | Distribute existing content across the shared feature workspace; retire it as the main multipurpose destination. |
| 3D Plant | Dedicated Plant panel with a process-diagram alternative, weather and equipment detail; reuse geometry and flow components. |
| V2 solar forecast | Weather panel with synthetic overhead view; concise linked context remains in Planning. |
| V2 schedule | Planning, sharing the selected decision and equipment. |
| V2 delivery/resource blocks | Deliveries & resources. |
| Metrics placeholder and V2 economics | Dedicated Economics panel with real available quantities. |
| Events | Dedicated Events panel, plus contextual timeline markers and incident inspector. |
| V2 provenance/research status | Evidence panel and expandable details. |
| Scenarios | Mission setup/new-run flow; initially the existing fixed V2 fixture. |
| Agent | Decision/execution explanation in the inspector; separately identify any future supervisor or HEPA output. |
| V1 controls and scenarios | Explicit legacy workspace reached through a secondary entry; preserve implementation and API semantics. |

Promote the unified V2 workspace to the primary demo entry after route compatibility
checks. Preserve old V1 destinations through explicit legacy routes/aliases. Avoid
mounting the V1 stream provider behind ordinary V2 navigation. The same visual shell
may host either runtime, but its session capability contract must distinguish them.

## Visual direction

Retain the repository's established industrial console identity. Core colors are
sage room `#d5dbca`, charcoal console `#0e120f`, raised surface `#1c221d`, bone controls
`#e3e8db`, light text `#dfe5d6` and olive accent `#b3cb92`. Keep existing carrier colors
consistent across 2D, 3D and timelines, while pairing alarm color with text/icons.
Use neutral chart styling or explicit legends when market price and gas flows appear
together so a reused color cannot imply a physical connection.

Keep Instrument Sans for labels/explanations and Martian Mono for aligned quantities
and clocks. Proposed type scale: 28 px view title, 18 px panel title, 14–16 px controls
and operational text, 12–13 px secondary labels. Essential chart labels must remain
legible on a projected screen; enlarge/reflow instead of shrinking a large diagram.

Use left-aligned composition, a dominant plant or shared time axis and one contextual
inspector. Avoid turning every V2 feature into an equal-weight card. At narrow widths,
stack the inspector beneath the main content and summarize equipment labels while
preserving full details on selection. Retain keyboard selection, visible focus,
reduced motion and the existing 2D fallback for unavailable WebGL.

Weather-driven visual change is a priority: a viewer should see the connection
between cloud shading, reduced solar availability and plant response. Selecting a
delivery or observed incident then connects the relevant equipment, earlier plan and
actual consequences across panels. Keep weather graphics informative and lightweight
so they do not obscure equipment labels or reduce interaction responsiveness.

## Backend work required for an honest interface

1. **Shared session and capabilities.** Lift V2 state above all feature panels. Include
   session/run/revision, displayed boundary, simulation versus recording mode and
   available actions. Preserve stale-command rejection, serialized writes and the
   last-complete-snapshot failure behavior. Controller reset and simulation advance
   are presently supported; bounded operator setpoints, live takeover and fallback
   are not interchangeable with those controls.
2. **Decision-linked public records.** Retain the public input issue, command, safe
   plan projection, completed physical result, economics and events under a common
   decision identity. Add commitment state/profile, predicted inventories/deliveries,
   per-resource settlement detail and exposure counters. Never serve raw private
   future schedules or solver/audit manifests to populate a chart.
   Add the weather presentation contract described above, keeping resource,
   equipment-health and curtailment effects distinct.
3. **Historical frames and timeline.** Store complete public decision frames, with
   bounded, separately fetched minute execution detail where needed. The six-number
   history currently served is insufficient for plant replay. Page large recordings
   and keep raw artifacts outside Git; commit only compact manifests and digests.
   Scrubbing reads a recording and cannot execute a controller or mutate the plant.
4. **Structured explanations.** Start with deterministic statements supported by
   source fields: mode/timer, target adjustment, observed shortage, plan output,
   measured exposure and settlement. Link each statement to its record. Assertions
   about why a particular objective caused a decision need controller diagnostics;
   causal superiority needs experiments. An LLM can later phrase bounded evidence,
   but cannot invent the missing attribution.
5. **Result admission.** Keep fixture recordings, research reports and eventual
   validated policies as distinct data products. Verify compatible identities before
   comparisons. Display missing arms, interrupted runs, failed gates and verification
   scope. Do not add a performance badge based solely on a digest or successful load.

For recording playback, an operator-perspective cursor shows only information that
was available at that time. Future incident markers may be shown in an explicitly
retrospective overview, but must never be presented as prior controller knowledge.
Detailed private research truth remains outside the ordinary public replay contract.

## Implementation sequence and acceptance

| Increment | Deliverable | Acceptance criterion |
| --- | --- | --- |
| 1. Cohesive workspace | Common V2 provider/header/controls and sidebar; compact Overview; separate Plant, Weather, Planning, Deliveries & resources, Economics, Events and Evidence as their data becomes available; explicit V1 entry | Changing panels or 2D/3D preserves run, clock, controller and selection. Overview is readable without scrolling at the presentation viewport. Every command targets the same V2 run. No experiment configuration changes. |
| 2. Visible weather and equipment behavior | Public weather adapter, 3D clouds/shading and synthetic overhead view; commitment/exposure/settlement projections; equipment inspector | At the same daylight sample, shading and solar readouts agree across views. Night-time unknowns and solar faults do not produce false cloud claims. A viewer can explain a startup and distinguish stored, produced and delivered methane. |
| 3. Connect decisions through time | Recorded public frames, original forecast issues, aligned plans/outcomes and incident timeline | Select an earlier decision and every panel uses the correct issue and boundary. Playback never steps physics; forecasts and plans never become observations. |
| 4. Rehearsable walkthrough | Bounded fixture recording, weather/decision bookmarks, dedicated outcome/economic/evidence panels and optional predeclared H/E-beta illustration | A five-minute walkthrough explains one operational decision and its consequences; every shown success/failure is traceable. A completed recording plays without a running optimizer. |
| 5. Commission polish | Presentation framing, projected-screen legibility, loading/failure handling and a portable demo bundle | Rehearsal on the actual presentation hardware; no accidental reset, mixed runtime, unreadable label or dependence on an untested connection. |

Increment 1 is the next implementation slice. Increment 2 closes the largest missing
V2 presentation features. The recording schema should be designed during those slices
so increment 3 does not require a second incompatible telemetry format.

Before increment 4 executes a full run, specify the disposable fixture, arms, maximum
duration, output location and interruption/failure behavior. This planning document
starts no simulation, selects no outcome and authorizes no long experiment. Do not
keep trying seeds or controllers until the walkthrough looks successful. If the
predeclared run fails, retain and explain it; revise the engineering plan openly.

The first walkthrough can follow: Overview mission → visible weather and solar
response → issued forecast and plan → actual delivery/cycle result → Economics →
Evidence. An observed outage is an optional
recorded chapter only if it occurs; a future deliberate fault exercise requires an
explicit V2 integration. A scripted chapter title does not promise successful recovery.

Registered training custody, 15-policy execution, development selection and held-out
evaluation remain a separate research sequence with the frozen three families and
five optimizer seeds unchanged. Their results later populate Evidence and inform
controller selection. The unified demo need not assume a PPO winner. Site calibration,
HEPA/RUL validation, governed V2 supervisor/operator intervention and HIL testing remain
subsequent milestones rather than graphical substitutes for absent integrations.

## Audit references

- Current public API and DTO: [`v2_demo.py`](../../backend/src/plant_backend/v2_demo.py),
  [`v2.ts`](../../frontend/src/types/v2.ts), [`V2_VISIBILITY_PREVIEW.md`](V2_VISIBILITY_PREVIEW.md).
- Current shell/runtime split: [`App.tsx`](../../frontend/src/App.tsx),
  [`views.tsx`](../../frontend/src/views.tsx),
  [`PlantProvider.tsx`](../../frontend/src/state/PlantProvider.tsx),
  [`useV2Demo.ts`](../../frontend/src/hooks/useV2Demo.ts).
- V1 task diagnosis and V2 task evolution: [`ENVIRONMENT_V1_AUDIT.md`](ENVIRONMENT_V1_AUDIT.md),
  [`ENVIRONMENT_V2_SPEC.md`](ENVIRONMENT_V2_SPEC.md),
  [`CYCLIC_MARKET_BETA_BUILD.md`](CYCLIC_MARKET_BETA_BUILD.md),
  [`CYCLIC_MARKET_BETA_CONTROLLER_BUILD.md`](CYCLIC_MARKET_BETA_CONTROLLER_BUILD.md).
- Equipment model and exposure: [`ELECTROLYSER_COMMITMENT_PROFILE.md`](ELECTROLYSER_COMMITMENT_PROFILE.md),
  [`commitment.py`](../../simulator/src/plant_sim/commitment.py),
  [`metrics.py`](../../backend/src/plant_backend/rl/v2/metrics.py).
- Public plan/command records: [`mpc_schema.py`](../../backend/src/plant_backend/rl/v2/mpc_schema.py),
  [`beta_mpc_schema.py`](../../backend/src/plant_backend/rl/v2/beta_mpc_schema.py),
  [`setpoint_execution.py`](../../backend/src/plant_backend/rl/v2/setpoint_execution.py).
- Weather and rendering boundaries: [`weather.py`](../../backend/src/plant_backend/rl/v2/weather.py),
  [`env.py`](../../backend/src/plant_backend/rl/v2/env.py),
  [`solar.py`](../../simulator/src/plant_sim/components/solar.py),
  [`visuals.ts`](../../frontend/src/components/plant-3d/visuals.ts),
  [`SunRig.tsx`](../../frontend/src/components/plant-3d/SunRig.tsx).
- Frozen research direction: [`CONSTRAINED_RL_BETA_PROTOCOL.md`](CONSTRAINED_RL_BETA_PROTOCOL.md),
  [`CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md`](CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md),
  [`ROADMAP.md`](ROADMAP.md), [`JOURNAL.md`](JOURNAL.md).
