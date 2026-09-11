# Unified commissioning workspace

Implemented 2026-09-10 on `codex/environment-v2-research`. The user requested one
product entry, independent feature panels, visible stochastic weather and preservation
of the project's visual identity. Panel titles use concise operational labels; tutorial subtitles and narrative copy are removed. This guide supersedes the plan's earlier proposal
for separately accessible legacy pages.

## Open and inspect

The local preview is at `http://127.0.0.1:5174/`. All eight destinations share one
V2 fixture run, selected equipment and completed-decision cursor:

- **Overview:** mission, next delivery, plant/weather and recent observed events.
- **Plant:** existing 3D geometry, process-diagram alternative, equipment selection,
  V2 commitment modes, timers and measured exposure.
- **Weather:** synthetic overhead site, procedural clouds driven by public observed
  solar attenuation, clear-sky/available history and aggregate solar forecasts.
  The previous decision's forecast can be overlaid with its original issue time.
- **Planning:** operating mode, predicted delivery shortfalls and resource outlook,
  exact decision input forecasts, four retained MPC target schedules,
  predicted inventories, delivery deadlines, first-interval actual/model comparison,
  requested/applied targets and minute execution/guard notes.
- **Deliveries & resources:** seven-day commitments, produced/stored/delivered methane,
  fixed opening inventory references and complete daily review quantities.
- **Economics:** reconciled signed EUR contributions, cumulative history and assumptions.
- **Events:** filtered observed events, equipment/settlement links through the shared
  cursor; optional whole-recording retrospective view is explicitly labelled.
- **Evidence:** actual controller availability, information access, recording outcomes,
  identity/checksums, export and unavailable integrations.

The established logo, Instrument Sans/Martian Mono fonts, sage/olive-charcoal/bone
palette and carrier colors are retained. Layout and information grouping change;
this is not a new brand. V1 geometry, source, backend APIs and tests are preserved.
`/v2`, `/plant-3d`, `/agent`, `/metrics` and `/scenarios` redirect into this workspace;
normal navigation never mounts the V1 stream provider.

Use **Mission & recordings** to open a saved run, export public history, return to
the current interactive simulation or reset with H/MPC. **Play recording** advances
stored frames; **Run simulation** executes real decisions. Pausing a simulation stops
new requests after any already accepted interval finishes. Historical inspection is
read-only. Reset retains the previous three runs in server memory; downloaded/saved
records survive restarts. Changing panels preserves the selected time.

## Rehearsal scope and observed outcomes

Exactly one H and one E-beta recording were declared before execution, each on seed
9,000,034 with a 168-hour target and 20-minute wall limit. Both use the same simulator,
minute dispatcher and independent guard. Raw recordings and attempt manifests live
in ignored `runs/commission-demo/`; compact manifests are committed alongside this guide.
No training, development selection or held-out evaluation took place. These are
historical outcomes; the separate product controller was scoped on 2026-09-11 in
[`COMMISSION_OPERATING_SCOPE.md`](COMMISSION_OPERATING_SCOPE.md).

| Controller | Retained time | Outcome | Delivered / unmet methane | Failed daily returns |
| --- | --- | --- | --- | --- |
| H | 168 h | Completed | 5,564.72 / 1,155.28 kg | 7 of 7 |
| E-beta | 9.25 h | Stopped: BetaMPCSolveError | 240 / 0 kg settled | No daily settlement reached |

These unequal-length records cannot establish comparative performance. No retry or
fallback replaced the failed E-beta attempt. Its terminal failure remains visible throughout
playback. Saved product MPC is now preferred when available, initially at the
existing 8-hour presentation cursor; otherwise historical recordings remain available.
H supplies the original complete seven-day history. Read the Evidence panel or
open Mission & recordings to switch between them.

The recordings were generated from a dirty implementation checkout, explicitly
recorded in each manifest with its parent Git revision and Python-source digest.
Their public JSON checksums and boundary sequences are validated when loaded. They
are presentation fixtures, not detached-verifier-approved research evidence.

## Runtime and contracts

Start the backend and frontend in two terminals from the repository:

```sh
PLANT_AGENT_ENABLED=0 .venv/bin/python -m uvicorn plant_backend.app:app --host 127.0.0.1 --port 8001
```

```sh
cd frontend
PLANT_API_TARGET=http://127.0.0.1:8001 npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

The optional optimisation dependency is required for product MPC. The recording CLI is
`python -m plant_backend.record_demo H` or `MPC` in the configured environment.
It uses exclusive attempt markers: existing attempts cannot silently be overwritten.
Do not delete a failed attempt to manufacture a replacement success.

The product's `operational-economic-mpc-v1` uses public mean solar/market forecasts,
exact mandatory shipments and sampled held-power bounds. A rolling 24-hour plan
maximizes delivery service, then minimizes economic cost with wear and depletion
priced against fixed episode-opening endowments. Daily restoration is measured,
not required for optimizer feasibility. Recovery means a current/projected resource
deficit or predicted shipment shortfall; it does not promise eventual restoration.
Typed solver/plan rejection selects an explicitly recorded heuristic fallback for
one interval and retries planning at the next boundary. Fallback publishes no plan
or inventory prediction. Unexpected errors and invalid/stale observations stop the
session. Every command uses the unchanged dispatcher, guard and minute physics.
The strict offline E-beta research controller is unchanged and no longer selectable
for new interactive sessions. Its old recordings retain their original outcomes
and strict daily-return labels.

The existing `/api/v2/demo` contract now retains all completed public frames, exact
original forecast issues, commitment state, full cycle settlements and applied minute
records. New read-only endpoints provide `/frames/{run_id}/{index}`, `/recording`,
`/recordings` and `/recordings/{id}`. No client-supplied seed, checkpoint, output path,
private future tape or research audit manifest is accepted. The `plant-public-recording-v1`
container adds a canonical JSON SHA-256; it establishes content integrity, not source
authenticity, policy admission or scientific superiority.

Weather is a presentation projection. Available/clear-sky power is attributed to
resource attenuation only in daylight with nominal solar equipment health. Night
and equipment-loss cases preserve unknown/unattributed status. Spatial cloud shapes,
land parcels and shading are synthetic; there is no satellite, wind or regional
forecast model. Forecast windows retain their published mean/p10 resolution. Public
market forecasts retain mean/p10/p90 hourly bins. No control observation, simulation
physics, registered protocol or benchmark selection rule changes.

## Explicit boundaries

The UI exposes an interactive V2 simulation and public replay. It is not a deployed
plant orchestrator. M24 remains an offline controller; beta PPO training/selection,
HEPA, natural-language scenario authoring, supervisory takeover, general V2 fault
injection and operator envelope changes remain separate integration work. The
product never presents them as active layers. Site calibration, credible commercial
savings and hardware commissioning remain unestablished.

Recording retention is bounded to 673 frames and three previous interactive runs;
saved files are limited to eight and 128 MB each. The current 16 MB H and 1.6 MB E-beta
recordings load as complete public files; pagination is a later scaling improvement.
The 3D scene loads lazily and has a process-diagram fallback. Its existing Three.js
stack still produces dependency deprecation/bundle-size warnings.

## Validation

Full Python suite: **1,853 passed** in 482.52 seconds (one Starlette/httpx deprecation
warning). The final frontend build/TypeScript and oxlint passed; compileall and Git
whitespace checks passed. Browser checks covered route redirects, the eight panels,
record loading and history, minute execution, original forecast issues, seven failed
resource settlements, keyboard focus containment and 390/1280-pixel layouts. The
Overview fits the 1280 × 720 presentation viewport without clipped mission content.
