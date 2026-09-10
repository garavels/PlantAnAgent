# V2 visibility preview

Status: implemented and checked as an interactive fixture preview; no registered controller outcome.

## Purpose and scope

The 2026-09-10 product decision is to make the existing V2 backend visible before
extending the research campaign. The hypothesis is that showing the public
forecasts, recurring deliveries, resource-return margins and guarded physical
execution together will reveal product gaps more effectively than inspecting
separate offline reports.

The first slice is an opt-in **V2 Lab** at `/v2`. It is a bounded interactive
fixture simulation, not a registered benchmark, verified artifact replay, trained
policy demonstration, or field deployment. The V1 control room remains available.

## Execution boundary

- The lazy `/api/v2/demo` session instantiates the existing cyclic-market beta
  environment with its unchanged default 168-hour task, 15-minute decisions and
  one-minute physics.
- Its only seed is `9,000,034`, inside the disposable fixture domain. The API
  accepts no seed, task configuration, checkpoint or output path.
- H uses the existing common-setpoint heuristic adapter. E-beta is an optional
  experimental controller when SciPy is installed. Changing controller resets
  the fixture; it is not a matched comparative study.
- The browser explicitly requests one or four decisions. There is no automatic
  background experiment or training process. A run/revision precondition prevents
  stale step requests from being applied to a different state.
- Failures stop the V2 session and remain visible. There is no controller fallback
  and no supervisor authority in this preview. This is not the later fail-safe
  live orchestration milestone.
- The API commits a detached public snapshot after each completed decision. If
  physical execution or presentation fails, it returns the last completed
  boundary and explicitly reports that a partial interval may have executed.
  Further stepping is refused; GET remains usable without reading damaged state.

## Presentation boundary

`frontend/src/types/v2.ts` defines the versioned operator-facing contract. It
contains current telemetry, public forecast issues, known recurring obligations,
realized economics, signed return margins, retained settlements, compact history,
and the latest decision's requested versus physically applied quantities.

Solar exposes four aggregate forecast windows with mean and p10 values; it has
no published p90. Market issues expose 24 hourly mean/p10/p90 bins. Only a retained
MPC plan is displayed as a future equipment schedule. The heuristic does not
acquire a synthetic plan or consume forecasts because they are visible to an
operator.

Private future weather, realized future market prices, future outage schedules,
random-stream manifests and solver audit blobs are never served as UI data.
Forecasts may be wrong. Mid-cycle negative resource margins are quantities to
restore by the next settlement, not immediate safety failures. Battery charging
may include recovered spill beyond the controller's requested charge ceiling.

The EUR ledger remains a synthetic operating-value model, including renewable
opportunity cost and wear/endowment assumptions. It is not an electricity trading
settlement, site saving, ROI estimate or environmental certification.

## Open the preview

Install the locked optional solver without removing existing extra dependencies:

```sh
uv sync --frozen --group optimisation --inexact
make dev
```

Open `http://localhost:5173/v2`. Use **Advance 15 min** or **Advance 1 hour**.
To inspect a real future schedule, select **Economic MPC (experimental)**,
choose **Reset and start new run**, and advance one decision. Controller changes
start a new fixture, rather than a scored comparison. **Explore in 3D** displays
the same physical snapshot as the process diagram.

For an isolated preview alongside an existing server, run the backend on port
8001 and start Vite with `PLANT_API_TARGET=http://127.0.0.1:8001` and `--port 5174`.
The default proxy remains port 8000. The preview has one in-memory V2 session per
backend process; another tab can advance it, so a stale tab must refresh before
it can issue a new command.

## Acceptance and remaining work

Acceptance checks cover real guarded interval execution, correct public-data
projection and units, rejection of unsupported inputs/stale requests, reset and
failure behavior, preservation of V1, frontend build/lint, and browser inspection
at desktop and narrow widths. The 12 API tests pass, including real E-beta plan
projection and partial-execution failures. The complete repository run returned
1,843 passes and one correctly rejected worktree-mutation test caused by concurrent
presentation edits; that exact test passed with the source fixed. Frontend build
and lint pass. Browser checks cover desktop/360px layouts, both controllers,
2D/3D views, advancement, reset and stale-tab recovery. The journal retains the
initial failure and exact verification boundary. Only fixture-domain checks are
part of this work; this is not a new clean-revision registered-study admission.

Verified artifact replay, full historical-frame scrubbing, four-arm scientific
comparison, trained PPO, HEPA, V2 natural-language authoring, immediate operator
takeover and governed fallback remain distinct later increments. The registered
training and development/held-out protocols remain unchanged and unopened.
