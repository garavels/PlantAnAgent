# Study execution capacity fixture

Status: **frozen before this fixture is executed; no registered execution authorized**

Date: **2026-09-11**

Identifier: **`plant-beta-constrained-rl-study-capacity-fixture-v1`**

## Purpose and scope

The training-capacity measurement at `4f505b8` remains unchanged. This separate
implementation fixture measures the physical/timing archive and execution costs
of the study runner at its actual 168-hour horizon, 15-minute decisions and
one-minute physics. It performs no optimizer update and opens no training,
development or held-out seed. It cannot select a controller or support an
economic, safety, service, latency-superiority or deployment claim.

Earlier 24-hour/hourly fixtures are insufficient: M24's lookahead changes from
24 to 96 intervals at the study cadence. An earlier E-beta fixture stopped after
12 completed decisions; its retained prefix is not full-horizon capacity evidence.
No controller, dispatcher, guard, plant parameter, solver setting, tariff,
disturbance or forecast is changed to make this measurement pass.

## Fixed fixture

- Exactly one episode seed: **9,100,001**. All four actors use it.
- Actor order: **H, M24, E-beta, PPO**. One fresh sequential process per actor.
- Full physical task equals the frozen registered template. Only training-only
  fields change: `beta_rl=null`, `total_timesteps=4`, `n_envs=1`, external output
  path and the explicit fixture note. Optimizer seed is 101; no optimizer runs.
- PPO is a newly written **untrained fixed-action checkpoint**, with four exact
  float32 zeros (half of each normalized target range). It is a serialization and
  execution proxy, not a trained PPO inference-capacity or value measurement.
- Child Torch intra-op/inter-op counts are **1/1**. The parent retains its own
  original thread configuration. Host, source and dependency identities are bound.
- One attempt per actor under this protocol and source revision. A failed or
  incomplete actor is retained. No alternative seed, action, shorter horizon,
  controller repair or retry is chosen from an observed result.

## Bounds and evidence

The external root must be new, canonical, outside Git and have **8 GiB** available.
Its fixed path is `/private/tmp/plantanagent-study-capacity-<first-12-HEAD-digits>`
on macOS (`/tmp` on Linux). Another destination at the same revision is refused.
Each child has a **600-second** wall limit and two **1 MiB** bounded logs. Four
child timeouts therefore total at most **40 minutes**, excluding parent bookkeeping.
Each physical archive is limited to **1 GiB stored / 4 GiB decoded**; timing to
**8 MiB stored / 64 MiB decoded**. These are probe caps, not proposed study budgets.

A child samples its process peak RSS every 0.1 seconds. At **12 GiB** it durably
records a resource-limit marker and exits. This is a sampled stop, not an operating
system hard reservation; momentary overshoot is possible. Normal completion also
records process peak RSS. Raw checkpoints, logs and archives remain outside Git.

The existing fixture-only single-arm producer and complete physical/timing verifier
are used without dependency injection. Each retained successful measurement binds
the exact archive descriptors and reports expected/completed decisions and physics
steps, completion/failure, timing-channel totals, peak RSS and wall time including
execution, publication and the producer's complete archive reconstruction. The
parent reopens byte descriptors and retains the complete per-cell inventory and
process result. It prints no economic result or policy ranking.

Only an arm with verified evidence, **672 completed decisions and 10,080 physics
steps** supplies a full-horizon storage/time observation. An incomplete or failed
arm has no full-horizon projection. Failure does not erase other actors' records.
Any source mutation or loss of root/storage custody stops later attempts.

## Interpretation and next gate

One fixture is a measured planning input, not a worst-case capacity guarantee.
Stored/decoded limits and timeouts for the actual study must be stated explicitly,
include uncertainty/headroom and preserve identical limits across all PPO families
and optimizer seeds and across development/held-out stages. The fixed policy does
not measure trained-network inference; the existing registered hardware and later
per-call latency admission remain mandatory.

If any actor lacks a complete observation, report the gap and diagnose it before
claiming complete study capacity. Do not silently multiply a retained prefix into
a full-week success. After capacity is resolved, complete the clean repository
release gate and prepare the concrete training authorization packet. This fixture
does not create that authorization or revise either frozen upstream protocol.
