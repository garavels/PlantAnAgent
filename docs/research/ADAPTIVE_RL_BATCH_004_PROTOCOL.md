# Adaptive batch 004: matched scalar and PI residual PPO

Prospective development design, 2026-09-13. Existing user research autonomy
authorizes this local experiment. Execute only after implementation review,
fabricated admission checks, exact clean source freeze, a passing revision-bound
gate and an exclusively claimed execution plan. The original registered
three-family/five-optimizer-seed study and its protected seeds remain unopened.
Earlier experiments, their caps and all negative results remain unchanged.

## Decision and hypothesis

The [closed batch 003](results/ADAPTIVE_RL_BATCH_003_RESULTS.md) retained the
incumbent MPC's strict safety on eight development weeks. Its mean EUR 104.98
gain accompanied 17 worsened individual deliveries, worsened daily resource
deficits in every scenario and approximately 1,107 kWh/week extra battery cycling.
The four pre-update diagnostics show learning and active exploration, without
establishing convergence. A small positive mean does not qualify this policy for
confirmation.

Test whether the existing PI-Lagrangian learner improves the economic/service/
resource tradeoff relative to scalar PPO around the same MPC prior at an equal,
fixed training budget. PI receives the already implemented four daily resource
cost channels; no new reward shaping, cost normalization, multiplier tuning or
physical model change is part of this batch. Neither method is expected by
assumption to solve the regressions. Scalar already has inventory incentives
through methane-equivalent endowment economics; PI adds separate resource-cost
optimization rather than introducing the first inventory incentive.

Both arms receive 65,536 transitions. Their matched comparison examines learner
method at that budget, conditional on this initialization and scenario panel.
Comparing either result to the shorter batch 003 cannot isolate a training-budget
effect because the training and development seeds also change. This remains a
development screen, not a convergence, algorithm-superiority or deployment claim.

## Unchanged task and control interface

Keep the full 168-hour task, 28 recurring deliveries totaling **6,720 kg methane**,
one-minute physics, 15-minute decisions, stochastic weather and public forecasts,
market truth/forecasts, private DAC outages, economic/wear ledger, daily resource
settlements, equipment constraints, dispatcher, safety guard and operator authority.
Do not increase demand or change the plant to create an apparent RL advantage.
Economic improvement at preserved service and resources remains the objective.

Use the exact batch 003 research-only MPC wrapper: the returned 135 public inputs
plus four causal cached MPC targets, **139 observations and four residual actions**.
Keep its 25% multiplicative adjustment of operating effort, unchanged zero commands,
unchanged electrolyser commands at/below the stable-running floor, current authority
caps and exact float64 MPC command execution for zero residual. Keep the current
planner, model, forecasts, solver stages, time limits and explicit fallback.
Recovery labels denote accepted plans with deficits/shortfalls, not solver failure.
Neither arm gains startup/off or independent battery-discharge authority.

The complete [batch 003 interface and replay contract](ADAPTIVE_RL_BATCH_003_PROTOCOL.md)
applies unchanged. No new simulator, live runtime or frontend behavior is included.
Checkpoint identity binds this wrapper and its complete control dependencies;
these artifacts remain ineligible for registered evaluation or live deployment.

## Two fixed learners

| Arm | Learner | Optimizer reward scale | Initial Gaussian standard deviation |
| --- | --- | ---: | ---: |
| A-mpc-residual-scalar | Existing scalar PPO | 0.001 | 0.2 |
| B-mpc-residual-pi | Existing PI-Lagrangian PPO | 0.001 | 0.2 |

For each arm: optimizer seed **101**, exactly zero actor output-head weights and
biases, four synchronous environments, **65,536 transitions**, eight rollouts of
8,192 and **2,560 minibatch optimizer steps**. Each environment advances 16,384
decisions: 24 complete 672-decision episodes plus a final 256-decision prefix.
Fresh matched allocator seeds are **9,630,000–9,630,099**, 100 resets per arm.
Physical seeds come only from that allocator. Ignored framework environment-seed
metadata must be null; optimizer RNG/action-space seeding does not seed environments.

Keep the current CPU MLP policy, learning rate 0.0003, 2,048 steps/environment,
minibatches of 256, ten epochs/rollout, reward discount 0.995, GAE lambda 0.95,
clip range 0.2, entropy coefficient zero, value coefficient 0.5 and max gradient
norm 0.5. No reward/constraint clipping or new optimizer settings are introduced.
Only optimizer rewards are scaled; raw EUR info, Monitor returns and all physical
constraint signals retain their original units and values.

PI retains four independent cost critics and multipliers: cost discount 1.0,
cost GAE lambda 0.95, value coefficient 0.5, two 64-unit cost hidden layers,
zero cost limits, initial integral/multiplier values of one, proportional gain
0.25, integral gain 0.01, proportional EMA coefficient 0.95 and bounds [0, 100].
The existing learner independently normalizes reward and each cost advantage.
Costs remain normalized excess daily deficits divided by seven; dual updates use
completed episode costs, with no update when a rollout closes no complete episode.
Sparse approximate constraint learning does not guarantee every daily boundary.
Scalar retains its stock PPO optimization path without active cost critics/duals.

Persist initial weights and their exact tensor inventory before learning. Actor
and reward-critic tensors, including log standard deviation, must match across
the two fresh arms; existing PI cost-critic initialization remains RNG-isolated.
After both training attempts close, the producer verifies reached initial artifacts
and exact shared tensor equality. The detached auditor additionally reconstructs
fresh initialization without a production reset to validate the complete seeded
tensor identity; it never creates replacement comparison artifacts.
Missing initialization is permitted only as explicit unavailability from an
ordinary failed attempt. Existing malformed/mismatched evidence aborts execution.

Preserve eight pre-update diagnostic snapshots, eight completed optimizer ledgers,
eight immutable allocator prefixes and all reset records for each successful arm,
plus exact initial/final checkpoint identities. Diagnostics precede the update
they describe; the last snapshot does not measure the final updated critic.
Use only the fixed final endpoint. No intermediate selection, resume, extension,
retry or substitute policy is allowed within this batch.

## Fixed evaluation and detached audit

After the two training attempts close and initial matching is checked, attempt
all seven actors on all eight fresh development seeds **9,700,300–9,700,307**:

- H;
- corrected OperatingMPC;
- strict M24;
- strict E-beta;
- initial MPC residual, supplied only by successfully closed arm A;
- final A-mpc-residual-scalar;
- final B-mpc-residual-pi.

Every learned-policy slot requires its successful closed training receipt and
recorded weights/interface. If A fails, the common initial comparator stays
unavailable even if B succeeds. All 56 evaluation and 56 replay slots remain
explicit, including ordinary child failures and unavailable replay. Integrity
loss or operator interruption aborts the campaign and preserves its partial graph.
M24/E-beta failures and stopped prefixes do not supply artificial weekly wins.

Save and verify full minute trajectories, actual public/policy observations,
nominal MPC evidence, mapped commands, applied equipment behavior, guard/fault/
startup records, economic components and all daily settlements. Detached physics
replay uses saved absolute commands; separately recompute the causal prior and
mapping without new learned-policy inference. Exclude only the already declared
solver elapsed-time fields and their hash descendants for recomputation parity;
all saved raw hashes and semantic evidence still validate.

Before performance analysis, the detached auditor must validate the frozen source,
gate, plan, two isolated supervisor histories, process closure, 114 ordinary child
receipts, checkpoint/allocator/diagnostic ledgers, artifact inventory and replay
bindings. Verify complete initial-residual/native-MPC physical equivalence on all
eight scenarios when the initial comparator is available. Prepare and test the
auditor/reporter on fabricated artifacts before production execution, including
failed arms, corrupt evidence, null framework seed metadata and PI ledger joins.

Report both final policies against exact initialization and every incumbent, plus
the paired PI-minus-scalar comparison. Retain all eight weeks, every one of 224
individual delivery outcomes and every one of 56 daily four-resource deficit
vectors per complete actor. Full-panel means require all eight complete, replayed
episodes. Include all pre-update diagnostics and optimizer rounds for both arms;
rollout rewards are not weekly evaluation EUR. Retain losing pairs and unavailable
comparators. Any forecast/fault mechanism diagnosis is descriptive unless a later
prospective intervention or ablation identifies it.

No formal selection or numeric benefit threshold is set by this screen. The
[parent research protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) still
requires absolute inherited hard safety, paired individual service/resource
noninferiority, prospectively frozen materiality and fresh sealed confirmation.
At least three training seeds are required before an algorithm-level claim.
Unresolved absolute resource restoration must remain explicit even if paired
noninferiority is eventually achieved. Development results cannot complete the
operational-benefit goal.

## Local execution and process custody

Root: `/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-004`.
Both training attempts run concurrently in **two isolated Python supervisors**,
each executing one existing serial process capture on its main thread. Each
trainer retains four synchronous environments and one Torch intra-op/inter-op
thread. Evaluation/replay children run sequentially afterward. The supervisor
also sets the exact frozen host thread configuration before admission checks.
Do not thread-pool signal handlers or process-wide child CPU accounting, and do
not switch environments to subprocess vectorization in this batch.

Before a trainer command executes, an execution barrier publishes its process
identity to the parent and requires acknowledgement after durable registration.
Each supervisor has a fixed 30-second startup bound; the trainer barrier allows
five seconds for acknowledgement. Heavy parent integrity checks resume after both
trainer identities are known or their attempts have closed. Unresolved startup
at its bound aborts the wave, preventing an indefinitely deferred integrity check.
The parent retains both supervisor and trainer process-group identities. Parent
loss, supervisor integrity loss, reserved exit 86 or operator interruption cancels
the remaining wave and closes/reaps its groups before inventory. Ordinary trainer
numerical/timeout failures remain local and do not cancel the other arm. Missing,
corrupt or inconsistent supervisor evidence is a fatal custody failure. No serial
fallback, orphaned background training or implicit retry is permitted.
If bounded cleanup cannot establish every launched group's closure, retain a fatal
record with identities and unavailable paths only; do not hash or publish an
artifact inventory from potentially live writers.

Freeze source/runtime/host/gate in the plan and verify them before/after children.
During the concurrent wave, separate responsive process/log/storage polling from
periodic complete integrity checks, with a fixed 30-second integrity interval.
The launch plan and supervisor admission record bind the final executable polling,
barrier and closure limits. Admission tests must exercise launch/acknowledgement
failure, parent/supervisor loss, partial failures and bounded group cleanup.

Fixed per-arm training cap: **12 hours**. Evaluation and replay each retain a
ten-minute child cap, 64 MiB per log stream, 256 MiB compressed episode limit and
10 GiB free-space floor. The batch 003 training receipt took 15,617.84 seconds for
32,768 transitions; a simple double-budget extrapolation is approximately 8.68
hours per arm before contention, followed by sequential comparison/replay. This
is a planning estimate, not a throughput guarantee. Concurrency may change solver
wall times and fallback incidence; both arms receive the same concurrency setting,
and all actual timing/fallback evidence is retained. No speedup or strict causal
algorithm claim follows from changed execution concurrency relative to old batches.

Exclusively claim the new root and every attempt; never overwrite earlier roots.
Freeze and retain analysis-tool digests before launch. Do not inspect live outcomes,
partial policies or diagnostics, or revise the design while experiments execute.
Process-only monitoring may inspect identities and closure existence. Remain quiet
while execution is unchanged. No cloud spending, external messages, frontend
change or live deployment is included.
