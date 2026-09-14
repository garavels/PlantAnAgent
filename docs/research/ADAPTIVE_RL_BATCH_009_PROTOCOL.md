# Batch 009: a deadline-bounded economic-objective diagnostic

Frozen prospectively on 2026-09-14 after the user allowed ten more hours for the
economic/reward investigation, with the existing trained policies as the fallback.
The window starts at **21:01:28 UTC on September 14** and ends at
**07:01:28 UTC on September 15 (09:01:28 Europe/Rome)**. No extension, cloud
resources, protected seeds or replacement of historical results is authorized.

## Question and fixed scientific comparison

The [saved-trajectory economic audit](results/batch009_economic_audit_plan.md)
examines valuation sensitivity and discounted versus undiscounted realized
returns. Its illustrative weights are not site calibration and do not change
the environment. The current accounting already credits reduced battery and
electrolyser exposure and lower accepted-solar opportunity cost. Observed delivery
losses outweigh those credits. A separate hypothesis is that gamma 0.995 gives
immediate savings and delayed service/refill different weights from the reported
undiscounted weekly economic objective. No causal explanation is established.

Run one small paired scalar-PPO screen, isolating the economic-discount change
from PI cost-advantage combination. Both learners start fresh from exact-MPC
initialization; neither imports, resumes or selects a previously trained endpoint.
The two arms differ only in reward discount. Keep all other learner, simulator,
dispatcher, guard, forecast, tariff, contract and economic assumptions unchanged.

| Contract | Frozen value |
| --- | --- |
| Arms | `A-scalar-gamma0995`, `B-scalar-gamma1000` |
| Algorithm | Existing scalar constrained-PPO implementation in both arms |
| Reward gamma | A: 0.995; B: 1.0 |
| GAE lambda | 0.95 in both arms |
| Initialization | Fresh matched actor/reward tensors, optimizer seed 101, zero actor output head, Gaussian std 0.2 |
| Reward scale | All seven economic terms multiplied uniformly by 0.001 for learning; evaluation in raw EUR |
| Budget | 16,384 transitions per arm, two 8,192-transition rollouts, 640 minibatch updates |
| Vectorization | Four synchronous environments per arm; two isolated training supervisors; Torch 1/1 threads |
| Training seeds | Reused development seeds 9,630,000–9,630,027; 28 allocated resets per arm including partial episodes |
| Plant task | Unchanged 168 hours, 28 deliveries, 6,720 kg methane and individual daily resource returns |
| Interface | Existing 139-input MPC-residual wrapper and four bounded actions |
| Evaluation seeds | Reused development panel 9,700,300–9,700,307 |
| New physical work | Both fixed final endpoints on all eight scenarios: 16 evaluations and 16 independent replays |
| Selection | Fixed final endpoints only; no interim inspection, checkpoint selection, retries or budget extensions |

Gamma 1.0 is already valid in the learner/configuration. No PPO algorithm code,
reward coefficients or cost trace is modified. This two-rollout screen is a
bounded diagnostic, not a convergence test or a fair budget-matched comparison
with the older 65,536-transition endpoints. The matched A/B contrast, all cases,
failure tails and uncertainty remain visible regardless of its direction.

## Existing comparison evidence

Use the immutable, already audited batch-008 evaluation panel as explicit
reference evidence for H, OperatingMPC, M24, E-beta, the initial residual, and
the two existing final policies. The full analysis JSON has SHA-256
`9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c`;
its audit has SHA-256
`22aff1c0aa3b7c4a3b9bf78d6ec9a7825c14b88534dbe77537021bd66f6d9536`.
Bind their source, scenario, configuration, checkpoint, trajectory and replay
provenance before launch. Verify unchanged inherited scientific sources and
identical physical manifests/public episode initialization when joining new
episodes. Do not present reused baseline episodes as freshly executed here.

Initial actor/reward tensor equality is required between both new arms. Check
their exact zero output heads, interface and matching to the inherited initial
actor before simulation. The existing eight-case initial/native-MPC physical
parity remains historical evidence, not a newly executed batch-009 parity test.
Every new endpoint must be reloaded, bound to its actual gamma, configuration,
allocator, optimizer ledger and source, then physically replayed independently.

M24 and E-beta's stopped episodes remain unavailable full-week comparisons, with
their original reasons retained. No numerical win against a stopped prefix is
allowed. Both previous final checkpoints stay immutable and available for the
user's fallback; a deadline or negative result does not erase their limitations.

## Deadline and custody

Use a new exclusive root under
`/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-009`.
Keep internal APFS active storage, 10 GiB free-space floor, 64 MiB per log,
256 MiB per compressed episode, AC power with at least 20% reserve and the
corrected exact accepted power states. No system power settings change.

Admission must finish and the campaign must launch before **23:15 UTC on
September 14**. Otherwise publish the completed economic audit and retain the
existing policies without starting this screen. Each trainer has a maximum
3.5-hour bound. Each evaluation/replay retains its ten-minute bound. All waits
and child timeouts are capped by the remaining absolute experiment budget;
training and evaluation must stop by **06:30 UTC on September 15**, reserving
time for process closure, audit and final reporting. Deadline cancellation is
recorded as an incomplete attempt, never admitted as a completed endpoint/panel.
The entire completion pipeline stops by the user's **07:01:28 UTC** deadline.

Freeze reviewed source, exact scoped checks, launcher and detached analysis
before the first production reset. Reuse tested training, physical evaluation,
replay and process-supervision functions with explicit new paths/configurations;
do not monkeypatch old runner globals. Historical producer schema names may
remain when unchanged producer functions are reused; the new outer plan must
identify batch 009 and bind each per-arm configuration explicitly.

Test deadline admission, remaining-time caps, elapsed-time exhaustion, cleanup,
per-arm gamma/endpoint validation, initialization matching and reused-reference
joins with small fixtures. Preserve all failures and test evidence. Prospective
post-closure content custody uses path/logical length/SHA with stable nonsymlink
single-link reads; allocation is recorded storage metadata. Keep the complete
training supervisor/monitor/allocator/optimizer graph and all raw artifacts
outside Git. Do not reclassify batch 007 as successful.

Remain outcome-blind and quiet while the new campaign runs. A frozen completion
worker may audit/report only after all writers close. Source changes stop during
execution. No source, calibration or protocol tuning in response to interim
results is permitted. The old models are the fallback if time, execution,
custody or scientific comparisons do not support a useful new result.

## Reporting and decision

Report undiscounted economic components, discounted trajectory returns, physical
intensity proxies, strict safety, every individual delivery and every daily
resource deficit. Require the complete eight-case paired panel. Retain exact
signs and all failures; sensitivity reweighting cannot waive a service/resource
regression or retrospectively define a winning objective.

No new final confirmation or policy deployment is included in this timebox.
The original registered three-family/five-seed study and protected seeds stay
unopened. Any positive A/B contrast is one-seed development evidence only.
The [parent protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) continues to
require prospective materiality, service/resource noninferiority, fresh sealed
confirmation and at least three optimizer seeds for broader benefit claims.
