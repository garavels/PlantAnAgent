# Adaptive batch 007: matched residual screen with ongoing training power checks

Prospective design, 2026-09-14, before implementation or any batch-007 reset.
The user explicitly requested resumption after restoring power. macOS currently
reports AC power and battery at 56%, charging. Resume the research with fresh
models under this new identity; batch 006 has no resumable trained checkpoint.
Its failure records and the [closed report](results/ADAPTIVE_RL_BATCH_006_RESULTS.md)
remain untouched.

## Scientific and execution contract

Inherit all scientific provisions of [batch 006](ADAPTIVE_RL_BATCH_006_PROTOCOL.md),
including its matched scalar/PI hypothesis, initialization, reward scaling,
cost channels, optimizer settings, comparator semantics and acceptance conditions.

| Contract | Fixed value |
| --- | --- |
| Task | 168 hours, 28 deliveries, 6,720 kg methane; unchanged daily returns/economics |
| Physics/control | One minute / 15 minutes |
| Interface | 135 public inputs plus four causal MPC targets; four bounded residual actions |
| Arms | A-mpc-residual-scalar and B-mpc-residual-pi |
| Initialization | Fresh matched actor/reward tensors, seed 101, exact zero output head, Gaussian std 0.2 |
| Budget per arm | 65,536 transitions, eight rollouts, 2,560 minibatch updates, four synchronous environments |
| Training seeds | 9,630,000–9,630,099; exactly 100 allocator resets per successful arm |
| Development seeds | 9,700,300–9,700,307 |
| Actors | H, OperatingMPC, M24, E-beta, successful A's initial residual, final A, final B |
| Comparisons | All 56 evaluations and 56 independent replays, sequential after training |
| Endpoint | Fixed final only; no retry, resume, extension or replacement within this attempt |

Training seeds are reused adaptive development material. Batch 005 consumed its
first 44 reset seeds per arm; batch 006 recorded its first 40 plus three completed
optimization rollouts per arm (24,576 transitions, 960 minibatch updates). Exact
partial fourth-rollout progress is unavailable. No rewards, costs, learning-quality
diagnostics or weights from batch 006 inform this replication. No partial learner,
optimizer or allocator state is loaded. The development evaluation panel and
original registered/protected seeds remain unopened.

Use the exclusive internal root
`/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-007`.
All active gates, tools, claims, logs, failure evidence, checkpoints and trajectories
remain on the same internal APFS volume under the inherited storage contract.
Keep its 10 GiB free-space floor, 12-hour trainer limit, ten-minute evaluation/replay
limit, 64 MiB log limit and 256 MiB compressed episode limit. Keep two isolated
training supervisors and Torch 1/1 threads. Lexar is only a verified post-closure
archive destination; retain internal originals through audit/publication.

## Bounded power repair

Batch 006 entered forced low-power hibernation on battery at 1% and failed closed
immediately after wake. There is no demonstrated prefix corruption or hashing
throughput defect. Retain the generic supervision implementation, schema-v2
receipts and all existing monitor/closure deadlines, including the 60-second
prefix-verification deadline. Do not reset deadlines after sleep or admit late
verification as a successful run.

Add a runner-only requirement for **AC power and at least 20% battery reserve**.
This reserve is a prospective operational margin, not a learned threshold.
Query `/usr/bin/pmset -g batt` with a two-second subprocess timeout and bounded
captured evidence. Reject battery operation, insufficient reserve, missing or
ambiguous battery/source records, reported discharge, command failure or timeout.
The declared command, limits and sampling policy are immutable gate/plan data.
Actual battery levels and observation timestamps are dynamic samples and must
not participate in immutable identity comparisons.

Check power before claiming the experiment root, at the existing 30-second root
training integrity callbacks, and at the existing entry/exit integrity checks
for each training/evaluation/replay stage. Sequential evaluation/replay children
retain their existing ten-minute bound and entry/exit power checks; this protocol
does not claim 30-second power sampling inside those children. Keep launch-time
`caffeinate -i -s`; it is not a guarantee against forced hibernation or power loss.

Record the accepted admission sample in the plan. An admission refusal leaves
the experiment root unclaimed and writes a unique bounded failure record in its
internal readiness directory. A later refusal writes a unique bounded record
under the run root and raises an identifiable integrity failure through existing
cancellation/receipt handling. Preserve the power observation, query failure/code
and observation timing. Never silently retry, continue on battery or resume from
partial state. Power checks cannot prevent abrupt power loss; invalid attempts
remain invalid even if the host subsequently recovers.

General monitor/callback timing instrumentation is outside this bounded repair:
batch 006 supplies no precise mechanism requiring a monitor change. The known
timing-evidence limitation remains explicit. No simulator, solver, forecast,
learner, reward, dispatcher, guard, frontend or operator-authority change is included.

## Admission, custody and interpretation

Commit this protocol before implementing the runner change. Use tiny fixtures
without production resets to test AC/20% boundaries, changing accepted charge
levels, battery/discharge/low-reserve/malformed/unavailable/timeout refusals,
preclaim versus in-run failure evidence, immutable power-policy joins, and
unchanged scientific/storage contracts. Retain existing cancellation/closure
coverage and exercise an actual internal-volume helper fixture with the updated
producer/tool joins. No real sleep, power disconnection or battery discharge test.

Commit reviewed source, pass an exact clean-revision scoped regression/static
gate, and freeze the adapted detached auditor/reporter/launcher/archive tools
with their checks before launch. Verify power policy independently in the gate
and plan before outcome access. Successful admission requires the recorded initial
power sample to satisfy the declared policy. Failed tools/checks remain evidence;
source/gate pin filling must not silently change tested tool behavior.

Remain quiet and outcome-blind while training runs. Monitor only process identity
and closure-marker existence. After all writers close, normal detached audit
precedes reporting. Aborted or incomplete custody permits only forensic
preservation. Retain every failed arm, stopped comparator, fallback and paired
delivery/resource/value result; neither this repair nor partial progress is a
learning result.

Preserve the parent requirement for absolute hard safety, paired individual
delivery/resource noninferiority, frozen materiality, fresh sealed confirmation
and at least three optimizer seeds before an algorithm-level claim. This remains
a single-seed adaptive development screen. No registered study, operational-benefit
claim, changed task or live deployment is authorized by this restart.
