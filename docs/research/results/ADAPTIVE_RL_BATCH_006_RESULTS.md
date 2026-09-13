# Adaptive batch 006: training interrupted by host hibernation

Both scalar and PI-Lagrangian MPC-residual arms recorded **24,576 of 65,536
transitions (37.5%)** before this attempt aborted. Neither saved a final trained
policy; evaluation never began. This is an execution interruption, with no new
performance result or basis for ranking the learners.

The attempt followed the unchanged [protocol](../ADAPTIVE_RL_BATCH_006_PROTOCOL.md)
and used source
`c59ba89aa301b863e50be7246fe5003c6fb731dc`, tree
`a4468ece118b9b2b4f67cb03e81207e929b61ca1`. All active artifacts are under
`/Users/filippogombac/Developer/PlantAnAgentResearch`, on internal APFS. Source
remained clean and unchanged through closure and forensic preservation. No partial
state has been resumed, no replacement policy selected and no next run launched.

## Admission and execution

The clean-source gate passed **189 Python tests in 164.09 seconds**, plus Black,
strict source MyPy, compilation, CLI and whitespace checks. This was a scoped
gate; the last complete repository/frontend gate remains `b5ea55e` (3,052 Python
tests plus static/frontend checks). The detached auditor/reporter passed 173/66
fixture tests and their static/CLI checks. The post-closure archive helper passed
49 tests and static/CLI checks. The actual internal-volume helper fixture checked
18 custody records and eight log hashes without production resets or learning.
Declared source/gate pins were verified by normalized AST equivalence; no full
retest after pinning is claimed. Prior unsuccessful preparation checks remain
within the frozen 70-file tool archive, whose manifest is an additional file.

Launch was **2026-09-13 15:07:12.693 UTC** (17:07 Rome), on AC with battery at
100%, using `caffeinate -i -s`. Report closure was **21:29:55.776 UTC** (23:29 Rome),
approximately 6h23m later. Ordinary process receipts record approximately 3h39m
of monotonic elapsed runtime; these clocks exclude much of the civil hibernation
interval. They are not an exact wall-clock measure of active learning time.

| Surviving progress metadata | Scalar A | PI B |
| --- | ---: | ---: |
| Completed optimization rollouts | 3 / 8 | 3 / 8 |
| Transitions in those rollouts | 24,576 / 65,536 | 24,576 / 65,536 |
| Minibatch optimizer steps | 960 / 2,560 | 960 / 2,560 |
| Recorded allocator resets | 40 | 40 |
| Final trained policy / result | Absent / absent | Absent / absent |
| Ordinary exit code | -15 (SIGTERM) | -15 (SIGTERM) |

Each persisted post-update ledger contains 8,192 transitions and 320 optimizer
steps; allocator prefixes corroborate 6,144 decisions in each of four environments.
Reset records contain development seeds **9,630,000–9,630,039**, ten per environment.
Only these allowlisted progress fields were interpreted after closure. Rewards,
costs, learning-quality diagnostics, policy tensors and trajectories were not
interpreted. Exact subsequent partial-rollout progress is unavailable. Initial
policy files survive, but there is no evaluable trained endpoint.

## Host event and failure mechanism

Narrowly filtered `pmset -g log` evidence records the following local Rome times:

| Time, September 13 | OS event |
| --- | --- |
| 20:45:21 | Low Power Sleep, using battery at 1%; reported duration 9,873 seconds |
| 23:29:54 | Wake from Hibernate, using AC at 20% |

The root reported `log_verification_overdue` for B stdout at `log.budget` and
closed immediately after wake. Both ordinary receipts retain `root_custody_lost`
and SIGTERM; both supervisors published completion records with exit 86. All
trainer/supervisor groups closed, and both arms remained unadmitted. Separate
ENOTCONN cleanup evidence remains preserved rather than suppressed.

The host power interruption is the highly likely precipitating cause. The monitor
failed closed on a verification age above its fixed 60-second deadline. It names
B because B stdout is the first nonempty stream in monitoring order; this does
not establish an arm-specific failure. Maximum recorded poll duration was
24.052 ms, maximum poll bytes 452,150 and maximum recorded backlog zero. These
records do not show hashing throughput exhaustion or prefix corruption.

The exact last verification, poll, failure timestamps and whole synchronous
integrity-callback duration were not persisted. The precise clock/callback
mechanism cannot be reconstructed. Later final verification succeeded, but it
does **not** undo the missed deadline or admit the experiment. Launch-time AC and
idle-sleep assertions did not establish continuous power throughout execution.

## Evidence and next requirement

The [compact index](adaptive_rl_batch_006_evidence_index.json) binds the full
post-closure inventory, OS query and selected progress metadata. Publication
rehashed all **207 files**: 126 run files, ten readiness files and 71 frozen tool
files. There are no Apple metadata files in these collections. Full process-table
snapshots before and after preservation establish current absence of known groups
and the exact runner module; they do not reconstruct continuous historical custody.
The normal auditor, reporter and archive-copy helper were not run on this abort.
Internal originals remain preserved.

| Record | SHA-256 |
| --- | --- |
| Source gate | `c538927a3fd026523ced6007d00cfeebe136306bb0be5031b1768cb6367dfbc5` |
| Frozen tool manifest | `056fed6ecc596e726244a334ee98987b54aca07087f162fbc9891ffa0ff55633` |
| Forensic inventory | `e19ff6d6f37c452af00e43125a0ab7be109e5218ad988c2fd1c0a40e1f9e66fe` |
| OS power evidence | `51b02b6fd7823534cf20053afe717bab1350e6e0733e100d80aac9deeea1365a` |

Before a fresh, separately admitted attempt, address reliable continuous power,
periodic power checks and bounded interruption/timing evidence. Keep the existing
integrity deadlines and preserve this failure. No new training settings are
justified by an attempt whose learner outcomes remain unopened.

The eight development evaluation seeds remain unopened, as do the original
registered study and protected seeds. The task remains 6,720 kg methane over
168 hours in 28 deliveries. The strongest completed screen remains
[batch 003](ADAPTIVE_RL_BATCH_003_RESULTS.md): EUR 104.98/week higher mean value
than OperatingMPC, with unacceptable individual delivery/resource regressions.
No policy has qualified for confirmation or demonstrated preserved-service
operational benefit.
