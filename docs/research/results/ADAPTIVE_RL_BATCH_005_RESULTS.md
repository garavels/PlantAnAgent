# Adaptive batch 005: interrupted without a training result

The matched scalar/PI MPC-residual attempt **ended without completion evidence or
a trained endpoint**. Neither arm can be evaluated or compared. Partial training
artifacts remain preserved, but there is no admitted learning or performance result
and no basis for ranking PI against scalar PPO.

The [prospective protocol](../ADAPTIVE_RL_BATCH_005_PROTOCOL.md) remains unchanged.
This repair replication used clean source
`776a98fa01b0cad5b85f0e532bbc6a3c660dbcd6`, tree
`83c502a4138f6c4c579d056b4cf998deb42368cc`, and the exclusively claimed root
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-005`. The attempt has not
been resumed or retried. Post-interruption analysis reads its artifacts without
modifying them.

## Admission

The exact source gate passed **435 Python tests in 559.14 seconds** (one retained
Starlette test-client deprecation warning), plus Black,
strict source MyPy, compilation, CLI, whitespace and frozen real-volume fixture
verification. This was a scoped dependency gate, not a repeat of the full
repository/frontend suite. The last complete gate remains `b5ea55e`, with 3,052
Python tests and static/frontend checks.

The detached tools passed **139 fabricated auditor tests in 427.02 seconds** and
**62 reporter tests in 31.94 seconds**, plus static and CLI checks. A coordinated
helper deadline repair changed a dependency during that run, so its original
check receipt correctly retained `passed=false` despite every test passing.
A fresh-process helper/auditor integration test subsequently passed against the
final unchanged helper, checking 18 custody-file identities and eight final log
hashes. The combined receipt explicitly distinguishes that recheck from the
unchanged fabricated/static results. Earlier unsuccessful checks remain archived.

The 36-file tool archive and its manifest were frozen before launch. Constant-only
source/gate pinning was checked by normalized AST equality; no tests were claimed
rerun after pinning. These checks established the implemented admission behavior;
they did not establish uninterrupted availability of the external storage device.

## Execution and surviving evidence

Both optimizer-seed-101 arms launched on AC power at **2026-09-13 10:03:31 UTC**,
with two isolated supervisors, four synchronous environments per arm and the
fixed 65,536-transition budget per policy. Evaluation and replay were to begin
only after training closed. The last retained live-process observation was at
root elapsed time 3 hours, 56 minutes, 55 seconds. No partial rewards, policy
weights or learning diagnostics were interpreted during execution.

By **14:21:59 UTC**, the five registered process IDs/groups were absent. The
forensic inventory independently checked their absence before and after hashing
the surviving files at 14:26 UTC. There is no recorded exact exit time, exit code
or termination signal. The host's boot-time evidence rules out a host reboot
during this attempt.

| Surviving record | Scalar A | PI B |
| --- | ---: | ---: |
| Allocator reset records | 44 | 44 |
| Allocator rollout-prefix files | 3 | 3 |
| Diagnostic files | 3 | 3 |
| Rollout-ledger files | 3 | 3 |
| Final trained checkpoint | Absent | Absent |
| Ordinary child receipt / supervisor completion | Absent / absent | Absent / absent |

Reset metadata records seeds **9,630,000–9,630,043** in each arm, with 11 resets
per environment and null ignored framework-seed fields. These are consumed
development seeds. The counts above describe surviving files, not a closed
training attestation or the exact final optimizer/transition count. Diagnostics
and policy tensors were hashed as opaque bytes; their performance contents were
not interpreted. Only files labelled `initial-policy.pt` remain, with no final
checkpoint or resumable learner state.

The batch has no `training-wave.json`, `report.json` or `initial-matching.json`.
There are no ordinary process receipts, supervisor completion records, final
endpoints or evaluation artifacts. The eight planned development seeds
9,700,300–9,700,307 were not evaluated. No registered or protected study was opened.

## Storage event and causal limits

macOS recorded the following events for the Lexar volume, UUID
`9A20639E-4B81-3097-8262-D9B68AE02E30`, on September 13. Times below are local
Europe/Rome time (UTC+02:00).

| Time | Recorded event |
| --- | --- |
| 16:08:48.726–.730 | USB termination, followed by removal of `disk4s1` |
| 16:08:48.953 | Volume unmounted |
| 16:08:50.839 | Device reappeared |
| 16:09:16.750–.849 | Filesystem repair reported successful, then volume mounted |

The device disappearance is concrete evidence of lost access to the volume that
held the campaign, logs and completion evidence. Its timing is consistent with a
shared storage failure interrupting the processes and preventing durable failure
publication. That is the supported working explanation, not a reconstructed
process-exit trace: the experiment records do not establish the exact signal,
order of process exits or reason for USB termination. The event does not show
that either learner failed algorithmically, nor does it establish a recurrence of
batch 004's forecast or inode-validation defects.

The [evidence index](adaptive_rl_batch_005_evidence_index.json) binds the preserved
OS query evidence separately from the execution artifacts. It retains the limits
of both sources rather than converting a later filesystem inventory into proof
of continuous execution custody.

## Evidence disposition

The closed-file forensic inventory preserves **128 ordinary run files and 139
macOS metadata files**. The normal-closure auditor and reporter were not run:
their admission requirements are not met. No policy was selected, no comparison
mean or optimizer ranking was computed, and no operational-benefit claim follows.
Publication rehashed all 363 preserved run, tool-archive and readiness files
(176 ordinary files and 187 metadata files), plus the OS evidence, without
interpreting learner payloads.

| External record | SHA-256 |
| --- | --- |
| Exact clean-source gate | `8ea2b4dbf999d083815c5527449c9d5cdc9a08e110be733989b8bb8ad2c639e3` |
| Frozen tool manifest | `fc32123ab519ec29337803e8c08301e98cfab4fd1862c0b84c603c68620c9b90` |
| Launch receipt | `92f1f2c122432630ebe7c78c3eb727c268c35d598c30a67314c94d673fa65540` |
| Forensic interruption inventory | `a768e110eb1b0c7b4e6cf4d8e449adef33bca3b73aa8d314344c47fec1bf434e` |
| Preserved OS event query | `7d4d97084054c116d6a8a88f2baae0a03a71023f61f601246a0cd36000281a5d` |

Preserve this failed attempt and its raw artifacts without resuming it. A further
execution requires its own protocol, exclusive root, source gate and prospective
analysis tools, with storage continuity addressed before launch. Demand remains
**6,720 kg over 168 hours in 28 obligations**. PPO still needs to demonstrate
economic, wear or recovery value while preserving service and resources; the
interruption neither provides that evidence nor resolves the research hypothesis.
