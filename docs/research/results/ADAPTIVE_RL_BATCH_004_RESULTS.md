# Adaptive batch 004: aborted before learning

The matched scalar/PI MPC-residual attempt **aborted before the first PPO update**.
It produced no final policy, no comparison episodes and no performance result.
This is an engineering failure of the forecast interface and execution evidence
path. It neither supports nor refutes the PI-versus-scalar hypothesis.

The unchanged [prospective protocol](../ADAPTIVE_RL_BATCH_004_PROTOCOL.md) and
source `4f908fcaeaeed417a793204778d87f62c29ec768` remain the record of this attempt.
The original root is `/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-004`.
Its files have not been repaired, resumed or reused.

## Admission and execution

The exact clean-source gate passed 698 Python tests in 1,140.75 seconds, plus
Black, strict source MyPy, compilation, CLI and whitespace checks. This scoped
dependency gate did not repeat the full repository/frontend suite; the last full
gate remains `b5ea55e`, with 3,052 Python tests and static/frontend checks.

Prospective detached tools passed 105 fabricated auditor tests in 406.07 seconds
and 62 reporter tests in 33.79 seconds. Superseded attempts and review corrections
were retained. Constant-only source/gate pinning was checked by normalized AST
equality; tests were not falsely claimed rerun after pinning. The 54-file tool
archive was frozen before either trainer started. These passing fixtures missed
the forecast sampling mismatch and Lexar's file-allocation identity transition.

Both seed-101 arms launched on AC power at **2026-09-13 07:24:38 UTC**, under
separate supervisors, each with the fixed 65,536-transition budget. Process-only
monitoring confirmed both were live; no partial rewards, weights or diagnostics
were inspected during execution. The wave closed at **08:00:47 UTC**. Root later
verified all five registered process IDs and groups absent before forensic reads.

| Arm | Retained outcome | Evidence limit |
| --- | --- | --- |
| A, scalar | Exit 86 after 2,165.68 seconds; learner count 4,048 | Zero completed rollout/optimizer ledgers; no final checkpoint |
| B, PI | Cancelled with signal 15 after 2,165.70 seconds | No final checkpoint or completed diagnostics; exact transition count unavailable |

Scalar recorded eight resets from 9,630,000 through 9,630,007, all with null
ignored framework-seed metadata. The four allocator consumption counts were
1,013, 1,013, 1,013 and 1,012. The residual wrapper physically advances a step
before priming its next MPC target; the failed fourth environment had reached
decision boundary 341 before that target calculation raised. The allocator and
vector learner record progress only after their wrapped calls return. The partial
snapshot is therefore not a resumable checkpoint. The eight development seeds
9,700,300–9,700,307 were never evaluated.

## Forecast failure

The public forecast producer uses minute-start samples of clear-sky solar power.
The MPC projection validated issued band energy against a continuous integral.
The [exact public-issue diagnosis](adaptive_rl_batch_004_forecast_diagnosis.json)
reproduced scalar's failure for already consumed seed 9,630,007 at day 4, 19:15,
without an environment reset, physical advance, policy inference or solver call.

| First-hour normalized energy | Value |
| --- | ---: |
| Issued mean | 0.007359442500906482 |
| Producer's sampled clear-sky mean | 0.007754685426807396 |
| MPC's continuous clear-sky mean | 0.00727015429953326 |
| Issued / sampled bound | 94.903% |
| Issued / continuous bound | 101.228% |

The forecast is valid against the producer's envelope. A seed-free forecast at
95% of the sampled envelope independently reproduces the old rejection. This
is not numerical roundoff, delivery infeasibility or terminal-horizon exhaustion.
The old mean and p10 projections share the analytic envelope; switching quantiles
would not correct its meaning. Existing fixtures built their input from that same
analytic helper, explaining the missed compatibility boundary.

## Supervision failures

The parent reported a generic `ValueError` without its message or location.
During cancellation, scalar's supervisor was interrupted while publishing its
completion, leaving an empty file. PI's supervisor raised ENOTCONN during socket
shutdown and never published completion. Both ordinary child receipts survived
separately, but the parent report contains zero receipts because it parsed the
missing/broken completion before attempting receipt retention.

A separately claimed, tiny real-volume probe showed that Lexar's exFAT/FSKit
can change the inode of an empty file during ordinary allocation/fsync; an open
reader remains attached to the same file. The monitor's cached initial inode
check is thus incompatible with this volume. Replacement leaves the reader on
the old object, while truncation can change path and reader metadata together.
This supports a descriptor-based monitor with bounded prefix verification. It
does **not** establish which live check caused the original parent ValueError:
the necessary first-failure detail was not retained.

## Evidence status and decision

The [closed-file inventory](adaptive_rl_batch_004_results.json) preserves all
44 ordinary-file and 55 macOS metadata-file digests, the original missing-receipt
state and both independently recovered receipts. It is a forensic inventory,
not a successful execution-custody audit. The normal-closure auditor/reporter was
not run on this aborted root. No averages, policy rankings or fallback-derived
weekly wins can be reported.

The [publication index](adaptive_rl_batch_004_evidence_index.json) binds the exact
gate, launch, frozen tools, forecast diagnosis and exFAT probe. Publication
rechecked every referenced file: 99 batch files, 54 tool files, three diagnosis
files and 12 probe files, plus gate logs. The Git copies of the inventory and
diagnosis are byte-identical to their closed external originals.

| External record | SHA-256 |
| --- | --- |
| Clean gate | `98efb720738e434bdde7ad7b35a87c8268e706af22eb59af9103164dd7dcd91b` |
| Frozen analysis manifest | `514d4c54482e864b45538361d94a839641b24c232799054928e2fe785ec8a60a` |
| Original aborted report | `f4b7c3a4796f99f45686db1ea6fe9d6250a260c79d4511094a7e88c7072a68f3` |
| Closed-file inventory | `43da491388c802597866703dc1030b0dacee30429e9c7de90a0b91f20dd9b017` |
| Forecast diagnosis manifest | `fc1ee1a173170e0c385eb8e79b3bc5e9c2bf884b2132b3fbe31bacdf6e1a806d` |
| exFAT probe manifest | `d7dc4dab0139813f40d776ac76fbf7de91d7879b4623eaabcf8462ceb235902d` |

Proceed with the [bounded repair scope](../ADAPTIVE_RL_BATCH_004_FAILURE_AND_REPAIR.md):
a separately versioned, sampled OperatingMPC projection plus reliable supervisor
monitoring and failure evidence. Preserve the old registered projections and all
failed artifacts. Any subsequent attempt needs a new protocol/root, source gate
and prospective tools; batch 004 is never retried or resumed.

Delivery remains 6,720 kg over 168 hours. PPO still needs economic, wear or recovery
benefit with service/resource preservation. No policy is qualified, no confirmation
or protected-study seed was opened, and no operational-benefit claim is justified.
