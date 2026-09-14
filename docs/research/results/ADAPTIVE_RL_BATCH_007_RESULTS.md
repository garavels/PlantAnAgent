# Adaptive batch 007: completed training, evaluation aborted by power guard

Both scalar and PI-Lagrangian MPC-residual arms completed **65,536 transitions**
and saved their fixed final checkpoints. The attempt later aborted during replay:
the power guard rejected `finishing charge` while macOS reported **AC power and
99% battery**. No performance comparison has been admitted or interpreted.

The attempt followed the [batch-007 protocol](../ADAPTIVE_RL_BATCH_007_PROTOCOL.md),
using source `61b97851637a0f5a16e56c479e1a436cd30cf68d`, tree
`311f8d3b5d048643e4a38b638c73ef95363b4d27`. Launch was
2026-09-13 22:48:47.964279 UTC; report closure was
2026-09-14 10:45:29.395990 UTC. Source stayed clean through closure and preservation.
All active artifacts and preserved evidence remain on internal storage under
`/Users/filippogombac/Developer/PlantAnAgentResearch`.

## Verified training completion metadata

| Recorded endpoint | Scalar A | PI B |
| --- | ---: | ---: |
| Transitions | 65,536 / 65,536 | 65,536 / 65,536 |
| Completed optimization rollouts | 8 / 8 | 8 / 8 |
| Minibatch optimizer steps | 2,560 | 2,560 |
| PPO epoch updates | 80 | 80 |
| Allocator resets | 100 | 100 |
| Decisions per environment | 16,384 | 16,384 |
| Final result published, September 14 UTC | 08:05:20.536991 | 08:04:14.180150 |
| Ordinary trainer exit code | 0 | 0 |

Each completed rollout records 8,192 transitions and 320 optimizer steps.
Published rollout records match the endpoint ledgers, and all 100 reset records
match each allocator snapshot, covering development seeds 9,630,000–9,630,099.
The frozen allocator verifier passed without executing a reset. Both endpoints
record no pending or aborted rollout, unledgered transitions or resume receipt.
Neither arm has a training-failure record. The wave records both supervisors and
trainers closed, both arms admitted successful, no fatal error and no cleanup error.

The producer published each result after its endpoint and checkpoint-reopen checks.
Post-closure opaque byte hashing confirms these final files match their result records:

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| A `final-policy.zip` | 972,651 | `4031802ee0d8b7042de4b5193ab82797479dfa800d6943f62b134aa0e415e524` |
| B `final-policy.zip` | 1,639,652 | `881140cb3ddfa52d975017bb6cbbcbcecb8fbeb147a91cdd6a7db8430fd8e05d` |

These are saved endpoint candidates, not a normal admission of the aborted experiment.
Post-closure inspection loaded no models and interpreted no rewards, costs,
learning-quality diagnostics, policy tensors or trajectories.

## Evaluation interruption and custody limitation

| Stage | Planned | Exit 0 | Integrity exit 86 | Unattempted |
| --- | ---: | ---: | ---: | ---: |
| Evaluation | 56 | 26 | 0 | 30 |
| Independent replay | 56 | 25 | 1 | 30 |

These are process counts, not accepted performance results. The exit check for
replay `9700303-initial-mpc-residual` recorded, at 10:45:29.220552 UTC, AC power, 99%,
`finishing charge`, battery present, command exit 0 and a 12.319166 ms query.
The frozen accepted-state list excluded that label, producing
`power_battery_state_rejected` and the existing integrity exit. The root then
closed with a generic `RuntimeError`. This sample identifies a power-guard false
positive; it supplies no evidence of power loss. The already-published replay
file was not interpreted. The refusal remains preserved.

A separate metadata-only call to the frozen supervision verifier failed A stdout's
allocated-byte identity: the completion record says 589,824 bytes; post-closure
storage reports 581,632. Logical length remains 580,389 bytes and SHA-256 remains
`74c355b7d7013b4d24589917e91ebe712f096092481e0aa0cdca914d0971b272`.
The other three ordinary training streams match their records. This allocation
discrepancy has not been waived and is not established as the cause of the abort.
Normal campaign audit and reporting were not run; complete admission is not claimed.

## Preserved evidence and next boundary

The `adaptive-rl-batch-007-analysis` directory retains the forensic inventory and
power-refusal evidence, including process-closure snapshots. Known groups were
absent in the post-closure process inspection; internal originals remain preserved.

| Preserved record | SHA-256 |
| --- | --- |
| `interruption-inventory.json` | `37d2ec6f25f0ff6d7f3dd949bfd97cd657378298ac93ab4ef067c7c09ef26f95` |
| `power-refusal-evidence.json` | `3bb591413ca3bffe09f1a3338713ef9fbab029caf0ff6ed2320ce6ec1c5957a4` |

There is no demonstrated need to retrain. Reusing these exact endpoints requires
a separately frozen evaluation, explicit treatment of the preserved custody
failure, checkpoint verification and a complete evaluation/replay closure graph.
No new run has launched. Previously opened development material must remain
identified as such; original registered and protected study seeds remain unopened.
The task remains 6,720 kg methane over 168 hours in 28 deliveries, with unchanged
physics, rewards and operational authority. No operational benefit is established.
