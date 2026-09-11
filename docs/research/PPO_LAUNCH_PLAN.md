# PPO launch plan

Prepared 2026-09-11. Execution is not authorized by this document. The final
readiness packet binds the exact clean revision and reports the release gate.

## Fixed campaign

| Item | Value |
| --- | --- |
| Families | PI-Lagrangian, I-Lagrangian, scalar PPO |
| Optimizer seeds, each family | 101, 211, 307, 401, 503 |
| Order | Family first, seed order above; 15 sequential children |
| Per policy | 1,048,576 transitions; 128 rollouts; 40,960 optimizer steps; 4 environments |
| Task | 168 hours; 672 decisions; 10,080 physical minutes; 135 observations; 4 actions |
| Source | Exact clean commit/tree in the generated readiness packet |
| New campaign root | `/Volumes/Lexar/PlantAnAgentResearch/registered-beta-v1` |
| Host | This Apple M4 Pro, 12 cores, 24 GiB; Torch 8 intra-op / 12 inter-op |
| Time | Historical projection 62.49 h; 1.25× / 1.50× scenarios 78.10 / 93.71 h |
| Hard timeout | 10 h per child, 150 h sum of child ceilings |
| Logs | 64 MiB per stream; at most 1.875 GiB across 30 streams |
| Opening training floor | 5 GiB, including the conservative checkpoint/log allowance |
| Failure | Retain every attempt; no replacement, retry, resume or outcome inspection between cells |

Lexar is ExFAT with 256 KiB allocation units. A bounded probe passed production
exclusive file/directory publication, `fsync`, duplicate rejection, identity change
on identical-byte replacement, and canonical compressed archive roundtrip. About
326.6 GiB was free at inspection. No existing user files were changed. This is a
filesystem compatibility check, not a power-loss guarantee or new training speed
measurement. Keep the disk connected throughout execution and verification.

The conservative evaluation proposal preserves the existing 32 MiB H/PPO and
64 MiB M24/E-beta physical archive caps, 2 MiB timing caps, 1 MiB per log,
one-hour child ceilings and the original decoded/report limits. Its filesystem
margin increases from 2 to 4 GiB for external-volume allocation overhead. It
requires **48.80 GiB** for both stages at development opening, plus training
artifacts. Lexar meets this planning reserve. Complete trained-policy/MPC stage
capacity is still unmeasured; these are caps, not promised runtimes or a guarantee
that no child reaches its cap. The budget becomes fixed in later stage authority,
unchanged between development and held-out. Compression experiments on saved
archives are retained; the production archive format is unchanged.

## Scientific decision

Retain the frozen task and research E-beta, including the three recorded opening
infeasibilities. This is a launch with an explicit risk of no admissible policy:
one ineligible E-beta development episode prevents a complete 32-pair primary
score for every PPO policy. The product's operating MPC cannot substitute for
E-beta. Do not change rewards, forecasts, resource returns or the selection gate
after seeing training outcomes. A negative development result remains useful
evidence of the environment's RL interface and its limitations; it does not
establish useful learned control. Held-out remains sealed when the gate fails.

## Before execution

1. Finish and retain the complete clean implementation gate: Python suite,
   changed-source strict typing/Black, compilation/imports and frontend build/lint.
2. Generate the read-only proposal with `docs/research/tools/ppo_readiness_packet.py`
   in a fresh process. It verifies clean source, frozen protocols, all 15 configs,
   exact recovered capacity bytes, matching host, absent campaign/authorization,
   and storage. It creates only the explicitly named proposal file.
3. Obtain explicit approval for the exact source/root/host/duration/budget and
   the retained scientific risk. The frozen custody protocol requires that
   concrete multi-day execution decision. Readiness work and permission to use
   Lexar do not create it.
4. Commit the canonical authorization with that implementation as its sole parent;
   only the protocol's enumerated authorization/status paths may change. Run the
   complete clean gate at the authorization revision, then execute the registered
   campaign once through `registered_campaign run` with its three assertions.

The final implementation result can be retained outside tracked source alongside
the exact proposal so the checked revision stays unchanged. Its compact journal
entry can accompany the later authorization/status-only commit. Do not amend the
approved source merely to embed its own gate result or hash.

## Closed-custody product workflow

The new **Learning** panel reads bounded, integrity-checked exports from ignored
`runs/commission-rl`. HTTP cannot load a checkpoint, inspect a running campaign,
authorize a stage or start PPO. Every exporter uses fresh production admission;
exports are exclusive and cannot silently replace prior evidence.

After training closure, on the unchanged authorized source/runtime:

```sh
.venv/bin/python -m plant_backend.publish_ppo training /Volumes/Lexar/PlantAnAgentResearch/registered-beta-v1/report.json
```

This publishes all 15 slots, failure codes and complete admitted training curves.
The objective is the sum of the frozen family rewards over 8,192 transitions,
not episode profit. Resource costs and multipliers retain the original ledgers;
private episode allocation and future truth are not exported.

After separately authorized, completed and verified development/held-out stages,
use the same command with `development` or `heldout` in place of `training`.
The exporter reconstructs the complete stage, preserves all 576 actor/episode
slots, and exposes execution, safety, cycle, selection and value outcomes. The
web server checks training/stage identities and refuses orphan held-out exports.
Public checksum validation is not a fresh custody verification or execution grant.

The separate future presentation rehearsal uses final PI-Lagrangian seed 101 on
fixture 9,000,034, fixed before training, for at most 168 hours/20 minutes wall time.
The explicit `rehearsal --assert-authorized` command records it through the same
dispatcher/guard, preserves stopped prefixes and attempt markers, and quarantines
a replay rejected by the post-run custody check. A failed fixed checkpoint has no
replacement. The rehearsal is descriptive and cannot select a winner or open a
protected seed. It is not executed during this readiness build.

## Evidence

- [Prospective implementation scope](PPO_READINESS_BUILD.md)
- [Storage and compression probes](results/v2_ppo_readiness_storage_20260911.json)
- [Historical full gate and capacity review](TRAINING_READINESS_REVIEW.md)
- [Frozen custody protocol](CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md)
- [Commission submission scope](COMMISSION_SUBMISSION_SCOPE.md)
