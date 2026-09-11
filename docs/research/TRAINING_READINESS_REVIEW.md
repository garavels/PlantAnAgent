# Registered study readiness review

Date: **2026-09-11**

Status: **implementation gate passed; final source and execution authorization pending**

The training and study custody implementation passes the complete repository gate
at clean `12936c7627940f2ede04405bbbba28603bb8b10f`. No registered policy has
been trained, selected or evaluated. This review is not an authorization record.
The three families, five optimizer seeds, task, learning budgets and protected
seed domains remain unchanged.

## Verified implementation

| Check | Result at `12936c7` |
| --- | --- |
| Complete Python suite | 3,002 passed, one deprecation warning, 2,037.91 seconds, exit 0 |
| Strict typing | All 36 source modules changed since `ad6ad1e` pass |
| Formatting | All 57 changed Python files pass Black |
| Compilation and CLI imports/help | Pass |
| Frontend production build and lint | Pass; existing bundle-size warning retained |
| Whitespace, dependency lock and registered task identities | Pass |

The source remained clean throughout those checks. Its tree is
`5163ffaa5e870de75100236ffaaa8b0c986f9c81`; the unchanged lock SHA-256 is
`080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`.
The later documentation commit records this gate; it does not claim a new full
test run at its own revision. The UI and controller behavior were not changed
during this review.

Training custody retains every one of the 15 attempts and admitted checkpoints.
The study implementation retains all 576 actor/episode slots per stage, including
failed arms, and independently reconstructs physical, economic, timing and
selection evidence. Passing fixture tests establishes implementation evidence,
not trained-policy performance or full-stage resource capacity.

## Recovered capacity evidence

The original temporary capacity directory had no regular files when inspected.
The exact 17,380-byte V3 report was recovered from historical tool output and its
unchanged deterministic cell contract. Acceptance required both frozen digests:

- Raw SHA-256: `3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa`.
- Report-domain SHA-256: `78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e`.

The recovered report and recovery record are stored outside Git at
`/Users/filippogombac/Developer/PlantAnAgentResearchArtifacts/capacity-4f505b8/`.
The production regular-file reader, both digests, complete static contract and
projection arithmetic have been rechecked. A fresh host identity exactly matches
the recorded capacity host. Recovery required no plant reset or optimizer step.
The original checkpoints and logs remain absent: this is report recovery, not
fresh detached verification of the full historical artifact set. The frozen
training authority requires these exact report bytes and digests.

## Training proposal for final review

| Item | Proposed or frozen value |
| --- | --- |
| New root, currently absent | `/Users/filippogombac/Developer/PlantAnAgentResearchArtifacts/registered-beta-v1` |
| Host | This Apple M4 Pro, 12 cores, 24 GiB memory, recorded macOS/runtime identity |
| Training Torch threads | 8 intra-op / 12 inter-op; five declared thread environment variables absent |
| Run order | PI-Lagrangian, I-Lagrangian, scalar PPO; seeds 101, 211, 307, 401, 503 per family |
| Per-policy budget | 1,048,576 transitions, 128 rollouts, 40,960 optimizer steps, four environments |
| Sequential training projection | 62.49 hours; mechanical 1.25x/1.50x scenarios 78.10/93.71 hours |
| Hard child timeout | 10 hours per policy; 150 hours summed child caps, excluding parent overhead |
| Logs | 64 MiB per stream; 1.875 GiB maximum across 30 streams |
| Opening free-space floor | 5 GiB; checkpoint-growth recommendation alone is 1.17 GiB |

These duration estimates are host-specific planning evidence. There is no
training root or tracked authorization file. Development and held-out evaluation
need separate later authority; training permission cannot open either stage.

## Evaluation budget and capacity gap

The compact evidence retains one **unfrozen planning scenario** constructed with
the production budget arithmetic. It gives every actor a one-hour child timeout,
1 MiB per log stream, 2 MiB stored timing and a 2 GiB filesystem reserve. Physical
archive caps are 32 MiB for H/PPO and 64 MiB for M24/E-beta; all 15 PPO policies
share the same limits. Decoded caps are retained in the evidence.

This scenario reserves 24,052,236,288 bytes (22.40 GiB) per stage. Development
opening reserves both stages plus the filesystem margin: **50,251,956,224 bytes
(46.80 GiB)**. At 05:38 UTC on September 11 the selected volume had
27,020,935,168 bytes (25.17 GiB) free, a **21.64 GiB gap**, before any further
training allocation. Available space is an observation, not reserved storage.
No files were deleted to close the gap.

Each stage has 576 hours of summed child caps and a separate 24-hour detached
verifier cap. These are ceilings, not expected durations, and exclude parent
admission, verification and publication overhead. The observed full-week H and
fixed-policy fixture runs do not measure trained inference or full-week MPC
execution. M24 and E-beta stopped early in both retained capacity attempts.
Consequently this budget is not yet a measured or frozen stage capacity claim.

## Decisions before final source freeze

1. **Resolve the primary comparator.** E-beta reached a saved sunset boundary
   1.16889234221253 kWh below its required battery return with no forecast solar
   left before settlement. This is a forecast/model-error feasibility problem;
   the earlier two-ULP representation repair did not solve it. The recommended
   next engineering scope is explicit forecast/model-error reserve handling,
   declared prospectively and tested only on fixture seeds before source freeze.
   No new controller strategy or capacity attempt is authorized by this review.
   Preserving the current baseline is also scientifically valid, with the risk
   accepted explicitly: one E-beta failure on a development episode prevents
   every PPO policy from having the required complete 32-pair score. Failure on
   the inspected fixture does not establish failure on unopened study seeds.
2. **Close evaluation capacity and storage.** Retain both negative capacity
   attempts. Agree a bounded prospective capacity scope if more measurements are
   needed, freeze defensible caps and provide the required space. M24's declared
   DAC limitation remains visible; context-arm failures do not by themselves
   block primary policy/E-beta pairs.
3. **Freeze and authorize the actual implementation.** After decisions and any
   changes, complete the clean implementation gate and present its exact commit,
   root, host and budgets. Only explicit training approval permits the canonical
   authorization commit, whose parent must be that approved implementation.
   The custody protocol then requires the same complete clean gate at the
   authorization revision before the single sequential 15-run campaign.

No positive controller outcome is required to call the custody software
implemented. A negative development result remains valid and leaves held-out
seeds sealed. Resolving the comparator before training avoids changing its
source after checkpoint provenance has been frozen.

The [compact evidence](results/v2_training_readiness_12936c7.json) records the
gate, recovery, proposed destination and full draft budget. Its raw SHA-256 is
`6a846d800613434126b654f5e36d93965492494c11bfabb42092cd9be437166f`.
The [custody protocol](CONSTRAINED_RL_REGISTERED_TRAINING_CUSTODY_PROTOCOL.md),
[analysis build](CONSTRAINED_RL_STUDY_ANALYSIS_BUILD.md) and
[retained capacity failure](results/v2_study_capacity_cfa16a6.md) remain the
underlying contracts and evidence.
