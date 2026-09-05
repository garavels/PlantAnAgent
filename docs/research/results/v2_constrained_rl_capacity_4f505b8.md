# Constrained-RL capacity attempt at `4f505b8`: verified campaign projection

Date: **2026-09-05**

Classification: **fixture-only infrastructure capacity evidence; no controller-performance claim**

## Result

The only capacity attempt at clean revision
`4f505b89c36dbb5542fd255e469e3c729791cee2` completed all three frozen cells once, sequentially,
in PI-Lagrangian, integral-only, and scalar-PPO order. Every child exited zero, every checkpoint
passed the complete production admission path, and the canonical V3 report issued the frozen
15-policy campaign projection. The runner's immediate verification and a later detached same-host
production verification returned the same report identities and completed status.

This result measures fixture-domain execution capacity only. It opened optimizer seed 101 and
fixture episode seeds 9,880,000--9,880,015 for each family. It opened no registered optimizer seed,
development seed, held-out seed, controller comparison, policy selection, or plant-performance
outcome. The three checkpoints remain permanently ineligible for registered evaluation.

## Pre-execution custody and release gates

Repair 002 was frozen and committed in `82a616c` before source changes. Its implementation was
committed separately as the exact clean revision `4f505b8`. Before the attempt:

- the complete Python repository passed **1,832 tests in 415.48 seconds** with the existing
  Starlette/httpx warning and trailing HiGHS diagnostics;
- the 69-test capacity suite, constrained-learner selection, locked integration selection, scoped
  strict typing, Black, Python compilation, Git whitespace, frontend production build, and
  frontend `oxlint` passed;
- the Repair-002 file matched frozen SHA-256
  `1809e88942a0d075409e80c12219a1352bb2abfe4210a73580446e2a82012b06`;
- the tracked and untracked worktree was empty;
- `/private/tmp/plantanagent-beta-rl-capacity-4f505b8` did not exist;
- no repository file or earlier artifact referenced this revision as a capacity attempt;
- a read-only process check found no capacity, training, controller-benchmark, or test process;
- all five bound thread-count environment settings were absent; and
- the operator supplied both required exclusivity and first-attempt assertions.

As an operator-custody statement, all other agents were already stopped or complete and no
repository edit, test, subagent task, or other experiment ran while the three cells executed; the
root agent only polled the parent process. The canonical report independently binds the narrower
assertion that no other repository benchmark was intentionally running.

## Execution and authenticated identity

The single execution command was:

```text
make v2-beta-rl-capacity \
  OUTPUT=/private/tmp/plantanagent-beta-rl-capacity-4f505b8 \
  ASSERT_EXCLUSIVE=yes ASSERT_FIRST_ATTEMPT=yes
```

The detached verification command was:

```text
make v2-beta-rl-capacity-verify \
  REPORT=/private/tmp/plantanagent-beta-rl-capacity-4f505b8/report.json
```

| Identity | Value |
|---|---|
| schema/domain | `plant-beta-constrained-rl-capacity-report-v3` |
| raw `report.json` SHA-256 | `3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa` |
| report-domain SHA-256 | `78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e` |
| deterministic pre-run evidence SHA-256 | `4676d4862b3d783be2d4990034bbcde31be66ef7a8d591331b007ace6651b27c` |
| runtime provenance SHA-256 | `970041ae08346527aa9e5fd31743fdf7be4d82ac805c96d6152bdb3881974a5d` |
| status | `completed` |

The report binds the original protocol, Repair 001 and both clarifications, and Repair 002 with
their exact raw hashes. Its source identity records a clean worktree at `4f505b8`, Python 3.12.7,
NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, Torch 2.13.0, SciPy 1.18.1, the exact
`uv.lock`, simulator source, and dependency-entrypoint digests.

The bound host was an Apple M4 Pro with 12 physical/logical cores, 24 GiB physical memory, eight
Torch intra-op threads, twelve Torch inter-op threads, and all five declared BLAS/OpenMP thread
environment settings absent. The projection applies to this recorded sequential host context; it
is not a claim about another machine or parallel execution.

## Admitted cell measurements

Each cell proves exactly 8,192 controller transitions, 122,880 physical minute steps, one rollout,
320 optimizer steps, 16 fixture allocator entries, and one admitted final checkpoint.

| Order | Family | Wall s | Child user s | Child system s | Rollout s | Fixed overhead s | Controller steps/s | Checkpoint logical bytes | Checkpoint allocated bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PI-Lagrangian | 129.550079666 | 126.012431 | 2.835579 | 117.208728250 | 12.341351416 | 69.892405816 | 1,245,824 | 1,253,376 |
| 2 | Integral-only | 130.401020125 | 126.757362 | 2.853473 | 117.977248541 | 12.423771584 | 69.437116913 | 1,245,805 | 1,253,376 |
| 3 | Scalar PPO | 128.532934000 | 124.870081 | 2.633286 | 116.023039708 | 12.509894292 | 70.606665888 | 578,575 | 581,632 |

The exact cell configuration identities are:

| Family | Configuration SHA-256 |
|---|---|
| PI-Lagrangian | `a008b9b14b6cb2468953fe98d907e92cebb18b84c7f5002fe3f0bdd93154be89` |
| Integral-only | `43134ca0f1d2905acceb5410ea6f1edd38c6e1938e035804fc534a29adcfef74` |
| Scalar PPO | `2748da89b653ea7ffba27de26b8fb8c83b4f07abd9118532d48d9b0b7f9ed36b` |

Every stderr record is empty and has the empty-file SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The three stdout logs
remain unopened. The authenticated report binds them without exposing their contents:

| Family | stdout bytes | stdout SHA-256 |
|---|---:|---|
| PI-Lagrangian | 1,225 | `bd8ad2535c9eedc1bf760926aaba43a1d6df7efd5f1f198dec83113a2d0eca84` |
| Integral-only | 1,224 | `f3b4e89326fcb12b89135f3fc20f43b69f806a045269c24d0ddb63ded66232ab` |
| Scalar PPO | 1,222 | `b72ec8664bd1b8ac8566289ead4e3533bd6452dac35596859917fa78e227c29f` |

## Frozen campaign projection

The registered protocol fixes 15 policies, 1,920 rollouts, 15,728,640 controller transitions,
235,929,600 physical minute steps, 614,400 minibatch optimizer steps, and 23,460 reset manifests.
Applying the frozen integer-nanosecond formula gives:

| Rollout-time factor | Sequential wall ns | Hours | Days |
|---:|---:|---:|---:|
| 1.00x | 224,960,145,645,820 | 62.4889 | 2.6037 |
| 1.25x | 281,153,588,285,660 | 78.0982 | 3.2541 |
| 1.50x | 337,347,030,925,500 | 93.7075 | 3.9045 |

The point-estimate family contributions are 20.8542 hours for PI-Lagrangian, 20.9910 hours for
integral-only, and 20.6437 hours for scalar PPO. The 1.25x and 1.50x rows mechanically inflate only
the rollout term; they are planning scenarios, not confidence intervals, runtime guarantees, or
measurements of thermal throttling. The matched per-family timings are operational capacity
measurements, not a statistical comparison of algorithms. No timing or checkpoint from the failed
`fab4a55` V1 or `27a47d0` V2 attempt enters this projection.

The structural checkpoint allowance projects:

| Family | Projected logical bytes/policy | Five-policy logical bytes |
|---|---:|---:|
| PI-Lagrangian | 12,463,136 | 62,315,680 |
| Integral-only | 12,463,117 | 62,315,585 |
| Scalar PPO | 11,637,391 | 58,186,955 |
| **Total** | — | **182,818,220** |

The projected checkpoint total is **174.35 MiB**. The frozen rule recommends at least
**1,256,560,044 bytes (1.17 GiB)** free for checkpoint growth alone. This conservative structural
allowance deliberately ignores ZIP compression and is not a prediction of exact ZIP encoding. Raw
artifacts, logs, reports, temporary serialization, development evaluation, and held-out evidence
require separate space.

## Decision and claim boundary

The report's only admissible result claim is:

> On the recorded host and clean source revision, three fixture-domain, registered-shaped first
> rollouts imply the reported sequential capacity scenarios. No registered seed or
> controller-performance outcome was opened.

The report publishes **62.49**, **78.10**, and **93.71 host-hours** as its declared sequential
planning scenarios. It does not establish which algorithm performs better, whether any policy
learns, whether constraints are satisfied, whether the environment is mastered, or whether RL,
HEPA, supervision, or the complete stack supports a safety, state-of-the-art, or commercial claim.

As a protocol-process decision rather than an experimental claim, the valid capacity report closes
the infrastructure measurement required before requesting registered execution. The next
admissible step is to present it with the already frozen registered protocol and request explicit
authorization for the 15-policy campaign. No registered training or development/held-out reset
starts merely because that request prerequisite is closed.
