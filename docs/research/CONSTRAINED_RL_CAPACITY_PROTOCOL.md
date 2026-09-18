# Constrained-RL beta capacity protocol

Status: **frozen before benchmark implementation or execution**

Date: **2026-09-05**

Protocol identifier: **`plant-beta-constrained-rl-capacity-v1`**

## Decision and claim boundary

Before any registered optimizer seed is consumed, one bounded fixture-only
measurement will estimate the sequential wall-clock and storage required by the
15-policy campaign frozen in `CONSTRAINED_RL_BETA_PROTOCOL.md`.

This is infrastructure capacity evidence, not controller evidence. It may inspect
only timings, host/runtime identity, checkpoint structure, byte counts, allocator
custody, and optimizer-accounting fields needed to prove that the measured work has
the registered shape. It must not publish or compare rewards, constraint outcomes,
actions, multipliers, losses, policy behavior, market regimes, outages, inventories,
or plant performance. Every checkpoint remains a fixture artifact and permanently
ineligible for registered evaluation.

The measurement does not authorize:

- a registered training reset or optimizer update;
- a development or held-out reset;
- a retry, extension, fastest-run selection, or parallel speed claim;
- a policy checkpoint choice or controller comparison; or
- a statement that any learner masters the task or improves plant value.

## Frozen cells

Exactly three cells run sequentially in fresh Python processes and in this order:

| Order | Algorithm | Optimizer seed | Fixture episode-seed base |
|---:|---|---:|---:|
| 1 | `plant-beta-ppo-pi-lagrangian-v1` | 101 | 9,880,000 |
| 2 | `plant-beta-ppo-i-lagrangian-v1` | 101 | 9,880,000 |
| 3 | `plant-beta-sb3-ppo-scalar-v1` | 101 | 9,880,000 |

The identical optimizer and episode stream makes the cells operationally matched;
their timing is not a statistical algorithm comparison. There is one attempt per
cell and committed source revision. A failed or slower cell is retained. There is
no replacement seed or outcome-dependent repetition.

Each child receives an exact 3,600-second wall-clock timeout. A timeout is a
retained failed attempt, including any partial captured standard output and error;
it is never authorization to retry the cell. Before execution, the operator must
separately attest both that no other repository benchmark is intentionally running
and that the current committed revision has no prior capacity attempt.

Each cell uses the production controller-neutral training entry point with:

```text
environment                 plant-env-v2-cyclic-market-beta-v1
run kind                    fixture
requested/collected steps   8,192 / 8,192
n_envs                      4, synchronous
n_steps                     2,048 per environment
batch size                  256
epochs                      10
learning rate               3e-4
reward gamma                0.995
reward GAE lambda           0.95
clip range                  0.2
entropy coefficient         0
reward value coefficient    0.5
maximum gradient norm       0.5
policy                      MlpPolicy, [64, 64] tanh
device                      cpu
checkpoint cadence          final only
resume                      forbidden
```

All plant, episode, constraint, reward, economic, market, disturbance, and cycle
objects remain their exact default beta configuration. Each cell is therefore one
registered-shaped rollout: 8,192 controller transitions, 122,880 one-minute
physical steps, one policy update, 320 minibatch optimizer steps, and 16 fixture
reset/manifest records.

The benchmark root must be a new absolute directory outside the repository. Each
cell receives a new non-overwriting output path. No benchmark cell may execute
concurrently with another benchmark or repository experiment.

## Source and host admission

Before creating the output root, the runner must:

1. reopen this protocol as a stable regular file and match its committed SHA-256;
2. require a clean Git worktree and capture the existing beta-training runtime
   provenance;
3. record the exact Git revision, dependency lock, interpreter, dependency
   entrypoints, simulator source, and platform identity;
4. record CPU brand where available, architecture, logical and physical CPU count
   where available, physical memory where available, Torch intra/inter-op threads,
   and the declared BLAS/OpenMP thread environment; and
5. bind operator assertions that the three cells are sequential, no other
   repository benchmark is intentionally running, and the committed revision has
   no prior capacity attempt.

The runner re-captures clean provenance after every child process and after report
publication. Any source, lock, dependency, or runtime change fails the measurement.
Host fields that cannot be portably observed are recorded as `null`, never guessed.
The immediate verifier independently re-captures the host identity and requires an
exact match with the recorded parent-process identity.

## Per-cell measurements and admission

The parent process uses `time.perf_counter_ns` around each complete training child.
The child CPU time is measured where the host exposes it. Standard output and error
are retained as files, but the report records only their byte counts and SHA-256
digests.

Every successful checkpoint is reopened through the production admission path. Its
metadata and beta evidence must prove:

- the exact expected family, optimizer seed, task, PPO configuration, and 8,192
  collected transitions;
- `run_kind == fixture`, no resume parent, and registered eligibility false;
- exactly one 8,192-transition rollout and exactly 320 optimizer steps;
- allocator mode/split `fixture`, base 9,880,000, exactly 16 reset records, and no
  seed outside 9,880,000 through 9,880,015; and
- runtime provenance identical to the clean outer capture.

The report retains:

- child exit status and attempt order;
- end-to-end wall time in integer nanoseconds;
- the rollout ledger's collection-plus-optimization time;
- fixed process/setup/save/reopen overhead, defined as end-to-end wall time minus
  ledger time and floored at zero;
- controller transitions and physical minute-steps per second;
- logical and allocated bytes plus SHA-256 for `policy.zip`, `metadata.json`, and
  `training_config.json`; and
- rollout-ledger and allocator-entry counts and maximum canonical entry sizes.

A child failure produces a failed report with no campaign estimate. It is not
silently retried. The fixture artifacts and logs remain diagnostic only.

## Frozen duration projection

The registered campaign contains 128 rollouts per policy, five policies per family,
and three families. For each successful cell:

```text
one-policy point estimate = measured fixed overhead + 128 * measured rollout time
family point estimate     = 5 * one-policy point estimate
```

For the integer-nanosecond sensitivity scenarios, the rollout term is calculated
as `ceil(factor * 128 * measured rollout nanoseconds)` using exact rational factors
`1/1`, `5/4`, and `3/2`; rounding is applied once after multiplying by 128.

The sequential campaign point estimate is the sum of the three family estimates.
Because one fixture rollout cannot measure long-run thermal throttling or later
checkpoint serialization, the report must also publish mechanical sensitivity
bands that multiply the rollout term, but not one-time overhead, by `1.25` and
`1.50`. These are planning scenarios, not confidence intervals or guarantees.

The fixed campaign arithmetic is reported alongside the estimate:

```text
policies                    15
controller transitions      15,728,640
rollouts                    1,920
minibatch optimizer steps   614,400
reset manifests             23,460
physical minute-steps       235,929,600
```

## Frozen storage projection

One-rollout fixture checkpoint size is measured exactly but is not presented as a
registered endpoint size. A registered artifact contains 128 rollout records and
1,564 allocator entries, versus one and 16 respectively in the fixture.

For transparent capacity planning, each family reports:

1. the exact measured one-rollout logical and allocated bytes;
2. a structural growth allowance calculated as
   `2 * (127 * 3 * max_rollout_entry_bytes + 1,548 * 4 *
   max_allocator_entry_bytes)`;
3. projected checkpoint bytes as measured logical bytes plus that allowance; and
4. five-policy family bytes as five times the projected checkpoint bytes.

The factors deliberately ignore ZIP compression, allow three copies of each future
rollout entry and four copies of each future allocator entry, and then double the
result. This is a conservative planning allowance, not a prediction of exact ZIP
encoding. Recommended free space is the greater of twice the projected 15-policy
total and that total plus one GiB. Raw artifacts, logs, reports, temporary
serialization, development evidence, and held-out evidence remain separate storage
budgets.

## Report and verification

The runner publishes one canonical, domain-hashed `report.json` with exclusive,
durable create semantics and immediately reopens it through a strict verifier. The
report schema binds this protocol, source/runtime/host identity, exact cell order and
configuration, every attempt including failures, measurement definitions, fixed
campaign arithmetic, projection formulas, assumptions, and claim boundary.

The verifier uses exact field sets, rejects duplicate JSON keys and non-finite
numbers, recalculates every rate/projection/digest, checks all referenced regular
files without following symlinks, and refuses a successful report unless all three
checkpoint artifacts independently pass production admission. Wall-clock fields are
excluded from deterministic-evidence identity but remain inside the report's full
content digest.

The result may support only this statement:

> On the recorded host and clean source revision, three fixture-domain,
> registered-shaped first rollouts imply the reported sequential capacity scenarios.
> No registered seed or controller-performance outcome was opened.
