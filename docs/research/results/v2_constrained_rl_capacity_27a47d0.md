# Constrained-RL capacity attempt at `27a47d0`: retained checkpoint-admission failure

Date: **2026-09-05**

Classification: **fixture-only infrastructure failure; no capacity projection and no controller
claim**

## Result

The only capacity attempt at clean revision
`27a47d0604ffb9624a40683a610f2f8f4bb6c58a` is complete and permanently retained. The three
children ran once, sequentially, in the frozen PI-Lagrangian, integral-only, and scalar-PPO order.
Every child exited zero and published a checkpoint directory. The parent then rejected every
checkpoint during independent admission with the fixed category `checkpoint_admission_error`.

The canonical V2 report therefore has status `failed`, null result claim, null campaign projection,
and null checkpoint/measurement/projection fields for every cell. It cannot support a timing,
storage, controller, safety, or commercial conclusion. It was not retried.

## Pre-execution custody and release gate

Before execution:

- the repair implementation and its two clarifications were committed as `27a47d0`;
- the tracked and untracked worktree was empty;
- `/private/tmp/plantanagent-beta-rl-capacity-27a47d0` did not exist;
- no earlier capacity artifact referenced that revision;
- a read-only process check found no training, capacity, pytest, or repository benchmark process;
- all five bound thread-count environment variables were absent; and
- the operator exclusivity and first-attempt assertions were explicitly supplied.

The exact committed revision passed the following gates before the attempt:

- focused learner/capacity: 89 tests;
- locked learner/checkpoint/provenance/production integration: 285 tests;
- complete Python repository: **1,820 tests in 416.74 seconds**;
- strict scoped MyPy, Black, Python compilation, Git whitespace, and all four frozen-document
  hashes; and
- frontend TypeScript/Vite production build and `oxlint`.

The complete repository emitted only the existing Starlette/httpx deprecation warning and known
HiGHS diagnostics. The frontend retained its existing greater-than-500-kB 3D bundle advisory.

## Execution and report identity

The one execution command was:

```text
make v2-beta-rl-capacity \
  OUTPUT=/private/tmp/plantanagent-beta-rl-capacity-27a47d0 \
  ASSERT_EXCLUSIVE=yes ASSERT_FIRST_ATTEMPT=yes
```

The report was immediately reopened by the runner and then verified again through the detached
production verifier on the same host. Both returned the same authenticated failed state.

| Identity | SHA-256 |
|---|---|
| raw `report.json` file | `b7b4a9e3be62f62c7f82d55584ec5d29a7df57e8566051ace62c7f4699b31abc` |
| report-domain digest | `e34facf8760e25c74adc1aff4cb4e35e7327e36bed0f23f341e28982f9a0e1f6` |
| deterministic pre-run evidence | `ef1344f59687d92f537985cbf61251f6a743e178f9ddd768a8965a7fc00a3c91` |
| runtime provenance | `085ac94fc9b6caf259f7d2c644844d96a25d6d1f4d2936f3d5f88bba1834211e` |

The bound host was an Apple M4 Pro with 12 physical/logical cores, 24 GiB physical memory, eight
Torch intra-op threads, twelve Torch inter-op threads, and no configured bound thread environment
variable. This identifies the failed attempt but does not authorize extrapolation from it.

## Retained cell evidence

| Order | Family | Wall seconds | Child user CPU seconds | Child system CPU seconds | Exit | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | PI-Lagrangian | 129.781126917 | 126.202643 | 2.733033 | 0 | `checkpoint_admission_error` |
| 2 | integral-only | 129.247395542 | 125.733830 | 2.758390 | 0 | `checkpoint_admission_error` |
| 3 | scalar PPO | 128.619359792 | 124.971711 | 2.534196 | 0 | `checkpoint_admission_error` |

These timings are retained failure diagnostics only. The protocol requires an admitted checkpoint
before constructing a measurement, so they cannot be projected to the registered campaign.

All stderr files are empty and have SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The three stdout files
remain unopened. Their report-bound identities are:

| Family | Bytes | stdout SHA-256 |
|---|---:|---|
| PI-Lagrangian | 1,225 | `69b5939e21411d78b23ef287bdacca133763779c889fbacd52ca9f5876065884` |
| integral-only | 1,224 | `10d4c47b8dae204a06e306435ec54da7a30870175d9184ad730f1b9b104a8c0f` |
| scalar PPO | 1,222 | `676abe8ad1c824ac109e5f22b9145d1489751019117d57ada50524e655fe4a87` |

The failed report does not admit or bind the checkpoint contents. For preservation, the existing
unadmitted diagnostic files have these independently calculated identities:

| Family | File | Bytes | SHA-256 |
|---|---|---:|---|
| PI-Lagrangian | `policy.zip` | 1,089,564 | `04a2d5e225d4080cf5b8d6fc7b03251b928bc411dde43591cd5abb6c8b38eca3` |
| PI-Lagrangian | `metadata.json` | 150,339 | `97e51eda04b8731cecb7891f577f429e4f3aff0614a7a4328ce413f924c90c51` |
| PI-Lagrangian | `training_config.json` | 5,923 | `f347ec50b328b5d866790c6d3751f454c31213889b57890304665408cf1f663c` |
| integral-only | `policy.zip` | 1,089,553 | `3acc6b1a588a1f52983d2110d2f72dd3a0ba7bec5043bc4c5a93e0c636e3502c` |
| integral-only | `metadata.json` | 150,331 | `0f5d7cefad919c081f07bc462d0e096361747419ff3b9f3dca59913e47d9d536` |
| integral-only | `training_config.json` | 5,921 | `8105172b7fe4cba51f932d7d75ac3e57eda0bebbce1334430db414d335ff79a0` |
| scalar PPO | `policy.zip` | 437,865 | `2f6e11892d5b96ad589c185354e40b4b7f0045665c2977b181f8d75742dc902f` |
| scalar PPO | `metadata.json` | 134,795 | `253d6f58bec90b54f19ff1dd10ef813ba5e2837e7a97d97a49edc1df05091d7c` |
| scalar PPO | `training_config.json` | 5,917 | `da2e409a078854f6444c3888d05c4beefeaa3097e3995868fe07b0e2f73c1a88` |

## Diagnosis without controller-outcome inspection

The canonical failure category intentionally suppresses arbitrary exception text. After retaining
and verifying the report, each existing checkpoint was passed once through the unchanged read-only
`_admit_checkpoint` function. All three returned exactly:

```text
CapacityBenchmarkError: checkpoint training config differs from capacity cell
```

The exact checkpoint `training_config.json` byte check immediately before that comparison had
already passed. A second read-only comparison printed only unequal static training-configuration
fields, not metadata evidence or controller outputs. In every family the only unequal top-level
fields were `episode` and `beta_task`. Their values were semantically identical: tuple-valued
in-memory sequences such as bounds, duration ranges, forecast windows, and efficiency anchors had
become ordinary lists after JSON serialization.

The failing check compared a JSON-loaded metadata mapping directly with `config.to_dict()`. JSON
has one array type and Python distinguishes lists from tuples, so raw object equality rejected a
canonical round trip even though both objects produce the same canonical configuration bytes. This
is a family-neutral admission representation defect after successful child serialization.

## Decision

Revision `27a47d0` and this external root are closed. No same-revision rerun, partial reuse,
projection, or controller inference is permitted. `CONSTRAINED_RL_CAPACITY_REPAIR_002.md` freezes a
narrow whole-object canonical-JSON comparison before any source change. It preserves the exact
standalone config-byte check and every later checkpoint/evidence check, versions the next report,
and permits at most one new clean-revision capacity attempt after complete release gates.
