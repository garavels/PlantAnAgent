# Constrained-RL capacity repair 002 — canonical metadata configuration comparison

Status: **frozen after the failed `27a47d0` attempt and before source repair or re-execution**

Date: **2026-09-05**

Repair identifier: **`plant-beta-constrained-rl-capacity-repair-002`**

## Preserved failure

The sole Repair-001 capacity attempt at clean revision
`27a47d0604ffb9624a40683a610f2f8f4bb6c58a` is permanently retained as a failed engineering
result. Its report-domain SHA-256 is
`e34facf8760e25c74adc1aff4cb4e35e7327e36bed0f23f341e28982f9a0e1f6`, and its raw report-file
SHA-256 is `b7b4a9e3be62f62c7f82d55584ec5d29a7df57e8566051ace62c7f4699b31abc`.
All three children exited zero and published their three expected checkpoint files, but the parent
classified every cell as `checkpoint_admission_error`. The canonical report contains no admitted
checkpoint, measurement, campaign projection, or result claim. The detailed record is
`results/v2_constrained_rl_capacity_27a47d0.md`.

No cell at `27a47d0` may be retried. Its observed timing and unadmitted checkpoint artifacts cannot
enter a capacity projection. None of its three stdout logs was opened. Read-only diagnosis opened
only static training-configuration metadata through the existing loader and did not inspect reward,
cost, action, multiplier, loss, policy output, market, outage, inventory, or ranking values.

## Root cause

Checkpoint admission already proves that the independently published `training_config.json` bytes
equal the exact canonical bytes expected for the capacity cell. It then performs a second check:

```text
metadata.training_config == config.to_dict()
```

`metadata.training_config` has crossed a JSON serialization boundary, so every JSON array is a
Python list. `config.to_dict()` preserves tuple-valued dataclass fields as Python tuples. The
objects therefore compare unequal even though both serialize to identical canonical JSON. The only
observed differences were representation-only list/tuple pairs inside the static `episode` and
`beta_task` configurations, consistently across all three algorithms.

This is a checkpoint-admission representation defect after successful training and serialization.
It is not an environment, controller, objective, optimization, seed, timing, or artifact-corruption
failure.

## Sole permitted source repair

The repair may change only the metadata copy of the training-configuration equality check:

1. Retain the existing exact regular-file and byte equality check for checkpoint
   `training_config.json`.
2. Canonically serialize the complete `metadata.training_config` mapping with the same canonical
   JSON function, UTF-8 encoding, and trailing newline used by `_training_config_bytes(config)`.
3. Require those canonical bytes to equal `_training_config_bytes(config)` exactly.
4. Apply normalization to the complete mapping; do not introduce field-specific tuple/list
   exceptions, lossy coercion, subset comparison, tolerance, or ignored keys.
5. Preserve every later metadata, evidence, runtime, environment, artifact, allocator, rollout,
   optimizer-step, and policy-load admission check unchanged.

The following remain byte-for-byte or semantically unchanged:

- every learner, rollout buffer, reward/cost calculation, GAE value, multiplier, gradient, and
  optimizer update;
- all PPO settings, algorithms, seeds, episode allocation, task/environment contracts, and
  registered campaign arithmetic;
- checkpoint serialization, `metadata.json`, `training_config.json`, and `policy.zip` production;
- report timing, measurement, storage, projection, failure, deterministic-identity, host, process,
  and operator-custody rules; and
- the failed V1 and V2 reports and all prior external artifacts.

## Report identity

The next capacity production report schema/domain must be
`plant-beta-constrained-rl-capacity-report-v3`. Synthetic reports must use a separately versioned
synthetic V2 schema/domain. Both protocol identities must append this repair identifier, path, and
exact raw file SHA-256 after Repair 001 and its two clarifications. A V1 or V2 report cannot be
relabeled or verified as V3.

## Required implementation evidence

Before another capacity attempt:

1. Prove that tuple-valued in-memory configuration and its canonical JSON list representation have
   identical canonical bytes while remaining unequal as raw Python objects.
2. Prove that the production metadata comparison accepts that representation-only difference.
3. Prove that changed scalar values, changed nested sequence values, missing keys, extra keys,
   wrong sequence order, and non-finite values remain rejected.
4. Retain exact rejection when checkpoint `training_config.json` bytes differ, even if a metadata
   mapping is internally self-consistent.
5. Retain the complete Repair-001 learner, capacity, checkpoint, provenance, production-integration,
   report, static, full-repository, and frontend gates.
6. Prove the V3/V2 schema and nested repair-chain hashes are exact and cross-version verification
   fails before runtime, host, or checkpoint access.

No real 8,192-transition rollout is permitted as a development test. The retained `27a47d0`
checkpoints may be used only for read-only root-cause confirmation; they cannot be admitted into a
capacity report or used to derive timing/storage projection evidence.

## Re-execution custody

After this repair and every required gate are committed, exactly one new three-cell capacity
attempt may run from that new clean revision into a new absolute external output root. It must keep
the original algorithm order, optimizer seed 101, fixture base 9,880,000, transition count,
environment/task, timeout, process isolation, and operator assertions unchanged.

If any cell fails, the new report is retained without projection or retry and another separately
frozen repair protocol is required. If all cells pass, only the complete V3 report may provide the
host-specific campaign projection. It remains fixture-only infrastructure evidence and cannot
support controller-performance, safety, state-of-the-art, or commercial claims.
