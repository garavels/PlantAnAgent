# Constrained-RL capacity repair clarification 001 — evidence issuance boundary

Status: **frozen after independent audit and before capacity-boundary code changes or repaired execution**

Date: **2026-09-05**

Clarification identifier: **`plant-beta-constrained-rl-capacity-repair-clarification-001`**

## Scope

Repair 001 remains the sole authority for changing rollout-cost aggregation. This clarification
does not authorize a learner, environment, seed, optimizer, measurement, projection, or campaign
change. It closes three report-integrity defects found while the Repair-001 implementation was
still uncommitted and before any V2 capacity report existed:

1. an injectable test executor could publish and self-verify the ordinary production report;
2. the purportedly timing-free digest included timing transitively through logs and checkpoints;
   and
3. failure evidence retained arbitrary Python exception class names.

The failed V1 report at revision `fab4a55` remains immutable and ineligible for projection. No V2
capacity cell has run. The repaired production report may therefore retain schema V2 provided it
binds this clarification and satisfies every rule below.

## Production and synthetic issuance

Only the non-injectable production entry point may build, publish, and immediately verify
`report.json` under the production V2 schema and domain. That path must bind the concrete
subprocess runner, checkpoint admission, clean-runtime capture, host capture, and monotonic clock;
none may be caller supplied.

Any dependency-injected execution seam is test-only. It may collect a synthetic trace, but it may
not call the production report builder, production publisher, or production verifier. If an
end-to-end synthetic fixture is serialized, it must use all of the following:

- a distinct filename, schema, and full-report domain;
- an explicit `synthetic-test-only` execution class;
- a null result claim and null campaign projection; and
- a synthetic-only verifier that refuses the production schema.

The production verifier must reject a synthetic schema before runtime capture, host capture, or
checkpoint admission. No caller-controlled Boolean, token, function identity comparison, underscore
naming convention, or echoed assertion may upgrade synthetic evidence to production evidence.
Pure parsing, arithmetic, and structural-verification helpers may be shared.

## Deterministic evidence identity

The field named `deterministic_evidence_sha256` is a digest of the deterministic **pre-run
measurement contract**, not a second digest of post-run artifacts. Its exact allowlist is:

1. the bound capacity protocol, Repair 001, and this clarification;
2. the capacity claim boundary;
3. the clean runtime-provenance digest;
4. host capacity identity with the observed clock description removed;
5. the three frozen operator assertions;
6. the normalized cell contract, including its normalized training-config template and static
   timeout/measurement/projection formulas;
7. the exact planned cell order, algorithm, optimizer seed, fixture seed base, and normalized
   config-template digest; and
8. the fixed registered-campaign arithmetic.

It must exclude every post-run field, including cell status, exit/failure evidence, observed
timestamps and CPU time, measurements, projections, stdout/stderr records, absolute-output config
digests, checkpoint presence, metadata/evidence/artifact/file hashes, ZIP or logical/allocated
sizes, and maximum serialized-entry sizes. These exclusions cover transitive timing from SB3
elapsed/FPS logs, metadata creation time, rollout `elapsed_compute_s`, ZIP timestamps, timeout
classification, and float-text width. The production report's full content digest continues to
bind every excluded field, file, measurement, projection, and failure.

Changing any bound protocol, runtime, non-clock host-capacity field, assertion, normalized task
contract, planned cell identity, seed, or campaign arithmetic must change the deterministic digest.
Changing only an output root or direct/transitive timing evidence must not.

## Closed failure vocabulary

The canonical report may retain only these fixed failure categories:

- `training_timeout`;
- `training_nonzero_exit`;
- `training_launch_error`; and
- `checkpoint_admission_error`.

Each category has one fixed phase and one fixed sanitized message. Actual exception class names,
exception messages, commands, paths, and child output text may not enter the canonical failure
record. Child stdout/stderr bytes remain available only as full-report-bound, hash-identified
external files.

## Static task identity versus observed outcomes

Static reward, cost, tariff, market, and economic parameters that are already part of the frozen
normalized training-config template are permitted because they define the task being measured.
Observed rewards, costs, losses, actions, multipliers, forecasts, market realizations, outages,
inventories, policy outputs, rankings, and controller conclusions remain forbidden. Code and
documentation must say **observed** when describing this exclusion so the immutable task contract
is not mischaracterized as an outcome leak.

## Required evidence before repaired execution

Before the one permitted repaired attempt:

1. prove injected hooks create no production `report.json`, production claim, or production-schema
   artifact;
2. prove cross-schema production verification fails before runtime, host, or checkpoint access;
3. freeze and test the exact deterministic allowlist;
4. mutate every direct and transitive timing class and prove the deterministic digest is unchanged
   while the full report digest changes;
5. mutate protocol/runtime/non-clock host/normalized contract/order/seed identity and prove the
   deterministic digest changes;
6. prove two different external roots have the same deterministic identity;
7. prove arbitrary exception class names and messages cannot enter a canonical failure record;
8. retain strict recomputation of every full-report measurement and projection; and
9. pass the complete Repair-001 learner, capacity, checkpoint, provenance, repository, static, and
   frontend release gates from one clean committed revision.

Only after those conditions are committed may exactly one new three-cell capacity attempt run.
