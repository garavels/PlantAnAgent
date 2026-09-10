# Registered training custody build

Status: **custody implementation validated at `ad6ad1e`; study preparation in progress; registered training closed**

The unchanged custody protocol and inherited status edits were committed as `8c3d535`
before this implementation. Its raw SHA-256 is
`af73687eefff32132409a8afcfc6d34528da68bf94c4788e697fdc62046a9801`.
The three families, five optimizer seeds, plant task, learning budgets and seed domains remain frozen.

## Admission audit

| Requirement | Implementation status | Remaining admission evidence |
| --- | --- | --- |
| Exact registered task and canonical configs | Implemented; complete immutable template and all-15 config bytes rechecked before authenticated child launch | Passed clean gate at `ad6ad1e` |
| Exact 15-cell matrix and paths | Implemented; durable prepublication and independent verifier reconstruction | Passed clean gate at `ad6ad1e` |
| Tracked authorization, scope, host and commit lineage | Read-only admission implemented; synthetic scope and Git fixtures | Final approved implementation, explicit user authorization and subsequent clean release gate |
| Durable claims, non-retry and sequential chain | Implemented; campaign state machine, single-use parent pipe, interruption and publication-failure fixtures | Passed clean gate at `ad6ad1e` |
| Sealed bounded subprocess logs and process groups | Implemented; tiny process fixtures and complete synthetic matrix continuation/abort tests | Passed clean gate at `ad6ad1e` |
| Artifact inventory and checkpoint admission | Implemented; production loader, retained partial/forbidden artifacts, substitution and post-load mutation checks | Passed clean gate at `ad6ad1e` |
| Closed failures and complete accounting | Implemented; all 15 cells, exact failure precedence, stopped/unstarted accounting | Passed clean gate at `ad6ad1e` |
| Production/synthetic separation | Implemented; separate report paths and schema rejection before dependency access | Passed clean gate at `ad6ad1e` |
| Detached verification and post-publication gate | Implemented; independent reconstruction and mandatory fresh-process invocation, including recapture failure | Passed clean gate at `ad6ad1e` |
| Protected seed isolation and full release gate | No protected reset or optimizer step used during this build | 2,285 Python tests and complete static/frontend gate passed at `ad6ad1e` |
| Development analysis and campaign schema binding | Numerical statistics, selection and sensitivity kernels implemented; 82 reset-free tests pass | Verified pairwise input binding, scorecards/latency, stage custody and detached development/held-out verifiers |

The generic registered `train()` entry point now requires a one-use permission delivered by
the live campaign parent. It reauthenticates the tracked authorization, complete config
registry, plan and open sequential claim before environment/model construction. No such
production authorization exists. Config validation, static plan inspection and read-only
forensic classification remain available; none grants permission to consume study seeds.

## Representation choices within the frozen protocol

The protocol fixes top-level evidence fields and several nested shapes. Remaining
representations are made explicit here before any registered outcome exists:

- Plan/config/claim/receipt/checkpoint/log paths are absolute canonical paths. Artifact
  inventory entries alone use normalized root-relative POSIX paths.
- A referenced authenticated JSON envelope uses exactly `path`, `raw_file_sha256`,
  and `sha256`. The first digest authenticates published bytes; the second is its
  schema-domain digest. File descriptors use the protocol's four specified fields.
- The execution contract contains `cell_count`, `sequential`, `timeout_seconds`,
  `log_limit_bytes_per_stream`, and `minimum_free_bytes` with the protocol's exact values.
- `approved_implementation` contains `commit`, `tree`, and `uv_lock_sha256`.
- `host_contract` contains the complete strict `host` mapping and the boolean
  `capacity_projection_applies`. False explicitly relinquishes the capacity projection;
  it does not waive exact binding to the newly authorized host.
- `capacity_report` contains `path`, `raw_file_sha256`, and `sha256`. The actual report
  must match the two frozen capacity digests before admission.
- `protocols` binds `constrained_rl`, `custody`, and the existing complete `capacity`
  protocol/repair/clarification chain with exact raw document hashes.
- Authorization identifiers use canonical UUIDs. Evidence timestamps use canonical
  UTC with six fractional digits and `Z`. Neither affects continuation or selection.
- The exact non-retry statement is: “Failed, timed-out, interrupted, or rejected cells
  cannot be retried or replaced at the same source revision.”
- A forbidden artifact root itself is represented by one forbidden `.` entry. Missing
  roots retain the protocol's empty, absent-root inventory. No partial bytes are repaired.
- The live parent sends an anonymous-pipe envelope with exactly `schema`, `parent_pid`,
  `claim_sha256` and `config_sha256`. Its schema is
  `plant-beta-constrained-rl-registered-training-child-launch-v1`. The child consumes and
  closes the inherited descriptor before any protected reset; the environment key
  `PLANT_BETA_REGISTERED_CUSTODY_FD` is removed. This transport adds no config override,
  mutable sidecar or thread-setting change. It prevents accidental standalone/repeated
  launches; it is not a security boundary against malicious code running as the same user.
- Process wall time covers launch through process-group closure and buffered-byte capture,
  ending before log fsync. A recorded interval at or above the frozen 36,000-second limit
  is a timeout. This makes the timeout classification independently reconstructible.
- Pre-claim storage exhaustion uses campaign abort code `storage_budget_unavailable`,
  stage `pre_child` and fixed message “The remaining campaign storage budget is unavailable.”
  It leaves every remaining claim absent and never becomes a child failure.
- Read-only `inspect` uses schema
  `plant-beta-constrained-rl-registered-training-forensic-inspection-v1`. It reports
  receipt syntax and declared states, with `verified: false`, `result_claim: null` and
  `retry_permitted: false`. An interrupted claim cannot be repaired or resumed by inspection.

No authorization file, production campaign root or registered checkpoint has been created.
The previously measured 62.4889–93.7075-hour sequential planning scenarios and 5-GiB
campaign admission floor remain visible planning constraints, not execution approval.

## Work still required before requesting the study

Close the development-analysis build required by the post-capacity audit, tracked in
[`CONSTRAINED_RL_STUDY_ANALYSIS_BUILD.md`](CONSTRAINED_RL_STUDY_ANALYSIS_BUILD.md). Its numerical
kernels are implemented; they confer no verified arm eligibility or stage authorization. The campaign runner
and detached verifier passed their complete clean release gate at
`ad6ad1e85a0269f592d42b7c9edcd6c64f2c0579`: 2,285 Python tests, strict typing of all 17 changed
source modules, Black on all 29 changed Python files, compilation, CLI help/import,
Git whitespace and lock cleanliness, and frontend production build/lint. This does not admit
any later implementation revision or authorize training.

The current registered policy loader requires the exact training source/runtime identity.
Consequently selection, sensitivity, uncertainty, latency registration and independent
development/held-out report verification must be in the frozen implementation before training;
post-training source edits cannot silently become the evaluation runtime. The later stage
authorization/evidence workflow must preserve this identity and keep stage permissions separate.

The comparator fallback
descriptor verifier gap is now closed with exact-field and declaration checks, eight new
reset-free adversarial tests, and a passing 43-test verifier regression run. The accepted E-beta stop diagnosis,
M24 product connection and explicit operational handover remain separate product work; none
constitutes evidence of PPO value. Preserve the existing UI identity and operational copy.

Only a complete clean release gate and a concrete root/host/budget authorization packet can
precede the user's later instruction to execute the 15 registered runs. Training, development,
held-out evaluation and live policy deployment are separate admissions.
