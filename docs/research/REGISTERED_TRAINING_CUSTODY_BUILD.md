# Registered training custody build

Status: **implementation in progress; registered training closed**

The unchanged custody protocol and inherited status edits were committed as `8c3d535`
before this implementation. Its raw SHA-256 is
`af73687eefff32132409a8afcfc6d34528da68bf94c4788e697fdc62046a9801`.
The three families, five optimizer seeds, plant task, learning budgets and seed domains remain frozen.

## Admission audit

| Requirement | Implementation status | Remaining admission evidence |
| --- | --- | --- |
| Exact registered task and canonical configs | Implemented; mutations across all immutable config leaves rejected | Integration with authenticated child launch |
| Exact 15-cell matrix and paths | Implemented; deterministic family-first order and config hashes | Durable prepublication and verifier reconstruction |
| Tracked authorization, scope, host and commit lineage | Read-only admission implemented; synthetic scope and Git fixtures | Final approved implementation, explicit user authorization and subsequent clean release gate |
| Durable claims, non-retry and sequential chain | Exclusive file/directory barriers and inert claim schemas implemented | Campaign state machine and crash-window integration tests |
| Sealed bounded subprocess logs and process groups | Implemented; tiny process fixtures cover both stream caps, timeout, signals and surviving descendants | End-to-end frozen matrix continuation/abort tests |
| Artifact inventory and checkpoint admission | Non-following inventory and fixed production-loader adapter implemented | Completed-cell integration, substitution and admission tests |
| Closed failures and complete accounting | All cell failure codes and precedence implemented | Terminal/report construction and all-15 accounting |
| Production/synthetic separation | Distinct report schemas declared | Separate execution/report paths and rejection before dependency access |
| Detached verification and post-publication gate | Pending | Independent reconstruction, tampering tests and mandatory fresh-process invocation |
| Protected seed isolation and full release gate | No protected reset or optimizer step used during this build | Full repository checks on a clean completed implementation |
| Development analysis and campaign schema binding | Pending | All-32 pairwise eligibility, selection, sensitivity, uncertainty, latency and detached verifiers |

The generic registered `train()` entry point deliberately refuses execution until the
campaign launch gate is complete. Config validation and static plan inspection remain
available. This is an interim closed boundary, not final training admission.

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

No authorization file, production campaign root or registered checkpoint has been created.
The previously measured 62.4889–93.7075-hour sequential planning scenarios and 5-GiB
campaign admission floor remain visible planning constraints, not execution approval.

## Work still required before requesting the study

Complete and adversarially test the campaign runner and detached verifier, then close the
development-analysis build required by the post-capacity audit. The comparator fallback
descriptor verifier gap is now closed with exact-field and declaration checks, eight new
reset-free adversarial tests, and a passing 43-test verifier regression run. The accepted E-beta stop diagnosis,
M24 product connection and explicit operational handover remain separate product work; none
constitutes evidence of PPO value. Preserve the existing UI identity and operational copy.

Only a complete clean release gate and a concrete root/host/budget authorization packet can
precede the user's later instruction to execute the 15 registered runs. Training, development,
held-out evaluation and live policy deployment are separate admissions.
