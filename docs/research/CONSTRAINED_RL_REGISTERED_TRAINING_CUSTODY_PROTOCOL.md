# Constrained-RL registered training custody protocol

Status: **frozen after capacity validation and before custody implementation, authorization, or any registered reset**

Date: **2026-09-05**

Protocol identifier: **`plant-beta-constrained-rl-registered-training-custody-v1`**

## Decision and authorization boundary

The authenticated V3 capacity report at source revision `4f505b8` closes the infrastructure
measurement required before a registered-training request. It does not authorize training. Static
readiness review at `2894175` found that the single-run learner and checkpoint path are implemented,
but a production campaign must not start through the generic `v2-train` command because three
outcome-custody requirements are absent:

1. registered validation does not yet require every plant, task, economic, and disturbance value to
   equal the frozen beta task;
2. no exact 15-run registry or durable attempt ledger enforces family/seed order and same-revision
   non-retry; and
3. no campaign report or detached verifier accounts for every configuration, process attempt, log,
   and admitted final checkpoint.

This document freezes the narrow build that closes those gaps. It authorizes fixture-only unit and
integration tests. It does **not** authorize a registered optimizer update, training-environment
reset, development reset, held-out reset, checkpoint selection, controller evaluation, or log/
outcome inspection. Production execution additionally requires an explicit user authorization
record committed after this implementation and its complete release gate.

## Bound upstream evidence

The custody implementation must bind:

- constrained-RL protocol `plant-beta-constrained-rl-protocol-v1`;
- capacity protocol `plant-beta-constrained-rl-capacity-v1` and both repairs;
- completed V3 report raw SHA-256
  `3b9074824c6e81733e18653fd87e0a470e51c478183ed1a0a656f3589c4851fa`;
- V3 report-domain SHA-256
  `78b6c1f067d88aa43b672e42dd94c41d37ff9444e3963a2ebcb47f85933d9c3e`;
- recorded capacity host: Apple M4 Pro, 12 physical/logical cores, 24 GiB memory, Torch 8 intra-op
  and 12 inter-op threads, with all five declared thread environment values absent; and
- sequential duration scenarios 62.4889, 78.0982, and 93.7075 hours plus the 1,256,560,044-byte
  checkpoint-growth recommendation; and
- the registered endpoint arithmetic of 128 rollouts, 40,960 minibatch optimizer steps, 1,564
  allocator entries, and 1,048,576 controller transitions per policy.

The projection remains host-specific planning evidence. A different host or thread configuration
requires an explicitly accepted loss of that projection or a separately frozen capacity protocol;
it cannot be silently called equivalent.

## Exact registered task template

For each registered run, canonical configuration is constructed from the current V1 types with:

```text
environment_id             plant-env-v2-cyclic-market-beta-v1
plant                      PlantConfig()
episode                    EpisodeConfigV2()
constraints                Constraints()
diagnostic_reward          RewardWeightsV2()
economic_assumptions       EconomicAssumptionsV1()
beta_task                  CyclicMarketTaskConfigV1()
total_timesteps            1,048,576
n_envs                     4
checkpoint_every           0
resume_from                null
ppo                        PPOConfig()
beta_rl.protocol_id        plant-beta-constrained-rl-protocol-v1
beta_rl.run_kind           registered
beta_rl.episode_seed_base  1,000,000
notes                      empty string
```

Only three fields vary: optimizer `seed`, `beta_rl.algorithm`, and the deterministic absolute
`output_dir`. The template digest is reproduced exactly as follows:

1. call `to_dict()` on the complete default registered configuration above;
2. remove the top-level `seed` and `output_dir` members and remove only `algorithm` from the nested
   `beta_rl` object;
3. serialize the remaining object with Python `json.dumps(..., allow_nan=False,
   ensure_ascii=True, separators=(",", ":"), sort_keys=True)` and no trailing newline; and
4. calculate SHA-256 over the ASCII bytes of
   `plant-beta-constrained-rl-registered-task-template-v1`, one zero byte, and the UTF-8 canonical
   payload bytes, in that order.

The frozen digest is:

```text
89abbe6288d512c254d8186f56c6d269b9f273fb49bbb13d24d7b02425bfa77b
```

Registered configuration validation must reconstruct and require that digest. Type validity or a
672-decision horizon alone is insufficient. Changing any plant rating, episode distribution,
delivery request, weather model, constraint, diagnostic reward, tariff/economic coefficient,
cycle rule, market setting, disturbance setting, PPO setting, protocol identity, run kind, seed
base, budget, environment count, checkpoint rule, resume state, or notes must fail before model or
environment construction.

Every configuration file is decoded first with an `object_pairs_hook` that rejects a duplicate at
any nesting depth and a `parse_constant` callback that rejects `NaN`, positive infinity, and
negative infinity. This strict parse occurs before examining `beta_rl`, so conflicting duplicate
`run_kind` or `beta_rl` keys cannot choose their own validation boundary. Historical alpha and
fixture configuration semantics otherwise remain compatible.

A file opened from the registered campaign's predeclared `configs/` paths is classified as
registered by its path and plan before any content is trusted. It must be the exact bytes produced
by `json.dumps(complete_mapping, allow_nan=False, ensure_ascii=True, indent=2,
separators=(",", ": "), sort_keys=True) + "\n"`, encoded as UTF-8. Duplicate keys, non-finite
values, a missing or different registered declaration, noncanonical bytes, or a semantically
equivalent alternative encoding fails before model or environment construction.

## Exact 15-run matrix and path contract

Family order and optimizer seeds are fixed:

```text
plant-beta-ppo-pi-lagrangian-v1: 101, 211, 307, 401, 503
plant-beta-ppo-i-lagrangian-v1:  101, 211, 307, 401, 503
plant-beta-sb3-ppo-scalar-v1:    101, 211, 307, 401, 503
```

The order is the Cartesian product above, family first and seed ascending, numbered 01 through 15.
For a user-authorized absolute campaign root `ROOT`, the deterministic layout is:

```text
ROOT/authorization.json
ROOT/plan.json
ROOT/configs/{order:02d}-{family_slug}-seed-{seed}.json
ROOT/logs/{order:02d}-{family_slug}-seed-{seed}.stdout.log
ROOT/logs/{order:02d}-{family_slug}-seed-{seed}.stderr.log
ROOT/attempts/{order:02d}-{family_slug}-seed-{seed}/claimed.json
ROOT/attempts/{order:02d}-{family_slug}-seed-{seed}/completed.json OR failed.json
ROOT/artifacts/{order:02d}-{family_slug}-seed-{seed}/checkpoint/{policy.zip,metadata.json,training_config.json}
ROOT/report.json
```

Family slugs are exactly `pi-lagrangian`, `i-lagrangian`, and `scalar-ppo`. No destination override,
alternate order, omitted cell, duplicate cell, additional seed, replacement seed, resume path, or
intermediate checkpoint is accepted.

## User authorization record

The future production runner requires one canonical JSON authorization file at the exact tracked
path `docs/research/authorizations/v2_constrained_rl_registered_training_v1.json` in the clean
source revision. `ROOT/authorization.json` is an exact durable byte copy, not the authority itself.
The authorization's exact top-level keys are `schema`, `authorization_id`, `authorized_at`,
`protocols`, `capacity_report`, `approved_implementation`, `campaign_root`, `host_contract`,
`campaign_contract`, `registered_training_authorized`, `development_evaluation_authorized`,
`held_out_evaluation_authorized`, `non_retry_statement`, and `sha256`; unknown or missing keys fail.
Its schema and domain are both
`plant-beta-constrained-rl-registered-training-authorization-v1`. Its domain digest uses the same
ASCII-domain, zero-byte, canonical-JSON construction as the task-template digest after removing
only `sha256`. The canonical authorization must bind:

- `plant-beta-constrained-rl-registered-training-authorization-v1`;
- a unique authorization identifier and date;
- the constrained-RL protocol and this custody protocol identifiers plus exact raw file hashes;
- the successful capacity report raw and report-domain hashes;
- the exact approved custody-implementation commit, its Git tree identity, and `uv.lock` SHA-256;
- the exact absolute new campaign root;
- the exact host/thread contract or an explicit declaration that the capacity projection does not
  apply;
- the exact 15-cell scope, sequential execution, 36,000-second per-cell timeout, 67,108,864-byte
  limit for each stdout or stderr log, and 5,368,709,120-byte whole-campaign minimum free space;
- `registered_training_authorized: true`;
- `development_evaluation_authorized: false` and `held_out_evaluation_authorized: false`; and
- an exact statement that failed, timed-out, interrupted, or rejected cells cannot be retried or
  replaced at the same source revision.

The authorization file is not created by this build protocol. It may be added only after an
explicit user instruction made with the measured duration and storage budget visible. Its commit
must have the approved implementation revision as its single parent. Relative to that parent, the
authorization commit may change only the canonical authorization file and the enumerated README,
roadmap, and journal status paths: `README.md`, `docs/research/README.md`,
`docs/research/ROADMAP.md`, and `docs/research/JOURNAL.md`. It may not change Python, simulator,
frontend, dependency,
Makefile, protocol, test, or configuration source. The production runner verifies the parent,
approved tree, lock hash, and exact changed-path allowlist before output creation. Runtime
provenance then captures the final authorization commit without relying on a self-referential Git
hash inside that commit.

## Durable attempt custody

The production runner must require a new absolute non-symlink campaign root outside the repository,
an existing writable parent, at least **5,368,709,120 bytes (5 GiB)** free, an exact canonical
tracked authorization file, an empty Git status, and explicit `--assert-exclusive`,
`--assert-first-attempt`, and `--assert-authorized` flags before creating the root.

The 5-GiB floor is a conservative whole-campaign admission rule, not measured usage. It covers the
1,256,560,044-byte capacity-derived checkpoint-growth recommendation, at most 2,013,265,920 bytes
for 30 bounded 64-MiB log files, 536,870,912 bytes for configs, plans, receipts, reports, and
inventories, 1,073,741,824 bytes for one active cell's temporary serialization, and 488,270,420
bytes of unallocated contingency. The runner rechecks that the unused portion of this declared
budget remains available before each claim: `statvfs.f_bavail * statvfs.f_frsize` for the campaign
filesystem plus the recursively measured allocated bytes already retained below `ROOT` must remain
at least 5,368,709,120. If it does not, the runner aborts the remaining unclaimed cells. This rule
does not reserve space for development or held-out evidence.

It exclusively publishes `authorization.json`, `plan.json`, and all 15 exact config files before the
first child. Every durable publication uses exclusive file creation, flushes the complete bytes,
calls `fsync` on the file, closes it, and `fsync`s the containing directory before the next state
transition. Before launching a child it publishes that cell's `claimed.json` through this complete
file-plus-directory persistence barrier. The child process cannot start before the barrier returns
successfully. The claim conservatively makes that cell non-retriable at the current source revision
even if the child fails before its first gradient. This is stricter than exercising the base
protocol's pre-gradient repair allowance and closes the otherwise unobservable crash window.

After the process returns, the parent publishes exactly one terminal receipt through the same
persistence barrier: `completed.json` after successful checkpoint admission or `failed.json`
otherwise. A claimed cell without a terminal receipt proves an interrupted attempt and cannot be
resumed or relaunched. The runner never overwrites a receipt, config, log, artifact, or report.
Restarting against an existing root is refused.

Cell-local training timeout, nonzero exit, or checkpoint-admission failure is retained and the
predeclared matrix continues without operator choice. A source/runtime/authorization/plan mutation,
loss of root custody, or inability to publish durable evidence aborts all remaining cells as not
started. These rules are fixed before outcomes and cannot be changed after inspecting a log.

## Process and log boundary

Each cell executes in a fresh child Python process, sequentially, with a fixed 36,000-second timeout.
The only child command is the controller-neutral training module with that cell's exact canonical
config; CLI seed, timestep, environment, resume, or destination overrides are prohibited.

Standard output and error are captured to the predeclared files without forwarding their contents
to the operator. Receipts and the final report record only paths, byte counts, and SHA-256 digests.
No stdout/stderr content, rollout return, constraint cost, multiplier, loss, action, policy output,
market, outage, inventory, service, safety, economic value, or ranking is displayed or used for a
continuation decision. Logs stay sealed until all 15 attempts and the campaign report close; later
diagnosis requires a separately journalled reason and can never authorize retry or selection.

Each stream is capped at exactly 67,108,864 bytes by the parent while being copied without decoding
or displaying it. Reaching the cap terminates the child, retains the exact capped bytes, and records
`child_log_limit_exceeded`; it never truncates a successful cell silently.

The child starts in a new process group. On timeout, log limit, or parent-handled interruption, the
parent terminates and then force-kills that complete group if necessary, waits until every child is
reaped, closes and fsyncs both log files and their directory, and only then captures process and
artifact evidence. No inventory is taken while a descendant can still mutate the output tree.

Before and after every child the parent recaptures clean runtime provenance and the authorized host
identity. Any difference is a campaign-integrity failure. Child wall time and CPU time are retained
as operational evidence, never as an algorithm-performance comparison.

## Checkpoint admission

A zero exit is insufficient. The parent must reopen the exact expected checkpoint using the
production registered-policy loader without resetting an evaluation environment. Admission must
prove:

- the exact canonical cell configuration and task-template digest;
- family identifier, optimizer seed, registered run kind, and no resume parent;
- exactly 1,048,576 collected transitions, 128 rollouts, and **40,960 minibatch optimizer steps**;
- the complete 1,564-entry registered allocator ledger, exact seed domain, manifests, final cursors,
  and absence of pending/unledgered state;
- exact endpoint and registered-evaluation eligibility;
- runtime/source identity and policy-load attestation; and
- exact regular files, logical/allocated byte counts, and hashes for the policy, metadata, and
  training configuration.

The campaign layer binds the config, authorization, plan, claim, log, and checkpoint identities; it
does not reinterpret or publish the checkpoint's outcome-bearing rollout values.

Before a child starts, its artifact directory must not exist. After every claimed attempt, the
parent recursively inventories every directory entry below that cell's artifact root using
`lstat`, without following symlinks. The inventory object has exactly `root`, `exists`, `entries`,
`regular_file_count`, `directory_count`, `forbidden_count`, `logical_bytes`, `allocated_bytes`, and
`tree_sha256`. Each inventory entry has exactly `path`, `node_type`,
`logical_bytes`, `allocated_bytes`, and `sha256`; `node_type` is `directory`, `regular_file`, or
`forbidden`, directory byte counts and digest are null, and forbidden-node digest is null. Paths
are normalized root-relative POSIX paths in bytewise ascending order. A regular file is hashed from
its exact bytes and must have link count one. Symlinks, hard-linked files, sockets, devices, FIFOs,
unreadable entries, path escapes, or any unlisted node type are forbidden. Nothing is deleted or
rewritten after the attempt's outcome is known.

Counts and byte totals reconstruct exactly from `entries`. `tree_sha256` is the domain-separated
digest of the complete entries array under
`plant-beta-constrained-rl-registered-training-artifact-inventory-v1`; the absent-root inventory is
represented by `exists == false`, an empty array, zero counts and bytes, and the digest of that
empty array. The detached verifier repeats the walk and rejects any difference.

An admitted cell's artifact inventory must contain exactly the checkpoint directory and its three
declared regular files. A failed, timed-out, or rejected cell retains and authenticates every
partial and unexpected entry; unexpected content makes the cell ineligible but remains inside the
evidence graph. If inventory cannot complete, the claim is left without a terminal receipt, root
custody is failed, and no later cell may start.

## Exact plan, claim, receipt, and failure contract

Every authenticated JSON document uses the task-template domain-digest construction after removing
only its top-level `sha256`. `plan.json` has schema/domain
`plant-beta-constrained-rl-registered-training-plan-v1` and exactly these top-level keys:
`schema`, `protocols`, `authorization`, `campaign_root`, `source`, `runtime_provenance`, `host`,
`execution`, `task_template`, `cells`, `deterministic_evidence_sha256`, and `sha256`. Each of its 15
cell objects has exactly `order`, `algorithm`, `family_slug`, `optimizer_seed`, `config_path`,
`config_sha256`, `output_path`, `stdout_path`, `stderr_path`, `claim_path`,
`completed_receipt_path`, and `failed_receipt_path`.

Every `source` object has exactly `authorization_commit`, `approved_implementation_commit`,
`approved_implementation_tree`, `uv_lock_sha256`, `authorization_parent`, and `changed_paths`; its
domain is `plant-beta-constrained-rl-registered-training-source-v1`. Every `runtime_provenance`
object is the complete strict `beta-training-runtime-provenance-v1` mapping already emitted by
`BetaTrainingRuntimeProvenanceV1.to_dict()`. Every `host` object has exactly `platform`, `system`,
`machine`, `processor`, `cpu_brand`, `physical_cpu_count`, `logical_cpu_count`,
`physical_memory_bytes`, `torch_intraop_threads`, `torch_interop_threads`, `thread_environment`,
and `clock`, using the capacity runner's strict nested thread/clock fields. The implementation may
factor these shapes into shared code but cannot add, omit, or reinterpret a field under V1.

The plan's `deterministic_evidence_sha256` uses domain
`plant-beta-constrained-rl-registered-training-plan-deterministic-evidence-v1` over an object with
exactly `protocols`, `authorization`, `campaign_root`, `source`, `runtime_provenance`, `host`,
`execution`, `task_template`, and `cells`, copied byte-for-byte from the plan. The final report must
carry this exact digest; it cannot substitute a post-outcome identity.

`claimed.json` has schema/domain `plant-beta-constrained-rl-registered-training-claim-v1` and exactly
`schema`, `protocols`, `authorization_sha256`, `plan_sha256`, `cell`, `source_sha256`,
`runtime_provenance_sha256`, `host_sha256`, `previous_terminal_sha256`, `claimed_at_utc`, and
`sha256`. `cell` must equal that cell's complete plan object; `previous_terminal_sha256` is null for
cell 01 and exactly the preceding cell's terminal domain digest thereafter. This forms one strict
sequential receipt chain; a missing preceding terminal makes the next claim impossible. The host
digest uses domain `plant-beta-constrained-rl-registered-training-host-v1`. Time is canonical UTC
with six fractional digits and trailing `Z` and is evidence only, never a continuation input.

Both terminal filenames use schema/domain
`plant-beta-constrained-rl-registered-training-terminal-v1` and exactly `schema`, `status`,
`protocols`, `authorization_sha256`, `plan_sha256`, `claim_sha256`, `cell`, `process`, `logs`,
`artifact_inventory`, `checkpoint`, `failure`, `source_before`, `source_after`, `runtime_before`,
`runtime_after`, `host_before`, `host_after`, `finished_at_utc`, and `sha256`. `completed.json`
requires `status == "admitted"`, null `failure`, and a non-null admitted `checkpoint`.
`failed.json` requires `status == "process_failed"`, `"checkpoint_rejected"`, or
`"campaign_integrity_failed"`, a non-null closed failure object, and a null checkpoint. `process`
contains exactly
`command`, `timeout_seconds`, `started`, `exit_code`, `termination_signal`, `wall_elapsed_ns`,
`child_user_cpu_ns`, and `child_system_cpu_ns`; absent or inapplicable values are explicit nulls.

`logs` contains exactly `stdout` and `stderr`. Each is a descriptor with exactly `path`,
`logical_bytes`, `allocated_bytes`, and `sha256`. A non-null admitted `checkpoint` descriptor has
exactly `path`, `metadata_sha256`, `checkpoint_evidence_sha256`,
`runtime_provenance_sha256`, `environment_contract_sha256`, `checkpoint_contract_sha256`, `files`,
`logical_bytes`, `allocated_bytes`, `rollout_entry_count`, `allocator_entry_count`,
`optimizer_steps`, and `registered_evaluation_eligible`. Each member of `files` has exactly `path`,
`logical_bytes`, `allocated_bytes`, and `sha256`; it lists the three checkpoint files in lexical
path order. A rejected cell always has null `checkpoint`; its complete inventory authenticates the
rejected bytes without publishing parsed training outcomes.

The closed failure codes and precedence are:

1. `source_mutation`, `runtime_mutation`, `host_mutation`, `authorization_mutation`,
   `plan_mutation`, or `root_custody_lost`, in that order, produce
   `campaign_integrity_failed` and abort all remaining cells;
2. `child_log_limit_exceeded`, `child_timeout`, `child_spawn_failed`, `child_signaled`, then
   `child_nonzero_exit`, in that order, produce `process_failed` and continue when custody remains;
3. `checkpoint_missing`, `artifact_inventory_invalid`, `checkpoint_malformed`, then
   `checkpoint_admission_failed`, in that order, produce `checkpoint_rejected` and continue; and
4. otherwise the cell is `admitted`.

A failure object has exactly `code`, `stage`, and `message`. `code` is one value above, `stage` is
one of `pre_child`, `child`, `post_child`, `inventory`, or `admission`, and `message` is a fixed
categorical sentence selected by code: it contains no exception text, path supplied by a child,
rollout value, controller outcome, or other unbounded payload. If several observations apply, the
first code in the precedence above is authoritative while the complete process, logs, inventory,
and mutation evidence remain recorded.

Failure to durably publish a claim before process creation means no attempt occurred and produces a
campaign-level `claim_publication_failed`. Failure to durably publish a terminal receipt or the
final report produces `evidence_publication_failed`; the already published claim remains consumed,
the root is permanently closed, and no later cell starts. An external hard kill or power loss may
also leave a consumed claim without a terminal receipt. Such a root cannot make a report claim or
be resumed; detached forensic inspection classifies it as `claimed_interrupted`. These three
campaign-level codes never masquerade as child outcomes.

## Production report and detached verification

Production uses schema/domain `plant-beta-constrained-rl-registered-training-report-v1`. Its exact
top-level keys are `schema`, `protocols`, `claim_boundary`, `result_claim`, `authorization`, `plan`,
`source`, `runtime_provenance`, `host`, `execution`, `cells`, `status_counts`, `campaign_status`,
`abort`, `deterministic_evidence_sha256`, and `sha256`. Each report cell has exactly `cell`, `state`,
`config`, `claim`, `terminal`, `logs`, `artifact_inventory`, `checkpoint`, and `failure`; nullable
members remain present. Unknown or missing fields fail. Any dependency-injected fixture execution
uses distinct schema/domain `plant-beta-constrained-rl-registered-training-synthetic-report-v1`,
the filename `synthetic-report.json`, and a null production claim. Hooks can never issue or verify
`report.json`.

The production report accounts for all 15 planned cells in exact order, including `not_started`,
`claimed_interrupted`, `process_failed`, `checkpoint_rejected`, `campaign_integrity_failed`, and
`admitted` states. It binds protocol,
custody, authorization, source/runtime/host, plan/config, receipt, log, and artifact identities,
plus complete status counts and deterministic pre-run identity. It publishes no controller outcome.
Its only possible successful claim is:

> On the authorized host and clean source revision, the frozen 15-run constrained-RL training
> matrix completed once and every final checkpoint passed registered production admission. No
> development seed, held-out seed, controller comparison, or policy selection was opened.

Any missing or failed cell yields null claim. A non-injectable detached verifier must reopen every
referenced regular file without following symlinks, reject duplicate keys/non-finite values and
unknown fields, recapture source/runtime/host, independently re-admit every completed checkpoint,
recalculate every digest/count/order/status, and reject any output/config/log/receipt/artifact
substitution. Verification performs no development or held-out reset.

After exclusive file-and-directory-fsynced report publication, the production runner recaptures
source, runtime, authorization, plan, root, and host identity once more and invokes the detached
non-injectable verifier unconditionally in a fresh process. A successful command exit or report
claim is not displayed as verified unless this immediate pass succeeds. The verifier independently
checks the post-publication identity against the report's pre-run and per-cell evidence. Failure
closes the root without retry; it cannot rewrite the report or authorize a new attempt.

## Required implementation evidence

Only fixture-domain tests are permitted. They must prove:

1. the frozen task-template digest and rejection of a mutation in every semantic configuration
   section;
2. exact 3-by-5 matrix order, uniqueness, paths, config bytes, and output-root separation;
3. registered config duplicate-key, non-finite, alternative-encoding, override, resume, and task
   substitution rejection before environment/model construction;
4. authorization absent/false/wrong-scope/wrong-root/wrong-host/wrong-protocol/untracked/dirty
   rejection before output creation;
5. durable exclusive claims precede child launch and any existing root or receipt prevents retry;
6. sequential subprocess order, fixed timeout, bounded sealed logs, exact 5-GiB admission, and fixed
   continuation/abort rules;
7. all 15 successes, every closed failure code and precedence collision, interrupted claim,
   checkpoint substitution, complete failed-artifact inventory, and remaining-cell accounting;
8. production/synthetic report separation and schema rejection before runtime/host/artifact access;
9. detached verification of config, receipt, log, checkpoint, inventory, plan, source, runtime,
   host, full-report tampering, and mandatory post-publication invocation; and
10. no test, dry run, plan construction, report verification, or checkpoint admission opens a
    registered, development, or held-out reset.

The complete Python repository, strict touched-module typing, formatting, compilation, Git
whitespace, dependency-lock cleanliness, CLI import/help, frontend production build, and frontend
lint must pass on one clean implementation revision. A later user authorization record and its
exact root/host contract require another clean committed revision and the same complete gate before
production execution.

## Downstream boundary

Completing this custody layer does not make development selection ready. Separate outcome-blind
software must implement policy/E-beta pairwise eligibility, all-32-or-no-score vectors, the frozen
IQM/CVaR/median/sign lexicographic selector, five-policy family retention, generic economic
sensitivity, registered latency, hierarchical uncertainty, and detached development/held-out
verification before any development seed is opened.
