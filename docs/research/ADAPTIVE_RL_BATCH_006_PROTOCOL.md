# Adaptive batch 006: matched residual screen on internal active storage

Prospective design, 2026-09-13, before any batch-006 reset or implementation.
Existing local research autonomy authorizes this fresh execution after admission.
Batch 005 lost its active USB volume and left no normally closed training endpoint
or comparison. Preserve that attempt as interrupted evidence; do not resume its
partial learner state or infer a learning result from the interruption.

## Scientific design

All scientific provisions of the [batch-005 protocol](ADAPTIVE_RL_BATCH_005_PROTOCOL.md)
and its inherited matched scalar/PI design remain fixed. The hypothesis is whether
the existing PI-Lagrangian resource objective improves the economic/service/resource
tradeoff relative to scalar PPO around the same MPC prior at the same budget.
This storage repair supplies no evidence for or against that hypothesis.

| Contract | Fixed value |
| --- | --- |
| Task | 168 hours, 28 deliveries, 6,720 kg methane; unchanged daily returns/economics |
| Physics/control | One minute / 15 minutes |
| Interface | 135 public inputs plus four causal MPC targets; four bounded residual actions |
| Arms | A-mpc-residual-scalar and B-mpc-residual-pi |
| Initialization | Fresh matched actor/reward tensors, optimizer seed 101, exact zero output head, Gaussian std 0.2 |
| Budget per arm | 65,536 transitions, eight rollouts, 2,560 minibatch updates, four synchronous environments |
| Training seeds | 9,630,000–9,630,099; exactly 100 allocator resets per successful arm |
| Development seeds | 9,700,300–9,700,307 |
| Actors | H, OperatingMPC, M24, E-beta, initial residual from successful A only, final A, final B |
| Comparison | All 56 evaluation slots and 56 independent replay slots |
| Endpoint | Fixed final only; no retry, resume, extension or replacement |

Retain reward scale 0.001, all optimizer/cost-critic/dual-update settings, the
four physical resource cost channels, 25% operating-effort adjustment bounds,
zero/floor/startup behavior, dispatcher and guard. Retain the sampled OperatingMPC
projection for both priors and the native comparator. Recheck initial/native MPC
physical parity across all eight development weeks. Registered M24/E-beta keep
their original semantics and all stopped comparisons remain unavailable.

The training range is reused adaptive development material. Batch 004 opened its
first eight reset seeds; batch 005 left 44 reset records per arm, validated from
reset metadata as the expected 9,630,000–9,630,043 allocator prefix.
Its exact final training progress and exit status are unavailable. No checkpoint,
optimizer state, partial allocator or diagnostic outcome informs initialization.
The planned development panel has never been opened; it remains development,
not sealed confirmation. Original registered families/protected seeds stay unopened.

## Active storage and bounded implementation change

Use the persistent internal base
`/Users/filippogombac/Developer/PlantAnAgentResearch`, outside Git, temporary
directories and cloud-synced folders. Exclusively claim its `adaptive-rl-batch-006`
child. Readiness gates, prospective analysis tools, launch claim/receipt, launcher
log, supervisor records, checkpoints, trajectories and analysis outputs all use
this internal base. No active check, output or failure publication may require
Lexar to remain mounted. Historical external evidence may be read and copied with
verified hashes before admission; its original identity must remain explicit.

The inspected base parent and repository share internal APFS volume UUID
`86093AA6-4234-4173-AD1C-F1762C98870E`, mounted at `/System/Volumes/Data`.
Admission must verify actual device identity and the OS-reported internal,
nonremovable volume status; a path prefix alone is insufficient. Device numbers
are observed host metadata, not portable fixed constants. Recheck canonical base,
root and gate paths, device identity and at least 10 GiB free space before creating
the run root. Reject symlinks, external gates and any existing run root. Record
the checked storage identity in the gate and plan, and validate it with the
existing periodic plan checks. Extend those checks to enforce the free-space floor
during execution and before subsequent stages. Storage failure closes the attempt;
there is no automatic move back to USB or continuation with weaker checks.

At planning time the internal disk has approximately 18.4 GiB free. Historical
batch 003 occupies about 809 MiB of ordinary/metadata file contents and batch 002
about 1,305 MiB; these are sizing observations, not a guarantee for this run.
Retain the existing per-log/episode bounds and fail explicitly if space becomes
insufficient. Do not delete user files to create space.

Change only the existing matched runner's root/schema, interruption/storage
provenance, storage admission/periodic validation and corresponding fixtures/tool
joins. Preserve its source at `776a98f` as the batch-005 implementation. The generic
supervision helper already accepts supplied paths and needs no behavior/schema
change. Old imported runners' dormant external-root constants remain untouched.
No simulator, forecast, solver, policy, reward, frontend or task change is included.

## Execution and archive policy

Preserve two isolated training supervisors, Torch 1/1 threads, four synchronous
environments per arm, then sequential evaluation/replay. Keep 12 hours per trainer,
ten minutes per evaluation/replay, 64 MiB per log, 256 MiB per compressed episode,
and all batch-005 process/integrity/acknowledgement/monitor/closure bounds exactly.
Fresh models and empty optimizer/allocator state are mandatory.

Remain quiet and outcome-blind while training executes. Only process identity and
closure-marker existence may be monitored. No partial diagnostics, rewards,
weights or trajectories may be inspected. Integrity failures remain failures;
no opportunistic retries or endpoint substitutions are permitted.

After all writer groups close, run the prospective detached auditor before the
reporter. A missing closure graph permits only forensic preservation. Retain every
failed arm, stopped comparator and all paired delivery/resource/value differences.
Reports and checkpoints are audited at their original canonical internal paths.

Only after confirmed closure, copy artifacts to a new exclusive Lexar archive with
a manifest verifying byte counts and SHA256 at both source and destination. Keep
the internal originals through audit/report/publication. Do not rewrite embedded
absolute paths or reinterpret relocated checkpoints. A missing destination volume
or failed archive copy cannot relabel an invalid experiment as valid; retain local
evidence and report archive status separately. No automatic local deletion is part
of this protocol.

## Admission and interpretation

Commit this protocol before implementing the storage change. Then review the
bounded implementation; test unchanged scientific settings, internal path/device
admission, external/symlink/existing-root/low-space rejection, and ongoing source,
gate, device and space failures. Exercise a tiny actual helper wave on internal
storage and a fabricated runner graph with no active `/Volumes` access. Test
archive refusal while groups live and immutable, hash-verified copies after closure.
Use fixtures without production resets or learner-outcome access.

Commit clean source, pass a revision-bound regression/static gate, and freeze
updated detached tools and validation evidence before launch. New tool pins/root
and storage-plan joins must be validated against actual producer fixtures; retain
all failed admission checks. There is no normal batch-005 result to reuse.

The [parent protocol](RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) still requires
absolute hard safety, paired individual delivery/resource noninferiority, frozen
materiality, fresh sealed confirmation and at least three optimizer seeds before
an algorithm-level claim. Absolute restoration failures remain disclosed. This
single-seed development screen does not authorize an operational-benefit claim,
live deployment, changed delivery demand or weakened incumbents.
