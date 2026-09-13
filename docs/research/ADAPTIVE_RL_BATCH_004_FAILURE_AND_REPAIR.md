# Batch 004 failure and bounded repair

Prospective repair scope, 2026-09-13, against clean source
`4f908fcaeaeed417a793204778d87f62c29ec768`. The diagnostic archives were closed
before implementation. Existing research autonomy authorizes these local repairs;
this document does not launch or resume an experiment.

## Observed failure

The matched scalar/PI screen launched at 07:24:38 UTC and aborted at 08:00:47 UTC.
Scalar stopped with `MPCResidualIntegrityError`: the prior rejected the public
0–1 hour solar forecast as exceeding clear-sky energy. Its learner reported 4,048
transitions and zero completed optimizer rounds. PI was cancelled when wave
custody failed. Neither arm produced a final checkpoint or policy evaluation.
All registered processes and groups were subsequently verified absent.

The parent recorded only `ValueError`, losing its initiating message/site.
Repeated cancellation interrupted A's completion publication, and B's socket
shutdown raised ENOTCONN before publication. Both ordinary receipts survived,
but the parent report omitted them because completion parsing failed first.
Read-only recovery of those receipts does not repair the original custody graph
or admit performance. The original root and all failed artifacts remain immutable.

## Forecast diagnosis and permitted repair

The forecast producer averages solar power sampled at the start of each physics
minute. The MPC projection compares that issued energy with the continuous
integral of a clear-sky sine curve. These envelopes differ near dawn and dusk.
An exact public-issue reproduction for already consumed seed 9,630,007, decision
341 (day 4, 19:15), gives a first-band mean of 0.007359442500906482, sampled clear
mean 0.007754685426807396 and analytic clear mean 0.00727015429953326. The valid
issue is 94.903% of the producer's clear envelope, but 101.228% of the validator's.
A seed-free sampled fixture reproduces the rejection as well. This is a forecast
representation incompatibility, not plant infeasibility or floating-point noise.
Diagnosis used no environment reset, policy inference or solver call.

Implement a separately versioned OperatingMPC projection,
`issued-mean-physics-left-samples-v1`, using the producer's exact public
`SolarArray.clear_sky_kw` sampling and timestep. Bind the geometry, timestep,
complete public issue and sampled trajectory in its identity. Keep full canonical
24-hour band energy and exact shortened-horizon prefixes without renormalization.
Use the same sampled envelope for firm-power rows. Impossible energy and malformed
issues still fail with the existing tight tolerance; no clipping or hidden fallback.

The new projection applies equally to the OperatingMPC comparator and the residual
policy's prior. Add only the narrow trajectory validation extension needed by the
existing model builder and local OperatingMPC settings metadata. Preserve the old
M24/E-beta mean/p10 projection algorithms, versions and golden digests. Bind the
new module and solar implementation/constants explicitly into residual checkpoint
dependencies. Existing checkpoints are not silently reinterpreted.

Fixtures must cover the exact failed public issue, independently constructed
sampled dawn/dusk/night issues, supported timing grids, all four bands, darkness,
capacity rejection, full/short prefix equality, p10-only public digest changes,
trajectory tampering, sampled firm rows, consistent settings metadata and legacy
golden identities. No weather producer, simulator physics, delivery obligation,
reward, learner setting, residual authority or registered comparator is changed.

## Supervision diagnosis and permitted repair

A tiny fresh-file Lexar probe established that exFAT/FSKit can change an empty
file's inode on ordinary allocation/fsync while its retained reader descriptor
continues to reference that file. Cached initial inode equality is therefore an
invalid invariant on this volume. Replacement leaves the retained reader on the
old object; truncation can change path and reader identities together. This proves
a monitor incompatibility, but the lost parent error prevents attributing its
first ValueError to that specific check retrospectively.

Retain nonsymlink read descriptors before admitting writes. Compare fresh,
bracketed path/descriptor metadata with bounded resampling for allocation
transitions. Preserve regular-file/link checks, monotone observed size and stream
caps. Check retained byte prefixes so replacement, observed truncation and a
persistent in-place rewrite cannot be silently accepted. Prefer a small, explicit
monitor implementation; no general scheduling or storage abstraction is needed.

The implementation budget is 64 KiB hash blocks, at most 2 MiB read per 0.1-second
poll, a soft 10 ms work slice and a 60-second prefix-verification deadline. Record
bytes, backlog and timing; verification backlog is an explicit failure. Final
verification must complete under a declared bound before a successful inventory.
Fixtures must verify the budget and real-volume behavior. These sampled checks
cannot prove that a byte-identical rewrite never occurred between observations.

Coalesce repeated cancellation during existing capture cleanup and protect final
completion publication. A disconnected socket must not suppress durable failure
evidence. Keep the initiating failure separate from cleanup errors, recording a
closed failure code, site, exception type and bounded message in schema v2.
Once all registered groups are known and closed, validate and retain ordinary
receipts independently of supervisor completion. Such retention remains fatal and
non-admitted when completion evidence is missing. Unknown/unclosed groups continue
to prohibit file reads and inventories from potentially live writers.

Preserve ordinary peer-surviving failures, execution acknowledgement, identity,
storage/log bounds and source integrity. Tests must exercise the observed exFAT
transition, replacement, truncation/regrowth, prefix mutation, repeated signals,
ENOTCONN, missing completions, recovered receipts and unresolved group closure.
One exclusively claimed tiny real-volume fixture is permitted; no training or
policy evaluation is part of this repair validation.

## Evidence and next admission

The [aborted result](results/ADAPTIVE_RL_BATCH_004_RESULTS.md) retains admission,
launch, partial progress, all file digests and the limits on causal attribution.
Raw checkpoints and logs stay outside Git. A new protocol, new exclusive root,
clean source/gate and prospective detached analysis tools are required for any
subsequent training. Do not retry or resume batch 004. Reuse of already consumed
development training seeds, if chosen for a repair replication, must be explicit;
it is not fresh or held-out evidence.

Delivery remains **6,720 kg over 168 hours, in 28 obligations**. The objective is
economic, wear or recovery improvement while preserving service/resources. No
demand increase, weakened incumbent, protected-seed opening, frontend change,
cloud spending or operational-benefit claim is included.
