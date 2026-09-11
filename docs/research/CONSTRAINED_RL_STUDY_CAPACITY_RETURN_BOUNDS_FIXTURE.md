# Study capacity after the terminal-bound numerical repair

Status: **frozen before execution; no registered training or evaluation authorized**

Identifier: **`plant-beta-constrained-rl-study-capacity-return-bounds-fixture-v1`**

This separate fixture checks the implementation repair at `9c76d9a`. It preserves
the negative original capacity result at `17331fe` and the saved-boundary
diagnosis. The original report raw SHA-256 is
`2a9fc73241ad4b4568938558b86bbbe182dea95fa611f27ff3ed342b96e81986`,
and its domain digest is
`4c10df0a87beb258d7dc6625229d259641158e63020dc39361e7b07ca796c56a`.
Its incomplete M24/E-beta observations must remain visible in every capacity
summary. This fixture cannot convert that original attempt into a success.

## Fixed difference and execution

The sole controller change is the versioned, mathematically equivalent terminal
bound representation documented in
[`BETA_MPC_CAPACITY_BOUNDARY_DIAGNOSIS.md`](BETA_MPC_CAPACITY_BOUNDARY_DIAGNOSIS.md).
The same four hard rows, minima, solver tolerances, objective and physical task
remain. No policy, parameter or learning-rule tuning follows from this fixture.
M24's unsupported degraded/offline DAC boundary remains unchanged and may fail
again; its result is required and cannot be omitted or replaced.

Repeat the exact four-actor fixture **H, M24, E-beta, fixed untrained PPO proxy**
on the same fixture seed **9,100,001**, once each in sequential fresh processes.
Reuse the full task, fixed action `(0,0,0,0)`, settings and all resource limits of
[`CONSTRAINED_RL_STUDY_CAPACITY_FIXTURE.md`](CONSTRAINED_RL_STUDY_CAPACITY_FIXTURE.md).
The launcher now binds this document and the original document separately and
uses this fixture's distinct schema. There is one new source-derived destination
and one claim per actor; no retry, alternative seed or post-outcome repair is
permitted within this fixture.

The bounds remain 600 seconds per child (**40 minutes maximum across four child
timeouts**, excluding parent bookkeeping), two 1-MiB logs per child, 8-GiB opening
free space, a sampled 12-GiB child RSS stop, 1-GiB/4-GiB stored/decoded physical
archives and 8-MiB/64-MiB timing archives. CPU identity checks are read-only. The
fixed policy performs no optimizer updates and measures no neural inference.
No protected training, development or held-out seed is used. Hold source and
documents unchanged throughout execution and complete archive reconstruction.

## Interpretation

The question is whether the numerical repair permits a complete E-beta storage
and timing observation, or exposes a different failure. Keep either result.
Only a verified 672-decision/10,080-minute actor is a full-week observation.
Prefixes cannot be extrapolated or represented as complete controller capacity.
M24 remains a contextual comparator with a declared modeling boundary. If it
remains incomplete, state that gap; it does not justify silently changing M24 or
declaring complete four-arm capacity. H/PPO completion likewise says nothing
about cyclic feasibility, learned-policy value or commercial readiness.

After this fixture, retain compact measurements and all failure records. Resolve
remaining gaps through an explicit capacity assessment before final budgets and
the complete clean repository gate. Training authorization remains a separate
later user decision; this fixture grants none.
