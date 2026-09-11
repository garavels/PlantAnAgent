# Full-geometry study capacity fixture

The single frozen fixture at clean `17331fe5b89a8e30769cdd656a826683071dae29`
completed its four sequential process attempts. **Complete study capacity is not
admitted:** M24 and E-beta retained incomplete controller executions. No process
timed out, hit a log/RSS cap or used a fallback. No optimizer ran.

The fixture used seed **9,100,001**, the unchanged full physical beta task and
Torch **1/1** in fresh children. PPO was an **untrained fixed-action proxy**.
It cannot measure trained-network inference or policy value.

| Arm | Decisions / 672 | Minutes / 10,080 | Child wall seconds | Peak RSS GiB | Physical archive bytes | Decoded bytes | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | 672 | 10,080 | 64.312 | 1.486 | 11,409,255 | 109,386,617 | Complete, cycle-infeasible |
| M24 | 36 | 540 | 55.028 | 0.557 | 1,608,962 | 14,616,419 | Unsupported degraded/offline DAC |
| E-beta | 52 | 780 | 30.299 | 0.529 | 1,626,242 | 15,313,263 | Selected plan violates hard battery cycle return |
| Fixed PPO proxy | 672 | 10,080 | 65.282 | 1.526 | 11,628,605 | 110,003,802 | Complete, cycle-infeasible |

Wall time includes process startup/closure. The retained measurement separately
records execution, publication and complete archive reconstruction. Both complete
actors failed the cyclic return criterion; completion here is only a capacity
observation. Prefix sizes and timings are **not** full-week measurements and no
complete-stage duration/storage projection is reported.

The initial command supplied 13 revision digits in the destination. The fixed-path
guard rejected it before creating a root or starting a child. The correct 12-digit
destination was then used; none of the four actor attempts was repeated.

## Evidence and remaining work

Raw report and all archives/checkpoints/logs are retained outside Git at
`/private/tmp/plantanagent-study-capacity-17331fe5b89a/`.
The [compact evidence](v2_study_capacity_17331fe.json) binds process metadata,
measurement and inventory digests and exact physical/timing byte descriptors.

Report raw SHA-256:
`2a9fc73241ad4b4568938558b86bbbe182dea95fa611f27ff3ed342b96e81986`.

Report domain SHA-256:
`4c10df0a87beb258d7dc6625229d259641158e63020dc39361e7b07ca796c56a`.

M24's declared unsupported-equipment boundary is retained as a contextual-arm
limitation. E-beta's failed selected-plan validation needs diagnosis using its
retained fixture inputs before a primary-comparator readiness decision. Do not
weaken a cycle constraint, change controller behavior or rerun this fixture to
manufacture capacity completion. Any subsequent repair/probe needs its own
explicit scope and retained link to this negative result.

Training capacity at `4f505b8` is unchanged. The stage budget, complete clean
release gate and explicit training authorization remain open. No registered
training, development/held-out reset, selection or deployment occurred.
