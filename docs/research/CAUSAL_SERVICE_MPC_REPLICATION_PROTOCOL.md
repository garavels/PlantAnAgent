# Causal service-MPC engineering replication protocol

Date: **2026-09-04**

Status: **frozen before any repaired 72-hour rollout or complete repaired matrix is
opened**

## Why this replication exists

The first registered matrix from clean revision `59e377c` is immutable and failed G1.
None of its six MPC arms completed, so no MPC performance aggregate was eligible. Four
arms exposed a one-ULP negative-curtailment invariant at their first transition; the two
longer arms exposed a mismatch between the adapter's absolute residual threshold and
the feasibility of status-zero mixed-integer solutions returned by HiGHS. The first
compact report also copied complete failed ledgers instead of retaining their validated
hashes.

This protocol freezes a narrow engineering replication after repairing those execution
contracts. It does not create a second chance to tune the controller against observed
delivery performance. The cells, controller objective, comparator arms, public
information boundary, development gates, and no-subset rule are unchanged.

The original result and its custody record remain authoritative:

- source revision: `59e377c4bda4c46fa1873abc8e3506c0f4215946`;
- report digest:
  `22fadb2da9c72baf7f61ddc9fa2373c18b317cb494462dd12c126823df1a0961`;
- interpretation: `G1_failed_no_performance_aggregate`; and
- human record: [`results/v2_service_mpc_matrix_59e377c.md`](results/v2_service_mpc_matrix_59e377c.md).

## Claim boundary and prior inspection

This is an **engineering replication on already inspected synthetic development
cells**. It is not prospective evidence, a holdout, a controller-generalization result,
field validation, an economic result, or a comparison with RL.

The following post-failure diagnostics occurred before this freeze and are disclosed:

1. the exact first transitions for seeds 1 and 2 at both rates were replayed to isolate
   negative curtailment at approximately `10^-15 kW`;
2. rate 35 / seed 0 was replayed through successive partial horizons while diagnosing
   numerical extraction, including a final 130-decision run that completed without an
   exception; and
3. no repaired complete 288-decision MPC rollout, final episode metric payload, paired
   contrast, six-cell aggregate, or gate result has been opened.

Those diagnostics mean even repaired execution success through 32.5 hours on 35/0 is
known. The replication therefore answers only whether the repaired implementation can
complete the already-declared engineering matrix and, if it does, what the original
descriptive development gate reports. Any later generalization claim requires new,
pre-generated and outcome-sealed weather, demand, plant-parameter, and fault regimes.

## Frozen repair scope

Exactly four change classes are admitted by the serialized matrix contract.

### R1 — non-negative physical bus power

The methanator's accepted bus power is the offered power capped by the power reconstructed
from actual methane production. A kW-to-kg-to-kW binary64 round trip may not report more
power than was offered. This restores the allocator identity and prevents an impossible
negative spill while leaving methane, H2, CO2, dispatch priority, component limits, and
reward logic unchanged.

Required regression evidence includes the exact previously failing binary64 offer and the
registered seed-1 first transition. Both must report zero, not merely approximately zero,
curtailment when all available power is accepted.

### R2 — explicit solver feasibility contract

The default settings now bind `mip_feasibility_tolerance = 1e-7`. SciPy deliberately
passes this currently unrecognized public-wrapper option to HiGHS verbatim. HiGHS documents
the option as the MIP primal-feasibility tolerance. The adapter suppresses only SciPy's
exact pass-through warning for this one declared option; other warnings remain visible.

The setting, its independent audit, and its projection rule are serialized into the
controller settings digest. The adapter validates both the finite raw solver vector and
the bound-canonicalized vector. It accepts no fallback and still requires status zero,
`success == true`, finite objective evidence, and both optimal stages.

For a variable value `x` with finite declared bounds, bound violation is accepted only
when

```text
violation <= a + r * max(1, abs(x), abs(finite bounds))
```

For a sparse row with coefficients `A_i`, values `x_i`, and finite row bounds, row
violation is accepted only when

```text
violation <= a + r * max(
    1,
    abs(finite row bounds),
    sum(abs(A_i) * max(1, abs(x_i)))
)
```

where `a = r = 1e-7`. Integrality remains an absolute `1e-7` check. The coefficient term
is retained even when a binary is zero; otherwise rows such as
`productive_power <= rated_power * running` would discard the solver's declared row
scale precisely at the off state.

Raw values within the corresponding bound tolerance are projected to the nearest finite
declared bound before solution hashing and plan extraction. The projected vector is audited
again. Plan-step electrical identities use the same absolute-plus-relative convention at
their physical interval-power scale. This makes numerical zero explicit without admitting
a materially infeasible raw solution.

The authoritative option definitions are the
[HiGHS option reference](https://ergo-code.github.io/HiGHS/dev/options/definitions/)
and its [KKT diagnostics guide](https://ergo-code.github.io/HiGHS/dev/guide/kkt/).

### R3 — failed-ledger validation and compact reporting

A failed MPC evidence record is no longer trusted or copied wholesale. Before compaction,
the reporter must validate:

- the run digest, status, controller, completed-decision count, outer failure phase,
  pending state, and zero fallback count;
- a contiguous successful event prefix and at most one terminal typed failure event;
- successful event, requested-action, residual, solver-stage, and failure identities;
- a contiguous application prefix and its decision/action/residual cross-bindings;
- request, guarded-action, interval-power, and application digests;
- reconciliation/count relationships and every intervention/gap aggregate; and
- decision and solver-stage latency summaries.

Only after that validation may the report replace `events` and `applications` with
canonical ledger SHA-256 values. The raw ledgers remain in the immutable external cell
artifacts. A failed arm still fails G1 and suppresses the whole aggregate.

### R4 — protocol/version separation

The cell execution, matrix contract, and report formats advance to replication version 2.
The matrix contract binds this prior result, repair scope, exact cell order, exact
controller order, contrasts, horizon, decisions per arm, and the repaired default MPC
settings digest. Version-1 artifacts cannot be silently mixed with version-2 artifacts.

At this freeze, the repaired default settings digest is
`0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2` and the complete
replication matrix-contract digest is
`39a085da394c5469484cfdd4489982f3021b4de029af2e513684f725f6db74eb`.

## Explicitly unchanged

The replication may not change:

- rates `35` and `42 kg CH4/h`, seeds `0, 1, 2`, or their rate-major order;
- the 72-hour horizon or 288 held 15-minute decisions per successful arm;
- A0 legacy heuristic, A2 issued-forecast controller, then M service-MPC arm order;
- the causal public observation and issued-mean forecast boundary;
- the 24-hour receding horizon, delivery-lock tolerance, terminal battery target, or
  secondary objective weights;
- plant, weather, delivery, safety constraint, or reward configuration;
- the frozen feasible-planner references `A0`, `F`, and `U`;
- the seven original development gates or their thresholds;
- the ban on fallback and successful-subset aggregation; or
- the requirement for one fresh process per cell and a common clean source/runtime/lock.

No delivery or latency outcome observed during the replication may trigger a code,
threshold, objective, or settings change inside this matrix. Any such change requires a
new result record and a separately versioned protocol.

## Exact replication matrix and stop rule

The only admitted execution order is:

1. `35/0`
2. `35/1`
3. `35/2`
4. `42/0`
5. `42/1`
6. `42/2`

Each cell runs A0, A2, then M in a fresh process and writes to a new absolute JSON path
outside Git. Cells must not overlap. The source worktree must be clean and unchanged from
the start of the first cell through report publication. All six artifacts are retained
whether their MPC arm succeeds or fails.

After the first registered replication cell starts, implementation changes stop. If any
cell fails, the remaining cells still run in order so the registered execution-reliability
result is complete. The report must admit all six artifacts or publish no matrix report.

## Unchanged gate interpretation

G1 requires six admitted artifacts, all 18 arms complete, all six MPC rollouts complete,
1,728 MPC decisions, 3,456 successful solver stages, and zero fallback. If G1 fails,
the complete aggregate is `null` and G2–G7 are `not_evaluated`.

Only if G1 passes are the original utility, service, safety, terminal-state, cold-start,
and latency gates evaluated. A pass remains descriptive evidence on these six contaminated
development cells. A failure remains a result and does not authorize a surviving-subset
claim.

## Authorization gate before execution

The replication may start only from a committed clean revision after all of the following
pass:

- exact methanator and registered first-transition regression fixtures;
- solver schema, sparse audit, adapter, controller, runner, and reporter tests;
- adversarial failed-ledger digest/count/cross-binding tests;
- the complete Environment V2/planner/MPC focused suite;
- the entire repository test suite;
- formatting, strict type checks on changed production/test surfaces, compilation,
  whitespace, dependency-lock, and clean-worktree checks; and
- a short disclosed 35/0 execution regression crossing every first-matrix failure point.

Passing this gate authorizes only the registered engineering replication above. It does
not authorize held-out generation, RL training, commercial claims, or autonomous plant
deployment.
