# Causal service-MPC reporting freeze

Date: **2026-09-04**

Status: **pre-outcome addendum; no registered 72-hour service-MPC cell has been
opened**

This addendum closes the reporting choices left implicit in
[`CAUSAL_SERVICE_MPC_PROTOCOL.md`](CAUSAL_SERVICE_MPC_PROTOCOL.md). It changes no
controller, solver, plant, demand, seed, utility threshold, or privileged reference.
Its purpose is to ensure that the first result cannot influence how the six-cell matrix
is admitted, summarized, or judged.

## Mandatory instrumentation correction

The pre-freeze audit found that `EpisodeMetricsV2` exposed battery discharge but not the
protocol's mandatory **battery throughput** diagnostic. The existing future-information
planner had computed throughput as the ordered, bus-side sum of accepted solar-to-battery
energy and accepted battery-support energy. The causal evaluator now records that same
quantity and arithmetic order at each decision. Causal result schema version **7** therefore
supersedes version 6 before any registered result exists. This is an instrumentation-only
change: throughput is diagnostic and does not enter the controller observation, action,
model, reward, or gate.

## Exact matrix and execution identity

The matrix contract has domain-separated SHA-256
`causal-service-mpc-development-matrix-contract-v1`. It binds:

- the ordered cells `35/0, 35/1, 35/2, 42/0, 42/1, 42/2`;
- the 72-hour horizon and 288 decisions per arm;
- controller order A0, A2, then M;
- the A2-minus-A0 and M-minus-A2 contrasts; and
- one fresh process per cell.

Each cell artifact records its matrix index and UTC evaluation start/completion instants.
The postprocessor requires non-overlapping intervals in the declared order. All six cells
must have byte-identical clean Git source snapshots, dependency-lock identity, and runtime
identity. The postprocessor itself must run from that same clean source revision.

Each input is a new, absolute, non-symlink JSON file outside the repository. Admission
requires strict UTF-8 JSON with unique keys and the exact registered serializer form. The
compact report records every source file's byte count and raw SHA-256. Neither cell nor
report output may overwrite an existing path.

## Privileged-reference boundary

The only admitted future-information reference is the committed file
`results/v2_feasible_planner_matrix_564002c.json`, with:

- raw file SHA-256
  `05bd4b66bf71c01847aad369fe3abbea99fe2efe38a5c15be49b131cb551065e`; and
- internal report SHA-256
  `33f7873b0fba1623d816109900f8a5f6a41929924f6dcf99bff5dedd856b1d64`.

For every cell, all three causal arms must share the frozen planner cell's exact manifest
and exogenous-episode digests. A successful A0 arm must reproduce the complete frozen A0
metric payload and the separately retained throughput exactly. Only after those checks may
`F` and `U` be attached. No planner schedule, action, witness, or future truth is passed to
the controller.

## Independent MPC evidence admission

A successful M arm is admitted only when it retains exactly 288 contiguous decisions and
applications. Every decision must have the shrinking horizon `min(96, 288-k)`, the frozen
settings digest, two ordered successful status-zero stages, a reconciled public residual,
and a requested action identical to its pre-guard trace. The postprocessor recomputes:

- requested-action trace hashes;
- action, residual, decision, application, and full-run hashes;
- request, guarded-action, and interval-power hashes;
- event/application indices and cross-references;
- solver stage and decision latency distributions;
- intervention, correction, range, refusal/reroute, and authority-distance aggregates; and
- the expected total of 1,728 decisions and 3,456 successful solver stages.

A failed arm remains a valid experimental outcome if its failure record is structurally
valid. It makes the complete matrix fail G1; it is never dropped or replaced.

## Frozen gate interpretations

The seven original thresholds retain their original values. The executable interpretation
is now fixed as follows:

1. **Execution/evidence:** all 18 paired arms must succeed; all six M arms, 1,728 decisions,
   and 3,456 stages must be valid; fallback count is zero. If this fails, the report retains
   all cells but sets the aggregate to `null`. G2--G7 become `not_evaluated`; no surviving
   subset statistic is calculated.
2. **Rate-42 causal capture:** an improved cell requires `M - A0 > 1e-9 kg`. At least two
   of three must improve. Capture is `(M-A0)/(F-A0)` where `F-A0 > 1e-9 kg`; all three
   rate-42 cells are eligible, and their arithmetic mean must be at least 0.20.
3. **Rate-42 service:** sum M's fully met commitments over the three cells. It must be at
   least the frozen A0 total, 26 of 36.
4. **Rate-35 preservation:** compare arithmetic mean delivery fill across all three cells;
   M may be at most 0.01 below A0. A frozen A0-saturated cell is one whose A0 service
   numerator equals its denominator; M must retain at least that many fully met commitments
   on both such seeds.
5. **Safety/validity:** for each M rollout, the accumulated reserve-breach reward component
   must be exactly zero and minimum SOC must not cross the active reserve; shutdown steps,
   out-of-range decisions, controller-invalid requests, and forced electrolyser stops must
   all be zero.
6. **Terminal/start limit:** each M terminal SOC must be at least 0.25 within `1e-12` numeric
   comparison tolerance, and cold starts must be no more than four.
7. **Latency:** use complete controller decision latency, not an individual solver-stage
   time. Empirical nearest-rank p99 must be at most 2 seconds and maximum at most 5 seconds
   both in the pooled 1,728-decision distribution and in every 288-decision rollout. This
   per-rollout requirement prevents one slow cell from being hidden by pooling.

Population standard deviation uses `ddof=0` and is descriptive only. Metrics and effects
are aggregated only for a complete 18-arm matrix.

## Retained result

The compact report retains complete physical metrics for A0, A2, and M; terminal
CH4-equivalent feedstock derived from the exact plant stoichiometry; M-A2, M-A0, A2-A0,
and eligible causal capture; both requested-action mechanism summaries; solver reliability
and latency; every gate observation; cell/raw identities; and compact MPC evidence digests.
The large 288-event and 288-application ledgers remain in the hash-bound external cell
artifacts.

The report is written to a new external path through a sibling temporary file, exact-byte
round trip, file fsync, non-overwriting hard link, directory fsync, and final byte check.
It receives its own domain-separated report digest. A human-readable Git report may be
written only after this machine report exists; it may summarize but not reinterpret the
frozen gates.

## Outcome-free verification boundary

Before execution, synthetic tests must demonstrate:

- one complete matrix passing all seven gates;
- one failed M arm suppressing every aggregate and marking G2--G7 not evaluated;
- an independent failure for each G2--G7 threshold;
- rejection of reordered cells, overlapping execution windows, mixed source revisions,
  hidden fallback, A0 replay drift, planner identity drift, non-canonical JSON, and output
  overwrite; and
- exact battery-throughput accumulation in the ordinary environment path.

These fixtures contain invented controller values and do not open a registered episode or
solver outcome.
