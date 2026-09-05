# Full-day horizon repair clarification 001: evidence and canonical labels

Status: **frozen before implementation and before any M24 action on seeds 20--24**

Date: **2026-09-05**

This clarification preserves
`FULL_DAY_HORIZON_REPAIR_PROTOCOL.md` and resolves five implementability questions found by
the pre-code admission audit. It does not change the intervention, matrix, controller,
objective, task, threshold, gate, or decision rule. The original protocol remains immutable;
this document is an additional input to the implementation contract.

## 1. Canonical inferred-delivery label

F4 says that the second through fourth first-decision targets are “labelled inferred.” The
existing typed target contract accepts exactly two source values:

```text
first target:  announced
later targets: inferred_recurring
```

Therefore “labelled inferred” means the existing canonical string `inferred_recurring`. The
required first-decision source sequence is exactly:

```text
announced, inferred_recurring, inferred_recurring, inferred_recurring
```

Changing this schema string would introduce an unrelated controller-contract intervention and
is forbidden.

## 2. First-boundary forecast evidence

The existing MPC run ledger retains the public-forecast and reconstructed-trajectory SHA-256
identities but not their complete serialized bodies. Those hashes alone do not prove the F4
requirement that event zero used the unchanged complete public issue and its 24-hour
projection.

Before requesting the M24 action, the new runner must therefore retain an immutable
`first_boundary_audit` containing:

- the complete detached public controller observation;
- the complete `PublicForecastIssueV2` payload;
- the complete `MPCForecastTrajectoryV1` payload reconstructed from that issue;
- their domain-separated identities; and
- one digest over the complete audit.

The cell verifier must reconstruct and check those identities, require exactly 96 consecutive
15-minute projection intervals spanning 24 hours, and bind all three hashes to M24 event zero.
It must also require event-zero public-observation and public-forecast identities to equal the
corresponding M6 event-zero identities in the exact reference cell. This is additional evidence,
not additional information supplied to the controller.

## 3. New interval evidence and legacy missingness

Runner obligation 6 applies literally to the new M24 trajectory. For every M24 controller
interval the runner must retain, in order:

- decision index and public-boundary identities;
- scalar reward;
- delivery transition;
- complete economic interval/component record;
- command/setpoint and minute physical execution evidence; and
- the next public-boundary identity.

The interval rows must be hash-bound, sum exactly to the episode reward, delivery, energy,
throughput, exposure, and component totals, and be independently recomputed by the verifier.

The immutable robustness artifacts predate this requirement. M6 retains complete physical
boundary, command, solve, execution, and residual evidence, so its four settlement deliveries
and post-settlement tank/battery states can be reconstructed. H retains its command and minute
application hashes plus aggregate physical/economic metrics, but not the underlying boundary
state bodies, interval delivery rows, or interval economic component rows. Consequently:

- H post-settlement H2, CO2, CH4, and battery values are `not_retained`;
- H individual settlement deliveries are `not_retained`;
- controller-specific MPC forecast, horizon, solve, residual, and target fields are
  `not_applicable` for H; and
- old per-interval economic components are `not_retained` for both H and M6.

These values must not be reconstructed by rerunning H or by inventing a distribution from its
aggregate totals. Every cell and per-rate summary must expose the explicit availability state
and missing count. H aggregate service, physical, and economic values remain available and are
the only H values used by F5 and F7. M6 aggregate values and reconstructable settlement/state
values remain available for F5 and F6. No gate is weakened: the unavailable legacy fields were
never inputs to a frozen pass threshold.

## 4. Meaning of old arm and artifact identities

The old controller arms are embedded in one cell JSON file rather than published as separate
files. “Raw, full, deterministic, and evidence-graph identities for all old and new artifacts”
therefore means:

- old report: raw-file, report-domain, deterministic-evidence, and contract identities;
- old cell: raw-file, cell-domain, and deterministic-evidence identities;
- old H arm: controller ID, retained H-evidence SHA-256, and a newly computed
  candidate-independent deterministic-arm digest over the exact embedded arm;
- old M6 arm: controller/settings IDs, run-evidence SHA-256, evidence-graph SHA-256, and the
  same candidate-independent deterministic-arm digest;
- new M24 cell/report: their raw-file, full-domain, deterministic-evidence, contract, interval-
  evidence, run-evidence, and evidence-graph identities.

The deterministic reference-arm digest is a citation derived from already opened bytes; it is
not a new outcome or a replacement for the old cell identity. Absolute filesystem paths and
wall-clock timings are excluded from deterministic evidence.

## 5. Detached semantic verification is mandatory

The existing `verify_mpc_setpoint_run_digests` routine verifies the outer run digest and compact
evidence-graph digest. It does not, by itself, reconstruct every nested event and recompute the
semantic run ledger. That level is insufficient for F1--F4.

Before admission, the implementation must add a detached codec/verifier that:

1. rejects duplicate JSON keys, noncanonical JSON bytes, symlinks, non-regular directory
   entries, missing/extra entries, and path escapes;
2. reconstructs every delivery target, solver audit/selection/stage, first plan step, setpoint,
   envelope, minute dispatch, guarded action, commitment boundary, minute application,
   interval execution, residual, and success/failure event through the existing typed classes;
3. requires every typed round trip to equal the supplied mapping exactly;
4. rebuilds the complete run evidence from the typed events and dispatcher timings;
5. requires the rebuilt counts, energy, target tracking, deficit attribution, event/execution/
   application identities, latencies, evidence graph, and outer digest to equal the supplied
   run; and
6. applies this stronger verification to both the old M6 reference arms and new M24 arms.

The exact old robustness directory must additionally match the registered raw report,
report-domain, deterministic-evidence, contract, dependency-lock, executed-source, and clean
worktree identities. Its expected `complete_gate_failed` research status is valid retained
evidence and must not be confused with artifact-verification failure.

## 6. Unchanged intervention and decision rule

M24 still changes only `planning_horizon_h` from 6 to 24. The first target boundaries remain
24, 48, 72, and 96; all objective weights, solver options, plant/economic/task identities,
controller inputs, command authority, dispatcher, guard, and simulator behavior remain fixed.
F1--F8, including the five-seed denominators and the requirement to report failures without a
surviving-subset aggregate, are unchanged.

No protected environment reset, controller action, solver outcome, policy training, or
heuristic replay is authorized by this clarification. Implementation and adversarial tests
remain outcome-free. A separate admission audit and execution registration are still mandatory
before the ten M24 trajectories may run.
