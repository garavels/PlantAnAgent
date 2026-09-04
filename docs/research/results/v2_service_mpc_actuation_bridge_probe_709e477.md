# Service-MPC actuation-bridge mechanism probe

Date: **2026-09-04**

Status: **B1 execution failure; current bridge rejected as a supervisory compatibility
layer from separately labelled post-outcome mechanism evidence**

## Claim boundary

This was the single development-contaminated rate-35/seed-0 mechanism probe authorized by
`CAUSAL_SERVICE_MPC_ACTUATION_BRIDGE_PROTOCOL.md`. It is not a replication, holdout result,
controller comparison, or commercial claim. B1 failed, so the preregistered protocol leaves
B2--B6 unevaluated. Partial-trajectory quantities below are retained engineering evidence
only and are not promoted into complete-episode performance results.

## Immutable execution identity

| Field | Value |
|---|---|
| Source revision | `709e477b8e5b16faee4a1463235aca2bc1b4e787` |
| Worktree identity | `c63e8224e6585db5c252febecbaa9fa1ef0ae8464322de5231282d2265b9adfe` |
| Dependency-lock SHA-256 | `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd` |
| Protocol raw SHA-256 | `653b118a5d0003e0cb5a37a8076a0f9d765c8eaa7db9a9b759616bf501f981c0` |
| Probe-contract digest | `83ad83b07f57e43fb9af25bf80b07c6cb33b149fdb3e82d2321a9003e63b321f` |
| Action schema | `service-mpc-requested-action-v2` |
| Cell | 72 h, 35 kg/h, seed 0, 15-minute decisions |
| Manifest SHA-256 | `1e036f69c0514ad34a307b2d213fab3c781561ab6ad1efd199ed8313eca8dae4` |
| Exogenous episode SHA-256 | `2e0b7fd35b85c2a0b9d220c32151a00f96adbd7e24e610bb3b17740345903368` |
| Runtime | CPython 3.12.7; macOS arm64; SciPy 1.18.1; HiGHS 1.12.0 |
| Started | `2026-09-04T13:22:10.069452Z` |
| Evaluation ended | `2026-09-04T13:24:10.814156Z` |
| Runner wall time | 121.278863 s |
| Raw artifact bytes | 6,686,495 |
| Raw artifact SHA-256 | `e5fac6425a92ffc511827f12cf47936b8a776cf879e99c8b019ddcc5aa2f4458` |
| Internal report digest | `e49c7866ddd9a56e71b0f9061aaa8d90d634a14b6b44e5fae6aa4a513ba72aa2` |

The source audit is valid: identity was complete and clean at both boundaries, the worktree
identity did not change, the dependency lock matched, and no fallback action was used. The
artifact was written once to a new external path and was not retried on this revision.

## Predeclared gates

| Gate | Status | Observation |
|---|---|---|
| B1 — execution | **fail** | 223/288 completed decisions; 446 successful stages; zero fallback; one typed terminal solver failure. |
| B2 — mechanism | not evaluated | Suppressed by B1 as preregistered. |
| B3 — safety | not evaluated | Suppressed by B1 as preregistered. |
| B4 — direction | not evaluated | Suppressed by B1 as preregistered. |
| B5 — accounting | not evaluated | Suppressed by B1 as preregistered. |
| B6 — latency | not evaluated | Suppressed by B1 as preregistered. |

The successful-decision latency prefix has p99 1.268296 s and maximum 1.373531 s. These
values are descriptive only because B6 was not opened after B1 failed.

## Exact execution failure

The run stopped during the secondary lexicographic solve at decision 223, after 55.75
simulated hours. HiGHS returned successful optimal status. The independent audit accepted
the raw solution and rejected the subsequently canonicalized solution:

```text
service-MPC secondary_tiebreak solution failed independent residual audit:
bounds=0, rows=3.49581e-05, integrality=0, worst_row='co2_balance[1]'
```

This distinguishes the defect from solver infeasibility. Bound canonicalization projects a
value close to a declared bound using a tolerance scaled by that variable's finite bounds.
The CO2 state has a 6,000 kg upper bound, so its zero-bound projection window is
`0.0006001 kg`. At this boundary the physical CO2 inventory is
`0.00003495807901682279 kg`. Projecting a future state from that scale to zero independently
of its coupled material-balance row creates the observed `3.49581e-05 kg` equality residual.
The raw-vector audit had already passed. The required repair is therefore
feasibility-preserving canonicalization, not a looser balance tolerance and not a claim that
HiGHS failed.

## Partial physical trajectory — descriptive only

The 223 completed decisions expose a strong mechanism signal:

| Quantity at 55.75 h | Bridge prefix |
|---|---:|
| Settled commitments | 9 |
| Commitments met | 9 |
| Requested / delivered | 1,890 / 1,890 kg |
| Cold / hot starts | 3 / 5 |
| Aborted starts | 0 |
| Forced stops | 0 |
| Controlled stops | 2 |
| Productive operating time | 85,500 s |
| Minimum battery SOC | 15.00% |
| Terminal battery SOC | 27.62% |
| Shutdown / out-of-range decisions | 0 / 0 |
| Guard interventions / corrected physics steps | 1,218 / 850 |
| Solar utilization | 99.8849% |

For context only, the frozen `3796eb5` controller delivered 647.7138 kg over its complete
72-hour episode and recorded 41 aborted plus 41 forced starts. The bridge prefix therefore
supports the narrow diagnosis that inadequate support authority caused the earlier repeated
startup failures. It does **not** establish that the implemented authority rule is a valid
way to realize the optimizer's plan.

## Frozen bridge accounting

All 223 comparable completed decisions pass the probe's digest and ordering audit. The run,
decision, action, residual, request, guarded-action, interval-power, and application
identities close; authority matches the versioned rule and requested-action trace; applied
support never exceeds guarded support. The separately retained totals are:

| Support quantity | Energy-equivalent total |
|---|---:|
| MILP-planned battery discharge | 28.499648 kWh |
| Contingent authority capacity | 42,722.660301 kWh |
| Authority remaining after the minute guard | 23,911.025270 kWh |
| Physically applied battery support | 14,410.008667 kWh |

The accounting violation count is zero under the frozen definition. That is an important
negative finding: ledger closure alone does not establish plan fidelity. Applied battery
support is **505.62 times** the MILP-planned discharge. The bridge makes the unplanned energy
visible, but its preregistered B5 rule did not place an acceptance bound on that difference.
It must not be retrospectively relabelled as a B5 failure; it is post-outcome evidence that
the next control contract needs an explicit target-tracking gate.

## Post-outcome target-fidelity audit

This audit was computed after B1 failed from the retained successful decision ledger. It was
not a preregistered gate and is reported separately.

Of 109 decisions with a positive planned electrolyser target:

- 22 track within `1e-7 kW`;
- 81 overshoot and 6 undershoot;
- mean error is **+717.568 kW**;
- the range is **-309.380 to +1,520.782 kW**;
- planned electrolyser energy is 44,360.845 kWh;
- applied electrolyser energy is 63,914.561 kWh; and
- absolute target error integrates to 20,227.057 kWh.

The net applied excess is 19,553.716 kWh. A representative decision plans 1,416.970 kW of
electrolyser power with no battery discharge, then applies 2,937.752 kW because the action
also exposes 1,416.970 kW of additive support authority.

The mechanism follows directly from the existing plant interface. Solar shares first route
some instantaneous PV to the electrolyser. `battery_support_kw` then offers additional
power up to stack and battery headroom; it is not a closed-loop target. Simultaneous
battery-charge/support requests may also cancel their overlap by redirecting that solar to
the stack. Consequently `authority = electrolyser target` does not mean “make up whatever is
missing to reach the target.” It means “permit as much as another target on top of the solar
route.” The unchanged equipment and reserve guards keep this physically bounded but cannot
recover the optimizer's intended setpoint.

## Decision

The implemented bridge is rejected as the long-term supervisory seam. This decision is not
a claim that B2--B6 failed; they were not evaluated. It follows from the probe's stated
purpose—testing whether the share interface can express the first MPC target—and the
separately labelled post-outcome fidelity audit showing that it cannot.

The programme will not repair the canonicalizer merely to rerun this bridge. Instead:

1. Preserve the `709e477` artifact and this failure without same-revision retry.
2. Repair solver canonicalization as an independent, outcome-disclosed engineering defect:
   a raw feasible vector may be projected only when the resulting vector remains feasible;
   no physical balance tolerance will be loosened.
3. Version the supervisory command around absolute process-power setpoints, including an
   explicit electrolyser target and declared process priorities.
4. Add a deterministic one-minute inner dispatcher that observes the currently available
   bus, routes solar first, supplies only the target shortfall from the battery, and remains
   subordinate to the same safety reserve and component limits.
5. Bind plan, setpoint, minute request, guarded request, and applied power in evidence. Add
   predeclared target-error and unplanned-energy limits, not merely inequality accounting.
6. Validate that actuator contract before objective tuning, added realism, PPO, or
   HEPA/supervisor experiments.

This is also the commercially relevant architecture: optimization chooses slower plant
intent under uncertain forecasts, deterministic control equipment realizes it at the fast
timescale, and an independent guard remains authoritative. A deployable product cannot ask
a forecast planner or learned policy to encode a closed-loop power target indirectly through
held fractions and an additive battery field.

## Artifact retention

The canonical raw artifact is retained for this research session at:

`/private/tmp/plantanagent-mpc-bridge.gUi8kC/probe.json`

The external path is not a durable product store. The raw and internal digests above are the
permanent identities; this compact Git report retains the conclusion, failure boundary, and
decision even if temporary storage is later removed.
