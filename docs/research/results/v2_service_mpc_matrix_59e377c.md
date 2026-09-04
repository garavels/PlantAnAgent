# Initial registered causal service-MPC matrix

Date: **2026-09-04**

Source revision: `59e377c4bda4c46fa1873abc8e3506c0f4215946`

Result: **development gate failed at G1; no controller-performance aggregate is
eligible**

## Evidence boundary

The exact pre-outcome protocol and reporting interpretation were committed before the
first cell. The source revision passed 986 repository tests (2 skipped), formatting,
strict type checks on the new production and test surfaces, compilation, dependency-lock
validation, and a clean-worktree check. All six cells then ran in separate processes in
the registered order `35/0, 35/1, 35/2, 42/0, 42/1, 42/2` without source changes or
overlap.

The admitted matrix reports:

- complete ordered cell set: `true`;
- common clean source revision: `59e377c4bda4c46fa1873abc8e3506c0f4215946`;
- common worktree identity:
  `9360ad08e1a851b700e58e0c5d9452644d066aeddd151f16c533d41eb1585027`;
- dependency-lock SHA-256:
  `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`;
- runtime: CPython 3.12.7 on `macOS-26.6.2-arm64-arm-64bit`;
- all twelve A0/A2 arms complete and all six M arms failed;
- zero fallback actions; and
- no surviving-subset aggregate.

## Registered cell outcomes

| Rate | Seed | M decisions completed | Failure phase | Retained failure |
|---:|---:|---:|---|---|
| 35 | 0 | 114 | `controller_compute_action` | HiGHS reported the secondary stage optimal; the independent audit rejected a `5e-7` bound violation (row residual `9.77707e-12`, worst row label `battery_balance[79]`). |
| 35 | 1 | 1 | `controller_transition_observation` | `observed_curtailed_kw must be non-negative` |
| 35 | 2 | 1 | `controller_transition_observation` | `observed_curtailed_kw must be non-negative` |
| 42 | 0 | 90 | `controller_compute_action` | HiGHS reported the secondary stage optimal; the independent audit rejected a `1e-6` bound violation (row residual `4.36984e-13`, worst row label `battery_balance[27]`). |
| 42 | 1 | 1 | `controller_transition_observation` | `observed_curtailed_kw must be non-negative` |
| 42 | 2 | 1 | `controller_transition_observation` | `observed_curtailed_kw must be non-negative` |

Every cell retained the failed M arm, partial metric state, requested-action prefix, and
controller evidence. No failed action was replaced by a heuristic fallback.

## Frozen gate result

G1 failed with 6/6 cell artifacts, 12/18 successful arms, 0/6 complete M rollouts, and
zero fallback actions. Per the preregistered no-subset rule, the machine report sets its
aggregate to `null`; G2 through G7 are `not_evaluated`. In particular, this run says
nothing yet about whether MPC improves delivery, preserves service, satisfies full-rollout
safety/terminal limits, or meets the latency target.

This is an engineering failure result, not evidence that MPC scheduling is worse than A0
or A2. The MPC did not complete any 72-hour paired rollout, so comparing its partial
trajectory to complete baselines would be invalid.

## Artifact custody

The immutable external artifacts were retained at
`/private/tmp/plantanagent-service-mpc.StFLr8/` during this research session:

| Artifact | Bytes | Raw SHA-256 |
|---|---:|---|
| `rate35-seed0.json` | 3,353,914 | `f2abca05b8e61c786e4752b81ec6e7068867aa555e6d7fcf00817ff97d91e071` |
| `rate35-seed1.json` | 338,834 | `f6a26aebd9f8b96aaeb8b6781b5786cc7c9711a96910f564de55efbd2a99d865` |
| `rate35-seed2.json` | 337,546 | `aa69ddd26ce6f7c8ba453b06bb0f1b1d75162c1417d7f244dfc60ce8a0279888` |
| `rate42-seed0.json` | 2,743,332 | `a937d5137fa545fb540151b9da5a7a768380a30e6d20545978c0a439f06657d6` |
| `rate42-seed1.json` | 339,229 | `83f46c18e86eaec4358e12f7aea2d3e34c6dc9475b7113662aaa74656d1e4419` |
| `rate42-seed2.json` | 339,700 | `3677078a2aa4df419f9f64f718d539096237b368b46dd7a6f29623c913b66edd` |
| `service-mpc-report.json` | 2,983,662 | `4c509fda07965ab92bfbf78a403822cf34b6e3dc4a492d6592f37ed924d379f2` |

The machine report's domain-separated internal digest is
`22fadb2da9c72baf7f61ddc9fa2373c18b317cb494462dd12c126823df1a0961`.
Its exact feasible-planner reference has raw SHA-256
`05bd4b66bf71c01847aad369fe3abbea99fe2efe38a5c15be49b131cb551065e`
and internal digest
`33f7873b0fba1623d816109900f8a5f6a41929924f6dcf99bff5dedd856b1d64`.

## Post-outcome diagnosis

A four-case, one-decision reproduction using the exact registered runtime and first M
request found observed curtailment means of `-1.894780628693601e-15 kW` for seeds 1 and
`-9.473903143468004e-16 kW` for seeds 2, at both delivery rates. The requested action sent
all solar to the methanator. A kg-to-kW floating-point round trip can make the component's
reported power one unit in the last place larger than its offer; the allocator then exposes
an equally tiny negative spill. The physical episode metric already clamps the interval
curtailment delta to zero, but the public interval telemetry did not. The strict residual
record correctly rejected the impossible negative physical quantity, revealing an
inconsistent invariant between two telemetry paths.

The two later failures expose a separate solver-adapter calibration issue. The independent
audit tolerance is `1e-7`, while status-zero HiGHS MIP solutions reached bound residuals of
`5e-7` and `1e-6`. These are negligible in physical units but cannot simply be ignored:
the next revision must bind the solver's declared feasibility tolerances, independently
audit against that explicit contract, and continue rejecting violations beyond it.

The first failed report also revealed a non-causal reporting defect: failed-run evidence was
copied with its full event/application ledgers instead of the compact hashes promised by
the reporting addendum. This did not affect G1, source admission, or any performance
statistic, but inflated the compact report to about 3 MB. The first report remains immutable;
the next reporting revision must compact failed evidence after validating its partial ledger.

## Next decision

The next work is a numerical-contract repair, not controller tuning:

1. restore non-negative, energy-consistent simulator curtailment at its physical source and
   add exact-runtime regression coverage;
2. declare and bind realistic HiGHS feasibility settings, with independent scaled and
   absolute residual tests around their acceptance boundary;
3. fully validate and compact partial failed-run ledgers;
4. rerun all short fixtures and the repository gate; and
5. preregister a fresh six-cell replication on a new clean revision while retaining this
   failed matrix as the first result.
