# Repaired causal service-MPC engineering replication

Date: **2026-09-04**

Source revision: `3796eb5c935b736f45d245f50175e96b2e776179`

Result: **execution-valid negative controller result; G1 and G7 pass, G2--G6 fail**

## Evidence boundary

This is the separately versioned engineering replication registered after the initial
`59e377c` matrix failed its execution gate. It tests only the four frozen numerical and
reporting repairs. The six cells had already been inspected during development, so this is
not holdout, prospective, site-generalization, or commercial evidence.

Before the first cell, the exact source revision passed 1,000 repository tests (2 skipped),
formatting, scoped strict type checks, compilation, dependency-lock validation, and a clean
worktree check. The cells then ran in fresh processes, sequentially in the registered order
`35/0, 35/1, 35/2, 42/0, 42/1, 42/2`, without source changes or overlap. The reporter
admitted all 18 controller arms, all 1,728 MPC decisions, and all 3,456 solver stages.
There were no fallback actions and no surviving-subset aggregate.

The common evidence identity is:

- worktree SHA-256:
  `32788dbc99d480de7587e3d90e9d8fe754ca2e1331e0873286373c46bd6564d7`;
- dependency-lock SHA-256:
  `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`;
- MPC settings SHA-256:
  `0e8896cfb9fe47dc3ac3b285fc729e029a80803e31eddf4a885795d2d04253f2`;
- runtime: CPython 3.12.7 on `macOS-26.6.2-arm64-arm-64bit`; and
- report format: `causal-service-mpc-engineering-replication-report-v2`.

## Frozen gate result

| Gate | Result | Observed |
|---|---|---|
| G1 execution validity | **Pass** | 6/6 cells, 18/18 arms, 1,728/1,728 decisions, 3,456/3,456 stages, zero fallback. |
| G2 stress-rate causal capture | **Fail** | MPC improved over A0 in 0/3 rate-42 cells; mean causal capture was -4.763, versus the registered minimum +0.20. |
| G3 stress-rate commitments | **Fail** | MPC met 9/36 rate-42 commitments; the registered A0 floor was 26/36. |
| G4 primary-rate non-inferiority | **Fail** | Mean rate-35 fill fell from 97.72% to 34.17%; both A0-saturated seeds lost complete commitments. |
| G5 operating safety | **Fail** | No reserve breach, shutdown, or invalid action occurred, but every rollout had forced electrolyser stops. |
| G6 terminal/cycling | **Fail** | Every terminal battery SOC was 100%, but cold starts ranged from 23 to 41, versus the maximum four. |
| G7 latency | **Pass** | Pooled p99 was 0.784 s and maximum 1.094 s; every rollout passed the 2 s/5 s limits. |

G1 means the performance failures are eligible evidence. It does not mean the controller
passed. The numerical repairs succeeded: the solver and telemetry paths completed reliably,
revealing a controller-to-plant mismatch that the first matrix could not evaluate.

## Controller outcomes

Descriptive means over the three development seeds are:

| Rate | Controller | Delivered kg | Fill | Commitments met | Cold starts | Forced stops | Terminal SOC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 35 | A0 legacy heuristic | 2,462.66 | 97.72% | 35/36 | 1.00 | 0.00 | 91.40% |
| 35 | A2 issued-forecast heuristic | 2,135.83 | 84.75% | 27/36 | 11.67 | 4.67 | 15.67% |
| 35 | M service MPC | 861.11 | 34.17% | 11/36 | 34.33 | 34.00 | 100.00% |
| 42 | A0 legacy heuristic | 2,569.51 | 84.97% | 26/36 | 1.00 | 0.00 | 91.40% |
| 42 | A2 issued-forecast heuristic | 2,468.18 | 81.62% | 26/36 | 10.00 | 4.67 | 15.79% |
| 42 | M service MPC | 891.78 | 29.49% | 9/36 | 34.33 | 34.00 | 100.00% |

MPC minus A0 delivered-product effects were negative in all six cells:

| Rate | Seed 0 | Seed 1 | Seed 2 | Mean |
|---:|---:|---:|---:|---:|
| 35 | -1,872.29 kg | -990.31 kg | -1,942.04 kg | -1,601.55 kg |
| 42 | -1,878.22 kg | -1,005.28 kg | -2,149.68 kg | -1,677.72 kg |

The MPC curtailed a mean 56.48 MWh at rate 35 and 56.64 MWh at rate 42, while drawing
only 204.0 and 181.0 kWh respectively from the battery. It ended every cell with a full
battery and approximately full CO2 storage. This is not evidence of an energy-scarce
schedule: usable energy and feedstock accumulated while the hydrogen-producing asset
failed to remain committed.

## Mechanism diagnosis

The failure localizes to the boundary between the 15-minute optimization model and the
one-minute authoritative plant, not to solver reliability or the primary delivery
objective.

1. At the first boundary of every cell, the MILP reports all four commitments in its
   24-hour horizon fully deliverable: 840 kg at rate 35 or 1,008 kg at rate 42, with 100%
   minimum fill. The complete plant rollouts achieve only 20.64--53.88% fill.
2. The MILP models `battery_support_kw` as expected battery discharge in a 15-minute mean
   power balance. The simulator defines the same action field as a *contingent top-up
   ceiling*: only an uncovered electrolyser request is actually discharged.
3. `realize_first_mpc_action` copies the expected discharge directly into that authority
   field. A start is therefore funded to exactly 600 kW in the mean
   (`20% * 3 MW`), with no authority for intra-interval solar error beyond the expected
   mean deficit.
4. The four solar shares are held for 15 one-minute physics steps and multiply the realized
   solar bus. The exact commitment kernel requires the start load at every minute and trips
   after two minutes of shortfall. A valid 15-minute mean balance therefore does not imply
   a valid one-minute commitment trajectory.

One retained seed-0 transition makes the mismatch explicit. At decision 192, after one
fully powered startup interval, the plan schedules 94.150 kW solar plus 505.850 kW battery
support: exactly 600 kW. The authoritative plant accepts 0 kW over the following interval
after sub-interval shortfall, aborting the start even though the optimization interval is
balanced.

Across all six rollouts, the plant recorded 206 cold starts, 204 aborted starts, and 204
forced stops. Seeds 0 and 2 never reached a productive MPC first interval at either demand
rate. Only seed 1 reached 34 productive decisions per rate. The near-identical failure by
weather seed across both demand rates is further evidence that actuation dominated the
service objective.

Guard behavior is not the primary rejection mechanism: every MPC request was in range,
there were no invalid actions, no reserve breach, and no plant shutdown. The exact plant
refused or rerouted power because the held mean-share command did not preserve the minute
commitment requirement. Guard clamping and forecast error remain secondary contributors
that must still be measured after this first mismatch is removed.

## Architecture decision

Do **not** tune objective weights, train PPO, or add more plant complexity on top of this
result. The current experiment cannot distinguish a weak optimizer from a lossy actuation
contract. The next bounded work package must separate three quantities that are presently
conflated:

- expected battery energy used by the planning model;
- contingent battery authority made available to the real-time dispatcher; and
- battery energy actually discharged by the plant.

The immediate implementation should introduce a versioned, commitment-aware actuation
bridge. Its first test is narrow: preserve a planned electrolyser target through one-minute
solar variation while the existing reserve guard remains authoritative. The MILP,
forecast, delivery objective, plant physics, and registered result stay unchanged. If that
bridge cannot close the mismatch without uncontrolled plan-energy deviation, the product
architecture should advance to explicit absolute process setpoints plus an inner-loop
dispatcher instead of retaining a share-only supervisory action.

Only after this seam is closed should the programme strengthen cyclic terminal economics,
add operational realism, and train RL against a credible equal-information MPC baseline.
RL remains a candidate for residual nonlinear, stochastic, and partially observed control;
it is not a remedy for an ambiguous actuator contract.

## Artifact custody

The immutable external artifacts are retained at
`/private/tmp/plantanagent-service-mpc-replication.AiZGHf/` for this research session:

| Artifact | Bytes | Raw SHA-256 |
|---|---:|---|
| `rate35-seed0.json` | 3,948,537 | `80f5d27af14d4d7633e194ed49307b31de96bd2c164485fcba51762d99333a13` |
| `rate35-seed1.json` | 4,011,083 | `c252d1f3f69cf0b0fc722431d4d164b010847174bb90fc9a5451e18161789630` |
| `rate35-seed2.json` | 3,936,562 | `dd8714db9349ba39254ec4b3fafd38b798dee5159f364dec360d9123d35bc444` |
| `rate42-seed0.json` | 3,947,679 | `6fa95928721a657a992e871b848c236fcb2073985b4c23046e17e2de4679005b` |
| `rate42-seed1.json` | 4,006,882 | `dd2c639bef844f7f6fd2b253d2fb41de26b606b20425eac839126aca2147b8b8` |
| `rate42-seed2.json` | 3,940,867 | `908407687454ca98ff3a38efd071baa3dab95c1275eaaf9e32ac6d0b038d345b` |
| `service-mpc-replication-report.json` | 207,908 | `bab822476d5b1b986ad03bdecbeaad6217727cfc3539e117af38af09bc458905` |

The report's domain-separated internal digest is
`5a5acf5f46ea0363bfee90925b1346d33b11ffc0766fb214ee55d8baaee2df13`.
The compact report is about 203 KiB, versus about 2.85 MiB for the initial failed report,
confirming that the version-2 failed-ledger compaction works while the full successful
ledgers remain in their hash-bound cell artifacts.
