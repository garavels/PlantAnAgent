# Feasible future-information planner development result

Date: **2026-09-03**

Status: **complete six-cell development evidence; privileged and non-prospective;
causal service MPC is now justified**

## What this study answers

The earlier causal-controller experiment established that the repository's stateless
deadline rule was not a credible production baseline. It did not establish whether its
failures came from physical scarcity, the five-field action interface, or poor temporal
planning. This study separates those questions with the frozen bracket:

```text
A0: exact legacy-controller replay
    <= F: best-known feasible future-information schedule in the exact simulator
    <= U: dual-certified delivery ceiling of the encoded continuous relaxation
```

`F` is selected with complete future truth and is not deployable. `U` is a formal bound
for the serialized finite-box linear program with a numerically validated—not bit-exact
certified—bridge to the Python simulator. Neither is an MPC or evidence that a learned
policy works. A fourth value, `E`, is the optimum of a *restricted* relaxation that must
restore initial battery, H2, CO2, and CH4 endowment. `E` is an anti-liquidation diagnostic
and is not a task upper bound.

The complete compact evidence package is
[`results/v2_feasible_planner_matrix_564002c.json`](results/v2_feasible_planner_matrix_564002c.json).
It retains every cell's artifact identities, source boundaries, task and runtime
identities, exact scalar bracket, complete A0 and selected metrics, deadline settlements,
terminal state, search counters, replay hashes, and all 288 selected action records.

## Frozen execution and integrity

The repaired matrix ran in the declared rate-major, seed-minor order from clean revision
`564002cd9a49761b7eb78a84424d5325e5b240c6`. All six parent artifacts report:

- `success=true`, `evidence_ready=true`, and `runner_failure=null`;
- byte-identical clean start/end source snapshots with source payload SHA-256
  `7d54ea4fb59310a19b72279318bf9af200dfc85117720eb2321e9e42b300cade`;
- worktree content identity
  `bd64e058d00659e7675519ad83142110bd04ff774ca9ae37a962e4dc74bc1ec1`;
- raw `uv.lock` SHA-256
  `080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd`;
- CPython 3.12.7, NumPy 2.5.2, SciPy 1.18.1, linked HiGHS 1.12.0,
  Gymnasium 1.3.0, and Darwin arm64; and
- the frozen action library digest
  `805bc4a6203026783a0d2d6514d25c83188d9ca4476b7b9c6c7763b6718456f2`.

Every exact schedule was replayed from a fresh reset through the ordinary safety guard and
one-minute plant path. Every cell completed 72 macro depths, 25,410 exact-simulator beam
edges, 1,440 coordinate trials, and a separate 288-decision finalist replay. Search took
3,559.0--3,780.3 seconds per cell (59.3--63.0 minutes) and peak reported RSS was
309.6--351.6 MB. This makes `F` a useful offline research reference, not an online control
method.

The compact report's domain-separated digest is
`33f7873b0fba1623d816109900f8a5f6a41929924f6dcf99bff5dedd856b1d64`;
its exact 1,158,332-byte file has raw SHA-256
`05bd4b66bf71c01847aad369fe3abbea99fe2efe38a5c15be49b131cb551065e`.
The report builder and endowment driver are retained under [`tools/`](tools/) with exact
source hashes `6696e09fdf9e05f1507d9356a47693a7c9fbe8b42cdaaac23c75eab5946c3bd9`
and `c98068b8deb380d04fe2e8817c79aa7313494062341068bb7e60886bd09eb3ea`,
respectively.

## Result

All values are kilograms delivered over one 72-hour episode. Service is the count of
fully met six-hour commitments. “Closed” is `(F - A0) / (U - A0)` and is undefined when
`A0 = U` within the frozen tolerance.

| rate kg/h | seed | A0 | F | U | F - A0 | U - F | closed | E | A0 service | F service |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 0 | 2520.000 | 2520.000 | 2520.000 | 0.000 | 0.000 | n/a | 2520.000 | 12/12 | 12/12 |
| 35 | 1 | 2347.973 | 2520.000 | 2520.000 | 172.027 | 0.000 | 100.00% | 2423.429 | 11/12 | 12/12 |
| 35 | 2 | 2520.000 | 2520.000 | 2520.000 | 0.000 | 0.000 | n/a | 2520.000 | 12/12 | 12/12 |
| 42 | 0 | 2552.208 | 3024.000 | 3024.000 | 471.792 | 0.000 | 100.00% | 2577.509 | 9/12 | 12/12 |
| 42 | 1 | 2382.421 | 2969.062 | 3024.000 | 586.641 | 54.938 | 91.44% | 2423.429 | 7/12 | 11/12 |
| 42 | 2 | 2773.894 | 3024.000 | 3024.000 | 250.106 | 0.000 | 100.00% | 2700.873 | 10/12 | 12/12 |

At 35 kg/h, A0 is already delivery-saturated on two seeds. The exact planner recovers the
entire 172.027 kg shortage on seed 1 and reaches all 36 commitments across the three
cells. This rate remains useful as a nominal case but is too easy to be the sole learning
benchmark.

At 42 kg/h, a material scheduling gap exists on every seed. Mean A0 delivery is
2,569.508 kg and mean `F` is 3,005.687 kg, a descriptive mean gain of 436.180 kg.
Commitment service improves from 26/36 to 35/36. `F` reaches `U` on seeds 0 and 2; on
seed 1 it reduces the 641.579 kg relaxed gap above A0 to 54.938 kg, closing 91.44%.
Across eligible cells, the planner closes 97.15% on average. These three inspected seeds
support an engineering decision only; they are not an estimate of field performance.

The deadline traces localize the remaining failures. The selected planner meets every
commitment except the last rate-42/seed-1 settlement, where it supplies 197.062 of
252 kg. A0 misses the final rate-35/seed-1 settlement by 172.027 kg and misses two to
five commitments in each rate-42 episode. Better temporal allocation, not merely a
different reward total, is therefore realizable through the existing guarded plant.

## Physical mechanism and objective limitation

The selected schedules use renewable energy far more aggressively. A0 curtails
3.25--6.44 MWh per episode; `F` curtails only numerical roundoff. `F` accepts
81.7--91.4 MWh of process energy and discharges 7.10--8.79 MWh from the battery, versus
72.5--82.5 MWh and about 2.0--2.1 MWh for A0. This is the operational source of the
delivery gain.

It is not free. Every selected schedule finishes with only 1.20--1.33 MWh in the 8 MWh
battery, close to the 15% reserve, whereas A0 finishes at 7.28--7.33 MWh. Every selected
schedule also makes four cold starts and one forced stop; A0 makes one cold start and no
forced stop. The current lexicographic objective ranks delivery before terminal endowment,
cycling, and throughput, so this behavior is valid for the frozen task but not yet a
sustainable operating policy.

The endowment-preserving relaxation makes that horizon effect explicit:

| rate kg/h | seed | E kg | E / requested |
| ---: | ---: | ---: | ---: |
| 35 | 0 | 2520.000 | 100.00% |
| 35 | 1 | 2423.429 | 96.17% |
| 35 | 2 | 2520.000 | 100.00% |
| 42 | 0 | 2577.509 | 85.24% |
| 42 | 1 | 2423.429 | 80.14% |
| 42 | 2 | 2700.873 | 89.31% |

The restricted relaxation restores full endowment in two of three 35 kg/h cells and no
42 kg/h cell. Mean restricted fill is 98.72% and 84.90%, respectively. Because `E` solves
a different, smaller feasible set, it may be below A0 when A0 itself spends endowment;
that is not a bracket violation. The finding is that most of the apparent rate-42
short-horizon delivery gain cannot be interpreted as indefinitely repeatable production.

The action contract is expressive enough to realize the demonstrated schedules, but not
cleanly transparent. Selected requests differ from applied process power by
38.8--45.5 MWh after routing, commitment, and guard effects. A causal optimizer must model
the accepted plant action rather than optimize request shares in isolation. If that mapping
proves too lossy or numerically awkward, the next interface change should expose bounded
absolute setpoints and explicit mode intent while preserving the independent safety layer.

## Instrumentation disclosures

The original frozen revision `0059626313866d7a2bda2a44f64926a757a9f5d6` produced one
valid failed rate-35/seed-0 artifact before the binary64 repair. Its embedded digest is
`9323d4c97205706d7c8a8c03f14734bc502f1056386d2ca7378b601a9b456ed1` and its
2,182,072-byte raw file digest is
`e1f621689c176ccd50b5433a2df14d41c963f11618f70be5eb3af4821a5f95d7`.
It remains a failed invocation, not one of the six repaired results.

The first endowment-driver invocation completed the rate-35/seed-0 solve but then failed
before publication because its wrapper passed a new custom domain to the schema's closed
`CanonicalPayloadV1` registry. No numeric result or artifact was exposed and the other five
solves did not start. That driver has SHA-256
`41ee77760fef6fdb40628fcdb09084b87d8638ca771a0a224d04819e9c2c6159`.
Only the wrapper hash path was corrected, yielding the retained driver identity above;
all six diagnostics were then run without selection on the unseen first value. This
duplicate solver call is disclosed rather than silently erased.

The 12 large source artifacts currently remain outside Git at
`/private/tmp/plantanagent-planner-564002c`; their embedded and raw identities are in the
committed compact report. That path is temporary custody, not durable archival storage.
They must be copied to permanent research storage before the temporary directory is
cleaned. The compact Git artifact deliberately retains selected schedules and evidence
identities rather than six redundant full LP witnesses.

## Decision

The development result passes the controllability entry gate for a causal service MPC:

1. **Physical headroom exists.** `U > A0` in four cells, including every 42 kg/h cell.
2. **The exact action space can realize better behavior.** `F > A0` in those same four
   cells, with five of six `F` values reaching `U` and the sixth closing 91.44%.
3. **The remaining problem is causal capture.** A deployable controller must recover a
   useful share of this gain using only the current public forecast and observation.
4. **The current objective is not commercially complete.** It permits terminal resource
   liquidation and does not price wear, energy, product, or reserve capacity.

The next implementation is therefore a deterministic receding-horizon **service MPC**,
not immediate RL training and not further tuning of the rejected deadline heuristic. Its
pre-outcome contract must:

- receive only `ControllerObservationV2`, public issued forecast, static plant/constraint
  data, and controller-local history;
- solve a declared multi-period model with explicit status—never silently fall back under
  the MPC label;
- cover the next renewable window and intervening commitments, rather than only one
  deadline;
- represent inventory, battery reserve, electrolyser commitment/ramp state, delivery
  slack, guard-compatible power, and terminal resource value;
- report solve latency, feasibility, guard intervention, cycling, throughput, service,
  and capture of the cell-specific `F - A0` gap; and
- be tested for information isolation by varying private future truth behind an identical
  public view.

The current six cells remain development-contaminated and may be used to debug this MPC.
Before any uninspected outcome is opened, a prospective protocol must freeze regimes,
sites/seeds, demand levels, endpoints, minimum useful gain, failure handling, and all
tuning boundaries.

## Product direction after the MPC gate

This result supports a controller-neutral product thesis: Plant An Agent can become an
auditable virtual-commissioning and supervisory-optimization layer for renewable-fuel
plants. It does **not** yet support the claim that RL wins. The product value is the ability
to expose latent production/service headroom, validate safe plans against a digital twin,
and preserve bounded fallback behavior under uncertainty.

After the causal MPC is working, realism should advance in attributed, switchable slices:

1. a sustainable-service task with cyclic terminal inventories or calibrated terminal
   shadow values, so performance cannot be bought by emptying the plant at the horizon;
2. empirical multi-site renewable traces and causal forecast vintages, followed by
   coherent scenario/risk-aware MPC;
3. explicit prices, offtake penalties, grid interaction, water/compression and quality
   constraints, and calibrated wear/degradation costs;
4. stochastic faults, maintenance choices, sensor noise/drift, and multiple shared units;
5. equal-information RL only where nonlinear control quality, scenario complexity, or
   solver latency leaves a measured residual opportunity.

HEPA then occupies a distinct predictive role: estimate calibrated fault, degradation,
and constraint-breach risk over useful lead times and expose it to the supervisor or plan
scorer. Natural language belongs at the supervisory/operator boundary: translate approved
goals and constraints into versioned machine contracts, explain plans and incidents, and
request human decisions. Neither HEPA nor language bypasses the safety guard or directly
allocates plant power. Their value must be established later through controller-only,
controller-plus-supervisor, and controller-plus-supervisor-plus-HEPA ablations.
