# Causal controller development results

Date: **2026-09-02**

Status: **complete preregistered development comparison; negative controller result**

Evidence scope: **three inspected development seeds per load; not held out, not an
RL/MPC comparison, and not commercial-performance evidence**

## Decision in one sentence

Reject the current deadline-closing delivery-planner family as a production baseline:
it changes every legacy decision, but it controls the stateful plant less effectively
than the simpler continuously operating heuristic. Build a trajectory-level model-based
benchmark and a separately labelled future-information feasible planner before training
or claiming value from RL.

## Frozen execution identity

The final A0--A3 implementation and protocol were committed before either final 72-hour
artifact was generated. Both runs began and ended from the same clean worktree.

| item | value |
|---|---|
| Git revision | `f6cb663c9c3aa2086f1639f88725031b2d010b7e` |
| worktree identity | `219a467df789795b0f5c528369e857c91c74caaf1d13405ca0596012e2bf098b` |
| dependency-lock SHA-256 | `9f77f6c33eb5bd71f8c8c5a49da66422463f6ca0825a4d46f946ec2fbd51d708` |
| runtime | CPython 3.12.7 on macOS arm64 |
| environment | `plant-env-v2-commitment-alpha`, spec 3 |
| plant simulator | 0.5.2 |
| commitment dynamics | `electrolyser-commitment-alpha-v3` |
| horizon and cadence | 72 h; 15-minute decisions; one-minute physics |
| development cells | 35 and 42 kg CH4/h; seeds 0, 1, 2 |
| matrix size | 2 rates x 3 seeds x 4 arms = 24 rollouts |

The commands were:

```text
python -m plant_backend.rl.v2.compare --seed-start 0 --count 3 --hours 72 \
  --delivery-rate 35 --json /private/tmp/plantanagent-v2-causal-35kgph.json
python -m plant_backend.rl.v2.compare --seed-start 0 --count 3 --hours 72 \
  --delivery-rate 42 --json /private/tmp/plantanagent-v2-causal-42kgph.json
```

The full strict-JSON artifacts remain outside Git because their 3,456 requested-action
rows and duplicated immutable provenance make them 1.4 MB each. Their identities are:

| cell | bytes | SHA-256 |
|---|---:|---|
| 35 kg/h | 1,411,840 | `2e1200c397c2aa5b23762a551d949e3787fd1bec5a73e7ee183fcd2d7b99fd55` |
| 42 kg/h | 1,426,064 | `e7b27f0269b2097b92bace6f435821b1088d8dc7548713546a65625cf7114de1` |

Both source audits are valid. All 24 planned rollouts succeeded. Each rate has exactly one
run for each declared seed/arm pair, the full and exogenous manifest identities match
within seed, and every predeclared adjacent contrast is valid. No seed was removed and no
post-outcome contrast was added.

## Arms

| arm | controller | information treatment |
|---|---|---|
| A0 | legacy heuristic | latest boundary telemetry only; no delivery context or forecast |
| A1 | delivery-aware fixed climatology | delivery context and frozen synthetic-prior resource expectation |
| A2 | issued mean forecast | A1 planner with current public aggregate mean issuance |
| A3 | risk-adjusted issued forecast | A2 planner with the preregistered 50% mean / 50% p10 mixture |

A1--A3 use the same planner and differ only in the declared availability input. This
supports attribution among those three implementations. A0 deliberately preserves the
historical behavior and has a narrower information set, so A1--A0 is a practical
controller comparison rather than a pure single-variable forecast contrast.

## Aggregate scorecard

Values are population means over three development seeds. `kWh/kg` is accepted
electrolyser + DAC + methanator energy divided by delivered methane. It must be read with
service and inventory changes, not as a stand-alone efficiency score.

### 35 kg/h primary development cell

| arm | fill | service | worst fill | unmet kg | CH4 produced kg | terminal H2 kg | terminal CO2 kg | terminal SOC | process kWh/kg | curtailed kWh | cold / forced starts-stops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 97.72% | 97.22% | 72.69% | 57.3 | 2,355.5 | 0.0 | 4,253.1 | 91.40% | 31.16 | 4,360.9 | 1.0 / 0.0 |
| A1 | 84.55% | 75.00% | 14.00% | 389.2 | 1,952.5 | 76.8 | 316.6 | 15.21% | 28.82 | 24,441.0 | 11.7 / 5.3 |
| A2 | 84.75% | 75.00% | 29.96% | 384.2 | 1,957.5 | 60.1 | 201.6 | 15.67% | 28.40 | 25,191.7 | 11.7 / 4.7 |
| A3 | 86.94% | 75.00% | 23.13% | 329.1 | 2,012.6 | 51.2 | 157.7 | 15.45% | 28.18 | 24,159.2 | 11.7 / 4.7 |

### 42 kg/h development stress cell

| arm | fill | service | worst fill | unmet kg | CH4 produced kg | terminal H2 kg | terminal CO2 kg | terminal SOC | process kWh/kg | curtailed kWh | cold / forced starts-stops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 84.97% | 72.22% | 14.63% | 454.5 | 2,355.5 | 0.0 | 4,253.1 | 91.40% | 29.89 | 4,360.9 | 1.0 / 0.0 |
| A1 | 81.43% | 72.22% | 11.66% | 561.5 | 2,248.6 | 17.8 | 221.5 | 15.61% | 27.12 | 19,494.5 | 9.0 / 5.0 |
| A2 | 81.62% | 72.22% | 12.29% | 555.8 | 2,254.2 | 16.3 | 169.7 | 15.79% | 27.09 | 19,344.1 | 10.0 / 4.7 |
| A3 | 81.59% | 72.22% | 7.07% | 556.8 | 2,253.2 | 17.3 | 248.7 | 15.61% | 27.09 | 19,508.9 | 9.7 / 6.3 |

## Predeclared paired effects

Raw deltas are treatment minus control. Percentage-point values are not relative
percentages. Negative unmet and curtailment deltas favor treatment; lower `kWh/kg` is not
interpretable as a benefit when service falls and stored endowments are liquidated.

### 35 kg/h

| contrast | fill pp | service pp | worst-fill pp | unmet kg | curtailment kWh | process kWh/kg | cold starts | forced stops | unequal requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 - A0 | -13.171 | -22.222 | -58.698 | +331.90 | +20,080.0 | -2.335 | +10.67 | +5.33 | 864 / 864 |
| A2 - A1 | +0.201 | 0.000 | +15.966 | -5.07 | +750.8 | -0.423 | 0.00 | -0.67 | 694 / 864 |
| A3 - A2 | +2.184 | 0.000 | -6.828 | -55.04 | -1,032.5 | -0.224 | 0.00 | 0.00 | 711 / 864 |

### 42 kg/h

| contrast | fill pp | service pp | worst-fill pp | unmet kg | curtailment kWh | process kWh/kg | cold starts | forced stops | unequal requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 - A0 | -3.538 | 0.000 | -2.977 | +106.98 | +15,133.6 | -2.766 | +8.00 | +5.00 | 864 / 864 |
| A2 - A1 | +0.187 | 0.000 | +0.637 | -5.65 | -150.4 | -0.025 | +1.00 | -0.33 | 768 / 864 |
| A3 - A2 | -0.034 | 0.000 | -5.229 | +1.02 | +164.8 | -0.002 | -0.33 | +1.67 | 769 / 864 |

The A2 treatment was expressed in 80.3% and 88.9% of matched decisions at 35 and
42 kg/h. A3 was expressed in 82.3% and 89.0%. The forecast arms therefore did not fail
because their inputs were accidentally ignored.

## Complete per-episode scorecard

This table retains every planned run. `H2`, `CO2`, and `CH4` are terminal kilograms;
`SOC` is terminal battery state of charge; `nonprod` is electrolyser startup plus standby
energy. Initial endowments are reported in the following table.

| rate | arm | seed | fill | service | worst | unmet kg | produced kg | H2 | CO2 | CH4 | SOC | kWh/kg | curtail kWh | discharge kWh | cold | forced | nonprod kWh |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | A0 | 0 | 100.00% | 100.00% | 100.00% | 0.0 | 2,394.6 | 0.0 | 4,366.3 | 5.9 | 91.60% | 29.87 | 6,443.9 | 2,014.5 | 1 | 0 | 3,904 |
| 35 | A1 | 0 | 82.64% | 75.00% | 14.20% | 437.4 | 1,951.2 | 84.5 | 413.7 | 0.0 | 15.15% | 28.29 | 27,438.2 | 22,372.5 | 11 | 5 | 5,966 |
| 35 | A2 | 0 | 78.44% | 75.00% | 11.60% | 543.4 | 1,845.2 | 96.3 | 425.4 | 0.0 | 15.75% | 28.44 | 30,094.6 | 22,339.3 | 12 | 5 | 6,023 |
| 35 | A3 | 0 | 86.94% | 75.00% | 17.29% | 329.2 | 2,059.4 | 37.0 | 113.9 | 0.0 | 15.00% | 27.11 | 27,005.5 | 21,971.3 | 11 | 4 | 6,166 |
| 35 | A0 | 1 | 93.17% | 91.67% | 18.08% | 172.0 | 2,175.7 | 0.0 | 4,734.5 | 0.0 | 91.02% | 30.88 | 3,385.7 | 2,094.4 | 1 | 0 | 3,765 |
| 35 | A1 | 1 | 85.05% | 75.00% | 13.84% | 376.8 | 1,970.9 | 49.6 | 191.0 | 0.0 | 15.10% | 28.29 | 20,247.4 | 17,822.0 | 11 | 6 | 5,326 |
| 35 | A2 | 1 | 87.50% | 75.00% | 37.72% | 314.9 | 2,032.8 | 17.6 | 12.0 | 0.0 | 16.07% | 27.48 | 20,277.7 | 17,156.3 | 11 | 5 | 5,349 |
| 35 | A3 | 1 | 87.57% | 75.00% | 37.27% | 313.2 | 2,034.6 | 20.4 | 12.0 | 0.0 | 15.98% | 27.44 | 20,296.5 | 17,422.9 | 11 | 5 | 5,186 |
| 35 | A0 | 2 | 100.00% | 100.00% | 100.00% | 0.0 | 2,496.3 | 0.0 | 3,658.4 | 207.6 | 91.58% | 32.72 | 3,253.1 | 2,028.7 | 1 | 0 | 3,612 |
| 35 | A1 | 2 | 85.97% | 75.00% | 13.96% | 353.5 | 1,935.2 | 96.3 | 344.9 | 0.0 | 15.38% | 29.89 | 25,637.3 | 21,727.4 | 13 | 5 | 6,200 |
| 35 | A2 | 2 | 88.33% | 75.00% | 40.57% | 294.2 | 1,994.5 | 66.5 | 167.4 | 0.0 | 15.18% | 29.29 | 25,202.9 | 21,861.1 | 12 | 4 | 6,647 |
| 35 | A3 | 2 | 86.31% | 75.00% | 14.84% | 345.0 | 1,943.7 | 96.3 | 347.1 | 0.0 | 15.38% | 29.98 | 25,175.7 | 21,777.8 | 13 | 5 | 6,482 |
| 42 | A0 | 0 | 84.40% | 75.00% | 0.00% | 471.8 | 2,394.6 | 0.0 | 4,366.3 | 0.0 | 91.60% | 29.49 | 6,443.9 | 2,014.5 | 1 | 0 | 3,904 |
| 42 | A1 | 0 | 81.56% | 75.00% | 11.71% | 557.6 | 2,308.8 | 13.6 | 418.7 | 0.0 | 16.49% | 26.61 | 21,318.2 | 14,094.6 | 9 | 5 | 4,755 |
| 42 | A2 | 0 | 82.03% | 75.00% | 11.10% | 543.3 | 2,323.1 | 0.0 | 284.6 | 0.0 | 16.58% | 26.45 | 21,326.6 | 14,058.3 | 11 | 6 | 5,283 |
| 42 | A3 | 0 | 82.10% | 75.00% | 7.60% | 541.4 | 2,325.0 | 0.0 | 478.6 | 0.0 | 15.54% | 26.51 | 21,313.9 | 12,976.5 | 11 | 7 | 4,968 |
| 42 | A0 | 1 | 78.78% | 58.33% | 0.00% | 641.6 | 2,175.7 | 0.0 | 4,734.5 | 0.0 | 91.02% | 30.44 | 3,385.7 | 2,094.4 | 1 | 0 | 3,765 |
| 42 | A1 | 1 | 78.01% | 66.67% | 8.75% | 665.0 | 2,152.3 | 2.1 | 193.6 | 0.0 | 15.31% | 26.90 | 17,795.4 | 13,302.9 | 10 | 7 | 4,777 |
| 42 | A2 | 1 | 78.39% | 66.67% | 11.88% | 653.6 | 2,163.7 | 2.3 | 199.3 | 0.0 | 15.74% | 26.87 | 17,497.1 | 13,582.2 | 11 | 7 | 4,720 |
| 42 | A3 | 1 | 78.43% | 66.67% | 1.25% | 652.2 | 2,165.1 | 0.0 | 215.4 | 0.0 | 16.09% | 26.78 | 17,825.7 | 11,453.1 | 11 | 8 | 4,489 |
| 42 | A0 | 2 | 91.73% | 83.33% | 43.90% | 250.1 | 2,496.3 | 0.0 | 3,658.4 | 0.0 | 91.58% | 29.73 | 3,253.1 | 2,028.7 | 1 | 0 | 3,612 |
| 42 | A1 | 2 | 84.73% | 75.00% | 14.51% | 461.9 | 2,284.6 | 37.8 | 52.1 | 0.0 | 15.01% | 27.85 | 19,370.1 | 18,096.7 | 8 | 3 | 5,079 |
| 42 | A2 | 2 | 84.44% | 75.00% | 13.90% | 470.6 | 2,275.8 | 46.6 | 25.2 | 0.0 | 15.05% | 27.96 | 19,208.7 | 19,483.1 | 8 | 1 | 4,838 |
| 42 | A3 | 2 | 84.23% | 75.00% | 12.35% | 477.0 | 2,269.5 | 52.0 | 52.0 | 0.0 | 15.19% | 27.99 | 19,387.1 | 18,551.3 | 7 | 4 | 4,572 |

### Initial endowments and forcing totals

These values are shared by all four arms within each rate/seed pair. CH4 differs by rate
because the initial-coverage distribution is expressed in hours of the contract rate.

| rate | seed | initial H2 kg | initial CO2 kg | initial CH4 kg | initial SOC | solar kWh |
|---:|---:|---:|---:|---:|---:|---:|
| 35 | 0 | 258.2 | 1,900.2 | 131.4 | 62.72% | 84,290.4 |
| 35 | 1 | 187.2 | 1,761.2 | 172.2 | 52.96% | 79,246.1 |
| 35 | 2 | 173.3 | 1,384.7 | 231.3 | 62.43% | 88,318.5 |
| 42 | 0 | 258.2 | 1,900.2 | 157.7 | 62.72% | 84,290.4 |
| 42 | 1 | 187.2 | 1,761.2 | 206.7 | 52.96% | 79,246.1 |
| 42 | 2 | 173.3 | 1,384.7 | 277.6 | 62.43% | 88,318.5 |

## Mechanism diagnosis

The result is not “forecasts have no value.” It is “this stateless deadline planner is
the wrong base policy for exposing forecast value in this stateful plant.” Five observed
mechanisms support that distinction.

1. **The treatment acted.** A1 differs from A0 at all 1,728 matched requests across the
   two cells; A2 and A3 also change most requests relative to their direct controls. There
   is no dormant-input explanation.
2. **The planner repeatedly loses commitment state.** A0 makes one cold start and no
   forced stop in every episode. The delivery-aware arms average 9.0--11.7 cold starts and
   4.7--6.3 forced stops, spend 4.68--6.01 MWh on startup/standby, and operate the stack
   for only 19.6--21.7 h on average versus 23.4 h for A0.
3. **It plans just in time, then pays for uncertainty and nonconvexity.** The planner
   closes the current mass deficit against equivalent forecast hours, but it does not
   optimize a sequence of on/off decisions, minimum-up/down state, buffer opportunity
   cost, or forecast recourse. Small boundary revisions therefore become repeated starts
   and stops rather than smooth schedule revisions.
4. **It liquidates flexibility while rejecting renewable energy.** A1--A3 end near the
   15% battery reserve instead of A0's 91%, discharge 11.5--22.4 MWh per episode, and yet
   curtail 17.5--30.1 MWh. Accepted process energy falls chiefly because DAC energy falls;
   the resulting feedstock balance is fragile. At 35 kg/h the planners end with H2 but
   almost no CO2, a direct signature of stranded intermediate inventory.
5. **A lower intensity ratio is not a win.** A1--A3 use 2.3--3.0 fewer accepted kWh per
   delivered kg than A0 on average, but they deliver less product after consuming roughly
   6 MWh more initial battery energy and about 4 t more initial/produced CO2 inventory.
   The ratio rewards endowment liquidation unless service and the full balance vector are
   considered simultaneously.

The stored action traces and an exact replay make the horizon defect concrete. In the
35 kg/h cell, A1--A3 miss exactly commitments 4, 8, and 12 on every seed: the three
settlements at local 06:00 immediately before solar returns. During the first six elapsed
hours (local 06:00--12:00), every delivery-aware arm requests zero electrolyser share,
zero DAC share, and zero battery support on all 24 decisions of every seed because the
initial feedstocks appear sufficient for the first deadline. During the first 12 hours,
the mean episode supplies 28.620 MWh solar. A0 accepts 25.753 MWh of process energy and
curtails only 0.174 MWh; A1 accepts 2.867 MWh and curtails 22.367 MWh. That first-half-day
curtailment is 91.5% of A1's complete 72-hour curtailment.

This follows from the implemented target, not forecast quality. At 35 kg/h one delivery
is 210 kg, and the largest planner target is only the current 210 kg, a 3% buffer, and one
210 kg inferred carry: 426.3 kg. The plant has 3,000 kg CH4 storage, while the interval
from the productive solar window to the next sunrise spans three settlements. The planner
therefore fills roughly two obligations, curtails the remaining solar, recognizes the
third obligation too late, and cannot use battery support for DAC or methanation
overnight. Its forecast queries end at the current deadline plus one six-hour period, so
the declared 12--24-hour band cannot affect an action in this task.

A post-outcome read-only diagnostic replay checked whether the simulator was often
curtailing solar in the exact same one-minute step in which it discharged the battery.
The overlap lower bound was only 0.7--38.0 kWh per rollout versus 11.5--22.4 MWh total
planner discharge and 17.5--30.1 MWh curtailment. The dominant defect is the short
planning horizon, compounded by commitment cycling and battery/feedstock coordination,
not a same-step charge/discharge-netting artifact.

## Hypothesis decisions

- **A1 over A0:** rejected in both development cells. A1 has lower mean fill in every one
  of the six paired episodes, with materially worse inventory, cycling, and curtailment
  behavior.
- **A2 over A1:** not supported. Mean forecast information changes actions but improves
  mean fill by only 0.20 and 0.19 percentage points. The sign of unmet-mass benefit varies
  by seed, and service level is unchanged.
- **A3 over A2:** not supported. It improves mean fill by 2.18 points at 35 kg/h, driven
  mainly by seed 0, while worsening worst-commitment fill by 6.83 points. At 42 kg/h its
  mean effect is essentially zero and forced stops increase.

Three seeds are too few for inferential claims. These decisions are engineering gates for
the inspected development task only.

## Direction unlocked by this result

The next controller must reason over a trajectory. The staged benchmark is:

1. retain A0 as the minimum credible operational baseline;
2. add an offline, explicitly future-informed **feasible planner** against the exact
   simulator to show whether materially better action sequences exist (a best-known
   feasible reference, never mislabeled as a mathematical upper bound);
3. add deterministic receding-horizon optimization using only the same current public
   forecast available to A2, with explicit commitment, inventory, energy-balance,
   delivery, and terminal-value terms;
4. add scenario/risk-aware MPC after the deterministic model is validated;
5. train RL only against the same controller observation and safety door, then report its
   gap to A0, MPC, and the future-information feasible reference;
6. advance the commercial environment in parallel with calibrated price/offtake,
   maintenance, degradation, grid/carbon, and quality constraints, but do not stack those
   additions before each one has a discriminative baseline and balance audit.

The project claim remains controller-neutral: Plant An Agent is developing an auditable,
safety-bounded supervisory optimization and virtual-commissioning layer for renewable
fuel plants. If MPC is the best controller, that is the product. RL earns a role only if
it closes a measured residual gap under equal information and survives held-out stress.
