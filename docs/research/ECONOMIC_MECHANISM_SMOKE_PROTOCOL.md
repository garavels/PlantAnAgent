# Economic task mechanism smoke: preregistration

Status: **frozen before either six-hour controller trajectory is opened.**

## Question

Does `plant-env-v2-economic-wear-alpha` produce an auditable nonzero economic separation
between the existing common-door heuristic and current service MPC when physical safety and
delivery are equal? This is a task-expression test, not an RL experiment or a controller
superiority study.

The prior 72-hour setpoint mechanism cell tied at 2,520 kg delivery while MPC used more energy
and battery throughput. This smoke checks whether the new ledger can represent that kind of
operational difference on a shorter fresh execution before resources are spent on a multi-seed
tournament.

## Frozen cell

| field | value |
| --- | --- |
| plant | default `PlantConfig` |
| constraints | default `Constraints` |
| environment | `plant-env-v2-economic-wear-alpha` |
| economic assumptions | default `EconomicAssumptionsV1` |
| diagnostic legacy reward | default `RewardWeightsV2`, objective version 3 |
| horizon | 6 hours |
| start | day 1, 06:00 |
| decision interval | 900 seconds |
| delivery | one 210 kg commitment at hour 6 (35 kg/h) |
| inventory/weather distribution | otherwise default `EpisodeConfigV2` |
| seed | 0 |
| expected decisions | 24 per arm |
| expected physical minutes | 360 per arm |

Six hours are long enough to include one actual settlement, an alkaline cold-start horizon,
falling morning-to-midday prices, storage movement, and the complete endowment liability. It
is intentionally much shorter than the earlier 72-hour mechanism run.

## Frozen arms

### H — common-door legacy heuristic

`LegacyHeuristicSetpointControllerV1` receives only the physical
`ControllerObservationV2`, active constraints, and plant configuration. Its source shares are
converted transparently to four held absolute targets and executed through
`ProcessSetpointCommandV1` plus the generic minute ledger. It receives no tariff, forecast,
delivery context, or economic coefficient.

### M — current causal service MPC

`ServiceMPCSetpointControllerV1` uses `ServiceMPCSetpointSettingsV1` with:

```text
task_horizon_h = 6
planning_horizon_h = 6
interval_s = 900
all other settings unchanged
```

It emits absolute setpoints and retains its rich MPC decision, solve, application, and
transition evidence. The environment's MPC execution door applies the same economic overlay
as the common command door. The controller remains the existing delivery-oriented MPC: it
does not optimize the new tariff or economic ledger and must not be called economic MPC.

This asymmetry is intentional for the mechanism question. Both arms ignore tariff when
choosing actions; the experiment tests whether the ledger distinguishes the physical paths
they already produce. A later economic MPC must consume `EconomicControllerObservationV1`
and receive a new controller/objective identity.

## Execution and failure policy

- The protocol and runner must be committed before execution.
- The worktree must be clean; source revision, dependency-lock digest, Python/runtime
  versions, environment/checkpoint/assumption identities, protocol digest, and output hash
  are recorded.
- Both arms use the same seed and must have identical physical exogenous and economic episode
  digests before either first action.
- H executes through `step_process_setpoints`; M executes through `step_setpoints` so its
  controller can reconcile the complete rich execution evidence. Both paths delegate to the
  same physical allocation kernel and guard.
- Arms run in independent fresh environment/controller instances. H runs first, then M.
- One execution is authorized for the committed revision. An exception or incomplete arm is
  retained as an engineering failure. No same-revision retry or parameter adjustment is
  permitted.
- The artifact is written outside Git with exclusive creation. A compact result may be added
  to Git only after the raw artifact hash is known.

## Predeclared gates

### E1 — complete execution

Both arms must complete exactly 24 decisions and 360 physical minutes, terminate normally,
and report no controller exception or unreconciled MPC transition.

### E2 — physical safety and authority

Both arms must report:

- zero shutdown steps;
- zero out-of-range decisions;
- zero forced stops;
- exactly 24 interval execution records and 360 minute applications;
- command, observation, and execution chains accepted by their typed constructors;
- for M, a complete deep `build_mpc_setpoint_run_evidence` result.

Guard intervention is reported, not required to be zero, because it may reflect the known
independent safety projection rather than a hard violation.

### E3 — comparable contracted service

Delivered and unmet methane must match between H and M within `1e-6 kg`. If they differ, the
economic ranking is confounded by service and E3 fails. The result remains useful but cannot
establish a clean cost tie-break.

### E4 — task expression

Conditional on E1--E3, absolute net-operating-value separation must exceed 1 EUR and at least
one of these absolute physical differences must be material:

- accepted solar energy > 1 kWh;
- battery throughput > 1 kWh;
- electrolyser equivalent operating exposure > 0.01 h;
- final CH4-equivalent deficit > `1e-3 kg`;
- final battery deficit > 0.1 kWh.

This is a minimum numerical-expression gate, not a commercial materiality threshold.

### E5 — accounting closure

For each arm:

- interval rewards sum to episode net operating value within `1e-8 EUR`;
- signed component totals sum to episode value within `1e-8 EUR`;
- delivered/unmet, accepted solar, battery throughput, and equipment exposure agree between
  interval and episode ledgers;
- cumulative methane and battery liability-change components equal the negative final-deficit
  liabilities within `1e-8 EUR`;
- physical exogenous, checkpoint, tariff, assumption, and economic episode identities match
  the frozen cell.

### E6 — interpretation boundary

If E1--E5 pass, the higher-value arm is reported descriptively with the exact component and
physical-driver decomposition. That is evidence the task is discriminative on one development
cell, not that the arm is generally better. If E3 or E4 fails, the next action is task diagnosis,
not RL training. If E1, E2, or E5 fails, the next action is engineering repair under a new
revision.

## Explicit exclusions

This smoke does not:

- tune any economic coefficient, heuristic rule, or MPC weight;
- train, select, or evaluate an RL policy;
- use a validation, held-out, or protected stress seed;
- establish statistical significance, tail risk, or generalization;
- test tariff-aware optimization;
- estimate plant savings, LCOCH4, degradation, or project return;
- evaluate HEPA, supervision, natural language, or deployment latency.

Only a clean E1--E5 pass permits a separately frozen multi-seed economic development
tournament. RL training remains downstream of that tournament's demonstrated residual gap.
