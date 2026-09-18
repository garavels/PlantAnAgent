# Economic/wear task robustness tournament: preregistration

Status: **frozen before any seed 20--24 controller trajectory is opened**

Date: **2026-09-04**

## Decision question

Does `plant-env-v2-economic-wear-alpha` distinguish materially different safe controllers
consistently across fresh development seeds, demand pressure, and episode horizon—and does that
distinction survive removal of the terminal methane-liability term that dominated the first
mechanism cell?

This is a task-robustness tournament, not an RL experiment or a controller-superiority study.
Its purpose is to decide whether the environment is mature enough to spend compute on a bounded
development PPO baseline and to justify implementation of a tariff-aware economic MPC comparator.

The admitted seed-0 smoke established only one-cell task expression. H and M delivered the same
210 kg, but M was 382.532 EUR higher under the alpha ledger. Avoided terminal methane liability
contributed 543.016 EUR—142% of the net gap—while additional battery and electrolyser reserves
offset 161.846 EUR. This tournament is designed to distinguish persistent operating headroom
from a seed-specific or endpoint-specific effect.

## Evidence and seed custody

- Seeds 0--3 have been directly inspected during earlier controller development.
- Seeds 0--19 have appeared in earlier Environment V2 baseline work and are not eligible as
  fresh evidence under the current generator.
- Seeds **20, 21, 22, 23, and 24** are frozen here as fresh *development* seeds. No trajectory,
  reset manifest, sampled inventory, weather realization, or controller result for these seeds
  is to be inspected before the protocol and runner are committed.
- Opening these seeds consumes them for development. They may never be relabelled as formal
  validation, held-out, site-generalization, or prospective evidence.
- Five seeds support an engineering go/no-go decision only. No p-value, confidence interval,
  stable tail estimate, or population claim is permitted.

Future formal evaluation must use separately frozen seeds and, more importantly, unseen demand,
weather, equipment, tariff, and plant-parameter regimes.

## Frozen task matrix

All cells use:

- default `PlantConfig`;
- default `Constraints`;
- `plant-env-v2-economic-wear-alpha`;
- default `EconomicAssumptionsV1` and its synthetic 24-hour tariff;
- default diagnostic `RewardWeightsV2` under objective version 3;
- start hour 06:00;
- 15-minute controller decisions and one-minute physics;
- six-hour recurring methane delivery commitments; and
- otherwise default `EpisodeConfigV2` inventory and weather distributions.

The Cartesian matrix is:

| Axis | Frozen values |
|---|---|
| episode horizon | 6 h, 24 h |
| delivery rate | 35 kg/h, 42 kg/h |
| seed | 20, 21, 22, 23, 24 |
| controller | H, M |

This creates **20 paired cells and 40 controller trajectories**. A six-hour arm must contain
24 decisions and 360 minute applications; a 24-hour arm must contain 96 decisions and 1,440
minute applications. Across the complete matrix, 2,400 controller decisions and 36,000 minute
applications are expected.

The 6-hour strata test replication of the original endpoint geometry on unseen development
episodes. The 24-hour strata include four settlements and a full tariff cycle, reducing—but not
eliminating—single-deadline and time-of-day endpoint sensitivity. Rate 35 is the calibrated
primary development load. Rate 42 is the previously selected commitment-stress load, not an
unseen regime.

## Frozen controller arms and information sets

### H — common-door legacy heuristic

`LegacyHeuristicSetpointControllerV1` receives the physical `ControllerObservationV2`, active
constraints, and plant configuration. Its source shares are transparently converted to four
absolute targets and executed through `ProcessSetpointCommandV1`, the generic minute dispatcher,
the independent guard, and the plant. It receives no issued forecast, delivery context, tariff,
or economic coefficient.

### M — causal service MPC

`ServiceMPCSetpointControllerV1` receives the causal physical observation, issued forecast,
public recurring-delivery state, constraints, and plant configuration. For every cell:

```text
task_horizon_h = episode horizon (6 or 24)
planning_horizon_h = 6
interval_s = 900
all other ServiceMPCSetpointSettingsV1 fields unchanged
```

M emits absolute process setpoints, executes through the same allocation kernel and guard, and
retains the richer transition-reconciled MPC evidence graph. It receives no tariff or economic
coefficient and remains a delivery-oriented controller.

The arms intentionally have unequal information and internal complexity. Therefore a directional
H-versus-M value difference cannot establish algorithm superiority. They are fixed behavioral
probes of whether the task expresses operating trade-offs.

No controller rule, solver setting, objective coefficient, or action contract may be tuned after
any tournament trajectory is opened.

## Frozen execution and artifact policy

1. The protocol and dependency-lock raw-file SHA-256 values must be embedded in the runner.
2. The protocol, runner, verifier, and outcome-free tests must be committed before execution.
3. The runner requires a clean, completely identified source and an absolute new output
   directory outside the repository.
4. Both H and M environments for a cell are constructed and reset before either arm acts. Their
   manifest, physical exogenous episode, economic episode, checkpoint contract, tariff, and
   assumptions identities must match exactly.
5. Cell order is deterministic: horizon 6 then 24; within horizon, rate 35 then 42; within rate,
   seeds 20 through 24; within cell, H then M.
6. Cells run sequentially. CPU parallelism is forbidden because solver contention would make
   latency evidence uninterpretable; this is not a throughput benchmark.
7. Every cell is published exclusively as a separate JSON artifact immediately after both arms
   and gate calculation. A final compact matrix report references their raw and internal hashes.
8. An exception or incomplete arm is retained as failure. The matrix continues so missingness is
   explicit, but no surviving-subset aggregate may replace the declared matrix.
9. No same-revision cell retry, seed substitution, coefficient adjustment, or controller repair
   is allowed. An infrastructure failure before Python, output creation, environment reset, or
   controller action does not consume a cell, but it must be journalled.
10. Source and dependency identity are rechecked after the matrix. Source drift fails the whole
    tournament even when controller gates pass.

Wall-clock controller, solver, dispatcher, cell, and tournament times are retained but excluded
from deterministic evidence digests.

## Required per-arm evidence

Each arm must retain:

- controller identity and declared information set;
- complete manifest and exogenous identities;
- decision and physical-step counts;
- normal termination/truncation/failure state;
- reward trace and additive economic metrics;
- physical service, safety, energy, inventory, commitment, and wear-driver metrics;
- every interval execution and minute application identity; and
- inference/solver/dispatcher timing outside deterministic outcome hashes.

M must additionally pass `verify_mpc_setpoint_run_digests` over every decision, both solver
stages, transition reconciliation, target allocation, guard/application chain, and latency
ledger. H must retain its decision, command, execution, and aggregate evidence digests.

## Frozen algebraic sensitivities

Physical trajectories are run once under the default assumptions. Because the episode economic
ledger is additive, the report then computes outcome-blind algebraic counterfactuals from each
arm's recorded components; controllers are not rerun and no coefficient is fitted.

For each paired cell, report M-minus-H value under:

1. the complete default ledger;
2. methane-endowment multiplier `0`, `0.5`, `1`, and `2`;
3. joint battery-throughput plus electrolyser-life multiplier `0`, `0.5`, `1`, and `2`;
4. solar-opportunity-cost multiplier `0`, `0.5`, `1`, and `2`;
5. no methane-endowment liability;
6. no declared wear reserve (battery and electrolyser components removed); and
7. service plus solar opportunity cost only.

Every counterfactual is a sensitivity of the synthetic accounting assumptions, not a physical
counterfactual, causal effect, or calibrated range. Report all ranking flips; do not select the
variant that favors either arm.

## Frozen per-stratum summaries

For each `(horizon, delivery rate)` stratum and each required scalar metric, report:

- all five paired seed values in seed order;
- arithmetic mean and population standard deviation;
- median, minimum, maximum, and worst seed;
- count positive, negative, and within numerical tolerance;
- complete/missing pair count; and
- for service, count satisfying the `1e-6 kg` comparability tolerance.

Primary paired metrics are:

- net operating value and every EUR component;
- delivered and unmet methane;
- accepted solar and process energy;
- curtailed energy;
- battery throughput, equivalent full cycles, and terminal energy;
- electrolyser equivalent exposure, cold/hot/aborted/forced starts, and operating time;
- terminal CH4, H2, CO2, and CH4-equivalent deficit;
- guard interventions, corrections, shutdown, and out-of-range decisions; and
- controller/solver/dispatcher latency, reported separately from performance.

No aggregate may mix horizons or delivery rates. A complete overall table may be shown only
after all four strata and all missing cells are visible.

## Frozen gates

### R1 — complete declared execution

Pass only if all 20 cells and 40 arms are present, every arm reaches its exact decision/minute
count and normal termination, all M transitions close, all expected M solver stages succeed,
no fallback occurs, and the source gate passes. Any missing or failed arm fails R1; no
surviving-subset result is eligible for later gates.

### R2 — identity and accounting closure

Pass only if every cell has exact H/M manifest and exogenous identity equality, only declared
cells/controllers appear, artifact/digest graphs verify, reward traces equal episode net value,
components sum to net value, interval drivers equal episode drivers, and cumulative endowment
liabilities equal final-deficit liabilities within `1e-8 EUR`.

### R3 — physical safety and authority

Pass only if every arm reports zero shutdown steps, zero out-of-range decisions, and zero forced
electrolyser stops; exact interval/minute evidence counts; no command/chain/accounting violation;
and, for M, zero requested target overshoot, reachable-tracking violation, unattributed deficit,
or support-authority violation. Guard interventions and fully attributed plant refusals are
reported and may be nonzero.

### R4 — service-comparable coverage

A paired cell is service-comparable when absolute H/M differences in both delivered and unmet
methane are at most `1e-6 kg`. Pass only if at least four of five cells are comparable in each
of the four horizon/rate strata. Economic differences from non-comparable cells remain reported
but cannot support a clean operating-cost tie-break.

### R5 — repeated task expression

A cell expresses the task when it is service-comparable, absolute M-minus-H default net value
exceeds 1 EUR, and at least one registered raw driver exceeds its materiality threshold:

- accepted solar, process energy, or battery throughput: 1 kWh;
- electrolyser equivalent exposure: 0.01 h;
- terminal CH4-equivalent deficit: `1e-3 kg`; or
- terminal battery deficit: 0.1 kWh.

Pass only if at least four of five cells express the task in every horizon/rate stratum. The
higher-value arm may differ by seed; R5 tests discrimination, not direction.

### R6 — nonterminal expression

For each cell, remove only the methane-endowment-liability component from both arms. Pass only if
at least four of five cells in every stratum remain service-comparable, retain more than 1 EUR
absolute separation, and have a material nonterminal driver among solar opportunity, battery
throughput, or electrolyser exposure. This is the direct guard against declaring readiness from
the term that dominated seed 0.

### R7 — horizon robustness

Pass only if both 24-hour strata independently pass R1--R6 and contain all four scheduled
delivery settlements per arm. Six-hour success cannot compensate for 24-hour failure. Report,
without a direction threshold, how the paired default and no-methane value distributions change
between horizons.

### R8 — assumption and ranking sensitivity

Always report the higher arm and sign flips for every frozen multiplier and leave-one-component
counterfactual, by seed and stratum. R8 has status `reported`, not pass/fail: the multiplier grid
has no calibrated probability or commercial threshold. A result whose direction depends on one
synthetic term cannot be used as a controller-selection claim even when R1--R7 pass.

## Decision rule

- If R1, R2, or R3 fails, repair execution/evidence/safety under a new revision; do not train.
- If R4 fails, service confounding is too common for this two-arm tie-break; revise the cell or
  add a service-normalized analysis under a new protocol.
- If R5 fails, the task remains too coarse or saturated; add physical/economic decision pressure
  before training.
- If R6 or R7 fails, terminal geometry dominates or task expression does not survive a full day;
  repair the objective/horizon before training.
- If R1--R7 pass, the task is admitted for a **bounded development-only PPO baseline** through
  the completed controller-neutral integration and for implementation of a tariff-aware economic
  MPC benchmark. Training seeds, budgets, checkpoint selection, and evaluation seeds still
  require a separate frozen protocol.
- R8 constrains interpretation in every case. It never authorizes commercial or controller
  superiority language.

## Explicit exclusions

This tournament does not:

- train, tune, select, or evaluate an RL policy;
- implement or test an economic MPC;
- use formal validation, held-out, prospective, site, or out-of-distribution evidence;
- calibrate the tariff, service value, wear, lifetime, or endowment assumptions;
- model grid import/export, product sales, ancillary services, CAPEX recovery, maintenance
  scheduling, DAC/methanator degradation, or plant availability;
- establish statistical significance, stable tail performance, or generalization;
- establish savings, profitability, LCOCH4, bankability, or commission superiority; or
- evaluate HEPA, natural-language supervision, operator workflow, or deployment fallback.

The strongest permitted positive conclusion is: **the alpha economic task repeatedly separates
the two declared safe development controllers across the frozen matrix, including without its
dominant terminal methane-liability term, and is ready for bounded development controller work.**
