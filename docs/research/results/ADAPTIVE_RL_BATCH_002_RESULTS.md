# Adaptive RL batch 002 — closed results

Bounded residual PPO learned a modest improvement over H, but did not establish
operational benefit against the stronger OperatingMPC. Both residual policies
retained H's five forced stops, individual service regressions and worsened daily
methane deficits. Direct PPO improved substantially over its initialization but
remained more expensive and failed strict safety in every scenario.

The unchanged obligation is **6,720 kg of methane per 168-hour episode**, settled
as 28 recurring deliveries. High contracted production and high fulfillment are
distinct: the current OperatingMPC fulfilled 93.94% on this stochastic panel.
There is no justification for increasing demand to manufacture an RL advantage.
The objective remains improved economics at preserved service and resource
performance, with strict safety and new confirmation evidence.

## Design and custody

The [frozen protocol](../ADAPTIVE_RL_BATCH_002_PROTOCOL.md) specified four fresh
131,072-transition cells, optimizer seed 101, four environments and sixteen
8,192-transition rollouts. Each cell performed 5,120 minibatch optimizer steps.
All optimizer rewards were raw interval EUR multiplied by 0.001, initial Gaussian
standard deviation was 0.2, and actor output-head weights and biases were exactly
zero. Physical accounting and constraints were unchanged.

| Cell | Learner | Policy interface |
| --- | --- | --- |
| A | PI-Lagrangian | 135 inputs, absolute targets |
| B | Scalar PPO | 135 inputs, absolute targets |
| C | PI-Lagrangian | 139 inputs, bounded H residuals |
| D | Scalar PPO | 139 inputs, bounded H residuals |

Training used fresh seeds 9,610,000–9,610,195. Every actor used all eight fixed
development seeds 9,700,100–9,700,107. The residual interface appends four H target
fractions to the actual public observation and restricts action corrections.
Its zero policy reproduced native H's complete physical and economic trajectory
on every development scenario, including all minute states and settlements.
The four added inputs and mapping constitute a method package; this experiment
does not isolate the causal contribution of either component.

Source was clean revision `a442a4a9f5b028568d36bbbddf21a34c3609a7cc`, tree
`7b9376db19948986501bd7cf52f677c8ffc73599`. Its readiness gate passed 511 scoped
Python tests in 656.33 seconds, Black, strict mypy, compilation, CLI and whitespace
checks. The inherited full repository/frontend gate was not repeated.

Execution ran from `2026-09-12T15:27:54.454314Z` to
`2026-09-12T19:34:57.336127Z`. All 164 child receipts closed without failure codes:
four training, eighty evaluation and eighty detached replay attempts. No outcomes
were opened during execution. The frozen post-closure audit verified 632 hashed
artifacts and the bindings to completed physical replay receipts. Sixty-four
episodes completed; sixteen comparator attempts retained stopped prefixes.
A successful process receipt is not a successful full controller episode.

## Every actor

Means require all eight complete and replayed episodes. The
[compact evidence index](adaptive_rl_batch_002_results.json) retains all eighty
slots, individual daily resource deficits, failures and artifact digests.

| Actor | Complete / 8 | Mean delivery fill | Mean value, EUR/week | Strict safety passes / 8 | All daily returns pass / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | 8 | 90.20% | −4,542.46 | 5 | 0 |
| OperatingMPC | 8 | 93.94% | 3,165.47 | 8 | 0 |
| M24 | 0 | — | — | 0 | 0 |
| E-beta | 0 | — | — | 0 | 0 |
| Initial absolute | 8 | 85.86% | −18,116.74 | 0 | 0 |
| Initial residual | 8 | 90.20% | −4,542.46 | 5 | 0 |
| A: absolute PI | 8 | 91.31% | −9,517.22 | 0 | 0 |
| B: absolute scalar | 8 | 93.02% | −6,322.48 | 0 | 0 |
| C: residual PI | 8 | 90.54% | −4,100.06 | 5 | 0 |
| D: residual scalar | 8 | 90.65% | −3,947.07 | 5 | 0 |

M24 stopped in every scenario because its model does not support degraded/offline
DAC (`ServiceMPCInputError`). E-beta stopped in every scenario at an infeasible
service-maximization LP (`BetaMPCSolveError`). Their prefixes cannot supply weekly
comparison means or numerical PPO wins. No fallback replaces either research arm.

The required OperatingMPC includes the previously fixture-tested unfinished-startup
correction. It had zero forced stops and passed strict safety in all eight new
scenarios. That is incumbent controller engineering, not an RL contribution.
Different scenario panels prevent attributing its improvement over batch 001
solely to that correction.

## Learned improvements and counterevidence

| Candidate minus comparator | Mean EUR/week difference | Worst paired EUR difference | Minimum paired delivered-kg difference |
| --- | ---: | ---: | ---: |
| B − A | 3,194.74 | 1,336.14 | 16.84 |
| D − C | 152.99 | −490.42 | −24.57 |
| C − H | 442.40 | −92.20 | −4.09 |
| D − H | 595.39 | −35.88 | −3.11 |
| C − OperatingMPC | −7,265.53 | −13,917.83 | −550.83 |
| D − OperatingMPC | −7,112.54 | −13,766.34 | −543.28 |

B improved both service and economics over A in all eight scenarios, but both
failed strict safety throughout. C and D improved mean economics over their exact-H
initialization. Their mean improvement does not satisfy the parent acceptance
conditions: each has a paired delivery regression, a worsened daily CH4 deficit
(maximum increases of 16.84 kg and 23.65 kg respectively), and five forced stops.
Their economics and delivery remain substantially behind OperatingMPC.

The C/D improvement over H comes from 22.88/30.51 kg of additional weekly
delivery and reduced shortfall penalties, with **higher** battery throughput
(+1,016.12/+1,427.09 kWh), wear and solar opportunity cost. It is not evidence of
lower operating cost at equal service. Against OperatingMPC, D loses EUR 5,301.90
in service value/shortfall components and EUR 2,219.63 in battery-throughput and
solar-opportunity components, partly offset by other ledger terms.

H, the initial residual and both trained residuals had forced-stop counts
`[0, 0, 0, 1, 3, 1, 0, 0]` across the ordered seeds. H's eight safe scenarios in
batch 001 did not generalize to this new panel. The bounded H mapping is therefore
an operating prior, not a safety certificate. All actors still failed at least
one daily resource return in every complete episode; simulated positive EUR must
not be presented as sustainable feasibility.

## Learning and startup diagnosis

An independent saved-data diagnosis bound all 64 diagnostic files, 64 rollout
ledgers and 32 trained-policy archives to the audit: all sixteen rollouts and
21,504 deterministic decisions. Final pre-update reward explained variance was
0.973/0.975/0.949/0.948 for A/B/C/D. Gaussian standard deviations remained
0.171–0.193 from an initial 0.2, and every saved deterministic action was inside
its bounds. The earlier raw-reward critic collapse is absent. Nonzero policy
updates and input-dependent outputs remain; neither convergence nor useful
forecast/fault adaptation is established by these diagnostics.

D's mean electrolysis adjustment relative to H on its own current trajectory was
only +8.12 kW. The mapping gave no direct electrolysis authority in 3,472 of
5,376 decisions, and all four targets equaled its current H command in 2,395
decisions despite nonzero network outputs. These are mapping restrictions, not
an untrained network. The policies can still change subsequent H decisions by
changing plant state.

The five H forced stops all followed a target below 600 kW during unfinished
cold startup, with 2,500 kW battery support available and SOC between 0.367 and
0.639. Both residuals inherited the same boundaries and electrolysis commands:

| Scenario seed | Decision index | Electrolyser target, kW |
| --- | ---: | ---: |
| 9,700,103 | 13 | 583.644849 |
| 9,700,104 | 11 | 516.041605 |
| 9,700,104 | 17 | 450 |
| 9,700,104 | 21 | 450 |
| 9,700,105 | 17 | 450 |

Each interval had one aborted startup and one forced stop. The saved evidence
identifies an underpowered command during startup, rather than depleted battery
support. OperatingMPC was already running at four corresponding boundaries and
held 600 kW during the remaining startup. Its different earlier trajectory means
this is not a paired counterfactual repair test.

PI diagnostics show intermittent gradient opposition and nonzero standardized
cost gradients even on some rollouts with zero observed cost. Bootstrapped cost
prediction errors and normalization are a hypothesis for later study, not an
established cause of the operating gap. More training, larger multipliers or
dropping a cost channel is not justified as a proven remedy.

## Evidence and limits

The audit's raw SHA-256 is
`a65ba53de363032caced6435a91543b782fcf993dbb93e242a6992f3d40406ea`.
Its separately identified canonical JSON digest used by the descriptive reporter
is `182d49901c00386aa3adbfb562b67b6d199c7705d88a010cf3efddc6938af426`.
The compact index binds the gate, plan, closure, audit, complete descriptive
report and frozen tools. Checkpoints, all sixteen training curves and minute
archives remain on Lexar under
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-002` and its analysis sibling.
The independent [learning diagnosis](/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-002-analysis/learning/plant-effort-learning-diagnosis.md)
and [operational diagnosis](/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-002-analysis/operation/plant-effort-operational-diagnosis.md)
retain every inspected rollout, action and event context with input digests.

Detached replay reconstructs outcomes from saved absolute commands and checks
residual input/mapping evidence. It does not establish that a checkpoint generated
each saved action; separate process/checkpoint custody supplies that evidence.
Reported native comparator exceptions are not authenticated by physical-prefix
replay. Training diagnostics are pre-update sample measurements, not final-policy
or actual Adam-update measurements. Raw EUR and scaled training rewards remain
distinct. All failures, incomplete attempts and per-seed regressions are retained.

This is one-optimizer-seed adaptive development. No formal policy selection,
held-out evaluation, operational-benefit claim, commercial savings or live
deployment follows. The original registered 15-policy study and protected seeds
remain unopened. New sealed confirmation and optimizer-seed replication are
still required; none of the present policies has qualified for that stage.
