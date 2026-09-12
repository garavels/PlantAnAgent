# Adaptive batch 002: operating effort around the heuristic

Prospective development design, 2026-09-12. The user's existing research autonomy
authorizes this local experiment. No production reset may occur until this
protocol, implementation and tests have a clean committed source and a passing
gate at that exact revision. The original registered three-family/five-seed
experiment and its protected seeds remain untouched.

## Decision from batch 001

All five previous cells completed, and all 80 recorded controller attempts had
their complete episodes or stopped physical prefixes independently replayed.
Reward scaling repaired critic conditioning but did not consistently improve
operation. Lower exploration made PI PPO's service competitive on average;
its economics were worse than H and OperatingMPC, every scenario failed strict
safety, and individual delivery/resource regressions remained. The principal
observed loss mechanism was battery use and electrolyser cycling: persistent
approximately 1.5 MW requests caused repeated starts near dawn/evening when the
battery was at its dispatcher reserve. The safety failure was forced stops,
not the battery safety threshold. Negative results remain in the batch 001
result packet.

The next hypothesis is that learning bounded adjustments to an existing causal
operating rule is more useful than learning all four absolute targets from a
near-midpoint initial policy. The missing scalar comparison at lower exploration
is included. This evaluates complete method packages; changing representation,
input features and initialization is not a single-factor causal experiment.
Within each representation, PI versus scalar is a matched comparison.

Residual control is motivated by [Johannink et al.](https://arxiv.org/abs/1812.03201)
and [Silver et al.](https://arxiv.org/abs/1812.06298). Those robotics results do not
establish benefit for this plant.

## Unchanged physical task and required comparators

Use the complete 168-hour stochastic beta task, one-minute physics, 15-minute
decisions, the same deliveries, public forecasts, prices, faults, economics,
resource settlements, dispatcher and guard. Do not raise demand or degrade an
incumbent to manufacture an advantage. All evaluation uses deterministic policies
and raw EUR. The primary opportunity is cost/wear reduction at preserved service.

Required comparators are H, the product OperatingMPC, M24 and E-beta. Preserve
M24/E-beta exceptions and incomplete episodes; they cannot supply numerical wins.
OperatingMPC's identified unfinished-startup/450-kW command defect is repaired
and fixture-tested before source freeze. This strengthens the incumbent; its
improvement is controller engineering, not RL evidence. The frozen code identity
records the exact resulting controller. No post-launch controller changes are
permitted.

## Four fixed training cells

Each cell starts fresh with optimizer seed **101**, **131,072 transitions**,
four synchronous environments, 16 rollouts of 8,192 transitions and the existing
PPO hyperparameters. All optimizer rewards are raw interval EUR multiplied by
**0.001**; physical accounting, Monitor returns and four constraint signals stay
unchanged. Initial Gaussian standard deviation is **0.2** for every cell.

| Cell | Learner | Policy interface |
| --- | --- | --- |
| A-absolute-pi | Existing PI-Lagrangian | 135 observations; four absolute targets |
| B-absolute-scalar | Existing scalar PPO | 135 observations; four absolute targets |
| C-residual-pi | Existing PI-Lagrangian | 139 observations; bounded H adjustments |
| D-residual-scalar | Existing scalar PPO | 139 observations; bounded H adjustments |

Set the actor output-head weights and biases to exactly zero before any learning
in all four cells. The absolute policy's deterministic initial output is therefore
the action-space midpoint, while the residual policy's deterministic initial
output is exactly H. Stochastic training still explores around those means.
PI/scalar actor and reward-critic initialization must match within each interface;
weights across different input dimensions are not asserted equal. Preserve all
initial weights, every rollout/diagnostic, final checkpoint, failures and receipts.
Use only the fixed final endpoint for comparisons.

All cells use the same fresh allocator sequence starting at **9,610,000**,
covering seeds through **9,610,195**. No batch 001 training seed is reused. The
existing allocator remains outside the residual wrapper so every physical
transition and reset is accounted for exactly once.

## Exact residual interface

Compute H once from the current public controller observation, installed plant
configuration and operator limits at each decision. Append its four normalized
absolute targets to the actual 135-vector returned by reset/step. This introduces
derived features, not privileged future information. Cache H's original command
in float64; the appended float32 features are not the authoritative command.

For each residual coordinate, clip the finite policy action to `[-1, 1]` and
retain clipping evidence. Let `H_i` be H's target, `L_i` the active command limit,
and `F = min_stable_load_frac × electrolyser_rated_kw` (currently 1,200 kW).
For electrolysis:

```
H_E <= F: target_E = H_E
H_E >  F: target_E = min(L_E, F + (H_E - F) * (1 + 0.5 * residual_E))
```

For DAC, methanator and battery-charge ceiling:

```
target_i = min(L_i, H_i * (1 + 0.5 * residual_i))
```

At exactly zero residual, execute H's original command and invalid flag, without
a normalize/de-normalize round trip. All zero H targets remain zero. H's low-power
electrolyser choices remain unchanged: its 450-kW battery-only target must not
be multiplied above the 600-kW cold-start threshold. Higher-load effort and
downstream allocation remain learnable using the complete public market/cycle
surface. The method cannot independently choose every possible startup/stop or
activate a zero H target; these are declared representational restrictions.

This mapping is not a feasibility supervisor or a safety guarantee. Changing
charging, inventories and load still changes future H decisions and minute power
availability. All existing safety/service/resource checks remain required.
There is no extra intervention whose effect can be silently attributed to RL.
The deterministic initial residual policy must reproduce H's full physical,
economic, commitment and settlement behavior in disjoint fixtures.

The residual checkpoint has an explicit separate identity covering its base
task, ordered 139 features, H implementation hashes, mapping, limits and exact-zero
semantics. Existing 135-observation production/registered loaders must reject it.
It is not a live deployment artifact.

## Evaluation, replay and analysis fixed before reset

After all four training attempts close, evaluate ten actor identities on all
eight fresh development seeds **9,700,100–9,700,107**: the four required comparators,
the initial absolute policy from A, the initial residual policy from C, and all
four final policies. Failed A/C training makes its respective initial comparator
unavailable; no replacement checkpoint or seed is chosen. All training admissions
and checkpoint/interface digests are checked before model loading.

Save every minute state, public observation, policy input/output, original H and
mapped command, raw EUR ledger, daily four-resource settlement and controller
failure. Detached replay must reconstruct physical outcomes from saved commands;
residual command mapping and policy-input evidence are additionally checked
against the public boundary and frozen H implementation. Replay does not prove
that a checkpoint generated a stored action; checkpoint/process/source custody
is separately recorded. Native comparator exception authenticity is not claimed
from physical-prefix replay.

Report all eight slots per actor. Full-panel means/paired economic differences
require all eight complete and replayed episodes. Report service for every seed,
strict safety including forced stops, and every individual daily resource deficit
before interpreting cost or wear gains. Also report battery/electrolyser wear,
startup energy, forced stops/aborted starts and commanded-versus-applied behavior
during outages. No partial-prefix or successful-subset comparison is a win.

Prespecified contrasts: B versus A, D versus C, C versus A, D versus B, each final
versus its own initialization, and each final versus H/OperatingMPC. Initial
residual versus native H checks the implemented operating prior. Keep all
rollouts, not a favorable checkpoint or diagnostic time. No policy is selected
for confirmation in this screen.

## Execution custody and limits

Root: `/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-002`.
Use one Torch intra-op/inter-op thread, sequential children and existing local
dependencies. Per-cell training cap: three hours. Per-evaluation/replay child:
ten minutes. Each stdout/stderr: 64 MiB. Compressed episode cap: 256 MiB.
Require at least 10 GiB free before admission. Expected duration is several
hours, based on batch 001; this is not a completion guarantee.

Exclusively claim the root and attempts. No retries, resumes, replacement arms or
overwrites. Bind exact clean source, runtime, host and passing readiness gate in
the plan. Recheck custody around every child. Keep local failures as explicit
slots; source/integrity loss or operator interruption aborts remaining execution.
Do not open outcomes or change the design while this batch is running. Analysis
preparation may use fabricated artifacts only until normal closure. No cloud
spending, purchase, live deployment or messages to others are included.

This remains adaptive development, with no held-out or commercial claim. A
promising result must still meet the parent protocol's strict safety, paired
service and individual resource noninferiority conditions. Practical materiality,
fresh sealed confirmation and at least three optimizer seeds remain necessary
before claiming operational benefit or an algorithm-level result.
