# Learned control follow-up: fixed plant, adaptive development, fresh final test

Authorized campaign begins **2026-09-17 23:23:43 UTC** and ends no later than
**2026-09-18 07:23:43 UTC**, using only this local machine. Target final-test
freeze is 05:53:43 UTC, reserving approximately 90 minutes for evaluation,
verification and the paper. The completed prior submission remains intact.

## Fixed scientific task

The user explicitly permits changing the training mixture while preserving
existing environment definitions. Reuse the original shared-battery,
dynamic-readiness plant and its existing ordinary and tail scenarios exactly.
Physics, capacities, action mapping, public observations, economic objective,
initial-stock generator, causal forecasts, seven-day duration and 180 kg
six-hourly deliveries stay fixed. No reserve floor, action shield, emergency
fallback, nighttime bonus, invented damage cost or additional plant mechanism
is added. All policy actions remain subject only to the existing physical model.

Conditions are `ordinary` (the original stochastic weather and DAC faults),
`control` (the existing healthy-equipment tail control), and the already defined
`pv6`, `pv18`, `pv36`, `meth24`, `meth48` interventions. These schedules remain
fixed: solar interruptions start at elapsed hour 48; methanator outages end at
hour 86. New scenarios vary the original weather and opening-stock draws, not
the intervention definitions. A policy may learn this benchmark's temporal
regularities. Success would therefore concern these known intervention families,
not surprise events at untested times or real-world event frequencies.

Only new seed admission and episode mixture selection are implemented in a
separate adapter. Exact parity and independent interval audits check that this
adapter preserves the parent definitions. Do not edit any prior frozen source.

## Development and test separation

* Training seeds: 50,000,000 through 50,999,999.
* Development seeds: 51,000,000 through 51,000,015.
* Final test seeds: 52,000,000 through 52,000,031.

Default training mixture: 50% ordinary episodes and 10% each of the five stress
conditions. `control` is a diagnostic evaluation condition. Development may
compare other explicitly recorded mixtures. No final seed is evaluated before
the selected recipe, checkpoint selection rule, baseline champion and case grid
are frozen into a content-hashed admission gate. The old inspected test sets
are historical evidence and may motivate hypotheses; they are not new test data.

Development is adaptive: all attempted recipes, learning curves, checkpoints,
teacher data, failures, selection decisions and compute times are retained.
The final test is opened once for the selected recipe, with all three prescribed
optimizer seeds reported. If the result is disappointing, it remains the result;
further final-test-driven development would require a separate future study.

## Initial candidates and resource allocation

1. Continue the existing direct PPO policies on the authorized mixture, with an
   explicit reduction in Gaussian exploration and deterministic development
   evaluation. Test longer advantage traces as a recorded training parameter;
   this changes learning, not the plant reward or action authority.
2. Fit a fresh neural policy to storage-controller demonstrations collected only
   on training seeds, validate its autonomous closed-loop behavior, then allow
   unrestricted PPO improvement. Optional dataset aggregation relabels states
   reached by the neural policy. There is no deployed teacher or fallback.
3. A replay-based learner or another controller architecture may be tried only
   when development diagnostics justify the compute. Every such attempt receives
   its own prospective recipe record. Avoid a broad unstructured search.

Imitation may use documented regression targets slightly beyond exact action
box endpoints so ordinary action clipping reproduces an exact OFF command.
This is initialization of the actor, not an additional runtime threshold or
action shield. State the use of demonstrations clearly; do not describe an
imitation-initialized policy as learning from random control behavior.

Each training attempt has an explicit step budget and absolute stop time,
immutable source/model/normalizer metadata, periodic recoverable checkpoints,
and a status distinguishing completed from deadline-limited attempts. Training
statistics are diagnostics; deterministic development performance selects the
recipe. Retain all three optimizer-seed outcomes for the final selected recipe.

## Strong comparator and development selection

Report both original daylight and storage baselines. Give the storage rule a
modest development-only tuning budget over its existing controller parameters:
methane buffer 48/72/96 hours, battery charge target 0.35/0.60/0.85, and upstream
replenishment time 3/6 hours (18 combinations). These parameters do not alter
plant capacities or restrict the learned policy. Screen on four development
seeds, then confirm the leading configurations on all 16. Teacher configuration
may be selected from this development evidence and is pinned for every dataset.

Select the baseline champion by the same default mixture contribution used in
training: 0.5 times ordinary mean plus 0.1 times each stress-condition mean.
Keep all attempted configurations visible. Prefer fewer unmet kilograms to
break an equal contribution score. `control` has zero selection weight and is
reported separately. A narrower or extended search must be recorded before
running those additional development cases, never after inspecting final data.

Learned candidates are ranked on development mixture contribution, with delivery
outcomes, ordinary contribution and consistency across optimizer seeds visible.
Prefer a single reproducible recipe to a collection of individually chosen
policies. Any continuation from old policies reports that initialization and
uses the corresponding three initial optimizer seeds, rather than keeping only
the strongest previously observed seed.

## Prospective final headline criterion

The main comparison is the selected three-policy recipe against the development-
selected baseline champion. Average the three fixed learned policies within each
base scenario/condition before paired inference. Report every policy separately;
scenario intervals do not estimate uncertainty over arbitrary retraining seeds.

A strong joint superiority claim requires:

* At least EUR 500/week higher mean contribution under the declared 50/50
  ordinary/stress mixture, with a strictly positive paired 97.5% interval lower
  bound for contribution difference.
* At least 20% lower mean unmet methane across the equally weighted five stress
  conditions, with a strictly negative paired 97.5% interval upper bound for
  unmet-mass difference.
* No lower stressed all-28-deliveries success rate, and no more than EUR 500/week
  lower ordinary-operation contribution.
* Each of the three optimizer seeds individually improves mixture contribution
  and lowers mean stressed unmet mass to support a consistency-across-seeds claim.

The two inferential endpoints use 97.5% two-sided percentile intervals (nominal
Bonferroni adjustment). Resample the 32 paired environmental seeds, keeping all
conditions together, for 10,000 replicates with a fixed recorded analysis seed.
Small-sample bootstrap limits remain. If the champion's mean stress unmet mass
is zero, the strict improvement criterion is not met by equality. No model or
metric is selected using final outcomes. A narrower empirical finding may be
reported when the joint criterion fails, with its scope stated explicitly.

Also report original baselines, the previous PPO reference where budget allows,
all condition-specific results, delivery rescues/regressions, ordinary controls,
mean and worst-quarter unmet mass, cost components and ending inventories. The
mixture is a designed research objective, not an estimated natural hazard rate.

## Integrity and deliverables

Audit every physical minute for energy, material, readiness and declared-fault
accounting, and independently reconcile economic ledgers and rewards. Retain
actual 149-dimensional observations, seven actions and normalized inference
witnesses. Prescribe complete physical traces for the first final seed in every
controller-condition panel; select replay examples before observing outcomes.

Finish with a concise research decision log, frozen selected-recipe record,
all development attempts, fresh final results, a two-page two-column LaTeX paper
by Filippo Gombac and Vittorio Garavelli, and a portable evidence package. State
the strongest supported result, including negative findings if necessary.
Publication or external submission is not part of this local campaign.
