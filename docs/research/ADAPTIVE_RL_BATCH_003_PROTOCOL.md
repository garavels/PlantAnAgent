# Adaptive batch 003: bounded residual PPO around OperatingMPC

Prospective development design, 2026-09-12. Existing user research autonomy
authorizes this local experiment. No production reset may occur before a clean
committed source, passing exact-revision gate and exclusively claimed execution
plan. The original registered three-family/five-seed study and protected seeds
remain untouched. Batch 002 is closed; none of its policies qualified for
confirmation.

## Decision and hypothesis

The [closed batch 002 results](results/ADAPTIVE_RL_BATCH_002_RESULTS.md) show
learning with well-conditioned critics and retained exploration, but the best
residual policy remains behind OperatingMPC by EUR 7,112.54 per simulated week.
H and its residual policies share five aborted startups: H commanded below the
600 kW startup requirement while startup was unfinished, despite available
battery support. The residual mapping cannot change those low-power commands.
The current OperatingMPC avoids those five events and is the stronger incumbent.

Test whether bounded operating-effort adjustments around this stronger causal
controller can learn useful behavior. This is one **short trainability and
feasibility screen**, not a convergence claim or an isolated comparison of
residual-H versus residual-MPC: prior, residual range, training budget and seeds
differ. Exact initial-policy equivalence to native OperatingMPC separates the
controller prior from subsequent learned changes.

The complete task remains unchanged: 168 hours, 28 recurring deliveries totaling
6,720 kg methane, one-minute physics and 15-minute decisions, original stochastic
weather/prices/outages, public information, operator authority, guard, economic
ledger and daily resource settlements. Do not increase demand, repair the plant
for one actor, weaken the incumbent or discard failed scenarios.

## Fixed learner and budget

Train one fresh scalar PPO cell, **A-mpc-residual-scalar**, optimizer seed **101**,
for **32,768 transitions**: four synchronous environments, four 8,192-transition
rollouts, 1,280 minibatch optimizer steps. Use the existing PPO hyperparameters,
optimizer reward scale **0.001**, initial Gaussian standard deviation **0.2**,
and exactly zero actor output-head weights and biases. Raw EUR accounting,
Monitor returns and four physical constraint signals remain unchanged.

Fresh training allocator seeds are **9,620,000–9,620,051**. This covers thirteen
episodes per environment, including the final partial episode. No existing
development trajectory becomes training data. Preserve initial weights, every
rollout/diagnostic, final checkpoint, allocator ledger and any failure receipt.
Use the fixed final endpoint only; no early checkpoint selection, continuation,
resume, retry or substitute arm is allowed within this batch.

Scalar PPO is chosen for a bounded first test of the stronger prior. Batch 002's
scalar residual had a modestly higher mean than PI, with a paired regression;
this is an adaptive engineering choice, not a finding of algorithm superiority.
No claim about PI/I-Lagrangian performance follows. The original family matrix
remains deferred and unchanged.

## Exact policy interface

The new research-only wrapper receives the actual returned 135-field public
observation and appends four cached OperatingMPC target fractions, yielding
**139 inputs and four residual actions**. Normalize appended targets by installed
ratings into `[0, 1]`. They are causal derived features, not private future truth.
Compute the stateful controller once at reset and each nonterminal decision
boundary. Reset its state for each physical episode. Never re-encode the base
observation or execute an unused terminal MPC decision; terminal features are
four zeros. The allocator remains outside the wrapper.

For finite residuals `r` clipped to `[-1, 1]`, let `M_i` be the cached MPC command,
`L_i` the active operator/equipment limit and
`F = min_stable_load_frac × electrolyser_rated_kw` (currently 1,200 kW):

```
M_E <= F: target_E = M_E
M_E >  F: target_E = min(L_E, F + (M_E - F) * (1 + 0.25 * r_E))
target_i = min(L_i, M_i * (1 + 0.25 * r_i))  # DAC, methanator, charge ceiling
```

Exactly zero residual executes the original float64 MPC command and invalid
flag, without reconstruction or a float32 target round trip. Keep all original
planner/fallback status and exceptions visible. Cache indices and authority must
match the physical decision. Record original and clipped residuals, nominal and
issued commands, current public inputs, controller evidence and physical effects.

This preserves the incumbent's low-power/startup commands and every zero target;
it does not grant independent startup/off authority or battery-discharge control.
High-load adjustments operate on excess above the stable-running floor, so a
negative residual cannot itself push a target below that floor. Future energy
and inventory effects can still cause safety/service/resource failures. This is
an action restriction, not a safety or feasibility guarantee.

Bind the full base task, ordered observations, mapping, OperatingMPC implementation
and its planning dependencies in a separate checkpoint/interface identity.
Neither the old 135-input loader nor the H-residual 139-input identity may admit
this artifact. It is not a live deployment checkpoint.

## Fixed evaluation and detached replay

After training closes, attempt all six actors on all eight fresh development
seeds **9,700,200–9,700,207**:

- H;
- current corrected OperatingMPC;
- strict M24;
- strict E-beta;
- initial MPC-residual policy;
- final A-mpc-residual-scalar policy.

Both policy comparators require the successful closed training receipt and
their exact recorded weight/interface identities. A failed training arm leaves
those slots explicitly unavailable; do not construct substitute initial weights.
M24/E-beta failures remain explicit prefixes and never produce full-week wins.

Save full minute trajectories, actual physical and policy observations, nominal
MPC evidence, mapped absolute commands, raw economic components, forced/aborted
starts, guard interventions, fault command/application behavior and all daily
resource settlements. Detached replay executes the saved absolute commands;
separately recompute the causal MPC prior to check its derived inputs and mapping.
Never reexecute the learned policy during physical replay. Checkpoint and process
custody are separate from physical reproducibility and native exception evidence.

MPC fallback stage timings vary between executions and affect the original
evidence/source hashes. Preserve and validate the complete saved records and
their internal hashes. Recomputed-prior comparison may exclude only explicitly
enumerated elapsed-time fields and hashes derived from those fields; targets,
authority, public bindings, plan semantics, invalid flags, operating mode and
failure classification must still match. Replay executes the saved original
command. Initial-MPC equivalence uses the same physical projection, not equality
of nondeterministic provenance identifiers. The implementation/tests must fix
that narrow projection before admission; no broad dropping of evidence is allowed.

Verify full initial-residual/native-MPC physical equivalence on all eight seeds.
Report all 48 slots, including missing, failed or stopped attempts. Full-panel
means and paired differences require all eight complete and replayed episodes.
Report the final policy against its exact initialization and every required
comparator, including every individual delivery outcome and each of 56 daily
four-resource deficit vectors. Retain all four training diagnostics and optimizer
ledgers; do not interpret rollout reward sums as weekly evaluation EUR.

No formal policy selection or practical-benefit threshold is set in this screen.
Any promising result still requires the parent protocol's absolute hard safety,
paired service and individual daily resource-deficit noninferiority, followed by
prospectively frozen materiality, fresh sealed confirmation and optimizer-seed
replication. A small positive mean is insufficient to complete the research goal.

## Runtime and execution custody

Root: `/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-003`.
Use existing local dependencies, one Torch intra-op/inter-op thread, four
synchronous environments and sequential children. A six-hour training cap is
prospective for this new batch; the old batch's three-hour limit is unchanged.
Evaluation and replay each retain a ten-minute per-child cap, 64 MiB per log
stream, 256 MiB compressed episode limit and 10 GiB free-space floor.

All eight previous OperatingMPC evaluation receipts completed 672 decisions in
median 351.50 seconds and maximum 389.04 seconds per week. Entire-receipt
extrapolation projects approximately 4.76–5.27 hours for this training budget,
plus evaluation and replay. These are observed week-average rates, not individual
decision latency percentiles or a bound on future solver difficulty. Four
synchronous environments do not divide this estimate by four. A timeout is a
retained failed arm, not grounds to extend the budget after observing outcomes.

Freeze exact clean source, runtime, host and gate in the plan, and recheck them
around every child. Exclusively claim root/attempts; never overwrite an earlier
batch. Preserve ordinary numerical failures as local failed slots. Allocator,
source, checkpoint or evidence integrity loss aborts remaining execution; operator
interruption stops the campaign. Do not inspect outcomes or revise the design
during execution. Prepare analysis only from fabricated artifacts until normal
closure. No cloud spending, purchase, messaging or live deployment is included.
