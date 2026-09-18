# Constrained-RL capacity repair 001 — canonical rollout-cost evidence

Status: **frozen after the failed `fab4a55` attempt and before repair implementation or re-execution**

Date: **2026-09-05**

Repair identifier: **`plant-beta-constrained-rl-capacity-repair-001`**

## Preserved failure

The one capacity attempt at revision `fab4a552a4fb2f1d1d6efbcb757a87e4f8a9cf73`
is permanently retained as a failed engineering result. Its report-domain SHA-256 is
`f3d2ab32dbb86c7c4268d3b114e2ee44aff5d29b02f2c4e4e725e162acd20895`; it contains no checkpoint,
measurement, campaign projection, or result claim. All three children exited after reporting the
same checkpoint-state invariant: `rollout cost scaling is incoherent`. The detailed non-outcome
record is `results/v2_constrained_rl_capacity_fab4a55.md`.

No cell at that revision may be retried. The observed wall times cannot enter a campaign estimate.
One raw fixture stdout log was accidentally inspected during diagnosis and contained ordinary SB3
reward/cost summaries. Those values are not reproduced and must not inform this repair, learner
settings, environment design, or any policy choice.

## Root cause

The transition contract is already exact. For each of four resources at each settlement:

```text
transition training cost = transition raw cost / 7 cycles
```

The collector independently summed the raw values and the already-divided training values across
the rollout. The checkpoint verifier then compared:

```text
sum(training values)  versus  sum(raw values) / 7
```

as though floating-point aggregation were associative. With many nonzero settlements these two
mathematically equivalent expressions can differ by several binary64 ULPs. The `1e-15` aggregate
tolerance was narrower than that ordinary summation drift. This is an evidence-representation
defect after optimization, not an objective, safety, controller, or learning failure.

## Sole permitted learner repair

The repair may change only rollout-level evidence aggregation:

1. Continue extracting, validating, and independently accumulating the exact raw and training
   transition costs during rollout collection.
2. At rollout finalization, compute the canonical four-vector once as
   `raw_rollout_total / BETA_EPISODE_CYCLE_COUNT` in binary64.
3. Before replacing the independently accumulated representation, require it to be numerically
   consistent with the canonical vector using `rtol=1e-12` and `atol=1e-12`. A larger discrepancy
   remains a poisoned rollout and must fail.
4. Store the raw total and canonical divided total in `_PendingRolloutV1`, the rollout ledger, and
   the corresponding diagnostic logger fields.
5. Make checkpoint-state verification require the stored ledger training vector to equal the same
   elementwise binary64 division of its stored raw vector. Do not relax evidence admission to hide
   arbitrary differences.

The following remain byte-for-byte or semantically unchanged:

- every environment transition and `CycleConstraintTransitionV1`;
- the float32 cost rollout buffer consumed by the optimizer;
- reward and cost GAE;
- completed-episode cost accumulation supplied to the PI/I multiplier controller;
- multiplier gains, limits, update order, and histories;
- actor, reward critic, and cost-critic losses or parameters;
- PPO configuration, network topology, seeds, rollout size, batch order, and optimizer steps;
- environment, market, fault, cycle, economic, command, guard, and plant behavior; and
- the three-cell capacity order and the entire registered campaign.

## Required implementation evidence

Before another capacity attempt:

1. Add a pure adversarial test that accumulates 84 synthetic nonzero settlement values through the
   two original arithmetic paths, proves their difference exceeds `1e-15`, and proves the new
   canonical ledger relation is exact.
2. Prove a materially inconsistent independently accumulated training vector still fails.
3. Prove zero and mixed-channel vectors retain channel isolation, finiteness, and non-negativity.
4. Retain all existing transition, GAE, multiplier, scalar-isolation, checkpoint-tampering, resume,
   capacity-report, and production-integration tests.
5. Pass Black, compilation, strict scoped MyPy, Git whitespace, the focused learner/checkpoint gate,
   and the complete repository release gate.
6. Extend the capacity report's protocol identity with this repair identifier, path, and exact raw
   SHA-256, and version the report/domain schema so a repaired attempt cannot be confused with the
   failed V1 report.

No real 8,192-transition rollout is permitted merely to make the unit test pass. The next admitted
capacity execution is itself the one-shot end-to-end test.

## Re-execution custody

After the repair and all gates are committed, exactly one new capacity attempt may run at that new
clean revision. It must use the unchanged algorithms, optimizer seed 101, fixture base 9,880,000,
three-cell order, configuration, timeout, and operator assertions, but a new absolute external
output root. The repaired report must bind both the original capacity protocol and this repair.

If any cell fails, the new report is retained without projection or retry and another repair
protocol is required. If all cells pass, only that complete repaired report may supply the frozen
capacity projection. It still cannot support a controller-performance, safety, or commercial claim.
