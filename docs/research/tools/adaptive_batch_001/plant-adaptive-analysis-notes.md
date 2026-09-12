# Closed-batch analysis of adaptive PPO batch 001

Prepared during the running batch at clean source
`238fe7ca763254b70af63a4ef0690a9d4523aa60`. No live training/evaluation outcome
values were opened while preparing these tools. Work stays outside the frozen
checkout. This analysis is descriptive development; it does not choose a
checkpoint, change the experiment, establish a materiality threshold or claim
confirmation.

These files were later archived in this directory after normal batch closure.
Their original bytes and pre-outcome fixture validation remain under
`/Volumes/Lexar/PlantAnAgentResearch/adaptive-analysis-tools-238fe7c`.
The auditor deliberately requires the exact clean `238fe7c` source/runtime and
gate; a later checkout must not bypass that check to reissue an audit. Run the
frozen external copy against its original source checkout to repeat that audit.
The report can be reproduced from the already verified `audit.json` without
running the simulator. The three `test_*.py` files use fabricated artifact graphs
and can run from this archive without accessing production experiments.

## Closure and reporting

The auditor refuses an absent or aborted final report before reading outcome
artifacts. For a normally closed batch it verifies the fixed five training cells,
80 evaluation slots, conditional replay receipts, exact source/gate identity,
initial/final policy hashes, matched initial tensors and all archived result
bindings. It retains failed process/controller cases, and separately reports
whether each complete physical episode has a successful replay receipt. The
auditor does not rerun a controller or simulator. AppleDouble metadata sidecars
on ExFAT do not count as duplicate receipts.

The report preserves every ordered development seed for every actor. Means and
paired differences require all eight eligible episodes; a stopped comparator
cannot provide a numeric win through a favorable subset. Strict safety and each
daily return remain separate. Every comparison retains all 56 four-resource
daily boundaries, including the largest deficit increase for each resource.
Battery deficits are kWh; H2/CO2/CH4 deficits are kg. An economic improvement is
insufficient when delivery, safety or a resource requirement worsens.

## Mechanism comparisons fixed before outcomes

| Hypothesis | Comparison | Evidence and interpretation |
|---|---|---|
| Reward conditioning impairs learning | B minus A; D minus C | Compare all 16 pre-update critic/actor gradient norms, common clip factors, tanh saturation and full-rollout explained variance. Reduced critic dominance with improved operation supports the hypothesis; better diagnostics without better operation show the repair is insufficient. |
| PI cost signals induce an undesirable resource/service tradeoff | A minus C; B minus D | Compare reward/cost gradient magnitudes and cosine, per-resource components, multipliers, cost prediction, episode costs and final daily deficits/delivery. A negative cosine alone is not a defect: a constraint can appropriately oppose profit. |
| Broad exploration and action-space clipping impair learning | E minus B | Compare expected/observed Gaussian bound clipping and action standard deviation, then final operation. Lower clipping alone establishes only that the intervention was expressed. Worse operation can indicate that narrower exploration hurt this setting. |

All comparisons use the fixed final checkpoint, all eight scenarios, and every
rollout index. Keep initialization, H and the product OperatingMPC comparisons;
retain M24 and E-beta failures. Never pick a favorable rollout, seed subset or
metric after seeing results.

The constructor-only check at the frozen source established equal actor/reward
parameter and Python/NumPy/Torch RNG digests across A-D, with reset patched to
raise and zero consumed decisions. Extra PI cost-critic construction uses
`torch.random.fork_rng` and checks RNG restoration. This supports matched
first-rollout states/actions as a source-level prediction, not an already
measured production-trajectory equality. After the first optimizer update,
visited states can diverge. E changes the initial Gaussian standard deviation.

Constructor-only actor/reward tensor digest:
`7cf3bdc1d8d9c9b6803488da86d00fab1f9552886d3c52dfea9553c038333cb8`.
Torch RNG digest:
`94aae829985494cd864c4203ac270c018355514435b63db280b44b6a80995cca`.
NumPy RNG digest:
`be97daf174f3e45c64055677575ef3c7d85c2711f79ae2b0406ec1e9a0ebf1d0`.
Python RNG digest:
`bea2da1487a9b579d11a364f4ca8042eb1bbbb697c7aecef26829a6bbc973e99`.

## Interpretation limits

- Diagnostics cover the first 256 environment-major samples before optimization.
  They are neither actual shuffled SGD minibatches nor Adam parameter updates.
- Gaussian action-space clipping differs from PPO likelihood-ratio clipping.
  The recorded pre-update probability ratio should be near one.
- Dividing critic MSE by reward scale squared restores squared-EUR units, but does not
  equalize visited states or bootstrapped targets. A smaller scaled MSE alone
  proves nothing; explained variance is more informative across reward units.
- A ledger's reward return is an immediate-reward sum; diagnostic return describes
  bootstrapped targets. Neither is the final evaluated weekly operating result.
- Scalar cost diagnostics are absent, not zero violations. Undefined variance
  or cosine remains unknown. PI and scalar approximate-KL aggregation differs.
- Physical replay authenticates saved physical consequences, not the provenance
  of a reported controller exception or a second independent controller run.
- One optimizer seed and adaptive scenarios cannot establish site generalization
  or an algorithm-level claim. No synthetic EUR result is calibrated site savings.

## Next decision

Resolve integrity/closure errors before interpreting affected evidence. If
conditioning changes without operational improvement, preserve that result and
isolate the remaining mechanism in a new prospectively specified batch. If an
arm improves economics/service while worsening resources, report the tradeoff.
If a candidate meets the required safety/service/resource conditions and shows
operational promise, freeze its method, practical-effect threshold, replication
seeds and a fresh confirmation panel before further confirmation data are opened.
If no arm helps, retain every negative result and revise the hypothesis.

The requested operational benefit remains unproven until fresh confirmation and
the complete requirement audit support it.
