# Learning: completed adaptive PPO evidence

The unified workspace's `/learning` page displays the completed adaptive research
record through `GET /api/v2/ppo/research`. This is a product evidence integration,
not a new experiment or operational controller release.

## Product scope

- **Training:** the four most recently evaluated endpoints, their actual 8/8/2/2
  rollout observations, training parameters, checkpoint identities and diagnostics.
- **Evaluation:** all 26 published ordered contrasts from batches 008 and 009,
  with eight individual development scenarios, absolute controller outcomes,
  every delivery and daily resource boundary, and all recorded regressions.
- **History:** the initial pilot and batches 001–009, including incomplete training,
  aborted campaigns, failed qualification and checkpoint reuse. The original
  registered-study panel remains available separately within History.

The existing workspace typography, colors, navigation and plant state remain in
use. Choosing a policy or scenario here only changes inspected evidence. It does
not activate a controller, reset the plant, infer from a checkpoint or start a run.

## Evidence boundary

The portable package file
[`adaptive_learning.json`](../../backend/src/plant_backend/data/adaptive_learning.json)
contains four policies, nine actors, 72 actor cases and 208 paired cases. Seven
actors are explicitly reused from batch 008; two are fresh batch-009 endpoints.
Only published ordered contrasts are exposed. The fixed seeds are development
seeds 9700300–9700307, not held-out confirmation. All four endpoints use optimizer
seed 101 and a 139-input MPC-residual interface with four residual actions.

The publisher reads fixed-SHA256 closed analysis/audit reports and compact research
records. The API reads only the packaged JSON, enforces its checksum and bounded
strict schema, and cross-checks every paired metric and regression against the
underlying actor cases. It never needs an external research disk at request time.
Checkpoint bytes and large research artifacts are not copied into Git.

Completed endpoint training and completed campaign admission are separate facts.
Batch 007's endpoints completed, but that campaign aborted after a power-guard
false positive and retained a separate allocation-metadata discrepancy. Batch 008
admitted and evaluated those exact endpoints without new training. Batch 006
aborted without completed endpoints; batch 005's interruption remains incomplete.

M24 and E-beta have stopped prefixes. Their full-week metrics and safety results
are **unavailable**, while recorded prefix safety, decisions and failure reasons
remain accessible. Recovery counts represent accepted optimized MPC plans; only
Fallback counts represent heuristic execution.

The two latest policies show positive average simulated value changes, with
negative scenario tails and individual delivery/resource regressions. All four
trained policies are experimental and none is promoted. The original 15-policy
registered study remains pending, with protected seeds unopened.

## Diagnostic units

Training reward is the sum of immediate scaled reward in each 8,192-transition
rollout. The reward scale is 0.001. It is not weekly evaluation profit or a
fixed-scenario performance curve. Explained variance and critic MSE are recorded
before the corresponding optimizer update; MSE uses squared scaled-reward units.
Missing diagnostics remain null. Scalar constraint-multiplier placeholders remain
null rather than being shown as active learned constraints. PI multipliers and
recorded normalized episode costs retain their actual values. Graph markers mark
recorded observations; line segments do not add measured samples.

Resource-regression counts include four resources at seven daily boundaries in
each of eight scenarios: 224 scalar comparisons. Battery deficits are kWh; gas
deficits are kg. Positive differences mean larger deficits. Delivery-regression
counts cover 28 obligations in each of eight scenarios: 224 commitments. Exact
recorded signs are preserved, including small floating-point differences; no new
post-hoc tolerance or selection criterion is introduced.

## Rebuild and verification

From the repository root, with the frozen research directory available:

```sh
PYTHONPATH=backend/src:simulator/src .venv/bin/python -m plant_backend.publish_adaptive_learning \
  --research-root /Users/filippogombac/Developer/PlantAnAgentResearch \
  --results-root docs/research/results
```

The exporter validates all source pins before publishing an atomic deterministic
replacement. A changed source fails rather than silently changing the product's
evidence. A normal checkout already includes the export and does not need to run
the publisher. Missing API data reports pending; malformed data returns 503.
Refreshing the page does not regenerate evidence.

The implementation's checks and visual inspection are recorded in the
[`journal`](JOURNAL.md). The controlling scientific results remain the
[`batch-008 report`](results/ADAPTIVE_RL_BATCH_008_RESULTS.md) and
[`batch-009 report`](results/ADAPTIVE_RL_BATCH_009_RESULTS.md).
