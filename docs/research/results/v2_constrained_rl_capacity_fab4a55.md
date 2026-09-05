# First constrained-RL capacity attempt — retained engineering failure

Date: **2026-09-05**

Status: **verified failed report; no capacity projection and no controller claim**

## Identity and scope

The first and only capacity attempt at source revision
`fab4a552a4fb2f1d1d6efbcb757a87e4f8a9cf73` executed the three fixture-domain cells frozen in
`CONSTRAINED_RL_CAPACITY_PROTOCOL.md`. It used no registered, development, or held-out seed. The
external evidence root is:

```text
/private/tmp/plantanagent-beta-rl-capacity-fab4a55
```

The retained canonical report has:

```text
schema                         plant-beta-constrained-rl-capacity-report-v1
raw file SHA-256               9b1bd155d70d8356dadc4a1faa86f9bb721c163af4f5e51004a3509362a97134
report domain SHA-256          f3d2ab32dbb86c7c4268d3b114e2ee44aff5d29b02f2c4e4e725e162acd20895
deterministic evidence SHA-256 dd41dac90b9fffae63b639d27fe07d31baf225ec8a0025d3a40e8dfd81d4491d
runtime provenance SHA-256     0f6fd3b9ba8dc5c0a40f37034a480cd44557f03ee8b099acc8971d90bdd5ffab
status                         failed
result claim                   null
campaign projection            null
```

The report passed its immediate strict verification in the original process. A separate
same-host verification with the same host-observation privileges reproduced the report identities
and failed status. Verification from the restricted development sandbox could not observe the CPU
brand or physical-core count and therefore correctly refused to call that partial host view equal
to the recorded host; this is a verifier execution-context difference, not a report mutation.

An initial Make invocation was refused before the runner or output root existed because the
restricted sandbox could not open uv's cache. It consumed no cell and is not an experimental
attempt. The admitted invocation then ran once with the required exclusivity and first-attempt
assertions.

## Retained cells

| Order | Family | Config SHA-256 | Wall seconds | User CPU seconds | System CPU seconds | Exit |
|---:|---|---|---:|---:|---:|---:|
| 1 | PI-Lagrangian | `58b845034376a24706b0f2c0c4a776a6526d23fcdde7253d62bd0efcfa590ca7` | 127.084657500 | 124.240814000 | 2.348562000 | 2 |
| 2 | Integral-only | `0d3ea97ec92d30dc9a20cf1f9c5d8210456d192f64f40ee6adb2b96cc30ba01d` | 133.561394000 | 130.170172000 | 2.555828000 | 2 |
| 3 | Scalar PPO | `0198b613fd84e9ed4082c945ba5c9ceef8357485705e3d08abc6ad65a9ba7d1f` | 130.659857167 | 127.406795000 | 2.198026000 | 2 |

All three failure records are categorical `training_process` / `ChildProcessError` records. No
checkpoint, measurement, per-family projection, campaign projection, or result claim survived.
Each stderr file is 71 bytes with the same SHA-256
`af0b82782468004843c654611226c87347c7daa5051fc3aaf9ad73ab9098d00c` and identifies the same
checkpoint-construction invariant: `rollout cost scaling is incoherent`. The raw stdout files are
retained externally and identified in the report but are not copied here.

These wall times are failure diagnostics only. The frozen protocol forbids using a failed cell to
produce the 15-policy campaign projection, so no extrapolation is made from them.

## Diagnosis

Every transition already proves the exact semantic relation
`training_cost = raw_cost / 7`. The rollout collector nevertheless accumulated raw and divided
training values through two independent binary64 summations. Checkpoint-state admission later
tested the aggregate relation at an absolute tolerance of `1e-15`. Floating-point addition is not
associative: summing the divided values and dividing the independently summed raw values can differ
by more than that tolerance after the 84 settlement records in one four-environment rollout. The
failure therefore occurs while serializing/verifying evidence after optimization, not because the
transition contract, controller, environment, or algorithm changed.

This diagnosis is family-neutral: the scalar arm also retains cost evidence for auditing while
keeping it outside its optimizer, and it failed the same serialization invariant. The repair must
canonicalize aggregate evidence only. It may not change transition costs, cost advantages,
multipliers, gradients, PPO settings, policy initialization, episode seeds, environment behavior,
or the registered campaign.

## Outcome-blindness incident

During failure diagnosis, the first raw stdout file was opened. Stable-Baselines3 had emitted its
ordinary rollout table, which includes fixture reward and cost-related summaries. This was an
operator-procedure breach of the capacity protocol's no-outcome-inspection boundary. No value from
that table is reproduced, analyzed, compared, or used to select the repair. The other two stdout
files were not opened. The exposed data used only fixture seeds 9,880,000--9,880,015 and is
permanently ineligible for registered evaluation, but the incident is retained here rather than
silently omitted.

The repair is frozen separately in `CONSTRAINED_RL_CAPACITY_REPAIR_001.md` before source changes.
It is restricted to mathematically canonical evidence aggregation and adversarial tests. Any new
capacity execution must occur once on a new clean revision and a new external root. Revision
`fab4a55` must never be retried.
