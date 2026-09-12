# Adaptive RL batch 001 — closed results

The five-policy screen completed and passed artifact audit. Reward scaling improved
critic conditioning, but this did not establish operating benefit. The low-exploration
PI policy recovered service relative to the initial policy, yet its economics were
worse than H and OperatingMPC and it failed strict safety in every scenario because
of forced electrolyser stops. No actor satisfied every daily resource return.

This is descriptive adaptive development evidence, not formal selection, held-out
confirmation, RL superiority, commercial savings or deployment readiness. The
original registered 15-policy study and protected seed sets remain unopened.

## Design and custody

The [frozen protocol](../RL_OPERATIONAL_BENEFIT_RESEARCH_PROTOCOL.md) used optimizer seed 101,
131,072 transitions per arm, four environments and sixteen rollouts. Each rollout
had 8,192 transitions and 320 minibatch optimizer steps: 5,120 optimizer steps per
arm. The task remained 168 hours, 135 observations, four actions, fifteen-minute
commands and one-minute physics. Training used the 9,600,000 seed stream; every
actor was evaluated on all eight fixed development seeds 9,700,000–9,700,007.

| Arm | Learner | Training reward scale | Initial Gaussian standard deviation |
| --- | --- | ---: | ---: |
| A | PI-Lagrangian | 1 | 1 |
| B | PI-Lagrangian | 0.001 | 1 |
| C | Scalar PPO | 1 | 1 |
| D | Scalar PPO | 0.001 | 1 |
| E | PI-Lagrangian | 0.001 | 0.2 |

A–D had identical initial actor and reward-critic tensors; E differed only in
initial log standard deviation. Evaluation always used raw EUR accounting.
The source was clean revision `238fe7ca763254b70af63a4ef0690a9d4523aa60`, tree
`becdfaa68e7924a7af75ca2794b32f09dbf8d4e2`. The launch gate passed 379 scoped Python
tests in 404.67 seconds, Black, strict mypy, compilation, CLI and whitespace checks.
It did not repeat the inherited 3,052-test complete gate or frontend checks.

The supervisor closed at `2026-09-12T14:23:56.520417+00:00`. All 165 process receipts
(five training, eighty evaluation and eighty replay) closed with empty failure
codes. The audit verified checkpoints, training ledgers, source bindings and all
slots: 64 complete physical episodes and 16 retained controller-failure prefixes.
A successful process receipt does not imply a complete controller episode.
Replay validates recorded commands and physical consequences; it does not
independently authenticate a reported controller exception.

## Every actor and every scenario

Weekly means below require all eight episodes to be complete and replayed. No
incomplete-controller prefix is included. The [compact JSON](adaptive_rl_batch_001_results.json)
retains all ten actors and all eighty seed slots, including failure text and exact
SHA-256 references for each summary, compressed minute archive and replay result.

| Actor | Complete / 8 | Mean delivery fill | Mean operating value, EUR | Strict safety passes / 8 | All daily returns pass / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial policy | 8 | 85.37% | −20,475.35 | 0 | 0 |
| H | 8 | 87.98% | −9,713.02 | 8 | 0 |
| OperatingMPC | 8 | 90.20% | −4,012.23 | 6 | 0 |
| M24 | 0 | — | — | 0 | 0 |
| E-beta | 0 | — | — | 0 | 0 |
| A: PI, raw | 8 | 76.45% | −32,070.27 | 0 | 0 |
| B: PI, scaled | 8 | 53.15% | −70,549.01 | 0 | 0 |
| C: scalar, raw | 8 | 57.55% | −62,371.54 | 0 | 0 |
| D: scalar, scaled | 8 | 73.55% | −37,592.42 | 0 | 0 |
| E: PI, scaled, low std | 8 | 90.53% | −11,637.03 | 0 | 0 |

M24 stopped in all eight scenarios with `ServiceMPCInputError`: the service MPC
model does not support degraded/offline DAC. E-beta stopped in all eight with
`BetaMPCSolveError`, an infeasible service-maximization LP. These are retained
comparator limitations, not evidence that a PPO policy beats a completed M24 or
E-beta episode. No fallback substitution was used to manufacture weekly results.

## Conditioning and operating counterevidence

| Arm | Final pre-update reward explained variance | Mean implied common clipping factor, all 16 rollouts | Final second reward-critic tanh saturation, absolute output ≥ 0.95 |
| --- | ---: | ---: | ---: |
| A | −0.000000059 | 0.0000254 | 100.00% |
| B | 0.92856 | 0.06685 | 20.47% |
| C | −0.000000017 | 0.0000522 | 100.00% |
| D | 0.93789 | 0.09654 | 12.58% |
| E | 0.97145 | 0.14355 | 9.24% |

With matched initial tensors, reward scaling reduced the first diagnostic critic
gradient norm from 8,818.53 to 8.56. B and D had higher explained variance than their
raw counterparts on rollouts 2–16 and larger implied actor norms after common
clipping on all sixteen rollouts. Every diagnostic still implied clipping.
These observations support poor numerical conditioning in the raw-reward setup;
they do not establish a useful control policy.

Operating evidence opposes a simple “better fit means better operation” conclusion.
B versus A lost 23.29 percentage points of mean service and EUR 38,478.73; D versus C
gained 16.00 points and EUR 24,779.12. Both directions held in every paired scenario.
PI versus scalar also reversed across reward scales: A exceeded C, while B fell
below D. Weighted cost/reward actor-gradient norm ratios averaged 2.61, 1.91 and
2.26 for A, B and E. Aggregate gradient cosine was negative in only 7, 5 and 1 of
sixteen diagnostics respectively. The multipliers did not explode or saturate.
This does not support persistent reward/cost opposition as a complete explanation.

Lower initial exploration reduced E's expected action-coordinate clipping to
approximately 0.00000026–0.00000057, compared with 0.306–0.321 for B. E improved over
B in every scenario and improved over the initial policy in every scenario.
However, E's mean additional deliveries cost EUR 1,924.01 versus H and EUR 7,624.79
versus OperatingMPC. Mean additional battery wear was EUR 2,520.19 / 3,229.16 and
electrolyser wear EUR 2,722.69 / 2,550.08 respectively. The adverse seed 9,700,005 is
retained: E delivered 103.22 kg less than H and 499.43 kg less than OperatingMPC,
with EUR 7,809.10 and EUR 18,037.91 lower operating value.

## Physical diagnosis

All forty trained-PPO evaluations failed strict safety because of forced
electrolyser stops. They respected the classifier's battery SOC threshold of
**0.05**, had no out-of-range actions and no unexpected critical/offline steps.
SOC near the dispatch floor of 0.15 is not a battery safety violation.

E had 88 forced stops, distributed `[14, 7, 12, 11, 11, 11, 12, 10]` across the eight
seeds. Sixty-nine stop intervals aborted a startup; 86 reached SOC at or below
0.151. Persistent positive targets near 1.5 MW and dawn/evening timing associate
these failures with insufficient startup-energy anticipation. H had no forced
stops. OperatingMPC had two: it reduced the target to 450 kW during an unfinished
startup. These observations motivate a prospective repair test, not a claim that
a repair has already succeeded.

E failed all 56 battery and H2 daily returns, 52 CH4 returns and one CO2 return.
It requested DAC power throughout all 4,503 declared outage minutes; the guard
removed unavailable allocations. There is no demonstrated learned fault adaptation.
Its predominately low-load operation and repeated starts explain substantial wear
components. Large terminal H2 accumulation also occurred in scalar arms, so it
cannot be attributed to PI constraints alone.

## Interpretation limits and evidence

All sixteen rollout curves, null values and per-resource diagnostics are retained
outside Git. The diagnostics use the deterministic first 256 env-major samples
before optimization; they are not actual shuffled SGD minibatches, Adam updates or
post-final-update measurements. The clipping factor applies jointly to actor and
reward critic in that diagnostic minibatch. Reward MSE is in scaled training units;
dividing by scale squared restores units but does not equate states or bootstrap
targets. Training lambda returns and sums of immediate training rewards are distinct
from evaluated weekly EUR. PI/scalar ledger KL uses different aggregation.

Checkpoint and ledger bindings were checked; diagnostic hashes and step/multiplier
joins were checked, but diagnostics are not embedded cryptographic commitments in
the final checkpoint payload. One optimizer seed and eight adaptive scenarios cannot
resolve algorithm-level reliability, causal mechanisms or site generalization.
No new outcome-tuned success threshold is introduced here.

The [compact evidence index](adaptive_rl_batch_001_results.json) records exact
bytes and SHA-256 digests for the launch gate, plan, closed report, audit, full
[descriptive report](/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001-analysis/descriptive/analysis.md),
[mechanism analysis](/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001-analysis/mechanism/plant-batch001-mechanism-results.md)
and [operational diagnosis](/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001-analysis/operation/plant-batch001-operational-diagnosis.md).
The raw audit SHA-256 is
`88048a9e5e5d1bb8bfd3d823824691daf82746fd488bbab5a1856ff57ab739a9`.
Its canonical JSON semantic digest, used by the descriptive report, is separately
recorded as `fccabb49ec22ff6e7dacbd32302c682034fd3f05cd18aefb25ddf93bea2d8e86`.
Raw checkpoints and minute archives remain on Lexar.
