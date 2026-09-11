# Constrained-RL study analysis build

Status: **outcome-blind implementation contract; no study execution authorized**

This build implements the development, sensitivity and held-out analysis sections of
`CONSTRAINED_RL_BETA_PROTOCOL.md`. The task, three families, five optimizer seeds, seed
domains, learning budget, primary contrast and ten-part success rule remain unchanged.
The complete implementation and its source contract must be committed and validated before
registered training, because the registered policy loader binds the training source/runtime.

## Required layers and evidence

| Layer | Required behavior | Completion evidence |
| --- | --- | --- |
| Numerical contract | Exact 32-pair summaries, Wilson intervals, paired and hierarchical uncertainty, fixed profiles | Independent arithmetic fixtures and strict contract identity |
| Selection | All-32-or-no-score, IQM/CVaR/median/positive-fraction/lower-seed key; retain all five policies | Missingness, each tie-break, failed training/report and family accounting cases |
| Economic sensitivity | Policy minus E-beta; fixed additive coefficient changes; retain every reversal | Independent rescoring, orientation and detached tamper checks |
| Pairwise admission | Reconstruct full completion/safety/cycle/service gates from independently verified arms | H/M24 context failures do not block valid primary pairs; every exclusion tested |
| Campaign execution | All 15 policies and H/M24/E-beta on all 32 seeds; baseline evidence once per seed | Exact registry, stage authority, durable non-retry custody and failure retention |
| Scorecards and latency | Every frozen operational field and stratum; 1/1 Torch threads, 1,000 warm-ups, nearest-rank p99 | Host/source binding, raw sample accounting and protected-seed-free fixtures |
| Detached development/held-out verification | Reopen referenced evidence; reconstruct selection, statistics, sensitivity and ten clauses | Substitution, rehashing, missingness and stage/claim separation fixtures |
| Release and authorization | Complete clean source gate; explicit later user permission with budgets visible | No authorization is created by this build |

## Numerical representations fixed before outcomes

The source contract uses domain/schema `plant-beta-constrained-rl-study-analysis-contract-v1`.
It binds the upstream protocol ID/hash and the following representations in addition to the
already frozen numerical values. None is chosen from controller outcomes.

- A policy paired vector has exactly 32 finite real values, in the registered episode order.
  Missing or ineligible values are explicit nulls in evidence and forbid that policy's value
  aggregate. Exact zero is a tie. Boolean values and numeric strings are not observations.
- Standard deviation uses the sample convention (`ddof=1`). IQM averages sorted positions
  8 through 23 inclusive; lower-20% CVaR averages the seven smallest observations. Statistics
  are float64; computed non-finite values are rejected rather than serialized as valid scores.
- Each bootstrap calculation starts a fresh `numpy.random.Generator(PCG64(20260905))` and uses
  10,000 replicates. One set of episode resamples is shared by every statistic/profile from
  a given calculation. Percentile endpoints use `numpy.quantile(..., method="linear")` at
  0.025 and 0.975. A policy bootstrap resamples 32 paired episode indices with replacement.
- A family retains five optimizer slots. Only policies with complete eligible 32-pair vectors
  enter conditional family value summaries; the admission denominator is always five. Report
  the exact contributing optimizer seeds and missing slots. With no contributing policies,
  family value summaries and intervals are null.
- Family point statistics are equal-weight means of the contributing policies' corresponding
  32-episode statistics. The two-stage bootstrap draws optimizer indices first and then a
  common 32-episode index block across the drawn optimizer rows. Sharing that block preserves
  the pairing of common exogenous episodes and the shared E-beta baseline. The family interval
  targets this equal-policy aggregate; it does not treat 160 cells as independent observations.
- PI-versus-scalar family inputs match the five optimizer identifiers exactly. A matched pair
  contributes only when both full vectors exist. No Cartesian pairing or replacement occurs.
- Empirical performance profiles report the fraction of paired EUR deltas strictly above each
  fixed threshold: `-10000, -5000, -2000, -1000, -500, -250, -100, -50, -25, -10, 0, 10, 25,
  50, 100, 250, 500, 1000, 2000, 5000, 10000`. This is an absolute paired-value profile;
  thresholds are not fitted or extended after outcomes. Family profile intervals use the same
  optimizer/episode bootstrap, never a 160-trial binomial interval.
- Binomial intervals use the two-sided 95% Wilson formula with
  `NormalDist().inv_cdf(0.975)`. A zero denominator reports null rate and interval. Counts and
  denominators are always retained. The policy probability of improvement is strictly-positive
  pair count divided by 32, with exact ties reported separately.
- Coefficient sensitivity uses the exact existing additive component names and one-factor
  multipliers `0, 0.5, 1, 2`. Each setting recomputes the weighted additive difference as
  policy minus E-beta. Methane liability changes leave unmet-service penalty unchanged; wear
  jointly changes battery-throughput and electrolyser-life reserve. Sign changes, including
  transitions to/from exact zero, are all retained. Sensitivity is not a new selection objective.

Pure numerical results cannot establish verified arm eligibility, grant stage permission or
issue a production campaign claim. Those responsibilities belong to the later fixed execution
and detached-verification layers. Fixture tests use explicit synthetic values only; they do not
instantiate a plant, load an admitted policy, inspect an outcome, or consume protected seeds.

## Source and stage boundary

The next evidence layer uses individually sealed arm records so H, M24 and E-beta can be
retained once per episode and referenced across all policy pairs. Its inner result uses the
new `plant-beta-constrained-rl-study-arm-result-v1` format/domain. It retains the existing
complete minute/decision/economic witnesses and own-arm classification, while removing the
legacy all-four `value_comparison_eligible` field and `paired_cell_ineligible` exclusion.
The historical benchmark format and its verification semantics remain unchanged.

An arm envelope binds the protocol, numerical contract, stage, episode seed, actor, full task,
PPO checkpoint descriptor (null for baselines), source provenance, canonical M24 settings,
manifest identity, audit manifest and complete inner result. Synthetic envelopes and pairs
have distinct schemas/filenames, fixture-only episode seeds and null production claims.
Pairwise analysis must independently reverify both referenced arm files, compare source,
episode identity, initial state and common public issue history, and apply the full
completion/safety/cycle/service gate before producing a delta or sensitivity result.
H/M24 outcomes remain outside the primary pair gate. Arithmetic fields cannot substitute for
the retained physical witnesses. Production study entry points remain closed until stage
authority, exact registry, checkpoint custody and the complete release gate are connected.

The numerical, family-selection and additive-sensitivity kernels are implemented in
`backend/src/plant_backend/rl/v2/study/`, with 82 reset-free numeric/adversarial tests.
The source contract digest is
`94c4ce4b665ab31228b124f9fe056c98fb37ab5b349a12f785c2368b4d095d72`.
Independent oracles cover trimmed positions, lower-tail rounding, linear bootstrap endpoints,
common episode clustering, optimizer variation and exact matched-seed contrasts. Selection
tests exercise every tie-break and retain failed/unverified/incomplete policies. Sensitivity
reconstruction rejects altered coefficients, orientation, arithmetic, settings and reversals.

The current selection and sensitivity result verifiers reconstruct calculations only. They
return `source_evidence_verified: false` and `result_claim: null`; source-ledger admission and
production report verification remain separate, incomplete layers. No numerical result is a
replacement for independently verified physical/minute/cycle evidence or campaign custody.

The standalone arm/pair layer is now implemented for explicit synthetic files only. Its 79
integration/adversarial cases reconstruct real minute/decision witnesses, demonstrate a valid
primary pair despite a failed H context arm, cover every independent pair gate and inclusive
service tolerance, retain failed/missing arms without value, and reject rehashed source/ledger/
report substitutions and identical file replacement during graph verification. These files
bind full checkpoint descriptors but do not reopen/admit registered policy artifacts. Their
verifier results explicitly carry `registered_checkpoint_verified: false` and null production
claim. Production stage authority, exact registry and training-custody binding remain required.

The shared verifier also checks exact initial economic/physical metrics for zero-interval
failures; previously that case skipped accounting reconstruction. The combined new evidence,
numerical and historical benchmark-verifier suites passed **204 tests in 331.05 seconds**.
Strict typing of the three changed source modules, formatting, compilation, whitespace and
dependency-lock checks passed. Both frozen upstream protocol hashes are unchanged. This is
an intermediate implementation regression, not the final clean study release gate.

Training custody is validated at `ad6ad1e`; full study integration remains pending. Its registered
authorization file is absent. Later training/development/held-out permissions remain separate.
The final workflow must execute against the admitted training source identity while retaining
stage evidence outside Git; it cannot edit the loader or report semantics after study outcomes.
No registered checkpoint, protected episode manifest/trace or sealed training log is opened.
