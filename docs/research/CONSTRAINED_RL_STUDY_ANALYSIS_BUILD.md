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

Training custody is validated at `ad6ad1e`; full study integration remains pending. Its registered
authorization file is absent. Later training/development/held-out permissions remain separate.
The final workflow must execute against the admitted training source identity while retaining
stage evidence outside Git; it cannot edit the loader or report semantics after study outcomes.
No checkpoint, manifest, forecast, market/outage trace or sealed training log is opened here.
