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

Historical beta evaluation routes now reject registered task/checkpoint declarations and
the frozen training/development/held-out seed domains before environment construction,
policy admission or reset. This includes the paired evaluators, direct policy/heuristic/cell
execution functions and legacy evaluation loaders. It is a rejection-only boundary with no
authorization or bypass option; training-custody checkpoint loading and historical evidence
verification are unchanged. Unregistered fixture and non-beta workflows remain available.
Seventy-three reset-free rejection cases and the existing benchmark/policy/CLI regressions
passed together: **119 tests in 7.74 seconds**, plus strict typing on all three changed source
modules. The fixed admitted study executor and its separate stage authority are still pending.

Training custody is validated at `ad6ad1e`; full study integration remains pending. Its registered
authorization file is absent. Later training/development/held-out permissions remain separate.
The final workflow must execute against the admitted training source identity while retaining
stage evidence outside Git; it cannot edit the loader or report semantics after study outcomes.
No registered checkpoint, protected episode manifest/trace or sealed training log is opened.

## Latency measurement representation

The separate study latency contract is fixed before protected execution. Every
fresh PPO arm process performs exactly 1,000 deterministic policy calls on an
all-zero float32 vector of the admitted 135-field interface. Warm-up calls never
reset a plant, enter inference samples, change controller state, or retry after a
failure. Their output digests and any failure are retained. Policy inference is
measured around the actual `policy.act` call, including unsuccessful calls, with
its decision index and exact normalized-input digest. Controller time spans the
whole command-producing call. End-to-end time starts immediately before that call
and ends when the simulated interval returns (after M24 transition reconciliation).
It includes command preparation, dispatch, guard and simulated physics, but excludes
report publication and detached verification. It is simulation pipeline latency,
not a measured field-network or physical-plant response time.

All clocks use `time.perf_counter_ns`. Raw integer timestamps retain nesting,
order, completion/failure and one sample per attempted call. Existing per-minute
dispatch and solver-stage measurements remain the source for those channels;
failed attempts remain visible rather than entering only successful-call timing.
Every p50/p95/p99 uses nearest rank `ceil(p*n)`, on all samples in its stated unit,
with null percentiles for empty channels. The inference gate is inclusive at
100,000,000 ns and may use no subset of a policy's admitted timed calls. Host
identity, CPU, OS, dependencies, clock properties and Torch 1/1 settings are bound
before and after measurement; the external stage authority must also bind the
operator's absence-of-concurrent-benchmark-jobs assertion. Measurement evidence
alone cannot supply that authority or satisfy the complete held-out success rule.

This measurement layer is implemented in `study/latency.py` and
`study/latency_evidence.py`, with optional instrumentation in the fixed private
generic and M24 arm runners. The legacy arm format remains unchanged; a separate
synthetic latency file references the independently verified complete arm. Its
verifier checks every attempted decision, normalized inference input, completed
minute, solver stage, timing interval and host/runtime identity, then reconstructs
all-channel summaries. Unknown CPU identity is rejected. The record retains actual
warm-up attempts/returns/output digests, failures and one-use semantics. No host or
concurrency registration is inferred by this synthetic path.

The latency contract digest is
`71678ee3f3578c6fc2a8fef115790498bbba64584029a5acfc6c8c3d3753d558`.
The 77 new numerical, host and instrumented integration cases include an identical
timed/untimed PPO physical/economic digest, actual failed H/PPO calls, failed solver
stages, rehashed substitutions, identical file replacement and the inclusive
100 ms boundary. Together with existing evaluation/CLI/legacy-verifier checks,
**239 tests passed in 239.80 seconds**. Strict typing, formatting, compilation and
whitespace checks passed; the dependency lock is unchanged. The host fixture needed
permitted CPU metadata reads because sandboxed sysctl returned no CPU identity;
the admission requirement was not relaxed. These remain fixture and intermediate
release results. Production stage authority and full operational scorecards remain
required, followed by the complete clean release gate and explicit authorization.

## Operational scorecard representation

Operational summaries retain every registered episode slot, including missing and
failed execution, independently of economic pair eligibility. Each fixed cycle
index and resource reports both verified feasible/32 and feasible/observed
settlements with Wilson intervals. Missing settlements are not imputed as physical
failures. Resource feasibility uses the simulator's existing tolerance plus
`CYCLE_FEASIBILITY_EPSILON`. Deficit distributions use linear min/p25/median/p75/
p95/max and retain every observed value and its episode/cycle identity. Cycles
within an episode are never treated as independent binomial trials.

Episode and cycle records retain all additive economic components; complete
episode records also retain the entire verified physical/economic metric maps.
Partial execution is explicitly labelled, with observed and expected decisions
and minutes. Guard correction, unexpected critical/offline, forced-stop and
invalid-command rates retain their actual count and exposure denominator, with
null rates at zero exposure; these serially dependent rates have no binomial CI.
Guard intervention counts are separately retained because multiple corrections
can occur in one minute. The invalid-command denominator is attempted decisions.
Safety verified over a complete episode is reported separately from safety of
an observed prefix; preflight failures have no observed safety trial. Cycles with
no observed interval retain null economic totals and signals, not invented zero
outcomes. The initialized episode's verified zero economic prefix is retained.

Natural scheduled outage/no-outage cycle strata include all cycles with known
manifests, retain episode seed as the cluster, and report complete-cycle value,
delivery, energy, wear and resource-return summaries. A fresh PCG64 generator
with seed 20260905 draws the same 32 episode indices for all stratum metrics in
10,000 replicates. The statistic is the pooled mean of observed complete cycles
in the stratum; a resample with no observations is undefined, retained in its
count, and omitted only from the conditional interval. Empty strata have null
statistics. Calm/volatile/scarcity are analogous episode-level post-opening
strata; incomplete episode totals remain visible separately from full-horizon
descriptive summaries. Neither stratum defines a selectable evaluation subset.

Outage recovery is elapsed simulation time from the scheduled outage end to the
start of the first subsequent minute with CO2 capture greater than 1e-9 kg/h,
before the next outage or observed execution end. Immediate capture has zero
delay. This includes deliberate controller idling and is not mechanical repair
time. Every scheduled outage retains start/end, encountered duration, recovery
or censoring status (not reached, incomplete outage, resumed, no capture before
next outage/execution end). No unobserved recovery time is imputed.

Training summaries retain all five optimizer slots per family, separate training
completion and checkpoint admission rates, and all admitted rollout records.
Raw/scaled cost exposure, completed-episode cost means (null when absent), reward,
cost losses, multipliers, proportional/integral state, saturation and optimizer
diagnostics remain available for each rollout. Saturation fractions use observed
rollouts and have no independent-trial CI. Arithmetic summaries do not admit a
checkpoint; production admission must reopen complete training custody and the
registered artifact before supplying these ledgers.

The operational projection, all-slot summaries, training-ledger diagnostics and
synthetic detached physical report are implemented. The operational contract
digest is `97c67d7bbae8fed259ece7ca0bed5710078fcb26a49b77e66c14bbae99f64959`.
Fifty-three reset-free arithmetic/trajectory cases and 26 physical report cases,
together with 79 existing arm/pair checks, passed **158 tests in 332.90 seconds**.
These cover all four controllers, missing and failed arms, complete versus prefix
safety, unobserved versus initialized-zero economics, strict capture/censoring
boundaries, independent clustered uncertainty, all-family admission accounting,
rollout-chain substitutions and rehashed physical/report/file-replacement attacks.
The synthetic detached report reopens full physical evidence; it does not admit
a registered checkpoint or production stage registry. Training diagnostics still
require the production stage verifier to reopen training custody and the complete
artifact/state attestation before supplying the rollout records.
Strict typing of all four changed source modules, formatting of all six changed
Python files, compilation and whitespace checks pass; the lockfile is unchanged.
The complete clean repository release gate remains separate and pending.

## Fixed campaign registry and aggregate report

Each stage has 576 slots in episode-major order: H, M24, E-beta, then the 15
policies in the frozen family/optimizer order for each of the 32 episode seeds.
Each baseline is executed once per episode and referenced across comparisons.
Training failures retain all 32 corresponding policy slots as unexecuted, with
their custody reason. Missing, failed and rejected arms cannot disappear from
the registry or acquire an aggregate through a surviving subset. Registry
construction is read-only and grants no permission to instantiate/reset a plant.

Development and held-out evidence roots are deterministic siblings of the
training root, with suffixes `.development-v1` and `.heldout-v1`. Their registry,
authority, opening claim, cell claims/terminals, raw arm/timing files and exclusive
report destinations are fixed before opening. The training root's closed layout
is preserved. Stage storage and time limits require their own registered budget;
the training-only 5 GiB minimum is not a study-evidence capacity allowance.

Primary aggregates use independently gated policy/E-beta pairs. Secondary
PI/scalar and I/scalar aggregates independently gate the actual two policies at
the identical optimizer/episode identifiers. They do not require E-beta to
succeed, and they do not subtract two primary vectors whose eligibility might
omit a valid secondary pair. Every complete secondary vector enters the same
equal-policy family estimator and hierarchical bootstrap. Incomplete vectors
remain null with all missing indices. No champion-to-champion contrast is
substituted for this matched-seed estimand.

Development selection is computed once from primary vectors. Held-out analysis
uses those frozen choices and cannot reselect from held-out values. All sixteen
coefficient settings are aggregated with the same all-32 rule, retaining every
per-episode and aggregate sign reversal from the default setting. Latency pools
all admitted raw samples in each channel across all 32 slots; it never averages
episode percentiles. Missing measurements and unsuccessful calls remain visible.
The ten held-out clauses are individually reported, with missing evidence
preventing success. Numerical clause calculations do not grant stage authority
or issue the restricted simulated-controller claim; the detached production
stage verifier must first reconstruct custody, identities and every source.

The registry and aggregate calculation are implemented in `study/registry.py`
and `study/campaign_analysis.py`. The physical/latency/campaign regression passed
**210 tests in 345.49 seconds**. The final campaign arithmetic suite passed **52
tests in 33.44 seconds**, including an additional check that the five PI rows
have distinct exact identities and that the frozen champion is their unchanged
member. Frozen identities are required for held-out success, and the development
identity condition includes registered hardware. All calculations retain a null
production claim; constructing a registry or satisfying numerical conditions
does not open either stage.

The remaining production connection must independently reopen training custody
and registered artifacts, bind each cell's physical/timing files to the exact
registry, execute through durable one-use claims, and reconstruct the complete
aggregate from those sources. Hardware admission must preserve the separate
training and evaluation thread contracts; changing the training verifier's host
requirements is not an acceptable shortcut to evaluation's Torch 1/1 setting.
Stage authority and evidence must remain outside Git so the admitted training
source/runtime identity stays unchanged through both evaluation stages. A final
clean repository gate and concrete execution/storage budget precede explicit
authorization; no registered training or stage authorization currently exists.

## Evidence storage and training admission bridge

Large physical and timing documents use lossless canonical JSON in a single
gzip member. The raw canonical bytes and compressed file each retain their own
SHA-256 and byte count. Publication is exclusive, mode 0600 and fsynced; failures
retain the consumed filename and partial bytes. Both stored and decoded byte
limits are mandatory. Readers reject duplicate/non-finite/noncanonical JSON,
corruption, extra members, trailing data, indirect paths and file replacement.
Compression has no scientific or authorization semantics. The remaining stage
capacity measurement must establish suitable limits and sufficient storage;
compression alone does not establish that the study fits this machine.
The fixed registry now names `arm.json.gz` and `latency.json.gz` and binds the
archive schema before any stage opening. Historical synthetic JSON filenames
remain unchanged.

The training-input bridge calls the same complete production reconstruction as
the detached training verifier, with no cached-verdict or dependency argument.
It then reopens the admitted metadata and reconstructs all 128 rollout records
for every admitted policy. Failed and unstarted slots remain in the fixed 15-slot
scorecard. Training completion means a closed successful child process (admitted
or checkpoint-rejected); it is separate from final checkpoint admission. Failed
slots do not acquire diagnostics from partial or unadmitted artifacts.

The bridge binds the source/runtime/training host, complete report, all checkpoint
descriptors and a stable metadata snapshot of the closed evidence tree. Parent
recapture rejects source/host changes and identical-byte file replacement. It
runs under the original training thread contract and grants no evaluation
authority. Its in-memory output is not a persisted admission token: detached
study verification must call the production bridge again.

The combined archive/bridge/training-custody/operational/campaign regression
passed **291 tests in 120.40 seconds**, including the corrected cross-family
ledger substitution. Provenance and model/parser substitutes are explicit in
the reset-free bridge fixtures; their passing status is not registered training
or production checkpoint evidence. Strict typing passes the three implementation
modules. The aligned registry passes **16 focused cases in 0.76 seconds**, with
two additional rehashed archive/path substitutions rejected. Final review made
the tree snapshot retain forbidden artifact nodes without following them, so a
properly recorded failed artifact does not hide other admitted policies. The
expanded bridge passes **33 tests in 33.32 seconds**. Strict typing passes all
four changed source modules; Black checks all seven Python files. Compilation,
whitespace, unchanged-lock and both frozen protocol digest checks pass. These
remain intermediate implementation gates; full stage closure and the final clean
repository release are still pending.
