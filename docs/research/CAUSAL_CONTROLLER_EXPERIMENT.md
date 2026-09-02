# Transparent causal-controller experiment

Status: **pre-outcome development protocol; no frozen 72-hour causal-controller
outcomes have been opened.**

This note defines the next transparent-controller comparison for commitment-alpha
Environment V2. It originated from the causal evaluation seam at source revision
`5a9b9aa` after the delivery-rate calibration and has been amended during pre-outcome
hardening. The candidate controllers, implemented defaults, failure rules, source audit,
and development analysis are declared below. This is not yet an execution-complete
**prospective** preregistration: the final clean controller revision, prospective sample
size, held-out seed list, statistical margins, and integrated validation record must be
reviewed and frozen in an amendment before any prospective outcomes are opened.

Nothing in this note is formal validation, held-out evidence, evidence that RL works,
an optimality result, or commercial-performance evidence. The synthetic weather and
technology-labelled commitment model remain engineering fixtures rather than
site-calibrated weather or vendor-calibrated equipment.

## Prior inspection and contamination boundary

The following conditions have already been inspected and are development data:

- delivery rates of **35 kg/h** and **42 kg/h**;
- seeds **0--2**;
- **72-hour** episodes; and
- the current synthetic weather regime and commitment-alpha equipment profile.

In addition, seed **3** under the current generator is excluded from future holdout use.
Its six-hour CLI smoke output was unintentionally printed when a schema-version assertion
failed during evaluator hardening. It is not part of the planned 72-hour development
matrix, but the conservative split rule treats the seed—not merely that horizon—as seen.

The normal causal-comparison CLI development allowlist is exactly seeds **0--2**. A seed
outside that allowlist is rejected unless the operator passes
`--acknowledge-seed-contamination`; using that acknowledgement declares every selected
seed to be development-contaminated and cannot create prospective evidence. Seed 3 stays
contaminated regardless of the requested horizon or any later CLI override.

The 35 kg/h cell was selected as the primary development target after inspecting the
legacy and fixed-controller calibration. The 42 kg/h cell was selected as a development
stress point after inspecting its failure behavior. Both may be used to implement,
debug, and tune the transparent rules, but neither the cells nor their already inspected
outcomes may later be relabelled as formal validation or held-out stress evidence. The
full provenance and earlier observations are recorded in
[`COMMITMENT_ALPHA_CALIBRATION.md`](COMMITMENT_ALPHA_CALIBRATION.md).

## Questions and hypotheses

The core experiment asks three ordered questions.

1. **Pragmatic delivery-control question (H1).** Does replacing the legacy V1 heuristic
   with a transparent, deadline-closing rule that consumes the causal delivery view and
   a fixed availability prior reduce unmet methane and improve delivery service under the
   same Environment V2 actuation, guard, commitment dynamics, and episode?
2. **Incremental issued-mean question (H2).** Within that same rule and temporal
   resolution, does replacing the fixed availability prior with the mean from the issue
   available at the current decision boundary improve reliability or, conditional on
   preserved reliability and inventories, operating efficiency?
3. **Incremental downside-band question (H3).** Holding the issued product, planner, and
   resolution fixed, does changing the availability value from issued mean to a frozen
   50/50 mean/p10 mixture improve tail reliability or, conditional on reliability and
   inventories, operating efficiency and commitment exposure?

H1 is a useful operational comparison, but it is a bundled intervention: the new rule
changes both controller logic and information. It does **not** identify the value of
delivery information alone. H2 and H3 use adjacent, matched-planner contrasts. The H2
arms differ only in whether canonical aggregate-band values come from a frozen prior or
the current issued mean. The H3 arms differ only in the zero versus 50% p10 weight applied
to that same current issue. Neither contrast establishes the value of forecasts in
general, and H3 does not establish that p10 is calibrated as a continuous-power bound.

The primary delivery endpoint is episode unmet methane in kilograms; lower is better.
At a fixed rate and horizon, aggregate fill is a monotone restatement of the same mass
balance and will be reported rather than counted as independent corroboration. The
primary risk endpoint for a sufficiently large prospective suite is upper-tail CVaR of
episode unmet kilograms. Service level, worst-commitment fill, lower-tail service,
energy intensity, and operating exposures are ordered supporting outcomes. The CVaR
tail fraction, smallest relevant effect, service non-inferiority margin, confidence
level, and multiplicity rule are deliberately not invented in this design draft; they
must be fixed before a held-out run.

## Core arms and clear-sky diagnostic

| ID | arm | declared online information | role |
| --- | --- | --- | --- |
| `A0` | legacy V1 heuristic through the V2 adapter | latest one-minute boundary telemetry and active safety constraints; no delivery view, preceding-interval aggregate, commitment profile, availability prior, or issued forecast | frozen historical reference |
| `A1` | delivery-aware fixed-climatology deadline-closing rule | detached structured V2 state, current delivery view, episode progress, decision cadence, static plant configuration including solar geometry, active constraints, commitment profile, canonical public forecast-band boundaries, one frozen unconditional resource factor, and the declared piecewise-constant band interpretation; issue explicitly removed | pragmatic comparator and matched-resolution H2 control |
| `A2` | otherwise identical issued-mean deadline-closing rule | exactly the A1 causal inputs, with values from the current public issued-forecast means replacing the fixed prior within the same bands | incremental issued-mean treatment |
| `A3` | otherwise identical risk-adjusted issued-forecast rule | exactly the A2 issue, with a frozen 50% p10 weight added to the same mean values | incremental downside-band treatment |
| `D1` | analytic clear-sky delivery-aware rule | the delivery-aware causal inputs and exact `PlantConfig.solar` geometry, with no issued forecast | non-core smooth-resolution diagnostic; the earlier three-arm draft called this A1 |

The A0 adapter must preserve the shipped legacy heuristic's behavior exactly. A1--A3 and
D1 use the same stateless physical planner and immutable settings schema:

| setting | implemented development value | meaning |
| --- | ---: | --- |
| `delivery_buffer_fraction` | 0.03 | extra methane targeted above the next requested delivery, capped by tank capacity |
| `feedstock_reserve_fraction` | 0.03 | additional stoichiometric H2/CO2 basis as a fraction of the next requested delivery |
| `climatological_resource_factor` | 0.6803843775134744 | A1 band value multiplier applied to analytic clear-sky energy; frozen unconditional expectation of the declared synthetic weather prior, not fitted episode truth |
| `forecast_p10_weight` | 0.00 in A1/A2/D1; 0.50 in A3 | issued-band value is `(1 - weight) * mean + weight * p10`; irrelevant to the non-issued availability sources |

The exact climatological factor is
`E[sigmoid(N(logit(0.70), 0.75²))]`, evaluated once by 128-node Gauss--Hermite
quadrature from the generator's declared stationary latent prior. It is a frozen synthetic
prior parameter, not a value selected from controller outcomes.

Every delivery-aware arm converts its declared availability source to equivalent
full-power hours over the current deadline. That value becomes the process planning
horizon, with a one-physics-interval floor. The same source is queried again over the
inferred next delivery period, and the shared transform
`clip(1 - equivalent_hours / period_hours, 0, 1)` determines how much of one inferred
recurring delivery is carried across settlement. There is no fixed 50% carry and no
clock-time-only horizon. The recurrence assumption remains necessary because the causal
boundary reveals only the current requested amount and cadence, not the private future
schedule. Electrolyser productive time uses the same availability transform after
excluding observed cold/hot startup, cooldown, and minimum-down delay. The held command
asks for the desired running endpoint; the one-minute equipment kernel, rather than the
15-minute controller, applies each ramp.

Methanator demand is capped by H2 and CO2 already available at the boundary so scarce
power cannot be trapped in a gas-starved product unit. Current solar is then allocated to
immediately feedable methanation, the electrolyser's applicable physical floor, DAC, any
remaining electrolyser target, and residual battery charging. Battery support is a
contingent electrolyser top-up limited to energy sustainable above the safety reserve over
the complete held decision. For a running stack, allocation first secures the direct-solar
complement that sustainable battery support cannot cover at the stable floor; intentional
charge/support overlap is not miscounted as additive battery discharge. When boundary
solar is zero, nonzero forward shares are built from the same current causal process
targets so solar appearing inside the held interval can displace support instead of being
curtailed. The algorithm and all implicit constants remain part of the source freeze; the
four settings above are not permission to change other behavior without versioning the
controller.

D1 obtains availability causally by analytically integrating the configured unit-peak
half-sine between `PlantConfig.solar.sunrise_h` and `sunset_h`, anchored at the current
boundary time. The periodic integral crosses sunrise, sunset, midnight, and multiple days
without sampling simulator truth. It is a known design envelope, not a weather forecast.

A1--A3 instead share the canonical issued-product intervals 0--1, 1--6, 6--12, and
12--24 hours from the decision issue. Each aggregate value is interpreted as
piecewise-constant within its band, so a queried interval receives
`overlap_duration * band_value`. A1's band value is the frozen climatological factor
times the band's analytic clear-sky average; A2 uses the issued mean; A3 uses
`0.50 * mean + 0.50 * p10`. Time beyond the published bands receives zero availability
credit.

Uniform-within-band disaggregation is an explicit estimator assumption needed to avoid
collapsing short or boundary-straddling horizons to one physics interval. The issue does
not reveal the actual sub-band timing of PV, so neither the controller nor report may
present this interpolation as observed continuous availability. D1 has analytic
continuous resolution while A1--A3 share the coarser piecewise-constant resolution;
D1-versus-any-band-arm is therefore a resolution-and-source diagnostic, not a clean
forecast estimand. H2 and H3 avoid that mismatch by comparing only identically resolved
band arms. A2/A3 receive no private issue object, future vintage, or future truth. Any
other planner, transform, band interpretation, setting, or state-behavior difference
invalidates the claimed adjacent information estimand.

The issued p10 is the tenth ensemble percentile **after member-wise aggregation over
a lead window**, expressed as a fraction of PV nameplate. It is not a guarantee that
continuous power at that level will exist, a lower bound on every minute in the window,
or a promise that startup, ramp, minimum-load, storage, or process-coupling constraints
can be satisfied. The controller and report must not reinterpret it as one.

MPC, perfect-foresight planning, learned policies, and RL are outside the core four-arm
experiment. A later perfect-information arm must be labelled an unattainable diagnostic
bound and must never share its future truth with A0--A3 or D1.

## Causal information boundary

`ControllerObservationV2` is the common algorithm-neutral online boundary. At decision
time it contains:

- a detached deep copy of `ControlStateV2`, including current boundary state,
  preceding-interval aggregates, and the public commitment snapshot;
- a narrow `DeliveryObservation`: steps to the next deadline, delivery cadence, next
  requested mass, and cumulative requested and delivered mass;
- either the current `PublicForecastIssueV2` or an explicit `None` ablation;
- episode progress; and
- the decision interval in seconds.

Static plant configuration, active safety constraints, and the commitment profile are
explicit controller-call arguments. The private delivery schedule beyond the current
view, environment or simulator object, episode seed, scenario identity, future realized
weather, unissued forecast vintages, raw forecast random-stream hash, complete replay
manifest, and future reward/accounting state are not controller inputs. Audit identities
remain evaluator-side. Forecast-blind execution calls `without_forecast()` before the
controller is invoked.

The 57-value RL observation is encoded from the same causal source, but that fact does
not establish representation equality. The structured controller interface exposes
typed, uncompressed values and separately supplied configuration, constraints, and
commitment metadata that are richer than the current normalized 57-vector. No core or D1
result may therefore be described as an equal-representation comparison with RL, and no
claim about RL follows from this experiment.

## Experimental unit, pairing, and execution

The independent experimental unit is an episode seed within a frozen regime, not a
decision step and not an individual delivery settlement. All inferential contrasts are
paired within that unit.

For development, the core matrix is 35 and 42 kg/h, seeds 0--2, and 72 hours per
rollout, with the existing six-hour delivery schedule, one-minute physics step,
15-minute controller cadence, and commitment profile enabled. This is a 24-rollout
development matrix: two rates by three seeds by four core arms. D1 is a separately
labelled diagnostic and is not silently added to a core contrast. The matrix is for
controller construction and failure diagnosis only.

For every arm and seed, the generic evaluator must:

1. construct a fresh `PlantEnvV2` with the same episode, plant, constraints, and reward
   configuration;
2. reset it with the paired seed and capture the immutable audit manifest before the
   rollout;
3. reset a fresh copy of the controller;
4. expose only the declared controller observation and static call arguments;
5. send the returned `Action` through `step_action`, so all arms share sanitization,
   the safety guard, physical routing, commitment dynamics, reward, and accounting; and
6. retain the controller's strict-JSON descriptor, label, information-set declaration,
   complete metrics, provenance, full manifest hash, exogenous episode hash, and the
   typed pre-guard requested-action trace, controller-side invalid/clipping flag, and
   canonical SHA-256 digest.

The prototype descriptor is validated before copying. Each fresh copy is reset and checked
against that descriptor before execution, and checked again after every decision. Copy,
reset, descriptor, computation, output-validation, and action-application exceptions are
retained as structured failed runs with phase, exception type, bounded message, completed
decisions, elapsed hours, partial metrics, and the requested-action prefix accumulated
before failure. A failure remains an attempted arm for execution accounting; unavailable
metric pairs are explicit and never imputed. Environment construction/reset failures that
prevent an audited attempted-run record are infrastructure failures and abort the
invocation rather than being mislabeled as controller outcomes.

For the complete core study, both the full manifest and exogenous episode hashes must
match across all planned arms for each seed, and every arm must be present exactly once
and complete successfully. A retained controller-side failure invalidates **all aggregate
core-study contrasts** for that invocation. Per-seed attempts, failures, partial metrics,
and any otherwise computable raw pair remain in the diagnostic artifact, but no successful
subset may be promoted to an aggregate headline. The CLI prints and, when requested,
writes that artifact, then exits nonzero.

A missing or duplicated arm, mismatched full/exogenous manifest, undeclared seed/controller
run, reused environment, or alternate action path is instead a structural pairing failure.
The evaluator fails closed and returns no artifact; `write()` independently revalidates
mutable results and refuses such evidence. Structural failure must not be represented as a
controller outcome.

The full manifest match is deliberately stronger than the exogenous match. In a later
commitment-off/on ablation, full manifests should differ because physics differ, while
the exogenous episode hash must still match. Such ablations must be reported separately
from the core same-environment contrast.

Source provenance is captured at both the start and end of execution. Each snapshot
includes the Git revision and dirty state, a canonical SHA-256 and byte count for the
reproducible full-index binary tracked diff, and every non-ignored untracked path with its
kind, byte count, and content SHA-256. An explicit completeness flag is false if any part
of that identity is unavailable or unreadable. The execution source audit compares the
two canonical worktree identities and is invalid if either is incomplete or if source
changes during the run. This is a fail-closed infrastructure abort: the evaluator raises,
returns no result artifact, and the CLI exits nonzero without writing one. Source contents
are not embedded in the result.

Prospective execution requires a complete identity and a clean committed source tree at
both boundaries; it fails closed otherwise. `--allow-dirty-development` is an explicit
development-only escape hatch for attributable local work and must never be used for a
prospective or held-out claim. Dirty-source identity is forensic provenance, not a
relaxation of the clean-source evidence rule.

## Outcomes and direction of benefit

Raw totals and rates are retained for every episode. Benefit-oriented paired deltas are
defined so positive means the treatment arm is preferable: treatment minus control for
delivered mass, fill, and service, and control minus treatment for unmet mass, energy,
curtailment, starts/trips, and safety interventions. Initial/terminal inventory and both
action-application gap measures are diagnostics with raw treatment-minus-control deltas but
**no automatic benefit sign**. Reports must show the original metric units so a sign
convention cannot hide a material trade-off.

| domain | required measures | preferred direction and interpretation |
| --- | --- | --- |
| delivery | requested kg, delivered kg, unmet kg, aggregate delivery fill rate, commitment service level, worst commitment fill rate | more delivered/fill/service and less unmet; settlement events remain clustered within episode |
| service tail | per-episode worst commitment fill; across-episode lower-tail fill/service and upper-tail unmet; worst episode; CVaR | better lower tail and lower upper-tail unmet; never infer a stable tail from the three development seeds |
| inventory balance | complete initial and terminal H2, CO2, and CH4 mass; initial and terminal battery SOC and stored energy; each inventory change | no automatic benefit sign; exposes delivery or energy gains obtained by liquidating feedstock, product, or battery stocks |
| energy intensity | physically accepted process energy per delivered kg, plus its numerator and denominator | lower only at comparable service and inventory change; zero-delivery episodes are undefined, never assigned a convenient finite value, and suppress the paired aggregate rather than yielding a defined-episode-only comparison |
| renewable use | available solar kWh, curtailed kWh, and solar utilisation | less curtailment at preserved service; report absolute energy as well as percentage |
| storage | battery discharged kWh, minimum SOC, and the initial/terminal SOC and stored-energy balance above | diagnostic constraint/efficiency outcomes, not free energy |
| commitment | cold and hot starts, aborted starts, controlled and forced stops, startup energy, standby energy, and nonproductive energy | fewer forced/aborted starts and less nonproductive energy, subject to delivery performance |
| operating exposure | low-load and operating time, ramp-limited intervals, absolute load movement, and per-mode time | mechanism and trade-off diagnostics; no unvalidated wear or monetary interpretation |
| safety guard | guard intervention count, corrected physics steps, out-of-range decisions, and request-to-guard intervention kWh | fewer interventions; hard safety is a constraint, not a reward trade |
| action application | guard-to-plant `dispatch_gap_kwh` and request-to-plant `request_application_gap_kwh` | raw diagnostic distances with no benefit sign; distinguish safety projection, routing, equipment refusal, and deliberate overlap cancellation |
| availability | shutdown steps and any hard-invariant failure | zero hard violations is required; failure cannot be compensated by scalar reward |

For energy intensity, the numerator is the non-double-counted physically accepted energy
of the electrolyser, DAC, and methanator, with battery support counted once through the
electrolyser total. Episode metrics now retain each channel, their sum as
`accepted_process_energy_kwh`, and
`accepted_process_energy_per_delivered_kg`. The ratio is JSON `null` before any product is
delivered; it is never assigned zero. The machine summary retains an explicit defined-value
count, the human report suppresses a mean unless every planned run defines the metric, and
the paired aggregate is suppressed unless every paired episode defines both values. Solar
availability, requested power, or a double-counted battery channel is not substituted for
physical process energy.

Delivery and energy intensity must be interpreted jointly with the complete paired
inventory-change vector. More delivery can be financed by ending with less CH4, H2, CO2,
or stored battery energy; a lower accepted-process kWh/kg can likewise reflect consumption
of material or energy accumulated before the episode. Neither supports an efficiency or
service claim unless service and terminal stocks are comparable under the predeclared
decision rule. Paired seeds establish equal starting states, but both initial and terminal
values remain in the artifact so that equality and liquidation are auditable rather than
assumed.

`electrolyser_nonproductive_energy_kwh` remains startup plus hot-standby energy.
`guard_intervention_kwh` is request-to-guard correction, while `dispatch_gap_kwh` is the
integrated guard-to-plant **routing/refusal distance**. The latter can include intentional
charge/support overlap cancellation and is not automatically a loss of control authority.
`request_application_gap_kwh` spans the requested-to-plant path and can inherit the same
mixed mechanisms. Neither gap has a universally favorable direction, so both receive raw
deltas only. The measures answer different questions and must not be collapsed into an
ambiguous “guard” measure. Scalar reward and reward components are diagnostics only and
cannot establish a hypothesis.

## Analysis plan

The three planned core paired contrasts, declared by controller order before execution,
are:

- **A1 versus A0:** pragmatic value of the delivery-aware fixed-climatology replacement.
  This must be described as bundled controller-plus-information value.
- **A2 versus A1:** incremental value of current issued aggregate means versus the frozen
  unconditional prior, at identical band resolution and within the otherwise identical
  deadline-closing rule.
- **A3 versus A2:** incremental value of adding the frozen p10 weight to the same issued
  aggregate product and rule.

A2 or A3 versus A0, A3 versus A1, and any D1 comparison may be displayed for context but
cannot isolate a single treatment named above. Results are first stratified by rate and
regime. They are not pooled across regimes unless weights and a target population are
declared before outcomes are seen.

The three-seed development screen will show all episode values and paired deltas; means,
medians, minima, and maxima may be descriptive, but confidence intervals, probabilities
of improvement, and CVaR from three seeds are not confirmatory evidence.

For a future adequately sized prospective suite, the analysis must report the paired
mean and median delta, interquartile mean, dispersion, worst paired delta, probability
of improvement, and a paired uncertainty interval. Resampling, if used, is over complete
seed/regime episode rows so the arms remain paired; decisions and settlements are never
resampled as independent trials. Upper-tail CVaR of unmet kilograms is the mean of the
predeclared worst tail fraction, and lower-tail CVaR of fill or service is the mean of
the corresponding lowest tail fraction. The tail fraction and finite-sample estimator
must be frozen with a sample size large enough to support them.

H1 is supported prospectively only by a predeclared improvement in unmet mass and
compatible fill/service behavior without a disqualifying safety or inventory trade-off.
H2 reliability is supported only if A2 improves the predeclared unmet/tail endpoint over
A1; H3 is supported only if A3 improves its corresponding predeclared endpoint over A2.
If reliability is statistically indistinguishable within a predeclared non-inferiority
margin, a separately labelled secondary efficiency claim may be supported by lower energy
per delivered kilogram or lower curtailment/nonproductive exposure only when the complete
terminal H2, CO2, CH4, and battery energy/SOC vector is also comparable. The margins and
decision rules must be frozen before the held-out suite; they may not be selected to fit
observed results.

## Development tuning discipline

Seeds 0--2 at 35 and 42 kg/h are the only outcome-bearing episodes authorized for rule
tuning under this draft. Deterministic unit fixtures may be used for correctness tests,
but must not become an undeclared outcome search over additional weather seeds.

Every tried controller version must record its source revision, strict-JSON settings,
information set, development cells evaluated, selection criterion, and outcome summary.
Failed and discarded variants remain in the research log. A1--A3 share the planner,
canonical band interpretation, availability-to-horizon transform, next-period
scarcity/carry transform, and operational settings. Only the predeclared band-value source
or functional differs: fixed climatology, issued mean, or issued 50/50 mean/p10. There is
no separate favorable retuning of any other A2 or A3 behavior. D1 is retained only as the
explicitly different-resolution diagnostic.

Before any prospective evaluation, freeze in a clean revision:

- controller source and all numerical/categorical settings;
- serialized controller descriptors and information-set text;
- environment, interface, checkpoint, weather-generator, commitment-profile, and
  dependency-lock identities;
- episode cadence, horizon, delivery schedule, plant, safety constraints, and reward
  configuration;
- metric formulas, energy accumulator, tail fraction, uncertainty method, effect and
  non-inferiority margins, exclusions, failure handling, and reporting script; and
- the exact prospective seed list, regime identifiers and parameters, sample size, and
  planned comparisons.

Any controller or analysis change made after prospective outcomes are opened consumes
that split. The changed experiment must use newly frozen, unseen seeds/regimes or be
reported as additional development. Infrastructure failures may be replayed from the
same immutable manifest only when the failure rule and repair are disclosed; a
controller failure is an outcome, not infrastructure censoring.

## Future prospective and held-out suites

Exact unseen seed outcomes are neither generated nor guessed here. Before execution, a
dated amendment must select seed IDs using a recorded deterministic procedure that does
not query simulator outcomes, and determine sample size from a precision/power target
using development information only.

The future evidence ladder is predeclared as follows:

1. **Nominal seed holdout:** the frozen 35 kg/h task and current generator on fresh seeds
   never used for tuning. Because the rate itself was selected on development results,
   this tests seed generalization at a development-selected task; it is not blind task
   selection.
2. **Development-selected stress confirmation:** fresh seeds at 42 kg/h. This may show
   prospective seed generalization at the named stress point, but 42 kg/h must never be
   called an unseen-regime held-out test.
3. **Regime holdout:** newly frozen, previously uninspected weather and demand regimes.
   The suite should span declared changes in cloud persistence/variability, forecast
   bias or calibration, delivery pressure, and initial inventory without selecting
   “interesting” realizations after preview.
4. **Mechanism/OOD stress:** separately frozen equipment/availability or capacity and
   parameter shifts outside the tuning envelope. These are stratified stresses, not
   pooled with nominal seeds to manufacture one favorable average.

The exact regime values are intentionally deferred until the simulator can name and hash
them. They must be fixed before generating their outcomes. Adjacent slices or forecast
vintages from one underlying profile cannot be split across tuning and held-out sets.

## Required ablations and falsification tests

The following checks are required before interpreting the core contrast:

- **Legacy parity:** the A0 adapter must reproduce the original baseline action and
  metric trace on the same manifest.
- **Forecast-blind invariance:** A1 and D1 must return the same action for an otherwise
  equal observation whether a forecast is absent, present, or numerically changed; normal
  evaluation additionally removes the forecast before invoking either arm.
- **Shared-planner identity:** A1--A3 must invoke the same deficit, commitment,
  availability-to-horizon, next-delivery scarcity/carry, allocation, and action code. They
  use the same canonical bands and piecewise-constant overlap calculation; only the
  declared band value differs. Given the same band values, their actions must be identical.
- **Issued-input requirement:** A2 and A3 must fail closed when a current
  `PublicForecastIssueV2` is absent rather than inventing a forecast, reading a private
  issuance, or silently switching information policy.
- **Future-truth counterfactual:** changing all truth after an issue boundary must leave
  the already-issued public forecast, current observation, and A2/A3 action unchanged.
- **Prior integrity:** A1's climatological factor must reproduce the frozen unconditional
  synthetic-prior expectation without reading a realized episode or controller outcome.
- **Resolution identity:** A1--A3 must use the same canonical bands, overlap duration, and
  beyond-horizon zero-credit rule. D1 must remain labelled as a smooth-resolution
  diagnostic rather than being substituted as the H2 control.
- **Boundary timing:** the controller observes boundary `t`, uses only the issue valid at
  `t`, holds one decision over `[t, t + decision_interval)`, and a deadline is settled
  after production in its final physical interval.
- **Information inspection:** controller-facing objects must contain no environment,
  simulator, private schedule, replay hash, seed, future truth, or unissued forecast.
- **Requested-action expression:** every attempted run must retain the typed pre-guard
  request trace and canonical digest. Each contrast reports exact decision alignment,
  unequal-request counts, solar-share L1 distance, and battery-support difference. A
  `not_expressed` treatment is a negative mechanism result, not evidence of forecast value.
- **Pairing/source audit:** core full/exogenous hashes match by seed, every planned arm is
  present exactly once and succeeds, and the complete start/end source identity is stable.

Supplementary mechanism ablations, frozen before their use, are:

- A1 with a predeclared null delivery view, to distinguish use of delivery context from
  unrelated changes in the new rule;
- a stale or cross-seed issued-forecast placebo that remains public and causal but is no
  longer matched to the episode, to test whether forecast skill rather than mere action
  variation drives the result; and
- commitment enabled versus compatibility-off on matched exogenous episodes, reported
  with expected full-manifest differences.

These ablations cannot replace the predeclared A2-versus-A1 or A3-versus-A2 contrasts,
and their parameters cannot be chosen after seeing prospective outcomes.

The causal interpretation is falsified or the run invalidated if any leakage, timing,
identity, parity, settings, or pairing check fails. The substantive hypotheses are not
supported if A1 does not improve the predeclared delivery endpoint over A0, if A2 does not
improve its predeclared endpoint over A1 and also fails the conditional efficiency
criterion, or if A3 does not improve its endpoint over A2 and also fails that criterion.
They are also not supported by gains that exist only in shaped reward, only after selective
seed removal, only on inspected development cells, or only because the guard/plant applies
the requested action differently. The two application-gap diagnostics have no benefit
sign; any material imbalance instead requires mechanism-level interpretation.

If A2 and A1, or A3 and A2, never request different actions, that treatment was not
expressed; this is a negative implementation/mechanism result, not evidence that forecasts
or uncertainty bands in general have no value. If requests differ but prospective paired
outcomes do not, the result is evidence only about this rule, issued aggregate, estimator
assumption, environment, and target distribution. If benefits disappear or reverse under
predeclared forecast placebos or new regimes, the limitation must be reported rather than
hidden by pooling.

## Reporting and claim boundary

The report must publish complete per-episode rows, benefit-oriented paired deltas,
controller descriptors, information sets, all configuration and provenance identities,
start/end source audit, full and exogenous hashes, requested-action traces, controller
invalid flags, and digests,
failures, the complete initial/terminal inventory vector, and the predeclared summary
statistics. It must separate development, nominal seed holdout, development-selected
stress, and true regime/OOD holdout tables. No best-seed trace or aggregate reward
substitutes for that record.

At most, a successful core experiment can establish that a particular transparent
deadline-closing rule with a fixed synthetic prior responds usefully to delivery context;
that the current causal issued mean adds incremental value over that prior; and that the
declared p10 mixture adds incremental value over the issued mean under the tested
synthetic task. It cannot establish true within-band PV timing, continuous-power
availability from aggregate p10, general forecast value, policy optimality, general
weather robustness, equipment-life or monetary benefit, superiority to MPC or RL,
equal-representation RL performance, site deployment readiness, or commercial value.

The implemented causal artifact declares adjacent core controller order exactly as
`A1-A0`, `A2-A1`, and `A3-A2`; no favorable post-hoc contrast is inserted. For each
scorecard metric it records treatment and control values by seed and the raw
treatment-minus-control delta. It adds a benefit-oriented delta only where direction was
declared in advance; inventory and request/application-gap diagnostics have none. The
development report shows mean, median, worst, and empirical lower-20%-tail benefit only
when the complete planned study and source audit are valid, with exact seed rows retained
even when aggregate summaries are invalid. The three-seed tail descriptions are
diagnostics only, not CVaR estimates or inferential evidence. Requested-action comparison
is reported separately as treatment-expression evidence and is never inferred from an
outcome difference.
