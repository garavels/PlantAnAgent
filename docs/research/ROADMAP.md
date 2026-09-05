# Environment V2 research roadmap

## Product and research thesis

Plant An Agent targets the scheduling and resilient operation of coupled industrial
processes powered by uncertain renewable energy. Its product proposition is a bounded,
layered autonomy stack:

```text
routine dispatch policy     fast, repeated operating decisions
predictive risk model       early warning across operational horizons
supervisory intelligence    diagnosis and high-level strategy changes
independent safety layer    non-negotiable action constraints and fallback
digital twin                reproducible validation before physical deployment
```

The near-term research goal is to show that this separation creates attributable value
on a benchmark whose optimal action depends on uncertain future conditions and delayed
equipment consequences.

## Progress snapshot — 2026-09-05

- Gate A is **partially complete**: the V1 tank-filling ceiling and heuristic
  saturation are reproduced, but the colleague's trained-policy artifact has not yet
  been supplied for identical replay.
- B1, B2, and B3 now have alpha implementations: private stochastic solar truth,
  causal forecast vintages, recurring delivery commitments, multi-rate accounting,
  and an opt-in pressurised-alkaline commitment state machine.
- The clean 24-rollout A0--A3 causal development matrix shows that the first stateless
  deadline controller family does not exploit the issued forecast and can underperform
  the persistent legacy heuristic. That negative result rejects parameter tuning as the
  next step and localizes the missing capability to trajectory-level planning.
- A replay-verified best-known feasible future-information planner and a dual-certified
  continuous relaxation now bound the exact 35/42 kg/h development cells. The planner
  improves delivery in four of six cells, including all three 42 kg/h seeds, and closes
  91.44--100% of every positive relaxed gap. At 42 kg/h it raises descriptive mean
  delivery by 436.180 kg and service from 26/36 to 35/36 commitments.
- The separate endowment-preserving relaxation exposes an objective limitation: its mean
  fill is 98.72% at 35 kg/h but only 84.90% at 42 kg/h, while selected exact schedules
  finish near the battery reserve. Much of the stress-rate gain spends starting resources
  and must not be presented as indefinitely sustainable output.
- Gate B is **passed as a development controllability gate**: the alpha environment has
  a material, exactly realizable scheduling gap and now justifies causal service MPC.
  It is not a prospective, field, economic, or site-generalization result, and the
  sustainable terminal objective must be strengthened before commercial benchmarking.
- The first registered causal service-MPC matrix is an auditable **G1 engineering
  failure**: all six MPC arms stopped before completion, no fallback was used, and no
  performance aggregate survived. Post-outcome diagnosis localized four failures to a
  one-ULP negative-curtailment invariant and two to an undeclared solver-feasibility
  scale mismatch. Narrow invariant, numerical-contract, and failed-ledger repairs were
  frozen for a separately versioned replication on the same contaminated cells.
- The repaired replication is now a complete, execution-valid **negative controller
  result**. G1 and G7 pass; G2--G6 fail. MPC loses 1.60--1.68 tonnes of mean delivery
  against A0, meets only 20/72 commitments, and aborts 204 of 206 physical starts. The
  optimizer schedules a 15-minute expected battery flow, but the plant action interprets
  that field as a contingent top-up ceiling and enforces commitment at one-minute
  resolution. The versioned commitment-aware bridge is now implemented and passes the
  complete 1,011-test repository gate. Its one-shot probe then failed B1 at decision 223
  when bound canonicalization broke a previously valid CO2 balance. The retained prefix has
  zero forced/aborted starts, but actual battery support is 505.62 times planned support and
  positive-target decisions overshoot by 717.57 kW on average. The bridge is therefore
  rejected as a target-realization seam; absolute process setpoints plus a deterministic
  one-minute dispatcher are next. The first prerequisite is now implemented without
  changing the legacy controller: the new versioned settings audit raw and projected
  solver vectors, retain the raw vector only when projection breaks feasibility, and bind
  that choice plus both residual summaries into stage evidence. The frozen CO2, gated-flow,
  material-infeasibility, and lexicographic-lock counterexamples pass. The immutable
  absolute command, pure minute dispatcher, and opt-in Environment V2 execution/evidence
  path are also implemented and outcome-free. A separately versioned causal MPC now produces
  those commands and closes them against the physical minute ledger and next public boundary.
  A frozen pre-outcome clarification resolves the incompatible old/new digest requirement
  through exact three-boundary structural equivalence. Its executable fixture and a deep
  run-level evidence/attribution validator now pass. The clean implementation revision was
  admitted and its one registered mechanism cell passed S1--S8, restoring the full 2,520 kg
  delivery but tying A0 and the certified ceiling. MPC uses 58.36% more battery throughput
  than A0, so the mechanism is accepted while controller superiority is rejected.
- Controller-parity phase 1 is complete without opening another outcome. A plain four-target
  value and one reserve-aware allocation kernel now define the physical authority that the
  heuristic, MPC, and future RL policy must share. The existing MPC dispatcher delegates to
  that kernel with byte-stable frozen boundaries.
- The controller-authority implementation gate is now closed without opening a performance
  outcome. `plant-env-v2-common-setpoint-alpha` reuses the exact 57 observations but exposes
  four normalized absolute targets under a distinct interface and checkpoint identity. The
  normalized policy path and a source-bound A0-setpoint adapter both cross the observation-
  bound common command, generic minute dispatcher, independent guard, and physical plant;
  MPC retains its complete original provenance through that same dispatcher. Cross-source
  fixtures prove identical targets produce identical physical allocations. Applied battery
  charging now distinguishes direct controller routing from process-refusal spill recovery.
  The complete repository passes 1,140 tests with 2 skips. No RL policy has been trained and
  no controller-performance claim follows. Economic/wear pressure and a sustainable terminal
  condition are the next task-design boundary before comparative training.
- That next task-design boundary is now implemented outcome-free as the separately versioned
  `plant-env-v2-economic-wear-alpha`. It keeps the four common absolute targets, appends five
  causal public tariff features to the unchanged 57-field plant observation, and returns an
  additive EUR ledger instead of modifying objective version 3. The ledger values delivered
  service, shortfall, physically accepted solar, bus-side battery throughput, declared
  electrolyser exposure, and changes in deficits below the episode's starting CH4-equivalent
  and battery inventories. Liability changes telescope exactly to the final deficit, preventing
  free liquidation without rewarding terminal surplus. Every coefficient and all 24 tariff
  points are checkpoint-bound and explicitly labelled as sensitivity assumptions. Thirteen new
  tests prove identity/physics isolation, exact tariff integration, accounting direction,
  terminal shaping, determinism, Gym compliance, rich MPC reconciliation, and legacy-door
  rejection; the complete repository passes 1,153 tests
  with 2 skips. This is still task infrastructure: no heuristic/MPC outcome, multi-seed claim,
  RL training, calibrated saving, or commercial return has been opened.
- Pre-smoke review closed the last execution asymmetry: normalized Gym actions, generic
  controller-neutral commands, and rich MPC setpoints now all receive the same economic
  overlay, while the inherited legacy-share method fails before state movement. The six-hour
  rate-35/seed-0 H-versus-M task-expression cell and E1--E6 decision rule are frozen in
  `ECONOMIC_MECHANISM_SMOKE_PROTOCOL.md`. The single admitted execution now passes E1--E5:
  both arms safely deliver 210 kg with zero unmet demand, and M is descriptively 382.532 EUR
  higher. Avoided methane-endowment liability contributes +543.016 EUR, partially offset by
  -161.846 EUR of extra battery/electrolyser reserves. This proves one-cell task expression;
  because the terminal liability dominates, multi-seed, horizon, and coefficient robustness
  remain mandatory before training or controller claims.
- The remaining software integration for learned four-target policies is now explicit and
  implemented behind a separate V2 path. Strict training configuration constructs either the
  57/4 physical ablation or 62/4 economic task; checkpoint admission binds field order,
  interface digest, complete task digest, and policy artifact bytes; deterministic inference
  binds its exact vector and checkpoint into `ProcessSetpointCommandV1`; and paired
  heuristic/policy evaluation retains the same generic execution ledger. Historical V1
  loading and runtime behavior remain unchanged. Tiny fixed-policy fixtures and one disposable
  two-transition train/four-transition resume/load/evaluate smoke have run. No research-scale
  four-target policy, comparative RL performance outcome, or selection result exists.
- The registered economic robustness tournament is complete with an execution-valid negative
  task-readiness result. R1--R3 pass across all 20 cells and 40 trajectories. Both six-hour
  strata achieve comparable service and material separation in 5/5 cells, but both 24-hour
  strata achieve comparable service in 0/5, so R4--R7 fail. Every full-day M run completes its
  first two commitments, only partially completes the third, and delivers zero at the fourth
  while retaining 7.1--8.0 MWh battery energy and stranded H2/CO2. The mechanism is the frozen
  six-hour lookahead plus methane-equivalent terminal value: it cannot distinguish deliverable
  CH4 from feedstock that cannot be converted during the sunless final window. Seeds 20--24 are
  now development-only. The alpha task is not admitted for substantive RL training.
- The next repair is now preregistered without opening an outcome. M24 changes only the causal
  service MPC planning horizon from 6 to 24 hours on the ten contaminated full-day cells and
  references the immutable H/M6 evidence rather than rerunning it. The probe asks whether all
  four public commitments are enough to restore fourth-settlement and full-day service while
  every objective weight, plant contract, command path, and economic assumption stays fixed.
  If horizon-only repair fails, the next protocol must add explicit deliverable-CH4 adequacy
  across renewable outages and/or change the energy-supply architecture. If it restores
  service but the task remains coefficient-dependent, calibrated operational pressure comes
  before PPO. A pre-code audit found that the old H artifact never retained intermediate tank
  states and that the existing detached MPC checker does not reconstruct nested event
  semantics. Clarification 001 now requires explicit legacy missingness, a complete M24 first-
  forecast/interval ledger, and a typed rebuild verifier for both old M6 and new M24 evidence.
  No protected M24 action has run.
- Gates C–E remain pending. No current result establishes RL superiority, HEPA value,
  full-stack value, site generalization, or a commercial return.

## Stage A — Establish the Environment V1 ceiling

### Hypotheses

- `H1`: RL/heuristic parity is caused by a low ceiling gap and a wide near-optimal
  action plateau.
- `H2`: the present scalar reward permits materially different operating behavior to
  receive similar returns.

### Required benchmark arms

1. Shipped heuristic.
2. Colleague's trained policy, with provenance.
3. Random safety-filtered policy.
4. Simple ablated heuristics.
5. Perfect-information optimizer or dynamic-programming/MPC upper bound where
   computationally tractable.

### Required reporting

- methane and other material totals;
- renewable utilization and curtailed energy;
- minimum and terminal battery SOC;
- feedstock shortages and critical/offline time;
- safety corrections and invalid actions;
- cumulative reward and every reward component;
- distribution over held-out seeds and scenario regimes;
- mean, dispersion, worst case, and lower-tail performance;
- policy decision latency;
- optimality or best-known-baseline gap.

## Stage B — Minimal discriminative Environment V2

Implement difficulty in separately switchable features so every contribution can be
ablated.

### B1. Exogenous uncertainty

- seeded multi-day solar/weather profiles;
- day-ahead and intra-day forecasts available to controllers;
- horizon-dependent forecast error;
- explicit separation of latent truth, forecast, and observed realization;
- held-out weather regimes rather than seed-only evaluation.

### B2. Temporal equipment commitment

- off, starting, running, and cooldown states for key process units;
- startup time and energy;
- ramp-rate limits;
- minimum up/down time;
- load-dependent efficiency;
- observable timers and state needed to preserve the Markov contract.

Current alpha status: implemented for one technology-labelled 3 MW pressurised-alkaline
archetype. Its startup, standby, ramp, minimum up/down, cooldown, and part-load parameters
are literature-bounded synthetic assumptions, not vendor calibration. The equipment
kernel is switchable, fingerprinted in the policy/audit contracts, and isolated from V1.
The present five-field share action expresses on/standby intent implicitly. A fingerprinted
1% stack-nameplate deadband applied to resolved offered kW makes an orderly stop reachable
to continuous policies without depending on PV/bus sizing, while exact guard-to-plant
refusal is accumulated separately from safety projection. The feasible-planner matrix
proves that this interface can express materially better exact schedules, but selected
requests still accumulate 38.8--45.5 MWh of request-to-application gap. The causal MPC
must model accepted power explicitly; an absolute-setpoint/mode interface is warranted if
that mapping prevents causal capture.

### B3. Product commitments

- methane offtake or delivery schedule;
- finite product storage and missed-delivery accounting;
- time-dependent value or service-level objective;
- terminal inventory valuation to prevent horizon artifacts.

These three additions are the minimum initial bundle because they create a decision that
cannot be solved purely from the current timestep: commit equipment using an imperfect
forecast while preserving enough inventory to meet a later obligation.

The alpha terminal-inventory tie-breaker does not prevent a controller from spending
starting endowment to maximize delivery. A sustainable-service beta must add cyclic
terminal constraints or calibrated terminal shadow values without retrospectively changing
the frozen alpha result.

## Stage C — Controller evidence

- upgrade the heuristic into a credible forecast-aware baseline;
- add deterministic and stochastic MPC where practical;
- retrain PPO under an immutable, versioned environment contract;
- tune only on training/validation regimes;
- perform held-out weather, demand, and parameter stress tests;
- investigate recurrent policies if partial observability demonstrably requires memory;
- require a Pareto or risk-adjusted improvement, not merely a larger shaped reward.

Stage C is now entered through the causal service-MPC work package. The same current public
forecast and `ControllerObservationV2` must feed every deployable comparator; the frozen
future-information planner remains only an offline reference. The MPC must span the next
renewable window and intervening commitments, expose solver status and latency, model
commitment/inventory/guard-compatible power, and report how much of each cell's `F - A0`
gap it captures. PPO training begins only after that equal-information comparator and a
sustainable terminal objective exist. RL earns a role only by closing a measured residual
gap, improving tail risk or latency, or handling nonlinear/scenario complexity that the MPC
does not capture.

The first service-MPC replication proves that equal information and a valid MILP are not
sufficient when the supervisory action contract loses commitment intent. Before comparing
controller families, Stage C must separate planned battery energy, real-time top-up
authority, and applied discharge. A bounded bridge repair will test that hypothesis on the
already contaminated development cells. Failure of that bridge is the trigger to replace
the share-only supervisory command with absolute process setpoints and a deterministic
inner-loop dispatcher. The one-shot bridge probe confirmed startup recovery but also showed
that additive authority cannot represent a target: it supplied 14.41 MWh against 28.50 kWh
planned over its valid prefix. The next contract must make target tracking and unplanned
energy explicit acceptance criteria. That contract is now frozen: it preserves the Gym
action and simulator, adds a versioned absolute setpoint seam, realizes it through a pure
reserve-aware one-minute dispatcher, and repairs solver projection only when feasibility is
preserved. The solver repair, typed setpoint, pure dispatcher, and opt-in environment
realization are complete under separately hashed contracts. The environment now records
every minute's boundary, dispatch, guard, physical application, commitment transition, and
energy attribution without changing the historical action doors. The separately versioned
MPC controller is now complete and cannot issue a second solve until the first command's
typed physical execution has been reconciled. A pre-outcome clarification now resolves the
protocol's impossible byte-identical cross-version model/plan condition by freezing exact
three-boundary identities plus structural equality of the forecasts, normalized models, and
complete plan quantities. The executable equivalence fixture and complete authorization
audit remain prerequisites to any full rollout. The fixture now reproduces every frozen
identity and exact structural equality. The run validator also closes ordered public and
physical boundaries, residual power/inventory values, minute latency cardinality, target
deficit attribution, energy totals, and deterministic evidence-graph identities.
The one-shot mechanism runner is now implemented without a tuning surface: it binds rate 35,
seed 0, the exact exogenous episode, protocol and lock digests, clean source identity,
non-overwriting external publication, and S1--S8. The implementation audit maps all twelve
outcome-free gates to tests. All gates now pass at clean implementation revision `a968a343`,
which was used for the one registered mechanism execution without retry. All S1--S8 gates
passed: the new path delivered the complete 2,520 kg contract with no aborted/forced start,
safety violation, target overshoot, or accounting defect. A0 and the certified ceiling also
deliver 2,520 kg on this cell, however, while A0 uses 3.856 MWh less battery throughput and
1.043 kWh/kg less accepted process energy. The actuation architecture is accepted, but the
task remains saturated and establishes no controller superiority.

The controller-neutral target value, normalized policy adapter, and common allocation kernel
now close the first authority phase without changing the historical Gym task. Before Stage D
changes the environment, Stage C must bind observations and executions to that common door,
route a transparent heuristic and opt-in RL environment through it, preserve MPC provenance,
and prove paired authority equivalence. A separately frozen multi-seed mechanism replication
then precedes controller claims. The first Stage D pressure should make the observed operating
tradeoff commercially legible: time-varying energy value and explicit cycling/equipment wear
are the leading candidate because service is tied while battery throughput differs by 58%.

## Stage D — Progressive operational realism

Add only after the preceding experiment identifies a missing decision pressure:

- cycling-dependent equipment wear and battery throughput degradation;
- thermal dynamics and restart restrictions;
- noisy, delayed, missing, and drifting sensors;
- maintenance actions and availability;
- electricity import/export and price exposure, if commercially in scope;
- water, compression, and balance-of-plant constraints;
- multiple units and shared bottlenecks.

Each feature must introduce a documented state variable, invariant, observable contract,
baseline response, test, and intended decision consequence.

## Stage E — Predictive and supervisory value

Only after Environment V2 produces meaningful uncertainty and event prevalence:

1. Collect HEPA trajectories from diverse behavior policies, not one controller.
2. Measure discrimination, calibration, false alarms, and useful warning lead time.
3. Compare controller-only, controller-plus-supervisor, and
   controller-plus-supervisor-plus-HEPA arms.
4. Condition risk on proposed actions/plans if unconditioned forecasting first proves
   useful.
5. Evaluate the rule supervisor reproducibly before attributing extra value to an LLM.

## Commercial evidence

Technical metrics must be translated into operational outcomes:

- additional saleable product from the same installed assets;
- reduced renewable curtailment;
- delivery reliability and avoided downtime;
- preserved equipment life and fewer damaging cycles;
- reduced need for conservative storage or generation oversizing;
- bounded fallback behavior and operator intervention rate;
- inference and deployment cost relative to MPC or manual operation.

No commercial claim is accepted until the assumptions, time horizon, plant sizing, and
counterfactual baseline are explicit.

## Decision gates

- **Gate A:** independently reproduce and explain Environment V1 parity.
- **Gate B:** demonstrate that the V2 benchmark has a material gap between naive,
  heuristic, and perfect-information behavior.
- **Gate C:** demonstrate controller generalization and safe fallback on held-out regimes.
- **Gate D:** demonstrate calibrated HEPA warning with operationally useful lead time.
- **Gate E:** demonstrate by ablation that the complete hierarchy improves a declared
  risk-adjusted operational objective.
