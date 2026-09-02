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

### B3. Product commitments

- methane offtake or delivery schedule;
- finite product storage and missed-delivery accounting;
- time-dependent value or service-level objective;
- terminal inventory valuation to prevent horizon artifacts.

These three additions are the minimum initial bundle because they create a decision that
cannot be solved purely from the current timestep: commit equipment using an imperfect
forecast while preserving enough inventory to meet a later obligation.

## Stage C — Controller evidence

- upgrade the heuristic into a credible forecast-aware baseline;
- add deterministic and stochastic MPC where practical;
- retrain PPO under an immutable, versioned environment contract;
- tune only on training/validation regimes;
- perform held-out weather, demand, and parameter stress tests;
- investigate recurrent policies if partial observability demonstrably requires memory;
- require a Pareto or risk-adjusted improvement, not merely a larger shaped reward.

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
