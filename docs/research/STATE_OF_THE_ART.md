# State of the art, product thesis, and evidence standard

## Executive conclusion

Plant An Agent should not be developed as a demonstration that reinforcement
learning can operate a simplified plant. That result has little scientific or
commercial value by itself. The stronger direction is a **safety-bounded
supervisory optimisation and virtual-commissioning platform** for renewable
power-to-X plants.

The initial operational problem is:

> Meet contracted methane deliveries under uncertain renewable generation while
> protecting equipment life and respecting a hard operating envelope.

The eventual research claim must compare the complete operating outcome, not a
private reward function:

> At equal safety and product-delivery reliability, the Plant An Agent controller
> reduces risk-adjusted production cost and asset wear relative to tuned,
> forecast-aware heuristics and stochastic model-predictive control on held-out
> conditions, and the benefit survives hardware-in-the-loop testing.

This is a hypothesis, not a current repository capability. If MPC or a hybrid wins
the controlled comparison, the product should use it. The platform's value is its
ability to qualify the best bounded supervisory strategy, not allegiance to one
algorithm.

## Real problem and prospective customer

Renewable power-to-hydrogen and power-to-methane plants couple volatile energy
availability to equipment with operational states, part-load efficiency, startup
costs, thermal limits, storage constraints, degradation, and delivery obligations.
An operator must repeatedly decide whether to produce now, preserve inventory,
commit a unit for a forecast opportunity, or avoid a cycle that is technically
possible but economically damaging.

The first plausible buyers and users are:

- power-to-X project developers comparing plant sizing and operating strategies;
- system integrators qualifying control logic before commissioning;
- plant operators evaluating dispatch recommendations in shadow mode;
- equipment vendors testing how their operating envelope affects system value.

The initial product wedge is above low-level PLC control. It proposes bounded
setpoints, evaluates schedules, and supplies traceable evidence while an independent
safety layer and deterministic fallback retain final authority. This scope reduces
integration and assurance risk and matches the repository's existing separation of
controller, supervisor, HEPA predictor, and safety guard.

Willingness to pay, the most urgent buyer, and the value magnitude have not been
validated. They require design-partner interviews, operating data, and ultimately a
paid pilot or letter of intent. Technical benchmark success is necessary evidence,
not market validation.

The market context makes cost and offtake risk more important than an “autonomous AI”
label. The IEA reports that low-emissions hydrogen passed only about 1% of global
production in 2026, while investment decisions slowed, the announced project pipeline
contracted, and renewable hydrogen remained more expensive than fossil production in
most regions. Its 2025 review also found that established industrial uses and
hydrogen-based fuels accounted for almost all announced firm private-sector offtake.
The first commercial discovery should therefore target projects with a real contracted
buyer, a costly operating decision, and accessible plant/design data—not an abstract
hydrogen operator persona.

## What the current evidence supports

### Supported by external evidence

- Renewable variability changes electrolyser operating state, efficiency, and life;
  scheduling therefore matters beyond instantaneous energy allocation.
- Startup, ramping, part-load, thermal, pressure, purity, and degradation behavior
  differ across electrolyser technologies and should not be represented as one
  generic linear converter.
- Model-predictive control is a strong baseline for constrained process control;
  learning is most defensible when it adds economic performance, adapts a model, or
  complements a predictive safety mechanism.
- Safe learning remains an open deployment problem. Reward penalties are not hard
  constraints; proposed and applied actions and all shield interventions must be
  recorded independently.
- Whole-chain power-to-methane operation couples hydrogen and carbon-dioxide supply,
  storage sizing, methanator startup/thermal behavior, product delivery, and site
  conditions.
- European renewable-fuel rules make temporal and geographic correlation,
  additionality, and greenhouse-gas accounting potentially material scheduling
  constraints.

### Not yet supported

- Plant An Agent has not shown lower LCOH or LCOCH4, lower degradation, or higher
  delivery reliability on a physical plant.
- No controller has yet beaten a tuned forecast-aware heuristic, economic MPC, or a
  perfect-information bound in this repository.
- The learned controller has not yet demonstrated out-of-distribution robustness,
  constraint guarantees, hardware-in-the-loop performance, or transfer between
  equipment or sites.
- The simulator has not been calibrated against physical plant data. Until that
  occurs, it should be called a **control-oriented simulator** or **digital-twin
  prototype**, not a validated digital twin.
- No customer demand, procurement route, or return on investment has yet been
  validated.

## Academic and industrial benchmark

The current literature sets a substantially higher bar than “RL reaches the same
production as a rule.” A recent green-hydrogen dispatch study combines a higher
fidelity electrochemical/thermal PEM model, several years of minute-level weather,
system sizing, PPO, rule-based control, MPC variants, and interpretability. Process
control benchmark work similarly emphasizes nonlinear dynamics, disturbances,
constraints, repeated trials, and an NMPC oracle. The differentiator for Plant An
Agent therefore needs to be whole-chain power-to-methane coupling, explicit safety,
forecast and delivery uncertainty, and deployment evidence rather than PPO alone.

The mandatory controller set is:

1. shipped Environment V1 heuristic, preserved only as a historical baseline;
2. a tuned, forecast-aware heuristic for the exact V2 objective;
3. deterministic economic MPC;
4. stochastic or robust MPC;
5. an offline perfect-foresight optimiser as a best-known bound;
6. independently trained RL policies with multiple training seeds;
7. a hybrid or residual RL-MPC strategy if standalone learning has a justified gap.

Every controller receives the same telemetry, forecasts, action authority, and safety
layer. Perfect future truth is available only to the offline oracle and is labelled as
such.

## Environment capability required for a meaningful comparison

### Exogenous data and uncertainty

- historical or traceable synthetic solar, ambient conditions, electricity prices,
  carbon intensity, and demand;
- explicit forecast issue time, horizons, quantiles or scenarios, and provenance;
- separate latent truth, measured realization, and controller-visible forecast;
- held-out years, sites, and extreme regimes instead of random-timestep splits;
- deterministic replay by immutable scenario identifier and seed.

### Plant dynamics

- technology-specific electrolyser modes: off, cold start, hot standby, and running;
- minimum stable load, ramp limits, startup time/energy, part-load efficiency, and
  degradation exposure;
- methanator startup and thermal/hotspot state, turndown, and product-quality limits;
- DAC cycle/weather sensitivity when evidence and calibration parameters exist;
- material balances, storage flow limits, delivery withdrawal, and terminal inventory
  value;
- noisy, delayed, missing, and drifting telemetry separated from physical truth.

Each feature requires units, invariants, parameter source, calibration status, tests,
and an ablation. Unsupported detail remains behind an explicitly experimental flag;
more equations do not automatically make a model more valid.

### Objective and constraints

Hard physical and safety constraints belong in the independent guard or a predictive
safety filter. Operational commitments belong in controller state and constrained
optimisation. Economic preferences belong in the objective. The benchmark should not
trade a hard violation against enough kilograms of rewarded methane.

The primary economic objective is risk-adjusted value of on-spec delivery:

```text
annual value = incremental on-spec product revenue
             + energy and curtailment savings
             + avoided degradation and replacement cost
             + avoided delivery and compliance penalties
             - deployment and compute cost
```

A learning reward may approximate this objective during training, but benchmark
conclusions must use physical and economic metrics.

## Evaluation standard

### Experimental design

- freeze environment, controller, reward, and scenario-set versions before evaluation;
- separate complete training, validation, held-out test, and stress regimes;
- use paired common-random-number scenarios for every controller comparison;
- use multiple independent policy-training seeds, not only evaluation seeds;
- record all hyperparameters, source revision, dependency lock digest, artifact hash,
  hardware, training time, inference latency, and checkpoint-selection procedure;
- publish negative results and predeclare the primary scorecard and success threshold;
- use ablations for forecasts, commitment dynamics, degradation, safety, recurrence,
  supervision, HEPA, and equipment sizing.

### Primary scorecard

| Dimension | Metrics |
| --- | --- |
| Delivery | on-spec kg, contract fill rate, unmet kg, shortage penalty, inventory resilience |
| Cost/value | LCOH or LCOCH4, EUR/MWh on-spec product, net operating margin, oracle gap |
| Asset health | starts, hot/cold cycles, low-load exposure, thermal excursions, equivalent battery cycles, estimated degradation |
| Energy/climate | kWh/kg, curtailment, renewable utilisation, qualified product fraction, GHG intensity, water and useful heat |
| Safety | hard violations, near misses, intervention magnitude/duration, fallback success |
| Control/AI | median and IQM outcome, bootstrap interval, probability of improvement, CVaR, OOD loss, sample efficiency, p95/p99 latency |

The hard-violation target is zero. A candidate does not win because its mean is larger
if it buys that mean with materially worse tail risk, delivery reliability, or asset
exposure.

## Evidence ladder and product releases

```text
verified simulator
    -> calibrated component models with uncertainty bounds
    -> held-out and OOD software-in-the-loop benchmark
    -> PLC/protocol hardware-in-the-loop, including latency and sensor failure
    -> physical shadow-mode recommendations
    -> bounded supervisory setpoints with deterministic fallback
```

The repository is currently at the first transition: verifying a simplified
control-oriented simulator and building a discriminative benchmark. Claims must be
limited to the achieved rung.

## Commission relevance

For a Clean Hydrogen JU or EU Innovation Fund type evaluation, the intended evidence
can contribute to innovation, maturity, replicability, cost efficiency, and quantified
greenhouse-gas avoidance. It does not yet satisfy those criteria. The exact call text
must be mapped before using commission-specific language.

The control-oriented simulator itself is not a sufficient novelty claim. EU-supported
renewable-methane projects such as TURN2X already report using a reactor digital twin
for design and physical-prototype reduction. Differentiation must come from the
validated whole-chain decision problem, comparative controller evidence, independent
safety, transferability, and a credible deployment path.

A future case would need to show:

- a novel whole-chain, uncertainty-aware and safety-bounded supervisory method;
- KPI movement at the system boundary, including energy use and degradation;
- calibrated/HIL evidence and a credible route to physical demonstration;
- transfer to another equipment configuration, vendor, or site;
- risk-adjusted EUR/kg or EUR/MWh improvement over credible industrial controls;
- qualified-product GHG accounting rather than renewable utilisation alone.

## Decision-relevant sources

1. Wang et al., “Water electrolyzer operation scheduling for green hydrogen
   production: A review,” *Renewable and Sustainable Energy Reviews* 203 (2024),
   [doi:10.1016/j.rser.2024.114779](https://doi.org/10.1016/j.rser.2024.114779).
2. Sayed-Ahmed et al., “Dynamic operation of proton exchange membrane
   electrolyzers—Critical review,” *Renewable and Sustainable Energy Reviews* 189
   (2024), [doi:10.1016/j.rser.2023.113883](https://doi.org/10.1016/j.rser.2023.113883).
3. Oh, “Quantitative comparison of reinforcement learning and data-driven model
   predictive control for chemical and biological processes,” *Computers & Chemical
   Engineering* 181 (2024),
   [doi:10.1016/j.compchemeng.2023.108558](https://doi.org/10.1016/j.compchemeng.2023.108558).
4. Hewing et al., “Learning-based model predictive control: Toward safe learning in
   control,” *Annual Review of Control, Robotics, and Autonomous Systems* 3 (2020),
   [doi:10.1146/annurev-control-090419-075625](https://doi.org/10.1146/annurev-control-090419-075625).
5. Gu et al., “A review of safe reinforcement learning: Methods, theories and
   applications,” *IEEE TPAMI* 46(12) (2024),
   [doi:10.1109/TPAMI.2024.3457538](https://doi.org/10.1109/TPAMI.2024.3457538).
6. Alshiekh et al., “Safe reinforcement learning via shielding,” *AAAI* (2018),
   [doi:10.1609/aaai.v32i1.11797](https://doi.org/10.1609/aaai.v32i1.11797).
7. Bloor et al., “PC-Gym: Benchmark environments for process control problems,”
   *Computers & Chemical Engineering* 204 (2026),
   [doi:10.1016/j.compchemeng.2025.109363](https://doi.org/10.1016/j.compchemeng.2025.109363).
8. Yun et al., “Interpretable reinforcement-learning energy dispatch co-optimized
   with system sizing for stable and efficient off-grid green-hydrogen systems,”
   *Applied Energy* 426 (2026),
   [doi:10.1016/j.apenergy.2026.128682](https://doi.org/10.1016/j.apenergy.2026.128682).
9. Bremer et al., “CO2 methanation: Optimal start-up control of a fixed-bed reactor
   for power-to-gas applications,” *AIChE Journal* 63(1) (2017),
   [doi:10.1002/aic.15496](https://doi.org/10.1002/aic.15496).
10. Bruns et al., “A multi-step framework for the design of flexible power-to-methane
    processes,” *Journal of Cleaner Production* 447 (2024),
    [doi:10.1016/j.jclepro.2024.141434](https://doi.org/10.1016/j.jclepro.2024.141434).
11. Bitencourt et al., “Verification and validation of digital twins: A systematic
    literature review for manufacturing applications,” *International Journal of
    Production Research* 63(1) (2025),
    [doi:10.1080/00207543.2024.2357741](https://doi.org/10.1080/00207543.2024.2357741).
12. Henderson et al., “Deep reinforcement learning that matters,” *AAAI* (2018),
    [doi:10.1609/aaai.v32i1.11694](https://doi.org/10.1609/aaai.v32i1.11694).
13. [Clean Hydrogen JU Strategic Research and Innovation Agenda KPIs](https://www.clean-hydrogen.europa.eu/knowledge-management/strategy-map-and-key-performance-indicators/clean-hydrogen-ju-sria-key-performance-indicators-kpis_en).
14. [Commission Delegated Regulation (EU) 2023/1184, consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A02023R1184-20240610).
15. [EU Innovation Fund scope and award criteria](https://climate.ec.europa.eu/areas-action/eu-funding-climate-action/innovation-fund/what-innovation-fund_en).
16. [IEA, *Global Hydrogen Review 2026: Production*](https://www.iea.org/reports/global-hydrogen-review-2026/production).
17. [IEA, *Global Hydrogen Review 2026: Cost acceptability*](https://www.iea.org/reports/global-hydrogen-review-2026/cost-acceptability).
18. [CORDIS TURN2X renewable-methane project report](https://cordis.europa.eu/project/id/101189219/reporting).
19. [IEA, *Global Hydrogen Review 2025: Executive summary*](https://www.iea.org/reports/global-hydrogen-review-2025/executive-summary).
20. [European Commission JRC, PVGIS hourly radiation and PV output](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en).

Literature-derived values are candidate priors, not automatically simulator
parameters. Equipment-specific values require a declared data source and calibration
status before they support a product claim.
