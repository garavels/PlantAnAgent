# Cyclic stochastic market beta protocol

Status: **implementation contract frozen; no controller outcome authorized**

Date: **2026-09-05**

## Decision

The next Environment V2 increment is a separately identified beta task. It does not
modify or relabel the frozen commitment, common-setpoint, or economic-wear alpha
tasks. The beta adds three independently switchable pressures:

1. a realized stochastic electricity price and causal, revisable 24-hour price
   forecast shared in the same representation by learned and model-based
   controllers;
2. uninterrupted 24-hour cycle accounting against one non-ratcheting opening
   inventory reference; and
3. bounded, seeded, finite DAC outages whose future schedule is private.

These additions target three concrete weaknesses in the alpha evidence. The alpha
price is perfectly known and repeats every day, economic MPC receives more of it
than PPO, and a finite-horizon policy may spend starting resources without proving
repeatable operation. A finite outage additionally tests whether inventory and
forecast-aware control can absorb a realistic process interruption without adding a
new simulator mechanism.

The beta is a completed task boundary, not a claim that RL is superior. Training and
comparative outcome protocols remain separate prospective decisions.

## Scientific rationale

Industrial-control benchmarks are useful only when they preserve properties that
make real operation difficult: delayed effects, partial observability, stochastic
disturbances, constraints, and competing objectives. The published Industrial
Benchmark was motivated by precisely the gap between interpretable toy tasks and
industrial control characteristics ([Hein et al., 2017](https://arxiv.org/abs/1709.09480)).
Work on safe RL and MPC likewise treats explicit constraints and uncertainty as
first-class rather than relying on reward alone
([Pfrommer et al., 2022](https://proceedings.mlr.press/v168/pfrommer22a.html)).

For power-to-gas specifically, recent dynamic and scheduling studies model coupled
conversion, storage, renewable variability, off-design operation, and operational
control rather than a memoryless conversion coefficient
([Energy Conversion and Management 305, 118257](https://doi.org/10.1016/j.enconman.2024.118257),
[International Journal of Electrical Power & Energy Systems 160, 110107](https://doi.org/10.1016/j.ijepes.2024.110107)).
Those sources motivate the kinds of state and uncertainty that matter; they do not
calibrate this repository's synthetic coefficients or establish a commercial result.

## Versioned task identity

The beta environment identifier is:

```text
plant-env-v2-cyclic-market-beta-v1
```

Its checkpoint contract must bind:

- the unchanged plant, safety, physical episode, weather, commitment, delivery, and
  four-target authority contracts;
- the unchanged economic coefficient set, with its 24-hour tariff used only as the
  public market generator's deterministic daily baseline;
- the complete market generator version, regime parameters, forecast cadence,
  forecast horizon, quantiles, and public field order;
- the cycle duration, inventory return fractions, absolute tolerances, boundary
  convention, and non-ratcheting reference semantics;
- the disturbance generator version, configuration, random-stream namespace, and
  visibility policy; and
- the exact observation and action field order.

An alpha checkpoint must fail admission to beta and a beta checkpoint must fail
admission to every alpha environment.

## Randomness and replay

The existing alpha root streams remain unchanged:

```text
0 inventories
1 weather
2 contracts
```

The beta deterministically takes later children from the same root seed without
altering the first three:

```text
3 market
4 disturbances
```

The beta episode identity is a new digest over the unchanged alpha exogenous episode
identity, the complete private market identity, the disturbance schedule identity,
and the cycle contract. The old alpha exogenous digest is retained under an explicit
name and is never presented as covering faults or prices.

The sampled market regime and future truth, future forecast vintages, disturbance
schedule, episode seed, and stream identities are offline audit data. They never
enter a controller observation.

## Stochastic market contract

### Truth

- One realized opportunity price applies to every controller interval.
- The deterministic 24-hour alpha tariff is the civil-time baseline, not future
  truth.
- A bounded autoregressive deviation creates serial correlation.
- Rare signed innovations create decaying price excursions.
- `calm`, `volatile`, and `scarcity` regimes have explicit immutable parameters.
- `mixed` selects one regime from the market stream; the selected label remains
  private.
- Every realized and forecast price is finite and clipped to the existing declared
  `[-300, 300] EUR/MWh` domain.

This is a synthetic engineering stress generator. It is not historical market data,
a calibrated price model, or a statement about a specific bidding zone.

### Public issue

A new issue exists at every controller decision boundary and contains only:

- current realized price;
- issue decision index and episode-relative time;
- 24 consecutive one-hour bins from the issue boundary; and
- mean, p10, and p90 price for each bin.

The fixed market vector therefore has 73 values in this order:

```text
current, hour_00(mean,p10,p90), ..., hour_23(mean,p10,p90)
```

Every forecast ensemble is conditioned on the current realized price and generated
from an issue-specific forecast stream. It may not inspect future realized price.
Changing truth strictly after an issue boundary while preserving the seed and prefix
must leave that issue byte-identical. Later issues are allowed to revise earlier
expectations after new prices are observed.

The typed public issue projects the same 24 hourly means onto the 15-minute economic
MPC grid. PPO receives those exact hourly means and quantiles in its vector. Neither
controller may substitute the private truth or a later vintage.

## Cyclic operation contract

The plant runs continuously across cycle boundaries. Simulator state, equipment
commitment, delivery ledger, weather, price, and fault lifecycle are never reset at a
cycle boundary.

At episode reset, the task records authoritative opening values for:

- battery energy;
- H2 inventory;
- CO2 inventory; and
- saleable CH4 inventory.

Every aligned 24-hour boundary is settled against fractions of that same opening
reference. The reference never ratchets down after a bad cycle. Each resource has an
explicit absolute numerical tolerance. A cycle is feasible only if all four deficits
are within tolerance. H2, CO2, and CH4 are checked separately: methane-equivalent
aggregation is not sufficient when conversion time or an outage makes feedstock
temporarily unusable.

The online cycle observation contains only:

```text
fraction of current cycle elapsed
signed battery return margin / installed battery capacity
signed H2 return margin / H2 tank capacity
signed CO2 return margin / CO2 tank capacity
signed CH4 return margin / CH4 tank capacity
```

The economic scalar remains the additive realized EUR ledger. Cycle violations are
reported as explicit constraint costs and hard evaluation failures; this increment
does not hide an uncalibrated constraint penalty inside the currency objective. The
existing dense endowment-liability terms remain visible and unchanged.

## Disturbance contract

The first disturbance profile uses only the simulator's existing `dac_outage`
physics. When enabled, each cycle independently samples at most one finite outage:

- occurrence probability is configuration-bound;
- onset and duration are restricted to that cycle;
- start and end are aligned to the one-minute physics grid;
- an outage never overlaps the next cycle; and
- the simulator's existing fault registry remains authoritative.

The controller sees only effects that are already present in ordinary public
telemetry and health state. It receives no warning, time-to-fault, schedule, or
private regime label. The no-disturbance ablation has an exactly empty schedule.

Solar-drop faults are excluded because stochastic solar truth already represents the
resource process. Sensor bias and permanent degradation are deferred until the beta
defines explicit measurement-quality and remaining-useful-life state; silently
mixing truth and reported state would invalidate the Markov and evidence contracts.

## Observation and action surface

The beta observation is the unchanged 57-field common physical observation followed
by the 73-field public market issue and the five cycle fields: 135 finite normalized
values in total. The action remains the existing four normalized absolute process
targets and crosses the same command, minute dispatcher, independent guard, and
simulator.

No controller receives future solar truth, future realized price, a future forecast
vintage, future outage schedule, hidden physical state, seed, or regime label.

## Implementation gates

The beta build is accepted only if short, outcome-free tests prove all of the
following:

1. alpha reset traces and contract digests are unchanged by the protected sampling
   hook;
2. market truth and every issue are deterministic for one configuration and seed;
3. forecasts are finite, ordered (`p10 <= mean <= p90`), bounded, complete, and
   causally invariant to changed future truth;
4. consecutive issues revise when newly observed truth differs;
5. market and disturbance streams cannot perturb alpha inventories, weather, or
   commitments;
6. disabled disturbances produce an empty schedule and the same physical episode as
   alpha under identical actions;
7. enabled outages activate and expire on their declared boundaries;
8. cycle settlement occurs once at every exact boundary without resetting the plant
   or changing the opening reference;
9. public vectors contain exactly 135 finite in-range values and no private fields;
10. normalized Gym actions and controller-neutral commands retain physical parity;
11. beta manifests and checkpoint contracts cover market, cycle, disturbance, and
    alpha identities; and
12. alpha/beta checkpoint substitution fails before physical movement.

Repository-wide tests, strict typing for new production modules, formatting,
compilation, dependency-lock cleanliness, and frontend build remain release gates.

## Evidence and experiment boundary

No performance experiment is authorized by this protocol. Seeds 0--24 are already
development-contaminated for the current V2 generator; 901--906 are fixtures. Because
the beta generator creates a new task, its future train/development/held-out split
must be selected and committed by a deterministic outcome-blind procedure before any
controller comparison.

The first future beta comparison must use episode seed as the experimental unit,
retain every failed arm, aggregate complete pairs only, require comparable delivery
service and zero hard safety/cycle violations before comparing value, and report
regime-level as well as seed-level performance. A trained PPO checkpoint may enter
only after a finite multi-training-seed budget and checkpoint selection rule are
frozen. No long run starts as part of this implementation.

## Explicit non-claims

Implementing this beta will not prove RL, MPC, HEPA, or the language supervisor adds
value. It will not calibrate a plant, tariff, fault rate, equipment lifetime, saving,
or return on investment. It creates the causal, cyclic, disturbance-bearing task on
which those claims can later be tested without the alpha environment's known ceiling
and information asymmetry.
