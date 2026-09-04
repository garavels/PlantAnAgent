# Economic and wear task alpha: frozen protocol

Status: **implemented research task; literature-anchored, explicitly synthetic, and not
vendor- or site-calibrated. No controller result has yet been produced on this task.**

## Decision

The next controller-discrimination task is separately identified as:

```text
plant-env-v2-economic-wear-alpha
```

It inherits the verified V2 physical dynamics, the 57-field causal plant observation, and
the four-target common controller door. It appends five public tariff features and replaces
only the scalar learning objective with an additive euro-denominated ledger. The historical
five-action V2 environment and the four-target authority-parity environment retain their
existing interfaces, rewards, checkpoint identities, and replay behavior.

This task exists because the registered service-MPC result saturated delivery just like the
heuristic while accepting 3.49% more process energy and 58.36% more battery throughput. A
delivery-only task cannot decide whether that trajectory is better. The economic alpha asks
the narrower and commercially relevant question:

> At equal hard safety and contracted delivery, can a controller reduce the opportunity
> cost of renewable electricity, storage cycling, equipment-life exposure, and depletion of
> the plant's starting endowment?

It does **not** claim to calculate a bankable LCOCH4, predict component failure, or represent
a specific offtake agreement.

## Evidence boundary

The structure is evidence-led, while the numerical scenario remains a declared benchmark:

- ACER reports frequent low and negative EU day-ahead prices as renewable penetration grows
  and identifies insufficient flexibility as a source of both low prices and price spikes.
  Therefore time-varying electricity value is a real scheduling variable, not a reward trick.
- ENTSO-E publishes day-ahead prices under the Transparency Regulation. A future calibrated
  task should replay one named bidding zone and period from that source; this alpha deliberately
  uses a deterministic stress shape so task mechanics can be verified before adding a dataset.
- The Clean Hydrogen JU alkaline-electrolyser KPI table reports 2024 targets of 49 kWh/kg,
  480 EUR/kW, 30-second hot-idle ramp, 900-second cold-start ramp, and 0.11% degradation per
  1,000 hours. It also states that the KPIs are interdependent and defines its capital-cost
  boundary. The alpha uses 480 EUR/kW only as an equipment-capital exposure anchor; it does
  not turn the reported steady-state efficiency loss into an unsupported per-start law.
- DOE's grid-storage assessment treats battery replacement, augmentation, cycle life, and
  calendar life as distinct cost drivers. Its Storage Innovations 2030 LFP reference uses
  2,640 cycles and 85% round-trip efficiency for one specific 100 MW/10 h case. The alpha
  uses 2,640 full cycles only as a visible sensitivity center, not a universal warranty.

Primary sources:

- [ACER 2024 wholesale-market integration report](https://www.acer.europa.eu/sites/default/files/documents/Publications/ACER_2024_MMR_Market_Integration.pdf)
- [ACER low/negative day-ahead price data item](https://aegis.acer.europa.eu/chest/dataitems/735/view)
- [ENTSO-E Transparency Platform day-ahead price procedure](https://consultations.entsoe.eu/markets/copy-of-tp-mop-v3r4-consultation/)
- [Clean Hydrogen JU SRIA technology KPIs](https://www.clean-hydrogen.europa.eu/knowledge-management/strategy-map-and-key-performance-indicators/clean-hydrogen-ju-sria-key-performance-indicators-kpis_en)
- [DOE 2022 Grid Energy Storage Technology Cost and Performance Assessment](https://www.energy.gov/cmei/2022-grid-energy-storage-technology-cost-and-performance-assessment)
- [DOE Storage Innovations 2030: Lithium-ion Batteries](https://www.energy.gov/sites/default/files/2023-09/2_Technology%20Strategy%20Assessment%20-%202%20Lithium-ion_508.pdf)

## Public information and tariff semantics

The task observation has 62 floats. Slots 0--56 are byte-for-byte the common-setpoint V2
plant observation. Slots 57--61 are:

1. current electricity opportunity price;
2. exact mean over `[t, t+1 h)`;
3. exact minimum over `[t, t+6 h)`;
4. exact mean over `[t, t+6 h)`;
5. exact maximum over `[t, t+6 h)`.

Each raw EUR/MWh value is divided by 300 and bounded to `[-1, 1]`. The tariff is public and
deterministic; it discloses no future weather, inventory, failure, or delivery settlement.
The default 24-hour profile repeats daily and spans -10 to 150 EUR/MWh, with low midday and
high evening values. It is a stress fixture, not a sampled historical day.

`EconomicControllerObservationV1` is the detached typed source for structured heuristic and
optimizer controllers. It contains the same physical `ControllerObservationV2`, the same five
raw tariff values encoded for Gym policies, the tariff ID, and the economic-assumption digest.
Thus non-Gym controllers do not need to scrape `info` or receive a weaker/different information
set. The physical `ProcessSetpointCommandV1` remains bound to the plant observation because
actuator safety and dispatch do not depend on prices; a future economic-controller decision
will bind the richer observation in its source evidence.

The controller interval must divide an hour exactly and its first boundary must align to that
interval. Consequently one hourly price applies to every physical minute under a held
command. A later historical-price task may integrate sub-hourly prices directly instead of
retaining this restriction.

The plant currently has no physical grid import/export port. The tariff is therefore a
shadow opportunity price for renewable electricity accepted by the battery or conversion
units. Because curtailment is available, negative export prices are visible to the policy but
floored at zero in the cost calculation; the simulator does not invent a payment for consuming
energy. Accepted solar is measured as generated minus curtailed kWh, not requested power.

## Objective

For controller interval `t`, the returned reward is:

```text
R_t = delivered service value
    - unmet-delivery penalty
    - accepted-solar opportunity cost
    - battery-throughput replacement reserve
    - electrolyser-life exposure reserve
    - change in CH4-equivalent endowment liability
    - change in battery-energy endowment liability
```

All components and raw signals are returned separately. The previous dimensionless V2 reward
is retained in `legacy_reward` for diagnosis only and cannot affect the returned scalar.
Physical metrics remain in the historical `metrics` payload; economic totals remain in the
separate `economic_metrics` ledger.

### Service and sustainable endowment

The default contract values one delivered kilogram at 4 EUR and one unmet kilogram at a
20 EUR penalty. These are synthetic service-contract assumptions, not methane market prices.
The penalty is deliberately larger than the credit so refusing a commitment cannot improve
value merely by preserving inventory.

CH4-equivalent inventory is:

```text
CH4 kg + min(H2 kg / 0.5, CO2 kg / 2.75)
```

where the ratios come from the simulator's frozen Sabatier balance. The alpha assigns a
4 EUR/kg liability to any deficit below the episode's starting CH4-equivalent inventory and
0.15 EUR/kWh to a battery-energy deficit below its starting energy. Surplus above the initial
endowment has no terminal credit, so gratuitous end-of-horizon overproduction is not rewarded.

Rather than paying a sparse terminal penalty, every interval pays the change in each liability.
Those changes telescope exactly: over the full episode their sum equals the final deficit cost
and is independent of controller cadence or the path taken. Depleting an initial inventory is
therefore not free, while restoring it produces only the reversal of the prior liability.

### Battery exposure

Bus-side battery throughput is physical charge plus discharge energy. Its linear reserve is:

```text
battery throughput EUR/kWh = replacement EUR/kWh / (2 * full-cycle life)
```

The factor of two means one equivalent full cycle contains one nameplate charge and one
nameplate discharge. Defaults are 300 EUR/kWh replacement cost and 2,640 cycles, producing
0.056818 EUR per throughput kWh. The replacement value is a synthetic scenario assumption;
calendar aging, depth of discharge, temperature, SOC dwell, chemistry, augmentation, and
efficiency-dependent cell-side throughput remain absent. Equivalent full cycles are also
reported as an unmonetized physical metric.

### Electrolyser exposure

The alpha converts existing exact commitment counters into equivalent operating hours:

```text
equivalent h = operating h
             + 0.5 * low-load h
             + 2.0 * cold starts
             + 0.25 * hot starts
             + 2.0 * aborted starts
             + 4.0 * forced stops
```

It then applies a linear capital reserve using 480 EUR/kW and 80,000 operating hours. Only the
480 EUR/kW anchor is directly taken from the Clean Hydrogen JU 2024 AEL target. The lifetime
and event-equivalent hours are explicit research assumptions and require sensitivity tests.
This is an exposure ranking proxy, not simulated health loss, maintenance timing, or a claim
that one cold start causes exactly two hours of degradation. Startup and standby electricity
are already captured through accepted energy; they are not charged twice as energy.

DAC and methanator operating-life costs are not yet included because the simulator exposes no
stateful operating/thermal/cycle evidence for those units. Adding arbitrary coefficients would
create precision without validity.

## Default assumption register

| assumption | default | status |
| --- | ---: | --- |
| delivered service value | 4 EUR/kg | synthetic contract value |
| unmet delivery penalty | 20 EUR/kg | synthetic contract consequence |
| CH4-equivalent endowment | 4 EUR/kg | synthetic replenishment liability |
| battery-energy endowment | 0.15 EUR/kWh | synthetic terminal liability; equals tariff ceiling in the default fixture |
| battery replacement | 300 EUR/kWh | synthetic sensitivity center |
| battery full-cycle life | 2,640 | external scenario anchor; not a universal warranty |
| electrolyser capital | 480 EUR/kW | Clean Hydrogen JU 2024 AEL target |
| electrolyser life | 80,000 h | benchmark assumption |
| low-load multiplier | 1.5 | synthetic exposure coefficient |
| cold/hot/aborted/forced event equivalents | 2/0.25/2/4 h | synthetic exposure coefficients |

Every value, tariff point, normalization, physical configuration, controller cadence, legacy
diagnostic reward, and action/observation field is embedded in the checkpoint contract. A
changed coefficient or tariff therefore cannot silently reuse a policy or claim compatibility.

## Required validation before controller claims

The implementation gate is outcome-free:

- prior environments retain their exact action and observation sizes and distinct identities;
- the physical episode digest is identical for the same seed across all three environments;
- changing only the tariff changes observation, reward, checkpoint identity, and economic
  episode identity, but not the physical transition under the same target;
- tariff windows integrate exactly across hour and day boundaries;
- reward components sum exactly to the returned scalar and episode ledger;
- negative prices cannot create fictitious consumption revenue;
- endowment liability changes telescope under depletion and restoration;
- battery throughput and equipment exposure monotonically increase their costs;
- reset and step remain deterministic and satisfy the Gymnasium contract;
- invalid cadence and malformed assumptions fail before an episode can move.

After this gate passes, the next experiment is a short paired smoke tournament using the
common-door heuristic and MPC. It must establish that the task produces a nonzero economic
separation while retaining zero hard violations and comparable delivery. Only then is a
multi-seed benchmark justified. RL training remains gated on that evidence; training a policy
before proving residual headroom would repeat the original non-discriminative experiment.

## Known limitations and direction

This alpha improves task discrimination, not physical fidelity. Commercial evidence still
requires historical/site tariffs, explicit grid and PPA/export constraints, equipment vendor
curves, calendar and nonlinear battery aging, DAC and reactor commitment/thermal states,
qualified-product and RFNBO accounting, maintenance/replacement scheduling, uncertainty in
prices and failures, and financial boundaries including financing and fixed O&M.

The intended progression is:

```text
verified synthetic economics
    -> paired heuristic/MPC separation
    -> frozen train/validation/test economic benchmark
    -> multi-seed RL comparison and ablations
    -> historical price/weather replay
    -> vendor/site calibration and shadow-mode validation
```

Until those steps are complete, results must be described as performance on a transparent
synthetic operational benchmark, never as plant savings or commission-ready commercial proof.
