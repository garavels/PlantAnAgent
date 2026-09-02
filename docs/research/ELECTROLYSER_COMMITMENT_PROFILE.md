# Electrolyser commitment alpha: evidence and frozen design

Status: **control-oriented benchmark profile; literature-bounded but not
vendor-calibrated physical truth.**

## Decision

The first stateful equipment profile is a 3 MW system-level pressurised alkaline
electrolyser with fixed balance-of-plant controls:

```text
ael_pressurised_fixed_bop_alpha_v1
```

Alkaline electrolysis is selected because its thermal startup and low-load gas-purity
limits create a real scheduling decision at the environment's 15-minute controller
cadence. PEM electrical response is commonly seconds to minutes. Assigning PEM an
hour-scale commitment merely to make RL useful would be an artificial benchmark.
PEM remains the intended contrasting profile for later transfer evaluation.

This profile is a scheduling archetype. It does not model temperature, pressure, lye
flow, gas crossover, product purity, or vendor control logic, and it is not a calibrated
digital twin of a commercial stack.

## Evidence boundary

The Clean Hydrogen JU 2020 state-of-the-art AEL table reports 50 kWh/kg at nominal
capacity, 3,600 seconds from cold standby to nominal hydrogen production, 60 seconds
from hot idle, and 0.12% steady nominal efficiency degradation per 1,000 hours. Its
declared system boundary is AC power and tap water to 30-bar, purity-5 hydrogen. The
existing simulator's 52 kWh/kg nominal value is therefore retained as a plausible
system-level proxy rather than changed solely for this experiment.

The JRC reports a practical AEL minimum load around 20--25%, warm response around
1--5 minutes, and much faster PEM response. A 250 kW experimental AEL study found a
42.0% minimum with lye-flow-only control, 21.2% with pressure control, and 15.6% only
with coordinated pressure and lye-flow control. Because the alpha has no pressure,
flow, or gas-purity state, a conservative 40% minimum is internally coherent. It must
not be generalized to every alkaline installation.

Relevant sources:

- [Clean Hydrogen JU SRIA technology KPIs](https://www.clean-hydrogen.europa.eu/knowledge-management/strategy-map-and-key-performance-indicators/clean-hydrogen-ju-sria-key-performance-indicators-kpis_en)
- [JRC, *Electrolysers as a flexible load*](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC139101/JRC139101_01.pdf)
- [JRC harmonised low-temperature electrolyser test protocols](https://publications.jrc.ec.europa.eu/repository/handle/JRC122565)
- [Hu et al., 250 kW AEL minimum-load control](https://doi.org/10.1016/j.etran.2023.100304)
- [Ding et al., AEL steady-state and cold-start energy flow](https://doi.org/10.1016/j.fuel.2023.128799)
- [Gu et al., experimental 250 kW AEL dynamics](https://doi.org/10.1016/j.jpowsour.2023.233920)

Reported startup endpoints differ: first current, first gas, nominal flow, nominal
temperature, pressure, and on-spec deliverable product are not interchangeable.
Likewise, stack-DC efficiency is not an AC system curve, and electrical-current ramp
is not a saleable-hydrogen ramp. The alpha therefore labels each parameter as either
an evidence anchor or a benchmark assumption.

## Frozen alpha parameters

| parameter | default | status and rationale |
| --- | ---: | --- |
| nameplate power | 3,000 kW | existing plant design |
| nominal specific energy | 52 kWh/kg | retained legacy system-level proxy; JU AEL reference is 50 kWh/kg |
| cold-start duration | 60 min | official 2020 AEL state-of-the-art anchor |
| cold-start power | 20% nameplate | synthetic powered-heating proxy; not a universal startup-energy claim |
| deliverable H2 during startup | zero | qualitative physical constraint; startup gas can be off specification or vented |
| hot-start duration | 5 min | conservative end of the JRC 1--5 minute warm range |
| hot-start power | 20% nameplate | synthetic balance-of-plant proxy |
| minimum stable running load | 40% nameplate | evidence-bounded fixed-control archetype; does not assume active pressure/flow optimization |
| hot-standby power | 2% nameplate | synthetic heat/pressure-retention proxy |
| commitment-command deadband | resolved offered power <= 1% stack nameplate | controller-interface constant, not equipment physics; creates a reachable off region for continuous policies and remains below standby demand |
| cooldown | 15 min, unpowered | synthetic restart-settling state; unpowered so the simulator never invents energy |
| minimum committed/up time | 60 min | synthetic unit-commitment challenge parameter |
| minimum down time | 30 min | synthetic unit-commitment challenge parameter |
| running ramp | +/-25% nameplate/min | synthetic system-production settling proxy, not an electrical-current limit |
| power-shortfall grace | 2 min | synthetic hysteresis against one-minute nuisance trips |
| low-load exposure threshold | 60% nameplate | reporting threshold only; no asserted physical damage rate |

Cold start at the frozen settings consumes 600 kWh if completed. That is a benchmark
consequence of the declared power and duration, not a claimed universal kWh/start.
The required sensitivity set is cold start 60/120/240 minutes, start power
10%/20%/40%, minimum load 20%/30%/40%, flat versus nonlinear yield, and commitment
disabled.

The provisional effective specific-energy curve is a deliberately synthetic AC-system
penalty for fixed auxiliaries:

| running load | multiplier | effective kWh/kg at nominal health |
| ---: | ---: | ---: |
| 40% | 1.12 | 58.24 |
| 60% | 1.06 | 55.12 |
| 80% | 1.02 | 53.04 |
| 100% | 1.00 | 52.00 |

Linear interpolation is used between anchors. This curve is not fitted to the cited
experiments because published stack-only and whole-system boundaries differ materially.
Every anchor is at least the legacy nominal specific energy, permitting a conservative
H2-storage headroom cap without discarding produced mass.

No start or cycle directly changes simulated physical health. Seven-day trajectories
cannot validate an equipment-life law, and steady nominal degradation cannot be
converted into a per-start coefficient. The simulator records exposure instead:
starts, aborted starts, trips, mode time, startup/standby energy, low-load time,
operating time, ramp limiting, and absolute load movement. A later economic wear proxy
must be explicit, ablated, and sensitivity-tested.

## State and interval semantics

The operating modes are:

```text
off -> cold_starting -> running <-> hot_standby
                            ^          |
                            |      hot_starting
                            +----------+

starting/running/standby -> cooldown -> off
```

The profile advances on the one-minute physics grid while the supervisory request is
held for 15 minutes. The existing five-dimensional action has no separate mode command,
so commitment intent remains implicit, but it no longer depends on hitting an exact
floating-point endpoint. Dynamics version 3 resolves feasible solar routing and battery
support into one offered stack power in kW after intake capping and overlap cancellation.
Intent is false when that offer is at or below 1% of electrolyser nameplate. This
synthetic, fingerprinted deadband lies below the 2% standby demand and gives continuous
policies a finite off-command region. It is an acknowledged interface assumption, not
vendor control logic. The earlier comparison of a dimensionless bus share with this
stack-relative threshold was invalid because resizing the upstream PV bus changed the
meaning of the same physically adequate stack offer.

- An eligible off unit begins a cold start only when the offered power reaches the
  fixed cold-start draw.
- Cold and hot starts consume their fixed accepted power and produce no deliverable
  hydrogen. A completed startup changes mode at the end of its final minute; running
  hydrogen begins no earlier than the following minute.
- A running request below minimum stable load but at or above standby power moves the
  unit to hot standby. Standby remains committed and continues its minimum-up clock.
- A running request is bounded by the previous one-minute load and the declared ramp.
  The equipment may refuse excess upward power; it never consumes electricity that
  was not physically routed to it.
- A request inside the declared deadband after minimum-up completion is an orderly stop. An early
  stop, fault outage, or persistent inability to sustain the required mode is recorded
  as an abort or forced trip.
- A request outside the deadband with insufficient power pauses startup/operation for at most the
  two-minute grace. Persistent shortfall enters cooldown. The grace counter is part of
  observable Markov state.
- Cooldown and remaining minimum-down time block restart. No cooldown power is
  consumed in this alpha.
- `electrolyser_online` retains its existing meaning: the unit is producing deliverable
  hydrogen. It is false during powered startup, hot standby, and cooldown.

The committed allocation path retracts unused battery support before discharge and
returns refused solar to the existing battery/curtailment path. These identities remain
mandatory each physical step:

```text
solar available = solar to battery + solar to electrolyser + solar to DAC
                + solar to methanator + curtailed solar

electrolyser power = solar to electrolyser + battery to electrolyser
```

## Compatibility and interface gates

Commitment is opt-in at the simulator constructor and must not add fields to
`PlantConfig` or `PlantState`. With the profile absent, the existing electrolyser and
dispatch arithmetic remain on the legacy path. V1, the live runtime, HEPA telemetry,
and the HTTP state payload therefore retain their current shape and behavior.

The pre-change weather-alpha trace at revision `e0a0ada` is frozen by SHA-256 in the
test suite. The compatibility-off environment must exactly reproduce the first 42
observations, plant states, rewards, delivery accounting, guard reports, and energy
accounting over all 96 decisions—not merely remain close within a tolerance.

The commitment-alpha observation appends mode one-hot state, prior applied load, ramp
headroom, startup/minimum-up/minimum-down timers, and power-shortfall grace after the
existing 42-field prefix. The full profile and calibration labels belong in the private
audit/checkpoint contract; fixed profile parameters need not be repeated in every
observation. Cumulative exposure counters are metrics, not policy state unless they
later influence dynamics or objective.

Every public commitment snapshot carries a canonical SHA-256 of the complete profile.
The observation encoder and interval accounting compare this digest, not only the human
profile label, so a numerically changed profile cannot silently reuse timer normalization.

The current action is deliberately retained for the first causal ablation. If physical
refusal and guard/router projection still dominate policy authority, the next interface
must separate desired equipment mode and target running load from fast bus allocation.
MPC and RL will not be compared until they receive an equivalent controller-neutral
command and forecast interface.

## Acceptance tests and next experiment

The increment is accepted only if tests pin duration/transition boundaries, real startup
energy and zero startup H2, hot restart, interruption/grace, minimum up/down behavior,
ramp bounds, nonlinear yield, fault trips, H2 headroom, reset/replay, and every energy
and mass identity. Explicit `None` and omitted commitment profiles must reproduce legacy
behavior exactly.

After implementation, demand is recalibrated on development seeds. The first scientific
experiment is factorial and paired:

```text
commitment off/on
    x forecast blind/issued/perfect
    x delivery-aware heuristic/MPC/feasible future-information planner
```

Primary outcomes are delivery reliability and unmet kilograms subject to zero hard
violations, followed by starts, trips, startup/standby energy, low-load exposure,
energy per delivered kilogram, curtailment, battery cycling, physical refusal, guard
intervention, solve time, and tail risk. No RL campaign begins until this experiment
shows a material controllable gap.
