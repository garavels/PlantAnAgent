"""Power routing: the single place that decides where each kW goes.

Dispatch rules, applied in this order every timestep:

1. **Split the solar bus** using the action's (already normalised) shares:
   battery, electrolyser, DAC, methanator. Anything the action leaves
   unallocated is curtailed on purpose.
2. **Resolve battery support.** The battery may discharge to top up the
   electrolyser, but never more than the stack can absorb and never more than
   the battery can deliver -- we do not discharge into a wall.
3. **Net out simultaneous charge/discharge.** Charging and discharging the same
   battery in one timestep is a pointless round trip, so the overlap is cancelled
   and that solar goes straight to the stack instead.
4. **Run the electrolyser**, capped by its rated power and by the room left in
   the H2 tank. Below its minimum load the stack stays off, battery support is
   retracted, and its solar spills back to the battery.
5. **Run the DAC**, capped by its rated power and by the room left in the CO2
   tank. Whatever it cannot usefully take spills back to the battery.
6. **Run the methanator.** CH4 output follows a fixed recipe (H2 + CO2 + power)
   and is capped by whichever input runs short -- gas in the tanks, room in the
   CH4 tank, or the power offered. The unit draws only the power its actual
   production needs; the rest spills back to the battery.
7. **Charge the battery** with its share plus every spill. What the battery
   cannot absorb is curtailed.

Energy is conserved by construction:
`solar_available = solar_to_battery + solar_to_electrolyser + solar_to_dac
                 + solar_to_methanator + solar_curtailed`
and `electrolyser_power = solar_to_electrolyser + battery_to_electrolyser`.

Material is conserved the same way: H2 and CO2 leave their tanks only through
the methanator, in the fixed ratio the recipe demands, and CH4 only appears
when both gases and power were actually there to make it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .components import Battery, DirectAirCapture, Electrolyser, GasStorage, Methanator
from .state import Action


@dataclass(frozen=True)
class PowerFlows:
    """Resolved flows for one timestep (kW), plus the resulting masses (kg)."""

    solar_available_kw: float
    solar_to_battery_kw: float
    solar_to_electrolyser_kw: float
    solar_to_dac_kw: float
    solar_to_methanator_kw: float
    battery_to_electrolyser_kw: float
    electrolyser_power_kw: float
    dac_power_kw: float
    methanator_power_kw: float
    solar_curtailed_kw: float
    h2_produced_kg: float
    co2_captured_kg: float
    ch4_produced_kg: float


def dispatch(
    action: Action,
    solar_kw: float,
    *,
    battery: Battery,
    electrolyser: Electrolyser,
    h2_tank: GasStorage,
    dac: DirectAirCapture,
    co2_tank: GasStorage,
    methanator: Methanator,
    ch4_tank: GasStorage,
    dt_h: float,
) -> PowerFlows:
    """Route `solar_kw` according to `action` and advance the components by `dt_h`.

    `action` is expected to be already sanitised (see `Action.sanitised`).
    """
    to_battery = action.battery_share * solar_kw
    to_electrolyser = action.electrolyser_share * solar_kw
    to_dac = action.dac_share * solar_kw
    to_methanator = action.methanator_share * solar_kw
    curtailed = max(0.0, solar_kw - to_battery - to_electrolyser - to_dac - to_methanator)

    # 2. battery support, clamped by both ends of the wire
    intake_limit_kw = electrolyser.intake_limit_kw(dt_h, h2_tank.headroom_kg)
    support_kw = min(
        action.battery_support_kw,
        battery.discharge_headroom_kw(dt_h),
        max(0.0, intake_limit_kw - to_electrolyser),
    )

    # 3. cancel the round trip: send that solar to the stack directly instead
    overlap_kw = min(support_kw, to_battery)
    to_battery -= overlap_kw
    to_electrolyser += overlap_kw
    support_kw -= overlap_kw

    # 4. electrolyser, capped by rated power and tank headroom
    spill_kw = max(0.0, to_electrolyser - intake_limit_kw)
    to_electrolyser -= spill_kw

    if to_electrolyser + support_kw < electrolyser.min_load_kw:
        # not worth starting the stack: retract support, spill its solar
        support_kw = 0.0
        spill_kw += to_electrolyser
        to_electrolyser = 0.0
        electrolyser.run(0.0, dt_h)
        h2_kg = 0.0
    else:
        support_kw = battery.discharge(support_kw, dt_h) if support_kw > 0.0 else 0.0
        h2_kg = h2_tank.store(electrolyser.run(to_electrolyser + support_kw, dt_h))

    # 5. DAC, capped by rated power and CO2 tank headroom; refusal spills
    dac_limit_kw = dac.intake_limit_kw(dt_h, co2_tank.headroom_kg)
    dac_kw = min(to_dac, dac_limit_kw)
    spill_kw += to_dac - dac_kw
    co2_kg = co2_tank.store(dac.run(dac_kw, dt_h))

    # 6. methanator: CH4 follows the fixed recipe, limited by the scarcest input
    result = methanator.run(
        to_methanator,
        dt_h,
        h2_available_kg=h2_tank.level_kg,
        co2_available_kg=co2_tank.level_kg,
        ch4_headroom_kg=ch4_tank.headroom_kg,
    )
    h2_tank.withdraw(result.h2_kg)
    co2_tank.withdraw(result.co2_kg)
    ch4_kg = ch4_tank.store(result.ch4_kg)
    spill_kw += to_methanator - methanator.power_kw

    # 7. battery charging with its share plus whatever the process refused.
    #    `support_kw > 0` implies `to_battery` is zero: step 3 cancelled the
    #    overlap. Spills while discharging are curtailed, never round-tripped.
    charge_request_kw = to_battery + spill_kw
    charged_kw = 0.0 if support_kw > 0.0 else battery.charge(charge_request_kw, dt_h)
    curtailed += charge_request_kw - charged_kw

    return PowerFlows(
        solar_available_kw=solar_kw,
        solar_to_battery_kw=charged_kw,
        solar_to_electrolyser_kw=to_electrolyser,
        solar_to_dac_kw=dac.power_kw,
        solar_to_methanator_kw=methanator.power_kw,
        battery_to_electrolyser_kw=support_kw,
        electrolyser_power_kw=electrolyser.power_kw,
        dac_power_kw=dac.power_kw,
        methanator_power_kw=methanator.power_kw,
        solar_curtailed_kw=curtailed,
        h2_produced_kg=h2_kg,
        co2_captured_kg=co2_kg,
        ch4_produced_kg=ch4_kg,
    )
