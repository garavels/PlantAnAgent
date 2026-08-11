"""Power routing: the single place that decides where each kW goes.

Dispatch rules, applied in this order every timestep:

1. **Split the solar bus** using the action's (already normalised) shares.
   Anything the action leaves unallocated is curtailed on purpose.
2. **Resolve battery support.** The battery may discharge to top up the
   electrolyser, but never more than the stack can absorb and never more than
   the battery can deliver -- we do not discharge into a wall.
3. **Net out simultaneous charge/discharge.** Charging and discharging the same
   battery in one timestep is a pointless round trip, so the overlap is cancelled
   and that solar goes straight to the stack instead.
4. **Run the electrolyser**, capped by its rated power and by the room left in
   the H2 tank. Below its minimum load the stack stays off, battery support is
   retracted, and its solar spills back to the battery.
5. **Charge the battery** with its own share plus any spill. What the battery
   cannot absorb is curtailed.

Energy is conserved by construction:
`solar_available = solar_to_battery + solar_to_electrolyser + solar_curtailed`
and `electrolyser_power = solar_to_electrolyser + battery_to_electrolyser`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .components import Battery, Electrolyser, GasStorage
from .state import Action


@dataclass(frozen=True)
class PowerFlows:
    """Resolved flows for one timestep (kW), plus the resulting H2 (kg)."""

    solar_available_kw: float
    solar_to_battery_kw: float
    solar_to_electrolyser_kw: float
    battery_to_electrolyser_kw: float
    electrolyser_power_kw: float
    solar_curtailed_kw: float
    h2_produced_kg: float


def dispatch(
    action: Action,
    solar_kw: float,
    *,
    battery: Battery,
    electrolyser: Electrolyser,
    h2_tank: GasStorage,
    dt_h: float,
) -> PowerFlows:
    """Route `solar_kw` according to `action` and advance the components by `dt_h`.

    `action` is expected to be already sanitised (see `Action.sanitised`).
    """
    to_battery = action.battery_share * solar_kw
    to_electrolyser = action.electrolyser_share * solar_kw
    curtailed = max(0.0, solar_kw - to_battery - to_electrolyser)

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

    # 5. battery charging with its share plus whatever the stack refused.
    #    `support_kw > 0` implies both are zero: step 3 cancelled the overlap and
    #    step 4 only spills when the stack is already at its intake limit.
    charge_request_kw = to_battery + spill_kw
    charged_kw = 0.0 if support_kw > 0.0 else battery.charge(charge_request_kw, dt_h)
    curtailed += charge_request_kw - charged_kw

    return PowerFlows(
        solar_available_kw=solar_kw,
        solar_to_battery_kw=charged_kw,
        solar_to_electrolyser_kw=to_electrolyser,
        battery_to_electrolyser_kw=support_kw,
        electrolyser_power_kw=electrolyser.power_kw,
        solar_curtailed_kw=curtailed,
        h2_produced_kg=h2_kg,
    )
