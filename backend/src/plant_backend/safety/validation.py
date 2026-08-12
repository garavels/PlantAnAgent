"""The guard between "what was asked for" and "what the simulator is told".

Every action -- from the heuristic controller or from an operator's slider --
passes through `validate()` on its way to the plant. It is an *interface* guard,
not a safety system: the simulator still enforces its own physics, and will
happily refuse power the equipment cannot use. What this layer does is stop
obviously invalid commands from ever getting that far, and say why:

* nothing negative, nothing NaN, no oversubscribed bus (`Action.sanitised`)
* no power routed past a component's rating or its configured constraint
* no power routed to an offline unit
* no discharging the battery below its reserve, and never past its rating

Clamping rather than rejecting is deliberate: the plant keeps running under a bad
command, and the adjustment is reported so the control room can see it happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plant_sim import Action, PlantConfig, PlantState

from ..controllers import Constraints


@dataclass(frozen=True)
class ValidatedAction:
    """The action as it will actually be applied, plus what had to change."""

    action: Action
    adjustments: tuple[str, ...] = ()

    @property
    def clamped(self) -> bool:
        return bool(self.adjustments)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "adjustments": list(self.adjustments)}


def validate(
    action: Action | dict[str, Any] | None,
    observation: PlantState,
    *,
    constraints: Constraints,
    config: PlantConfig,
) -> ValidatedAction:
    """Clamp `action` into the operating envelope for the observed plant."""
    safe = Action.sanitised(action)
    adjustments: list[str] = []

    bus_kw = max(0.0, observation.solar_available_kw)
    offline = {name: health.offline for name, health in observation.health.items()}
    limits = (
        ("electrolyser_share", "electrolyser", constraints.electrolyser_limit_kw(config), "Electrolyser"),
        ("dac_share", "dac", constraints.dac_limit_kw(config), "DAC"),
        ("methanator_share", "methanator", constraints.methanator_limit_kw(config), "Methanator"),
    )

    shares: dict[str, float] = {}
    for key, unit, limit_kw, label in limits:
        share = getattr(safe, key)
        if offline.get(unit, False):
            if share > 0.0:
                adjustments.append(f"{label} is offline: allocation dropped")
            shares[key] = 0.0
            continue
        requested_kw = share * bus_kw
        if requested_kw > limit_kw:
            shares[key] = limit_kw / bus_kw if bus_kw > 0.0 else 0.0
            adjustments.append(
                f"{label} allocation clamped to its {limit_kw:.0f} kW limit "
                f"(asked for {requested_kw:.0f} kW)"
            )
        else:
            shares[key] = share

    support_kw = _support_kw(safe, observation, constraints, config, offline, adjustments)
    # what the process gives back stays with the battery rather than vanishing
    freed = sum(getattr(safe, key) - shares[key] for key, _, _, _ in limits)
    battery_share = min(1.0, safe.battery_share + max(0.0, freed))

    return ValidatedAction(
        Action(
            battery_share=battery_share,
            electrolyser_share=shares["electrolyser_share"],
            dac_share=shares["dac_share"],
            methanator_share=shares["methanator_share"],
            battery_support_kw=support_kw,
        ),
        tuple(adjustments),
    )


def _support_kw(
    action: Action,
    observation: PlantState,
    constraints: Constraints,
    config: PlantConfig,
    offline: dict[str, bool],
    adjustments: list[str],
) -> float:
    """Battery discharge the plant is allowed to lend the stack right now."""
    requested_kw = action.battery_support_kw
    if requested_kw <= 0.0:
        return 0.0
    if offline.get("electrolyser", False):
        adjustments.append("Electrolyser is offline: battery support dropped")
        return 0.0

    reserve_soc = constraints.reserve_soc(config)
    if observation.battery_soc <= reserve_soc:
        adjustments.append(
            f"Battery at its {reserve_soc:.0%} reserve ({observation.battery_soc:.0%}): support dropped"
        )
        return 0.0

    usable_kwh = (observation.battery_soc - reserve_soc) * observation.battery_capacity_kwh
    limit_kw = min(constraints.support_limit_kw(config), usable_kwh / config.dt_h)
    if requested_kw > limit_kw:
        adjustments.append(
            f"Battery support clamped to {limit_kw:.0f} kW (asked for {requested_kw:.0f} kW)"
        )
        return limit_kw
    return requested_kw
