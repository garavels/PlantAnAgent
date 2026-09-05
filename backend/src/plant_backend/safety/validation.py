"""The guard between "what was asked for" and "what the simulator is told".

Every action -- from the heuristic controller or from an operator's slider --
passes through `validate()` on its way to the plant. It is an *interface* guard,
not a safety system: the simulator still enforces its own physics, and will
happily refuse power the equipment cannot use. What this layer does is stop
obviously invalid commands from ever getting that far, and say why:

* nothing negative, nothing NaN, no oversubscribed bus (`Action.sanitised`)
* no power routed past a component's rating or its configured constraint, solar
  and battery support counted together
* no power routed to an offline unit
* no discharging the battery below its reserve, and never past its rating

Clamping rather than rejecting is deliberate: the plant keeps running under a bad
command, and the adjustment is reported so the control room can see it happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
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


@dataclass(frozen=True)
class BatterySupportAvailability:
    """Pure reserve-aware support envelope shared by dispatchers and the guard."""

    available_kw: float
    support_limit_kw: float
    reserve_soc: float
    usable_energy_kwh: float
    energy_limited_kw: float
    electrolyser_headroom_kw: float
    battery_available: bool
    electrolyser_available: bool

    def __post_init__(self) -> None:
        for name in (
            "available_kw",
            "support_limit_kw",
            "reserve_soc",
            "usable_energy_kwh",
            "energy_limited_kw",
            "electrolyser_headroom_kw",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reserve_soc > 1.0:
            raise ValueError("reserve_soc must not exceed one")
        for name in ("battery_available", "electrolyser_available"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        expected = (
            min(
                self.support_limit_kw,
                self.energy_limited_kw,
                self.electrolyser_headroom_kw,
            )
            if self.battery_available and self.electrolyser_available
            else 0.0
        )
        if not math.isclose(
            self.available_kw,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("available_kw disagrees with the support envelope")

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "available_kw": float(self.available_kw),
            "support_limit_kw": float(self.support_limit_kw),
            "reserve_soc": float(self.reserve_soc),
            "usable_energy_kwh": float(self.usable_energy_kwh),
            "energy_limited_kw": float(self.energy_limited_kw),
            "electrolyser_headroom_kw": float(self.electrolyser_headroom_kw),
            "battery_available": self.battery_available,
            "electrolyser_available": self.electrolyser_available,
        }


def battery_support_availability(
    observation: PlantState,
    *,
    constraints: Constraints,
    config: PlantConfig,
    electrolyser_headroom_kw: float,
) -> BatterySupportAvailability:
    """Return the exact bus-side battery support available for one plant step."""

    if not isinstance(observation, PlantState):
        raise TypeError("observation must be PlantState")
    if not isinstance(constraints, Constraints):
        raise TypeError("constraints must be Constraints")
    if not isinstance(config, PlantConfig):
        raise TypeError("config must be PlantConfig")
    if isinstance(electrolyser_headroom_kw, bool) or not isinstance(
        electrolyser_headroom_kw, Real
    ):
        raise TypeError("electrolyser_headroom_kw must be a real number")
    headroom_kw = float(electrolyser_headroom_kw)
    if not math.isfinite(headroom_kw) or headroom_kw < 0.0:
        raise ValueError("electrolyser_headroom_kw must be finite and non-negative")
    if not math.isfinite(config.dt_h) or config.dt_h <= 0.0:
        raise ValueError("plant timestep must be finite and positive")

    reserve_soc = constraints.reserve_soc(config)
    usable_kwh = max(0.0, observation.battery_soc - reserve_soc) * max(
        0.0, observation.battery_capacity_kwh
    )
    energy_limited_kw = usable_kwh * config.battery.discharge_efficiency / config.dt_h
    battery_health = observation.health.get("battery")
    electrolyser_health = observation.health.get("electrolyser")
    battery_available = battery_health is None or not battery_health.offline
    electrolyser_available = (
        electrolyser_health is None or not electrolyser_health.offline
    )
    support_limit_kw = constraints.support_limit_kw(config)
    available_kw = (
        min(support_limit_kw, energy_limited_kw, headroom_kw)
        if battery_available and electrolyser_available
        else 0.0
    )
    return BatterySupportAvailability(
        available_kw=available_kw,
        support_limit_kw=support_limit_kw,
        reserve_soc=reserve_soc,
        usable_energy_kwh=usable_kwh,
        energy_limited_kw=energy_limited_kw,
        electrolyser_headroom_kw=headroom_kw,
        battery_available=battery_available,
        electrolyser_available=electrolyser_available,
    )


def validate(
    action: Action | dict[str, Any] | None,
    observation: PlantState,
    *,
    constraints: Constraints,
    config: PlantConfig,
    dispatch_solar_available_kw: float | None = None,
) -> ValidatedAction:
    """Clamp ``action`` into the operating envelope for the plant.

    ``observation`` remains the controller-visible telemetry used for health,
    inventory, and reserve checks. A guarded runtime should additionally pass
    the exact bus its next physical step will dispatch. This keeps a policy's
    causal information set separate from a fast interlock's obligation to
    enforce absolute kW caps at actuation time.
    """
    safe = Action.sanitised(action)
    adjustments: list[str] = []

    if dispatch_solar_available_kw is None:
        bus_kw = max(0.0, observation.solar_available_kw)
    else:
        if isinstance(dispatch_solar_available_kw, bool) or not isinstance(
            dispatch_solar_available_kw, Real
        ):
            raise TypeError("dispatch_solar_available_kw must be a real number")
        bus_kw = float(dispatch_solar_available_kw)
        if not math.isfinite(bus_kw) or bus_kw < 0.0:
            raise ValueError(
                "dispatch_solar_available_kw must be finite and non-negative"
            )
    offline = {name: health.offline for name, health in observation.health.items()}
    limits = (
        (
            "electrolyser_share",
            "electrolyser",
            constraints.electrolyser_limit_kw(config),
            "Electrolyser",
        ),
        ("dac_share", "dac", constraints.dac_limit_kw(config), "DAC"),
        (
            "methanator_share",
            "methanator",
            constraints.methanator_limit_kw(config),
            "Methanator",
        ),
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

    # A cap on the stack is a cap on the stack: solar and borrowed battery power
    # both land on the same rectifier, so they share the one limit.
    headroom_kw = max(
        0.0,
        constraints.electrolyser_limit_kw(config)
        - shares["electrolyser_share"] * bus_kw,
    )
    support_kw = _support_kw(
        safe, observation, constraints, config, offline, adjustments, headroom_kw
    )
    # what the process gives back stays with the battery rather than vanishing
    freed = sum(getattr(safe, key) - shares[key] for key, _, _, _ in limits)
    battery_share = min(1.0, safe.battery_share + max(0.0, freed))

    # Redistributing a disabled process share back to the battery can put the
    # ordered IEEE-754 sum one ULP above 1.0 even though the allocation is
    # mathematically conserved.  Canonicalize the final guard output through
    # the same sanitizer used by ``PlantSimulator.step`` so "validated" means
    # byte-for-byte what the plant will record, not merely numerically close.
    guarded_action = Action.sanitised(
        Action(
            battery_share=battery_share,
            electrolyser_share=shares["electrolyser_share"],
            dac_share=shares["dac_share"],
            methanator_share=shares["methanator_share"],
            battery_support_kw=support_kw,
        )
    )
    return ValidatedAction(guarded_action, tuple(adjustments))


def _support_kw(
    action: Action,
    observation: PlantState,
    constraints: Constraints,
    config: PlantConfig,
    offline: dict[str, bool],
    adjustments: list[str],
    headroom_kw: float,
) -> float:
    """Battery discharge the plant is allowed to lend the stack right now."""
    requested_kw = float(action.battery_support_kw)
    if requested_kw <= 0.0:
        return 0.0
    if offline.get("electrolyser", False):
        adjustments.append("Electrolyser is offline: battery support dropped")
        return 0.0

    availability = battery_support_availability(
        observation,
        constraints=constraints,
        config=config,
        electrolyser_headroom_kw=headroom_kw,
    )
    if not availability.battery_available:
        adjustments.append("Battery is offline: battery support dropped")
        return 0.0

    reserve_soc = availability.reserve_soc
    if observation.battery_soc <= reserve_soc:
        adjustments.append(
            f"Battery at its {reserve_soc:.0%} reserve ({observation.battery_soc:.0%}): support dropped"
        )
        return 0.0

    limit_kw = availability.available_kw
    if requested_kw > limit_kw:
        adjustments.append(
            f"Battery support clamped to {limit_kw:.0f} kW (asked for {requested_kw:.0f} kW)"
        )
        return limit_kw
    return requested_kw
