"""What the frontend / an agent / an RL policy sees and sends.

`Action` in, `PlantState` out. Both are immutable and JSON-serialisable so they
can cross a process boundary (HTTP today, anything later) without translation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0


def _finite(value: Any, fallback: float = 0.0) -> float:
    """Coerce anything action-shaped into a usable float (NaN/inf/str -> fallback)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class Action:
    """How the available power should be split this timestep.

    `battery_share` + `electrolyser_share` are fractions of *currently available
    solar power*. They are normalised down if they sum above 1.0; anything left
    unallocated is curtailed. `battery_support_kw` lets the caller pull stored
    energy back out of the battery to feed the electrolyser -- which is the whole
    point of having a battery on an intermittently powered plant.
    """

    battery_share: float = 0.5
    electrolyser_share: float = 0.5
    battery_support_kw: float = 0.0

    @classmethod
    def sanitised(cls, action: "Action | dict[str, Any] | None") -> "Action":
        """Never trust the caller: clamp shares to [0, 1] and support to >= 0."""
        if action is None:
            return cls(battery_share=0.0, electrolyser_share=0.0)
        raw = action if isinstance(action, dict) else asdict(action)
        battery = _clamp(_finite(raw.get("battery_share")), 0.0, 1.0)
        electrolyser = _clamp(_finite(raw.get("electrolyser_share")), 0.0, 1.0)
        total = battery + electrolyser
        if total > 1.0:  # oversubscribed bus -> scale both back proportionally
            battery /= total
            electrolyser /= total
        return cls(
            battery_share=battery,
            electrolyser_share=electrolyser,
            battery_support_kw=max(0.0, _finite(raw.get("battery_support_kw"))),
        )

    @classmethod
    def all_to_battery(cls) -> "Action":
        return cls(battery_share=1.0, electrolyser_share=0.0)

    @classmethod
    def all_to_electrolyser(cls) -> "Action":
        return cls(battery_share=0.0, electrolyser_share=1.0)

    @classmethod
    def idle(cls) -> "Action":
        return cls(battery_share=0.0, electrolyser_share=0.0)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PlantState:
    """Complete observable plant state after a timestep.

    Levels (`*_soc`, `*_level_kg`) describe the plant *now*; flows (`*_kw`,
    `*_kg_h`) describe the timestep that just finished.
    """

    step: int = 0
    time_s: float = 0.0

    # --- generation -------------------------------------------------------
    solar_available_kw: float = 0.0
    solar_to_battery_kw: float = 0.0
    solar_to_electrolyser_kw: float = 0.0
    solar_curtailed_kw: float = 0.0

    # --- battery ----------------------------------------------------------
    battery_soc: float = 0.0
    battery_energy_kwh: float = 0.0
    battery_power_kw: float = 0.0  # >0 charging, <0 discharging
    battery_to_electrolyser_kw: float = 0.0

    # --- electrolyser -----------------------------------------------------
    electrolyser_power_kw: float = 0.0
    electrolyser_load_frac: float = 0.0
    electrolyser_online: bool = False
    h2_production_kg_h: float = 0.0

    # --- hydrogen storage -------------------------------------------------
    h2_level_kg: float = 0.0
    h2_capacity_kg: float = 0.0
    h2_fill: float = 0.0

    # --- bookkeeping ------------------------------------------------------
    energy_produced_kwh: float = 0.0
    energy_curtailed_kwh: float = 0.0
    h2_produced_total_kg: float = 0.0
    action: Action = field(default_factory=Action.idle)

    @property
    def time_h(self) -> float:
        return self.time_s / SECONDS_PER_HOUR

    @property
    def time_of_day_h(self) -> float:
        return self.time_h % HOURS_PER_DAY

    @property
    def day(self) -> int:
        return int(self.time_h // HOURS_PER_DAY)

    @property
    def clock(self) -> str:
        """`D1 13:45` -- the label the control room shows."""
        minutes = int(round(self.time_of_day_h * 60.0)) % (24 * 60)
        return f"D{self.day + 1} {minutes // 60:02d}:{minutes % 60:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "time_h": self.time_h,
            "time_of_day_h": self.time_of_day_h,
            "day": self.day,
            "clock": self.clock,
        }
