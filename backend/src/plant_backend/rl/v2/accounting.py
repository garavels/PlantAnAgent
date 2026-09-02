"""Decision-interval requested/applied power accounting for V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plant_sim import Action, PlantState

POWER_CHANNELS = ("battery", "electrolyser", "dac", "methanator")
ALL_POWER_CHANNELS = (*POWER_CHANNELS, "battery_support")


def _zeros() -> dict[str, float]:
    return {name: 0.0 for name in ALL_POWER_CHANNELS}


def _commanded_kw(action: Action, bus_kw: float) -> dict[str, float]:
    return {
        "battery": action.battery_share * bus_kw,
        "electrolyser": action.electrolyser_share * bus_kw + action.battery_support_kw,
        "dac": action.dac_share * bus_kw,
        "methanator": action.methanator_share * bus_kw,
        "battery_support": action.battery_support_kw,
    }


@dataclass
class IntervalPowerAccounting:
    """Integrate requested and physically applied power over one decision."""

    hours: float = 0.0
    requested_kwh: dict[str, float] = field(default_factory=_zeros)
    guarded_kwh: dict[str, float] = field(default_factory=_zeros)
    applied_kwh: dict[str, float] = field(default_factory=_zeros)

    def record(
        self,
        requested: Action,
        guarded: Action,
        *,
        before: PlantState,
        after: PlantState,
        dt_h: float,
    ) -> None:
        bus_kw = max(0.0, before.solar_available_kw)
        requested_kw = _commanded_kw(requested, bus_kw)
        guarded_kw = _commanded_kw(guarded, bus_kw)
        applied_kw = {
            "battery": max(0.0, after.solar_to_battery_kw),
            "electrolyser": max(0.0, after.electrolyser_power_kw),
            "dac": max(0.0, after.dac_power_kw),
            "methanator": max(0.0, after.methanator_power_kw),
            "battery_support": max(0.0, after.battery_to_electrolyser_kw),
        }
        step_hours = max(0.0, float(dt_h))
        self.hours += step_hours
        for name in ALL_POWER_CHANNELS:
            self.requested_kwh[name] += requested_kw[name] * step_hours
            self.guarded_kwh[name] += guarded_kw[name] * step_hours
            self.applied_kwh[name] += applied_kw[name] * step_hours

    @property
    def guard_intervention_l1_kwh(self) -> float:
        """Absolute request/guard gap without double-counting support."""
        return sum(
            abs(self.requested_kwh[name] - self.guarded_kwh[name])
            for name in POWER_CHANNELS
        )

    @property
    def dispatch_gap_l1_kwh(self) -> float:
        """Absolute guard/plant gap caused by routing and physical throttling."""
        return sum(
            abs(self.guarded_kwh[name] - self.applied_kwh[name])
            for name in POWER_CHANNELS
        )

    @property
    def request_application_l1_kwh(self) -> float:
        """End-to-end request/application gap without double-counting support."""
        return sum(
            abs(self.requested_kwh[name] - self.applied_kwh[name])
            for name in POWER_CHANNELS
        )

    def to_dict(self) -> dict[str, Any]:
        divisor = max(self.hours, 1e-12)
        return {
            "hours": self.hours,
            "requested_kwh": dict(self.requested_kwh),
            "guarded_kwh": dict(self.guarded_kwh),
            "applied_kwh": dict(self.applied_kwh),
            "requested_mean_kw": {
                name: value / divisor for name, value in self.requested_kwh.items()
            },
            "guarded_mean_kw": {
                name: value / divisor for name, value in self.guarded_kwh.items()
            },
            "applied_mean_kw": {
                name: value / divisor for name, value in self.applied_kwh.items()
            },
            "guard_intervention_l1_kwh": self.guard_intervention_l1_kwh,
            "dispatch_gap_l1_kwh": self.dispatch_gap_l1_kwh,
            "request_application_l1_kwh": self.request_application_l1_kwh,
            "intervention_channels": list(POWER_CHANNELS),
            "battery_support_is_also_in_electrolyser_total": True,
        }


__all__ = ["ALL_POWER_CHANNELS", "POWER_CHANNELS", "IntervalPowerAccounting"]
