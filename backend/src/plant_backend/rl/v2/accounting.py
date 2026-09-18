"""Decision-interval requested/applied power accounting for V2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
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


def _require_finite_nonnegative(name: str, values: dict[str, float]) -> None:
    for channel, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0.0
        ):
            raise ValueError(
                f"{name} power channel {channel} must be finite and non-negative"
            )


@dataclass
class IntervalPowerAccounting:
    """Integrate requested and physically applied power over one decision."""

    hours: float = 0.0
    requested_kwh: dict[str, float] = field(default_factory=_zeros)
    guarded_kwh: dict[str, float] = field(default_factory=_zeros)
    applied_kwh: dict[str, float] = field(default_factory=_zeros)
    _guard_intervention_l1_kwh: float = field(default=0.0, init=False)
    _dispatch_gap_l1_kwh: float = field(default=0.0, init=False)
    _request_application_l1_kwh: float = field(default=0.0, init=False)

    def record(
        self,
        requested: Action,
        guarded: Action,
        *,
        after: PlantState,
        dt_h: float,
    ) -> None:
        if isinstance(dt_h, bool) or not isinstance(dt_h, Real):
            raise TypeError("dt_h must be a real number")
        if not math.isfinite(float(dt_h)) or dt_h <= 0.0:
            raise ValueError("dt_h must be positive and finite")
        # Share actions are fractions of the bus that actually materialises in
        # this physical interval. Using the prior boundary's solar value would
        # misclassify an exogenous weather ramp as controller/dispatch error.
        if (
            isinstance(after.solar_available_kw, bool)
            or not isinstance(after.solar_available_kw, Real)
            or not math.isfinite(float(after.solar_available_kw))
            or after.solar_available_kw < 0.0
        ):
            raise ValueError("realized solar bus must be finite and non-negative")
        bus_kw = float(after.solar_available_kw)
        requested_kw = _commanded_kw(requested, bus_kw)
        guarded_kw = _commanded_kw(guarded, bus_kw)
        applied_kw = {
            "battery": after.solar_to_battery_kw,
            "electrolyser": after.electrolyser_power_kw,
            "dac": after.dac_power_kw,
            "methanator": after.methanator_power_kw,
            "battery_support": after.battery_to_electrolyser_kw,
        }
        _require_finite_nonnegative("requested", requested_kw)
        _require_finite_nonnegative("guarded", guarded_kw)
        _require_finite_nonnegative("applied", applied_kw)
        step_hours = float(dt_h)
        self.hours += step_hours
        for name in ALL_POWER_CHANNELS:
            self.requested_kwh[name] += requested_kw[name] * step_hours
            self.guarded_kwh[name] += guarded_kw[name] * step_hours
            self.applied_kwh[name] += applied_kw[name] * step_hours
        self._guard_intervention_l1_kwh += step_hours * sum(
            abs(requested_kw[name] - guarded_kw[name]) for name in POWER_CHANNELS
        )
        self._dispatch_gap_l1_kwh += step_hours * sum(
            abs(guarded_kw[name] - applied_kw[name]) for name in POWER_CHANNELS
        )
        self._request_application_l1_kwh += step_hours * sum(
            abs(requested_kw[name] - applied_kw[name]) for name in POWER_CHANNELS
        )

    @property
    def guard_intervention_l1_kwh(self) -> float:
        """Legacy absolute gap between interval channel-energy totals."""
        return sum(
            abs(self.requested_kwh[name] - self.guarded_kwh[name])
            for name in POWER_CHANNELS
        )

    @property
    def dispatch_gap_l1_kwh(self) -> float:
        """Legacy absolute gap between guarded/applied interval totals."""
        return sum(
            abs(self.guarded_kwh[name] - self.applied_kwh[name])
            for name in POWER_CHANNELS
        )

    @property
    def request_application_l1_kwh(self) -> float:
        """Legacy absolute gap between requested/applied interval totals."""
        return sum(
            abs(self.requested_kwh[name] - self.applied_kwh[name])
            for name in POWER_CHANNELS
        )

    @property
    def guard_intervention_integral_l1_kwh(self) -> float:
        """Integral of the per-step request/guard L1 distance."""
        return self._guard_intervention_l1_kwh

    @property
    def dispatch_gap_integral_l1_kwh(self) -> float:
        """Integral of per-step guard/plant refusal and routing distance."""
        return self._dispatch_gap_l1_kwh

    @property
    def request_application_integral_l1_kwh(self) -> float:
        """Integral of the per-step end-to-end control-authority distance."""
        return self._request_application_l1_kwh

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
            "share_command_basis": "realized solar bus in each physical interval",
        }


__all__ = ["ALL_POWER_CHANNELS", "POWER_CHANNELS", "IntervalPowerAccounting"]
