"""Solar array: the plant's only power source, and the reason it is intermittent."""

from __future__ import annotations

import math
from typing import Any

from ..config import SolarConfig
from ..state import HOURS_PER_DAY, SECONDS_PER_HOUR
from .base import Component


class SolarArray(Component):
    """Deterministic half-sine day curve: zero at sunrise/sunset, peak at solar noon.

    Weather is a *disturbance*, not a fault: a `solar_drop` derates the array,
    so `potential_kw` keeps reporting the clear-sky curve while `power_kw`
    reports what the plant actually gets. The gap between them is exactly the
    uncertainty an intermittently powered plant has to live with.

    Extension point: swap `power_at` for an irradiance/weather model (seasons,
    forecast noise) without touching anything downstream.
    """

    name = "solar"

    def __init__(self, config: SolarConfig | None = None) -> None:
        self.config = config or SolarConfig()
        self.power_kw = 0.0
        self.potential_kw = 0.0

    def reset(self) -> None:
        self.power_kw = 0.0
        self.potential_kw = 0.0

    @property
    def output_factor(self) -> float:
        """Fraction of the clear-sky curve currently reaching the bus, 0..1."""
        return max(0.0, self.health)

    def clear_sky_kw(self, time_s: float) -> float:
        """Undisturbed power at an absolute simulation time (kW)."""
        hour = (time_s / SECONDS_PER_HOUR) % HOURS_PER_DAY
        sunrise, sunset = self.config.sunrise_h, self.config.sunset_h
        if not sunrise < hour < sunset:
            return 0.0
        daylight = sunset - sunrise
        return self.config.peak_kw * math.sin(math.pi * (hour - sunrise) / daylight)

    def power_at(self, time_s: float) -> float:
        """Power actually available at `time_s`, after any active disturbance."""
        return self.clear_sky_kw(time_s) * self.output_factor

    def update(self, time_s: float) -> float:
        """Advance the array to `time_s` and return what it is generating."""
        self.potential_kw = self.clear_sky_kw(time_s)
        self.power_kw = self.potential_kw * self.output_factor
        return self.power_kw

    def snapshot(self) -> dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "potential_kw": self.potential_kw,
            "output_factor": self.output_factor,
            "peak_kw": self.config.peak_kw,
            **self.health_snapshot(),
        }
