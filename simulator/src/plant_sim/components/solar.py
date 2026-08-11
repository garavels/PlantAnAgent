"""Solar array: the plant's only power source, and the reason it is intermittent."""

from __future__ import annotations

import math
from typing import Any

from ..config import SolarConfig
from ..state import HOURS_PER_DAY, SECONDS_PER_HOUR
from .base import Component


class SolarArray(Component):
    """Deterministic half-sine day curve: zero at sunrise/sunset, peak at solar noon.

    Extension point: swap `power_at` for an irradiance/weather model (clouds,
    seasons, forecast noise) without touching anything downstream.
    """

    name = "solar"

    def __init__(self, config: SolarConfig | None = None) -> None:
        self.config = config or SolarConfig()
        self.power_kw = 0.0

    def reset(self) -> None:
        self.power_kw = 0.0

    def power_at(self, time_s: float) -> float:
        """Available power at an absolute simulation time (kW)."""
        hour = (time_s / SECONDS_PER_HOUR) % HOURS_PER_DAY
        sunrise, sunset = self.config.sunrise_h, self.config.sunset_h
        if not sunrise < hour < sunset:
            return 0.0
        daylight = sunset - sunrise
        return self.config.peak_kw * math.sin(math.pi * (hour - sunrise) / daylight)

    def update(self, time_s: float) -> float:
        """Advance the array to `time_s` and return what it is generating."""
        self.power_kw = self.power_at(time_s)
        return self.power_kw

    def snapshot(self) -> dict[str, Any]:
        return {"power_kw": self.power_kw, "peak_kw": self.config.peak_kw}
