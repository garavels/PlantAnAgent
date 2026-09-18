"""Solar array: the plant's only power source, and the reason it is intermittent."""

from __future__ import annotations

import math
from typing import Any

from ..config import SolarConfig
from ..state import HOURS_PER_DAY, SECONDS_PER_HOUR
from .base import Component


class SolarArray(Component):
    """Deterministic half-sine day curve: zero at sunrise/sunset, peak at solar noon.

    ``resource_factor`` represents exogenous irradiance/weather relative to
    the clear-sky curve without making the equipment itself degraded.  An
    active ``solar_drop`` fault contributes the component health separately;
    the two effects multiply.  ``potential_kw`` therefore remains clear-sky
    power while ``power_kw`` reports what the bus actually receives.

    Extension point: improve ``clear_sky_kw`` with location or seasonal
    geometry without touching downstream dispatch.
    """

    name = "solar"

    def __init__(self, config: SolarConfig | None = None) -> None:
        self.config = config or SolarConfig()
        self.power_kw = 0.0
        self.potential_kw = 0.0
        self.resource_factor = 1.0

    def reset(self) -> None:
        self.power_kw = 0.0
        self.potential_kw = 0.0
        self.resource_factor = 1.0

    @property
    def output_factor(self) -> float:
        """Combined resource and equipment factor reaching the bus, 0..1."""
        return self.resource_factor * max(0.0, self.health)

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

    def update(self, time_s: float, *, resource_factor: float = 1.0) -> float:
        """Advance the array to `time_s` and return what it is generating."""
        if isinstance(resource_factor, bool):
            raise TypeError("resource_factor must be a real number")
        try:
            resource = float(resource_factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("resource_factor must be a real number") from exc
        if not math.isfinite(resource):
            raise ValueError("resource_factor must be finite")
        if not 0.0 <= resource <= 1.0:
            raise ValueError("resource_factor must be between 0 and 1")
        self.resource_factor = resource
        self.potential_kw = self.clear_sky_kw(time_s)
        self.power_kw = self.potential_kw * self.output_factor
        return self.power_kw

    def snapshot(self) -> dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "potential_kw": self.potential_kw,
            "resource_factor": self.resource_factor,
            "output_factor": self.output_factor,
            "peak_kw": self.config.peak_kw,
            **self.health_snapshot(),
        }
