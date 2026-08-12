"""Direct-air capture: turns electricity into captured CO2, linearly."""

from __future__ import annotations

from typing import Any

from ..config import DACConfig
from .base import Component


class DirectAirCapture(Component):
    """Consumes power anywhere between zero and rated, captures CO2 linearly.

    Unlike the electrolyser there is no minimum load: the unit modulates
    freely, so every kW offered is either used or refused. The CO2 it captures
    must fit in the storage tank -- a full tank throttles the unit, exactly
    like the H2 tank throttles the stack.

    Extension point: sorbent capacity cycles, humidity/temperature effects,
    fan-vs-heat split of the energy demand, degradation.
    """

    name = "dac"

    def __init__(self, config: DACConfig | None = None) -> None:
        self.config = config or DACConfig()
        self.power_kw = 0.0
        self.online = False

    def reset(self) -> None:
        self.power_kw = 0.0
        self.online = False

    @property
    def load_frac(self) -> float:
        return self.power_kw / self.config.rated_kw

    def intake_limit_kw(self, dt_h: float, co2_headroom_kg: float) -> float:
        """Max power the unit can usefully take: rated, capped by tank room."""
        headroom_kw = co2_headroom_kg * self.config.specific_energy_kwh_per_kg / dt_h
        return min(self.config.rated_kw, max(0.0, headroom_kw))

    def run(self, power_kw: float, dt_h: float) -> float:
        """Run for `dt_h` at `power_kw`; returns CO2 captured (kg).

        The caller is expected to clamp `power_kw` to `intake_limit_kw` first,
        so everything offered here is genuinely captured.
        """
        self.power_kw = min(max(0.0, power_kw), self.config.rated_kw)
        self.online = self.power_kw > 0.0
        return self.power_kw * dt_h / self.config.specific_energy_kwh_per_kg

    def snapshot(self) -> dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "load_frac": self.load_frac,
            "online": self.online,
            "rated_kw": self.config.rated_kw,
        }
