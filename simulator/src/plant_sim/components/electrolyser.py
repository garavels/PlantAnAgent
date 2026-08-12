"""Electrolyser: turns electricity into hydrogen, but only within its load window."""

from __future__ import annotations

from typing import Any

from ..config import ElectrolyserConfig
from .base import Component


class Electrolyser(Component):
    """Consumes power between `min_load` and `rated_kw`, produces H2 linearly.

    The minimum load is what makes power allocation interesting: a trickle of
    solar is worth nothing to the stack, so an agent has to decide between
    buffering it in the battery or discharging the battery to lift the stack
    above its floor.

    Degradation is the V4 seam: a degraded stack keeps drawing the same power
    but converts less of it, so `specific_energy_kwh_per_kg` rises with the
    fault severity and the stack runs hotter -- the diagnostic an agent can
    correlate with the missing hydrogen.

    Extension point: efficiency curve vs load, warm/cold start-up delays,
    standby power draw, thermal limits.
    """

    name = "electrolyser"

    def __init__(self, config: ElectrolyserConfig | None = None) -> None:
        self.config = config or ElectrolyserConfig()
        self.power_kw = 0.0
        self.online = False

    def reset(self) -> None:
        self.power_kw = 0.0
        self.online = False

    @property
    def min_load_kw(self) -> float:
        return self.config.min_load_frac * self.config.rated_kw

    @property
    def load_frac(self) -> float:
        return self.power_kw / self.config.rated_kw

    @property
    def specific_energy_kwh_per_kg(self) -> float:
        """Electricity per kg of H2 *now*: nominal, worsened by degradation."""
        return self.config.specific_energy_kwh_per_kg / max(self.health, 1e-6)

    @property
    def efficiency(self) -> float:
        """Yield relative to the nameplate stack, 0..1."""
        return self.health

    @property
    def temperature_c(self) -> float:
        """Stack temperature: ambient when idle, rising with load and degradation."""
        if not self.online:
            return self.config.ambient_temperature_c
        return (
            self.config.nominal_temperature_c
            + self.config.fault_temperature_rise_c * self.effect.degradation
        )

    def intake_limit_kw(self, dt_h: float, h2_headroom_kg: float) -> float:
        """Max power the stack can usefully take: rated power, capped by tank room."""
        if self.offline:
            return 0.0
        headroom_kw = h2_headroom_kg * self.specific_energy_kwh_per_kg / dt_h
        return min(self.config.rated_kw, max(0.0, headroom_kw))

    def run(self, power_kw: float, dt_h: float) -> float:
        """Run for `dt_h` at `power_kw`; returns hydrogen produced (kg).

        Below the minimum load the stack simply stays off and produces nothing --
        the caller is responsible for re-routing the power it did not use.
        """
        if self.offline or power_kw <= 0.0 or power_kw < self.min_load_kw:
            self.power_kw = 0.0
            self.online = False
            return 0.0
        self.power_kw = min(power_kw, self.config.rated_kw)
        self.online = True
        return self.power_kw * dt_h / self.specific_energy_kwh_per_kg

    def snapshot(self) -> dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "load_frac": self.load_frac,
            "online": self.online,
            "rated_kw": self.config.rated_kw,
            "efficiency": self.efficiency,
            "temperature_c": self.temperature_c,
            **self.health_snapshot(),
        }
