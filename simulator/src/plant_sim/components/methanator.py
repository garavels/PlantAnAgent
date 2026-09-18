"""Methanator: turns H2 + CO2 + electricity into CH4 (simplified Sabatier)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import MethanatorConfig
from .base import Component


@dataclass(frozen=True)
class MethanatorResult:
    """What one `run()` actually did, in kg. The caller moves the mass."""

    ch4_kg: float
    h2_kg: float
    co2_kg: float


class Methanator(Component):
    """Produces CH4 at a fixed recipe, limited by whichever input runs short.

    Production in one timestep is the minimum of what each input allows:

        power  -> power_kw * dt / specific_energy
        H2     -> h2_available / h2_kg_per_kg_ch4
        CO2    -> co2_available / co2_kg_per_kg_ch4
        tank   -> ch4_headroom (the product has to go somewhere)

    If either gas is unavailable, production stops -- there is no partial
    chemistry. The unit only draws the power its actual production needs, so
    power allocated beyond what the gases allow is handed back, not burned.

    An outage stops the reaction outright: no CH4, and -- just as importantly --
    no H2 or CO2 drawn from the tanks, so the feedstocks pile up while the
    reactor is down.

    Extension point: minimum turndown, thermal management, catalyst
    degradation, product-quality limits.
    """

    name = "methanator"

    def __init__(self, config: MethanatorConfig | None = None) -> None:
        self.config = config or MethanatorConfig()
        self.power_kw = 0.0
        self.online = False

    def reset(self) -> None:
        self.power_kw = 0.0
        self.online = False

    @property
    def load_frac(self) -> float:
        return self.power_kw / self.config.rated_kw

    @property
    def available_kw(self) -> float:
        """Rated power after faults: zero while the reactor is offline."""
        return self.config.rated_kw * max(self.health, 0.0)

    def max_power_kw(self, dt_h: float, h2_available_kg: float, co2_available_kg: float, ch4_headroom_kg: float) -> float:
        """Largest power draw that would still produce: what the gases allow."""
        if self.offline:
            return 0.0
        by_h2_kg = h2_available_kg / self.config.h2_kg_per_kg_ch4
        by_co2_kg = co2_available_kg / self.config.co2_kg_per_kg_ch4
        ch4_possible_kg = min(by_h2_kg, by_co2_kg, max(0.0, ch4_headroom_kg))
        return min(self.available_kw, ch4_possible_kg * self.config.specific_energy_kwh_per_kg / dt_h)

    def run(
        self,
        power_kw: float,
        dt_h: float,
        *,
        h2_available_kg: float,
        co2_available_kg: float,
        ch4_headroom_kg: float,
    ) -> MethanatorResult:
        """Run for `dt_h` at up to `power_kw`; returns the mass balance (kg)."""
        if self.offline:
            self.power_kw = 0.0
            self.online = False
            return MethanatorResult(ch4_kg=0.0, h2_kg=0.0, co2_kg=0.0)
        offered = min(max(0.0, power_kw), self.available_kw)
        by_power_kg = offered * dt_h / self.config.specific_energy_kwh_per_kg
        ch4_kg = min(
            by_power_kg,
            max(0.0, h2_available_kg) / self.config.h2_kg_per_kg_ch4,
            max(0.0, co2_available_kg) / self.config.co2_kg_per_kg_ch4,
            max(0.0, ch4_headroom_kg),
        )
        # draw only the power the actual production needs; the rest spills back
        reconstructed_power_kw = ch4_kg * self.config.specific_energy_kwh_per_kg / dt_h if dt_h > 0 else 0.0
        # The kg -> kW round trip can exceed ``offered`` by one binary64 ULP.
        # Power accepted from the bus is authoritative, so never manufacture
        # energy or a negative downstream spill from that representation noise.
        self.power_kw = min(offered, reconstructed_power_kw)
        self.online = ch4_kg > 0.0
        return MethanatorResult(
            ch4_kg=ch4_kg,
            h2_kg=ch4_kg * self.config.h2_kg_per_kg_ch4,
            co2_kg=ch4_kg * self.config.co2_kg_per_kg_ch4,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "load_frac": self.load_frac,
            "online": self.online,
            "rated_kw": self.config.rated_kw,
            "available_kw": self.available_kw,
            **self.health_snapshot(),
        }
