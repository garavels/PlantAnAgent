"""Static plant design parameters.

Everything the plant *is* (sizing, limits, efficiencies) lives here.
Everything the plant *does* lives in `state.py`.

Units used throughout the simulator:
    power   kW
    energy  kWh
    mass    kg
    time    s (state), h (internal integration)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SolarConfig:
    """A deterministic PV array. No weather, no clouds -- yet."""

    peak_kw: float = 5_000.0
    sunrise_h: float = 6.0
    sunset_h: float = 19.5


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 8_000.0
    max_charge_kw: float = 2_500.0
    max_discharge_kw: float = 2_500.0
    charge_efficiency: float = 0.96
    discharge_efficiency: float = 0.96
    min_soc: float = 0.05
    initial_soc: float = 0.35


@dataclass(frozen=True)
class ElectrolyserConfig:
    rated_kw: float = 3_000.0
    #: Below this fraction of rated power the stack cannot run and shuts down.
    min_load_frac: float = 0.15
    #: Electricity needed per kg of H2 (a flat number stands in for a real curve).
    specific_energy_kwh_per_kg: float = 52.0


@dataclass(frozen=True)
class StorageConfig:
    """Generic buffer tank. Reused later for CH4, CO2, water, ..."""

    capacity_kg: float = 2_000.0
    initial_kg: float = 120.0


@dataclass(frozen=True)
class PlantConfig:
    """Full plant design. Pass a modified copy to `PlantSimulator` to re-size."""

    #: Length of one simulation timestep in seconds.
    dt_s: float = 60.0
    solar: SolarConfig = field(default_factory=SolarConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    electrolyser: ElectrolyserConfig = field(default_factory=ElectrolyserConfig)
    h2_tank: StorageConfig = field(default_factory=StorageConfig)

    @property
    def dt_h(self) -> float:
        return self.dt_s / 3600.0

    @property
    def steps_per_day(self) -> int:
        return int(round(24.0 * 3600.0 / self.dt_s))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
