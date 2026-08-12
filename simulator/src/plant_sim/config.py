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
    #: Stack temperature when idle, when healthy under load, and fully degraded.
    ambient_temperature_c: float = 22.0
    nominal_temperature_c: float = 62.0
    fault_temperature_rise_c: float = 45.0


@dataclass(frozen=True)
class DACConfig:
    """Direct-air capture: power in, captured CO2 out.

    A flat specific energy stands in for the real sorbent cycle. No minimum
    load -- the unit modulates freely between off and rated power.
    """

    rated_kw: float = 1_200.0
    #: Electricity needed per kg of captured CO2.
    specific_energy_kwh_per_kg: float = 2.0


@dataclass(frozen=True)
class MethanatorConfig:
    """Sabatier reactor: CO2 + 4 H2 -> CH4 + 2 H2O, plus some electrical power.

    The stoichiometry is fixed and mass-based (per kg of CH4 produced):
    real molar masses give 0.503 kg H2 and 2.744 kg CO2 per kg CH4; we round
    to 0.5 and 2.75 and keep them here so the balance is easy to audit.
    """

    rated_kw: float = 180.0
    #: Electrical energy needed per kg of CH4 (compression, controls, ...).
    specific_energy_kwh_per_kg: float = 1.2
    h2_kg_per_kg_ch4: float = 0.5
    co2_kg_per_kg_ch4: float = 2.75


@dataclass(frozen=True)
class StorageConfig:
    """Generic buffer tank. Reused for H2, CO2 and CH4."""

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
    dac: DACConfig = field(default_factory=DACConfig)
    co2_tank: StorageConfig = field(
        default_factory=lambda: StorageConfig(capacity_kg=6_000.0, initial_kg=800.0)
    )
    methanator: MethanatorConfig = field(default_factory=MethanatorConfig)
    ch4_tank: StorageConfig = field(
        default_factory=lambda: StorageConfig(capacity_kg=3_000.0, initial_kg=0.0)
    )

    @property
    def dt_h(self) -> float:
        return self.dt_s / 3600.0

    @property
    def steps_per_day(self) -> int:
        return int(round(24.0 * 3600.0 / self.dt_s))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
