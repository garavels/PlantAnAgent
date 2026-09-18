"""Repeatable situations: a starting condition plus a schedule of faults.

A scenario is data, not code. Loading one resets the plant, moves the clock to
the scenario's start hour, sets inventories, and schedules its faults relative
to that start -- so running the same scenario twice with the same actions gives
the same trace.

    sim.load_scenario("electrolyser_degradation")
    sim.run(240, action)

Adding a scenario means appending one `Scenario` to `SCENARIOS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .faults import FaultSpec
from .state import SECONDS_PER_HOUR


@dataclass(frozen=True)
class InitialState:
    """Inventory overrides applied right after the reset. `None` keeps the design default."""

    battery_soc: float | None = None
    h2_kg: float | None = None
    co2_kg: float | None = None
    ch4_kg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_soc": self.battery_soc,
            "h2_kg": self.h2_kg,
            "co2_kg": self.co2_kg,
            "ch4_kg": self.ch4_kg,
        }


@dataclass(frozen=True)
class Scenario:
    """One reproducible situation the plant can be put into."""

    id: str
    name: str
    description: str
    #: Time of day the scenario starts at, in hours.
    start_hour: float = 6.0
    initial: InitialState = field(default_factory=InitialState)
    #: Faults scheduled relative to the scenario start (`start_s` = offset).
    faults: tuple[FaultSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "start_hour": self.start_hour,
            "initial": self.initial.to_dict(),
            "faults": [spec.to_dict() for spec in self.faults],
        }


def _h(hours: float) -> float:
    return hours * SECONDS_PER_HOUR


SCENARIOS: dict[str, Scenario] = {
    scenario.id: scenario
    for scenario in (
        Scenario(
            id="clear_day",
            name="Clear day",
            description="Nominal operation from sunrise. No faults, no disturbances -- the baseline every other scenario is read against.",
        ),
        Scenario(
            id="electrolyser_degradation",
            name="Electrolyser degradation",
            description="Three hours of normal running, then the stack loses 40% of its yield: same kW in, much less H2 out, and a rising stack temperature.",
            faults=(FaultSpec(type="electrolyser_degradation", severity=0.4, start_s=_h(3.0)),),
        ),
        Scenario(
            id="cloud_cover",
            name="Sudden cloud cover",
            description="Two hours in, generation falls by 60% for three hours. The battery carries the plant, or it does not.",
            faults=(FaultSpec(type="solar_drop", severity=0.6, start_s=_h(2.0), duration_s=_h(3.0)),),
        ),
        Scenario(
            id="dac_failure",
            name="DAC failure",
            description="The capture unit goes offline an hour in and stays down. CO2 inventory drains through the reactor until methanation becomes CO2-constrained.",
            initial=InitialState(co2_kg=400.0),
            faults=(FaultSpec(type="dac_outage", start_s=_h(1.0)),),
        ),
        Scenario(
            id="sensor_drift",
            name="H2 sensor drift",
            description="Two hours in, the hydrogen level transmitter starts over-reporting by 17% of full scale. Physics is untouched: the only thing wrong is what the plant says about itself, and the level no longer agrees with the flows.",
            faults=(
                FaultSpec(
                    type="sensor_drift", severity=0.17, start_s=_h(2.0), channel="h2_level_kg"
                ),
            ),
        ),
        Scenario(
            id="multiple_problems",
            name="Multiple problems",
            description="A degraded stack meets a cloudy afternoon: less H2 per kWh, and fewer kWh to spend.",
            faults=(
                FaultSpec(type="electrolyser_degradation", severity=0.35, start_s=_h(1.0)),
                FaultSpec(type="solar_drop", severity=0.5, start_s=_h(2.0), duration_s=_h(2.0)),
            ),
        ),
    )
}


def catalogue() -> list[dict[str, Any]]:
    """Every scenario, for menus and API discovery."""
    return [scenario.to_dict() for scenario in SCENARIOS.values()]


def get(scenario: "Scenario | str") -> Scenario:
    """Resolve a scenario id (or pass a `Scenario` straight through)."""
    if isinstance(scenario, Scenario):
        return scenario
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return SCENARIOS[scenario]


__all__ = ["InitialState", "SCENARIOS", "Scenario", "catalogue", "get"]
