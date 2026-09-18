"""What an episode is judged on, beyond the reward.

The reward is what a policy optimises; these are the numbers a *person* compares
two controllers with, and they are deliberately plant quantities rather than
learning quantities: kilograms of methane, kilowatt-hours wasted, how long the
plant could not produce, how often the guard had to step in.

Accumulated per step by `PlantEnv`, so the heuristic controller and a policy are
measured by exactly the same code on exactly the same rollout path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plant_sim import PlantConfig, PlantState
from plant_sim.state import CRITICAL, OFFLINE

#: Feedstock below this counts as a shortage when the reactor is asked to run (kg).
SHORTAGE_KG = 0.05


@dataclass
class EpisodeMetrics:
    """Running totals for one episode. `record` is called once per timestep."""

    steps: int = 0
    hours: float = 0.0
    reward_total: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)

    ch4_kg: float = 0.0
    h2_kg: float = 0.0
    co2_kg: float = 0.0
    #: Solar energy generated, and how much of it was actually used.
    solar_kwh: float = 0.0
    curtailed_kwh: float = 0.0
    battery_discharged_kwh: float = 0.0
    min_battery_soc: float = 1.0
    #: Steps where the plant made no methane at all.
    idle_steps: int = 0
    #: Steps where methanation was commanded but a feedstock was missing.
    h2_shortage_steps: int = 0
    co2_shortage_steps: int = 0
    #: Steps in a `critical` or `offline` plant condition.
    shutdown_steps: int = 0
    #: Guard adjustments, and steps where at least one was needed.
    corrections: int = 0
    corrected_steps: int = 0
    #: Steps where the policy's raw output was outside its own action range.
    out_of_range_steps: int = 0
    #: On/off transitions of the electrolyser, DAC and methanator.
    cycles: int = 0

    def record(
        self,
        before: PlantState,
        after: PlantState,
        *,
        config: PlantConfig,
        reward: float,
        components: dict[str, float],
        adjustments: tuple[str, ...],
        out_of_range: bool,
        commanded_methanation: bool,
    ) -> None:
        dt_h = config.dt_h
        self.steps += 1
        self.hours += dt_h
        self.reward_total += reward
        for name, value in components.items():
            self.reward_components[name] = self.reward_components.get(name, 0.0) + value

        ch4_kg = max(0.0, after.ch4_produced_total_kg - before.ch4_produced_total_kg)
        self.ch4_kg += ch4_kg
        self.h2_kg += max(0.0, after.h2_produced_total_kg - before.h2_produced_total_kg)
        self.co2_kg += max(0.0, after.co2_captured_total_kg - before.co2_captured_total_kg)
        self.solar_kwh += max(0.0, after.energy_produced_kwh - before.energy_produced_kwh)
        self.curtailed_kwh += max(0.0, after.energy_curtailed_kwh - before.energy_curtailed_kwh)
        self.battery_discharged_kwh += max(0.0, -after.battery_power_kw) * dt_h
        self.min_battery_soc = min(self.min_battery_soc, after.battery_soc)

        if ch4_kg <= 0.0:
            self.idle_steps += 1
        if commanded_methanation:
            if before.h2_level_kg <= SHORTAGE_KG:
                self.h2_shortage_steps += 1
            if before.co2_level_kg <= SHORTAGE_KG:
                self.co2_shortage_steps += 1
        if after.plant_status in (CRITICAL, OFFLINE):
            self.shutdown_steps += 1
        self.corrections += len(adjustments)
        if adjustments:
            self.corrected_steps += 1
        if out_of_range:
            self.out_of_range_steps += 1
        self.cycles += sum(
            1
            for pair in (
                (before.electrolyser_online, after.electrolyser_online),
                (before.dac_online, after.dac_online),
                (before.methanator_online, after.methanator_online),
            )
            if pair[0] != pair[1]
        )

    # --- derived ----------------------------------------------------------
    @property
    def solar_utilisation(self) -> float:
        """Fraction of generated solar energy the plant actually used, 0..1."""
        if self.solar_kwh <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.curtailed_kwh / self.solar_kwh)

    @property
    def ch4_per_solar_kwh(self) -> float:
        """kg of methane per kWh generated: the efficiency headline."""
        return self.ch4_kg / self.solar_kwh if self.solar_kwh > 0.0 else 0.0

    @property
    def idle_fraction(self) -> float:
        return self.idle_steps / self.steps if self.steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "hours": self.hours,
            "reward_total": self.reward_total,
            "reward_components": dict(self.reward_components),
            "ch4_kg": self.ch4_kg,
            "h2_kg": self.h2_kg,
            "co2_kg": self.co2_kg,
            "solar_kwh": self.solar_kwh,
            "curtailed_kwh": self.curtailed_kwh,
            "solar_utilisation": self.solar_utilisation,
            "ch4_per_solar_kwh": self.ch4_per_solar_kwh,
            "battery_discharged_kwh": self.battery_discharged_kwh,
            "min_battery_soc": self.min_battery_soc,
            "idle_steps": self.idle_steps,
            "idle_fraction": self.idle_fraction,
            "h2_shortage_steps": self.h2_shortage_steps,
            "co2_shortage_steps": self.co2_shortage_steps,
            "shutdown_steps": self.shutdown_steps,
            "corrections": self.corrections,
            "corrected_steps": self.corrected_steps,
            "out_of_range_steps": self.out_of_range_steps,
            "cycles": self.cycles,
        }


def aggregate(runs: list[EpisodeMetrics]) -> dict[str, Any]:
    """Mean of every numeric metric across runs, plus the run count.

    Averaged, not summed: a comparison over five seeds should read the same as
    one over ten, and a metric that is already a fraction cannot be summed.
    """
    if not runs:
        return {"runs": 0}
    dicts = [run.to_dict() for run in runs]
    numeric = [key for key, value in dicts[0].items() if isinstance(value, (int, float))]
    summary: dict[str, Any] = {"runs": len(runs)}
    for key in numeric:
        summary[key] = sum(float(entry[key]) for entry in dicts) / len(dicts)
    components: dict[str, float] = {}
    for entry in dicts:
        for name, value in entry["reward_components"].items():
            components[name] = components.get(name, 0.0) + float(value) / len(dicts)
    summary["reward_components"] = components
    return summary


__all__ = ["EpisodeMetrics", "aggregate"]
