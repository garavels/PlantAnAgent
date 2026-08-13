"""What one episode is: which scenario, which starting inventory, how long.

A `Scenario` is already the simulator's unit of reproducible situation (start
hour, inventories, fault schedule), so an episode is not a new concept -- it is a
*sampled* scenario. `EpisodeConfig` says which scenarios are in the pool and how
wide the starting conditions may vary; `sample()` draws one with a NumPy
generator, and `PlantEnv.reset(seed=...)` is what owns that generator. Same seed,
same draw, same rollout: nothing here reads a global RNG or the clock.

Variation is deliberately about *operating conditions*, not about faults. The
pool defaults to the two scenarios that are ordinary operation and ordinary
weather (`clear_day`, `cloud_cover`) plus an optional mild solar disturbance,
because the RL controller's job is routine allocation. Diagnosing a degrading
stack stays the supervisory agent's job (V6), and training a policy mainly on
severe faults would teach it to run a broken plant rather than a working one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
from plant_sim import FaultSpec, InitialState, PlantConfig, Scenario
from plant_sim import scenarios as scenario_library
from plant_sim.state import SECONDS_PER_HOUR

#: Ordinary operation and ordinary weather: what routine allocation has to handle.
DEFAULT_SCENARIOS: tuple[str, ...] = ("clear_day", "cloud_cover")


def _span(name: str, value: Any) -> tuple[float, float]:
    """Read a `(low, high)` range from config, keeping it ordered and numeric."""
    try:
        low, high = (float(number) for number in value)
    except (TypeError, ValueError) as reason:
        raise ValueError(f"{name} must be a (low, high) pair") from reason
    return (low, high) if low <= high else (high, low)


@dataclass(frozen=True)
class EpisodeConfig:
    """The episode distribution: horizon, scenario pool, starting conditions."""

    #: Simulated hours per episode. Two days: enough for the battery and the
    #: tanks to matter overnight, short enough to roll out quickly.
    horizon_h: float = 48.0
    #: Scenario ids to draw from, uniformly.
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    #: Starting battery state of charge.
    battery_soc: tuple[float, float] = (0.15, 0.75)
    #: Starting H2 / CO2 / CH4 inventory, as a fraction of tank capacity.
    h2_fill: tuple[float, float] = (0.0, 0.4)
    co2_fill: tuple[float, float] = (0.05, 0.5)
    ch4_fill: tuple[float, float] = (0.0, 0.2)
    #: Chance of adding one mild extra cloud-cover disturbance on top of the scenario.
    disturbance_probability: float = 0.3
    #: Severity and length of that disturbance.
    disturbance_severity: tuple[float, float] = (0.2, 0.6)
    disturbance_duration_h: tuple[float, float] = (0.5, 3.0)

    def steps(self, config: PlantConfig) -> int:
        """Episode length in timesteps for this plant's timestep."""
        return max(1, int(round(self.horizon_h * SECONDS_PER_HOUR / config.dt_s)))

    def validate(self) -> "EpisodeConfig":
        if self.horizon_h <= 0.0:
            raise ValueError("horizon_h must be positive")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        for scenario_id in self.scenarios:
            scenario_library.get(scenario_id)  # raises on unknown ids
        if not 0.0 <= self.disturbance_probability <= 1.0:
            raise ValueError("disturbance_probability must be between 0 and 1")
        for name, (low, high) in (
            ("battery_soc", self.battery_soc),
            ("h2_fill", self.h2_fill),
            ("co2_fill", self.co2_fill),
            ("ch4_fill", self.ch4_fill),
            ("disturbance_severity", self.disturbance_severity),
            ("disturbance_duration_h", self.disturbance_duration_h),
        ):
            if low > high:
                raise ValueError(f"{name} must be a (low, high) pair, got ({low}, {high})")
            if low < 0.0:
                raise ValueError(f"{name} must not be negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "EpisodeConfig":
        if not values:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown episode settings: {', '.join(sorted(unknown))}")
        updates: dict[str, Any] = {}
        for key, value in values.items():
            if key == "scenarios":
                updates[key] = tuple(str(item) for item in value)
            elif key in ("horizon_h", "disturbance_probability"):
                updates[key] = float(value)
            else:
                updates[key] = _span(key, value)
        return replace(cls(), **updates).validate()


@dataclass(frozen=True)
class Episode:
    """One drawn episode: the scenario to load, and the seed that produced it."""

    scenario: Scenario
    seed: int
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": self.scenario.to_dict(), "seed": self.seed, "steps": self.steps}


def sample(
    episode: EpisodeConfig,
    config: PlantConfig,
    rng: np.random.Generator,
    *,
    seed: int,
) -> Episode:
    """Draw one episode: a base scenario with sampled inventories and weather.

    The result is a plain `Scenario`, so it is loaded through
    `PlantSimulator.load_scenario` like any other -- the env adds no second way
    of putting the plant into a starting condition.
    """
    base = scenario_library.get(str(rng.choice(episode.scenarios)))
    initial = InitialState(
        battery_soc=float(rng.uniform(*episode.battery_soc)),
        h2_kg=float(rng.uniform(*episode.h2_fill)) * config.h2_tank.capacity_kg,
        co2_kg=float(rng.uniform(*episode.co2_fill)) * config.co2_tank.capacity_kg,
        ch4_kg=float(rng.uniform(*episode.ch4_fill)) * config.ch4_tank.capacity_kg,
    )
    faults = base.faults
    if rng.random() < episode.disturbance_probability:
        duration_h = float(rng.uniform(*episode.disturbance_duration_h))
        start_h = float(rng.uniform(0.0, max(0.0, episode.horizon_h - duration_h)))
        faults = faults + (
            FaultSpec(
                type="solar_drop",
                severity=float(rng.uniform(*episode.disturbance_severity)),
                start_s=start_h * SECONDS_PER_HOUR,
                duration_s=duration_h * SECONDS_PER_HOUR,
            ),
        )
    scenario = replace(
        base,
        name=f"{base.name} (episode seed {seed})",
        initial=initial,
        faults=faults,
    )
    return Episode(scenario=scenario, seed=seed, steps=episode.steps(config))


__all__ = ["DEFAULT_SCENARIOS", "Episode", "EpisodeConfig", "sample"]
