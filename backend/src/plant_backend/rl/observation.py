"""Telemetry -> the vector a policy reads.

The encoder reads exactly what the heuristic controller reads -- `PlantState`
telemetry, which a sensor fault is allowed to distort -- plus the two things a
controller is entitled to know about its own situation: the envelope it is
operating inside (`Constraints`) and how far through the episode it is.

It is stateful for one reason: `solar_trend` is the change in generation since
the previous observation, and a policy that cannot tell a clearing sky from a
darkening one has to guess. `reset()` drops that memory, so the first
observation of an episode reports a zero trend rather than a step change from
the previous episode.

Every entry is finite and clipped into the bounds declared in `spaces.py`;
nothing else in the stack has to defend against NaN reaching the network.
"""

from __future__ import annotations

import math

import numpy as np
from plant_sim import PlantConfig, PlantState

from ..controllers.base import Constraints
from .spaces import observation_high, observation_low


def _health(observation: PlantState, unit: str) -> float:
    health = observation.health.get(unit)
    if health is None:
        return 1.0
    return 0.0 if health.offline else max(0.0, 1.0 - health.degradation)


def _ratio(value: float, reference: float) -> float:
    return value / reference if reference > 0.0 else 0.0


class ObservationEncoder:
    """Turns telemetry into the normalised observation vector, in field order."""

    def __init__(self, config: PlantConfig) -> None:
        self.config = config
        self._low = observation_low()
        self._high = observation_high()
        self._previous_solar_kw: float | None = None

    def reset(self) -> None:
        """Forget the previous step, so `solar_trend` starts an episode at zero."""
        self._previous_solar_kw = None

    def encode(
        self,
        observation: PlantState,
        *,
        constraints: Constraints,
        progress: float = 0.0,
    ) -> np.ndarray:
        config = self.config
        peak_kw = config.solar.peak_kw
        previous_kw = (
            observation.solar_available_kw
            if self._previous_solar_kw is None
            else self._previous_solar_kw
        )
        self._previous_solar_kw = observation.solar_available_kw

        h2_max_kg_h = _ratio(
            config.electrolyser.rated_kw, config.electrolyser.specific_energy_kwh_per_kg
        )
        co2_max_kg_h = _ratio(config.dac.rated_kw, config.dac.specific_energy_kwh_per_kg)
        ch4_max_kg_h = _ratio(
            config.methanator.rated_kw, config.methanator.specific_energy_kwh_per_kg
        )
        day_angle = 2.0 * math.pi * observation.time_of_day_h / 24.0

        values = (
            math.sin(day_angle),
            math.cos(day_angle),
            _ratio(observation.solar_available_kw, peak_kw),
            _ratio(observation.solar_potential_kw, peak_kw),
            observation.solar_output_factor,
            _ratio(observation.solar_available_kw - previous_kw, peak_kw),
            observation.battery_soc,
            _ratio(observation.battery_power_kw, config.battery.max_discharge_kw),
            observation.battery_soc - constraints.reserve_soc(config),
            observation.h2_fill,
            observation.co2_fill,
            observation.ch4_fill,
            observation.electrolyser_load_frac,
            1.0 if observation.electrolyser_online else 0.0,
            _health(observation, "electrolyser"),
            observation.dac_load_frac,
            1.0 if observation.dac_online else 0.0,
            _health(observation, "dac"),
            observation.methanator_load_frac,
            1.0 if observation.methanator_online else 0.0,
            _health(observation, "methanator"),
            _ratio(observation.h2_production_kg_h, h2_max_kg_h),
            _ratio(observation.co2_capture_kg_h, co2_max_kg_h),
            _ratio(observation.ch4_production_kg_h, ch4_max_kg_h),
            _ratio(constraints.electrolyser_limit_kw(config), config.electrolyser.rated_kw),
            _ratio(constraints.dac_limit_kw(config), config.dac.rated_kw),
            _ratio(constraints.methanator_limit_kw(config), config.methanator.rated_kw),
            _ratio(constraints.support_limit_kw(config), config.battery.max_discharge_kw),
            constraints.reserve_soc(config),
            progress,
        )
        vector = np.asarray(values, dtype=np.float32)
        # a drifting sensor can report a level above full scale, and a policy
        # must never be handed a NaN: clip into the declared space either way
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(vector, self._low, self._high).astype(np.float32)


__all__ = ["ObservationEncoder"]
