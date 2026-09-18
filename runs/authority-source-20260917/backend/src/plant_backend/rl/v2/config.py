"""Seeded episode and delivery-contract construction for Environment V2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from plant_sim import (
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    DeliveryCommitment,
    ElectrolyserCommitmentProfile,
    InitialState,
    PlantConfig,
    Scenario,
    SolarArray,
)
from plant_sim.state import SECONDS_PER_HOUR

from .weather import WeatherConfigV2, WeatherEpisodeV2, generate_weather_episode

STREAM_NAMES = ("inventories", "weather", "contracts")
CLEAR_SKY_ENVELOPE_ID = "plant-sim-half-sine-v1"


def _span(name: str, span: tuple[float, float], *, unit_interval: bool = False) -> None:
    low, high = span
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name} bounds must be finite")
    if low < 0.0 or low > high:
        raise ValueError(f"{name} must be an ordered non-negative pair")
    if unit_interval and high > 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")


def _ticks(seconds: float, dt_s: float, *, name: str) -> int:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    ticks = int(round(seconds / dt_s))
    if ticks <= 0 or not math.isclose(seconds, ticks * dt_s, abs_tol=1e-9):
        raise ValueError(f"{name} must align to the {dt_s:g} s physics timestep")
    return ticks


def _aligned_start(seconds: float, dt_s: float) -> None:
    tick = round(seconds / dt_s)
    if not math.isclose(seconds, tick * dt_s, abs_tol=1e-9):
        raise ValueError(f"start_hour must align to the {dt_s:g} s physics timestep")


@dataclass(frozen=True)
class EpisodeConfigV2:
    """Commitment-alpha task distribution and explicit controller cadence."""

    horizon_h: float = 168.0
    decision_interval_s: float = 900.0
    start_hour: float = 6.0
    delivery_period_h: float = 6.0
    #: Provisional benchmark load. It is a hypothesis to calibrate, not plant data.
    delivery_rate_kg_h: float = 40.0
    battery_soc: tuple[float, float] = (0.25, 0.65)
    h2_fill: tuple[float, float] = (0.05, 0.30)
    co2_fill: tuple[float, float] = (0.10, 0.40)
    #: Starting methane expressed in hours of contracted average delivery.
    ch4_coverage_h: tuple[float, float] = (3.0, 9.0)
    weather: WeatherConfigV2 = field(default_factory=WeatherConfigV2)
    #: Stateful equipment physics are part of the task contract, not PlantConfig.
    electrolyser_commitment: ElectrolyserCommitmentProfile | None = (
        DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )

    def validate(self, config: PlantConfig) -> "EpisodeConfigV2":
        if not math.isfinite(config.dt_s) or config.dt_s <= 0.0:
            raise ValueError("plant physics timestep must be positive and finite")
        if not math.isfinite(config.solar.peak_kw) or config.solar.peak_kw <= 0.0:
            raise ValueError("solar peak power must be positive and finite")
        if not (
            math.isfinite(config.solar.sunrise_h)
            and math.isfinite(config.solar.sunset_h)
            and 0.0 <= config.solar.sunrise_h < config.solar.sunset_h <= 24.0
        ):
            raise ValueError("solar daylight hours must satisfy 0 <= sunrise < sunset <= 24")
        if not math.isfinite(self.start_hour) or not 0.0 <= self.start_hour < 24.0:
            raise ValueError("start_hour must lie within [0, 24)")
        if not math.isfinite(self.delivery_rate_kg_h) or self.delivery_rate_kg_h <= 0.0:
            raise ValueError("delivery_rate_kg_h must be positive and finite")
        if self.electrolyser_commitment is not None and not isinstance(
            self.electrolyser_commitment,
            ElectrolyserCommitmentProfile,
        ):
            raise TypeError(
                "electrolyser_commitment must be an "
                "ElectrolyserCommitmentProfile or None"
            )
        _span("battery_soc", self.battery_soc, unit_interval=True)
        _span("h2_fill", self.h2_fill, unit_interval=True)
        _span("co2_fill", self.co2_fill, unit_interval=True)
        _span("ch4_coverage_h", self.ch4_coverage_h)

        physics_steps = self.physics_steps(config)
        decision_stride = self.decision_stride(config)
        delivery_stride = self.delivery_stride(config)
        _aligned_start(self.start_hour * SECONDS_PER_HOUR, config.dt_s)
        if physics_steps % decision_stride:
            raise ValueError("horizon_h must contain a whole number of controller decisions")
        if delivery_stride % decision_stride:
            raise ValueError(
                "delivery_period_h must contain a whole number of controller decisions"
            )
        if delivery_stride > physics_steps:
            raise ValueError("horizon_h must contain at least one delivery commitment")
        if physics_steps % delivery_stride:
            raise ValueError(
                "horizon_h must contain a whole number of delivery periods"
            )
        max_initial_ch4 = self.ch4_coverage_h[1] * self.delivery_rate_kg_h
        if max_initial_ch4 > config.ch4_tank.capacity_kg + 1e-9:
            raise ValueError("ch4_coverage_h exceeds the methane tank capacity")
        delivery_amount = self.delivery_amount_kg
        if not math.isfinite(delivery_amount):
            raise ValueError("delivery amount must be finite")
        if delivery_amount > config.ch4_tank.capacity_kg + 1e-9:
            raise ValueError(
                "one delivery commitment exceeds the methane tank capacity"
            )
        self.weather.validate()
        if self.electrolyser_commitment is not None:
            if not math.isclose(config.dt_s, 60.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "commitment alpha requires a 60 s physics timestep so ramp "
                    "headroom and transition semantics remain exact"
                )
            self.electrolyser_commitment.validate_timestep(config.dt_s)
        for start_h, end_h in self.weather.forecast_windows_h:
            if start_h > 0.0:
                _ticks(
                    start_h * SECONDS_PER_HOUR,
                    config.dt_s,
                    name=f"forecast window start {start_h:g} h",
                )
            _ticks(
                end_h * SECONDS_PER_HOUR,
                config.dt_s,
                name=f"forecast window end {end_h:g} h",
            )
        return self

    def physics_steps(self, config: PlantConfig) -> int:
        return _ticks(self.horizon_h * SECONDS_PER_HOUR, config.dt_s, name="horizon_h")

    def decision_stride(self, config: PlantConfig) -> int:
        return _ticks(self.decision_interval_s, config.dt_s, name="decision_interval_s")

    def delivery_stride(self, config: PlantConfig) -> int:
        return _ticks(
            self.delivery_period_h * SECONDS_PER_HOUR,
            config.dt_s,
            name="delivery_period_h",
        )

    def decision_steps(self, config: PlantConfig) -> int:
        return self.physics_steps(config) // self.decision_stride(config)

    @property
    def delivery_amount_kg(self) -> float:
        return self.delivery_rate_kg_h * self.delivery_period_h

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weather"] = self.weather.to_dict()
        payload["electrolyser_commitment"] = (
            None
            if self.electrolyser_commitment is None
            else self.electrolyser_commitment.to_dict()
        )
        return payload


@dataclass(frozen=True)
class EpisodeV2:
    """Private sampled episode plus the data needed for exact replay."""

    scenario: Scenario
    commitments: tuple[DeliveryCommitment, ...]
    seed: int
    stream_spawn_keys: dict[str, tuple[int, ...]]
    physics_steps: int
    decision_stride: int
    delivery_stride: int
    weather: WeatherEpisodeV2

    def exogenous_manifest(self) -> dict[str, Any]:
        """Controller-independent sampled conditions for paired ablations."""
        return {
            "seed": self.seed,
            "random_streams": {
                name: {"spawn_key": list(key)}
                for name, key in self.stream_spawn_keys.items()
            },
            "start_hour": self.scenario.start_hour,
            "initial": self.scenario.initial.to_dict(),
            "commitments": [item.to_dict() for item in self.commitments],
            "physics_steps": self.physics_steps,
            "decision_stride": self.decision_stride,
            "delivery_stride": self.delivery_stride,
            "weather_episode_sha256": self.weather.episode_sha256,
        }

    @property
    def exogenous_episode_sha256(self) -> str:
        canonical = json.dumps(
            self.exogenous_manifest(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def audit_manifest(self) -> dict[str, Any]:
        return {
            **self.exogenous_manifest(),
            "scenario": self.scenario.to_dict(),
            "weather": self.weather.private_audit_manifest(),
            "exogenous_episode_sha256": self.exogenous_episode_sha256,
        }


def sample_episode(
    episode: EpisodeConfigV2,
    config: PlantConfig,
    *,
    seed: int,
) -> EpisodeV2:
    """Sample independent inventory/weather/contract streams from one seed.

    Inventory, weather truth, and forecast uncertainty use independent named
    child streams so changing one generator cannot silently perturb the others.
    """
    episode.validate(config)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    child_sequences = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    streams = dict(zip(STREAM_NAMES, child_sequences, strict=True))
    inventory_rng = np.random.default_rng(streams["inventories"])

    initial = InitialState(
        battery_soc=float(inventory_rng.uniform(*episode.battery_soc)),
        h2_kg=float(inventory_rng.uniform(*episode.h2_fill)) * config.h2_tank.capacity_kg,
        co2_kg=float(inventory_rng.uniform(*episode.co2_fill)) * config.co2_tank.capacity_kg,
        ch4_kg=(
            float(inventory_rng.uniform(*episode.ch4_coverage_h))
            * episode.delivery_rate_kg_h
        ),
    )
    physics_steps = episode.physics_steps(config)
    decision_stride = episode.decision_stride(config)
    scenario = Scenario(
        id="v2_commitment_alpha",
        name="Environment V2 commitment alpha",
        description=(
            "Recurring methane commitments with seeded solar-resource truth and "
            "causal probabilistic forecasts plus optional stateful electrolyser "
            "commitment dynamics."
        ),
        start_hour=episode.start_hour,
        initial=initial,
    )

    # The simulator's legacy flow convention samples clear-sky power at the
    # start of each physical interval. Forecast windows use the same interval
    # envelope, extended through 24 h after the final decision issue.
    final_issue_step = physics_steps - decision_stride
    max_forecast_h = max(end_h for _, end_h in episode.weather.forecast_windows_h)
    forecast_horizon_steps = _ticks(
        max_forecast_h * SECONDS_PER_HOUR,
        config.dt_s,
        name="maximum forecast horizon",
    )
    required_envelope_intervals = final_issue_step + forecast_horizon_steps
    solar = SolarArray(config.solar)
    start_s = episode.start_hour * SECONDS_PER_HOUR
    clear_sky_peak_fractions = tuple(
        solar.clear_sky_kw(start_s + interval * config.dt_s) / config.solar.peak_kw
        for interval in range(required_envelope_intervals)
    )
    truth_seed, forecast_seed = streams["weather"].spawn(2)
    weather = generate_weather_episode(
        episode.weather,
        physics_steps=physics_steps,
        dt_s=config.dt_s,
        decision_stride=decision_stride,
        truth_seed=truth_seed,
        forecast_seed=forecast_seed,
        clear_sky_peak_fractions=clear_sky_peak_fractions,
        clear_sky_envelope_id=CLEAR_SKY_ENVELOPE_ID,
    )
    delivery_stride = episode.delivery_stride(config)
    commitments = tuple(
        DeliveryCommitment(
            id=f"contract-{index:04d}",
            due_step=due_step,
            requested_kg=episode.delivery_amount_kg,
        )
        for index, due_step in enumerate(
            range(delivery_stride, physics_steps + 1, delivery_stride), start=1
        )
    )
    return EpisodeV2(
        scenario=scenario,
        commitments=commitments,
        seed=seed,
        stream_spawn_keys={name: tuple(sequence.spawn_key) for name, sequence in streams.items()},
        physics_steps=physics_steps,
        decision_stride=decision_stride,
        delivery_stride=delivery_stride,
        weather=weather,
    )


__all__ = [
    "CLEAR_SKY_ENVELOPE_ID",
    "EpisodeConfigV2",
    "EpisodeV2",
    "STREAM_NAMES",
    "sample_episode",
]
