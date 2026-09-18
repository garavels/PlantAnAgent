"""Boundary telemetry, delivery ledger, and forecast -> V2 observation vector."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from plant_sim import (
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    ElectrolyserCommitmentProfile,
    ElectrolyserMode,
    PlantConfig,
)

from ...controllers.base import Constraints
from ..observation import ObservationEncoder as V1ObservationEncoder
from . import spaces
from .boundary import COMMITMENT_MODE_NAMES, ControlStateV2
from .control import ControllerObservationV2, DeliveryObservation
from .weather import PublicForecastIssueV2


class ObservationEncoderV2:
    """Compose V2 boundary telemetry, delivery, and issued weather forecasts."""

    def __init__(
        self,
        config: PlantConfig,
        commitment: ElectrolyserCommitmentProfile | None = (
            DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
        ),
    ) -> None:
        if commitment is not None and not isinstance(
            commitment,
            ElectrolyserCommitmentProfile,
        ):
            raise TypeError(
                "commitment must be an ElectrolyserCommitmentProfile or None"
            )
        self.config = config
        self.commitment = commitment
        self._base = V1ObservationEncoder(config)
        self._low = spaces.observation_low()
        self._high = spaces.observation_high()

    def reset(self) -> None:
        self._base.reset()

    def encode(
        self,
        state: ControlStateV2,
        *,
        constraints: Constraints,
        progress: float,
        delivery: DeliveryObservation,
        forecast: PublicForecastIssueV2,
    ) -> np.ndarray:
        telemetry = state.encoder_telemetry()
        base = self._base.encode(
            telemetry,
            constraints=constraints,
            progress=progress,
        )
        boundary = state.boundary
        if delivery.steps_to_next is None:
            time_to_next = 1.0
        else:
            time_to_next = delivery.steps_to_next / max(1, delivery.delivery_period_steps)
        next_amount = delivery.next_requested_kg
        next_amount_fraction = next_amount / max(boundary.ch4_capacity_kg, 1e-12)
        coverage = boundary.ch4_level_kg / next_amount if next_amount > 0.0 else 0.0
        fill_rate = (
            delivery.cumulative_delivered_kg / delivery.cumulative_requested_kg
            if delivery.cumulative_requested_kg > 0.0
            else 1.0
        )
        forecast_values = tuple(
            value
            for window in forecast.windows
            for value in (
                window.mean_peak_power_fraction,
                window.p10_peak_power_fraction,
            )
        )
        commitment_values = self._commitment_values(state)
        vector = np.concatenate(
            (
                base,
                np.asarray(
                    (time_to_next, next_amount_fraction, coverage, fill_rate),
                    dtype=np.float32,
                ),
                np.asarray(forecast_values, dtype=np.float32),
                np.asarray(commitment_values, dtype=np.float32),
            )
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(vector, self._low, self._high).astype(np.float32)

    def encode_controller_observation(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
    ) -> np.ndarray:
        """Encode the same causal object consumed by non-learning controllers."""
        if observation.forecast is None:
            raise ValueError("the V2 RL observation contract requires an issued forecast")
        return self.encode(
            observation.state,
            constraints=constraints,
            progress=observation.progress,
            delivery=observation.delivery,
            forecast=observation.forecast,
        )

    def _commitment_values(self, state: ControlStateV2) -> tuple[float, ...]:
        snapshot = state.electrolyser_commitment
        if snapshot is not None and not isinstance(snapshot.mode, ElectrolyserMode):
            raise TypeError("commitment snapshot mode must be ElectrolyserMode")
        active_mode = "legacy" if snapshot is None else snapshot.mode.value
        if active_mode not in COMMITMENT_MODE_NAMES:
            raise ValueError(f"unknown commitment mode: {active_mode!r}")
        one_hot = tuple(float(name == active_mode) for name in COMMITMENT_MODE_NAMES)
        if snapshot is None:
            return one_hot + (0.0,) * 8

        profile = self.commitment
        if profile is None:
            raise ValueError("commitment snapshot is present but encoder profile is disabled")
        if (
            snapshot.profile_id != profile.profile_id
            or snapshot.profile_sha256 != profile.profile_sha256
        ):
            raise ValueError(
                "commitment snapshot/profile identity mismatch: "
                f"snapshot={snapshot.profile_sha256}, profile={profile.profile_sha256}"
            )

        def fraction(value: float, reference: float) -> float:
            return value / reference if reference > 0.0 else 0.0

        raw_values = (
            snapshot.previous_power_kw,
            snapshot.ramp_up_headroom_kw,
            snapshot.ramp_down_headroom_kw,
            snapshot.startup_remaining_s,
            snapshot.cooldown_remaining_s,
            snapshot.min_up_remaining_s,
            snapshot.min_down_remaining_s,
            snapshot.shortfall_elapsed_s,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0.0
            for value in raw_values
        ):
            raise ValueError("commitment snapshot state must be finite and non-negative")

        rated_kw = max(self.config.electrolyser.rated_kw, 1e-12)
        normalized = (
            snapshot.previous_power_kw / rated_kw,
            snapshot.ramp_up_headroom_kw / rated_kw,
            snapshot.ramp_down_headroom_kw / rated_kw,
            fraction(
                snapshot.startup_remaining_s,
                max(profile.cold_start_s, profile.hot_start_s),
            ),
            fraction(snapshot.cooldown_remaining_s, profile.cooldown_s),
            fraction(snapshot.min_up_remaining_s, profile.min_up_s),
            fraction(snapshot.min_down_remaining_s, profile.min_down_s),
            fraction(snapshot.shortfall_elapsed_s, profile.shortfall_grace_s),
        )
        if any(value > 1.0 + 1e-9 for value in normalized):
            raise ValueError("commitment snapshot state exceeds its declared bound")
        return one_hot + normalized


__all__ = ["DeliveryObservation", "ObservationEncoderV2"]
