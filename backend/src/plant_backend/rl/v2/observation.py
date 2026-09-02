"""Boundary telemetry, delivery ledger, and forecast -> V2 observation vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plant_sim import PlantConfig

from ...controllers.base import Constraints
from ..observation import ObservationEncoder as V1ObservationEncoder
from . import spaces
from .boundary import ControlStateV2
from .weather import ForecastIssueV2


@dataclass(frozen=True)
class DeliveryObservation:
    """Only contractual information available at the current decision boundary."""

    steps_to_next: int | None
    delivery_period_steps: int
    next_requested_kg: float
    cumulative_requested_kg: float
    cumulative_delivered_kg: float


class ObservationEncoderV2:
    """Compose V2 boundary telemetry, delivery, and issued weather forecasts."""

    def __init__(self, config: PlantConfig) -> None:
        self.config = config
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
        forecast: ForecastIssueV2,
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
        vector = np.concatenate(
            (
                base,
                np.asarray(
                    (time_to_next, next_amount_fraction, coverage, fill_rate),
                    dtype=np.float32,
                ),
                np.asarray(forecast_values, dtype=np.float32),
            )
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(vector, self._low, self._high).astype(np.float32)


__all__ = ["DeliveryObservation", "ObservationEncoderV2"]
