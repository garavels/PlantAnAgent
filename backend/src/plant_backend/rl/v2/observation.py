"""Boundary telemetry and delivery ledger -> Environment V2 observation vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plant_sim import PlantConfig, PlantState

from ...controllers.base import Constraints
from ..observation import ObservationEncoder as V1ObservationEncoder
from . import spaces


@dataclass(frozen=True)
class DeliveryObservation:
    """Only contractual information available at the current decision boundary."""

    steps_to_next: int | None
    delivery_period_steps: int
    next_requested_kg: float
    cumulative_requested_kg: float
    cumulative_delivered_kg: float


class ObservationEncoderV2:
    """Compose the frozen V1 telemetry transform with four delivery features."""

    def __init__(self, config: PlantConfig) -> None:
        self.config = config
        self._base = V1ObservationEncoder(config)
        self._low = spaces.observation_low()
        self._high = spaces.observation_high()

    def reset(self) -> None:
        self._base.reset()

    def encode(
        self,
        state: PlantState,
        *,
        constraints: Constraints,
        progress: float,
        delivery: DeliveryObservation,
    ) -> np.ndarray:
        base = self._base.encode(state, constraints=constraints, progress=progress)
        if delivery.steps_to_next is None:
            time_to_next = 1.0
        else:
            time_to_next = delivery.steps_to_next / max(1, delivery.delivery_period_steps)
        next_amount = delivery.next_requested_kg
        next_amount_fraction = next_amount / max(state.ch4_capacity_kg, 1e-12)
        coverage = state.ch4_level_kg / next_amount if next_amount > 0.0 else 0.0
        fill_rate = (
            delivery.cumulative_delivered_kg / delivery.cumulative_requested_kg
            if delivery.cumulative_requested_kg > 0.0
            else 1.0
        )
        vector = np.concatenate(
            (
                base,
                np.asarray(
                    (time_to_next, next_amount_fraction, coverage, fill_rate),
                    dtype=np.float32,
                ),
            )
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(vector, self._low, self._high).astype(np.float32)


__all__ = ["DeliveryObservation", "ObservationEncoderV2"]
