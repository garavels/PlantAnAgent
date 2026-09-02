"""Algorithm-neutral online control boundary for Environment V2.

The Gym vector, transparent comparators and future optimisation policies all
start from this causal object.  It intentionally contains no simulator handle,
episode tape, future forecast vintage, delivery schedule, or replay identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Protocol, runtime_checkable

from plant_sim import Action, ElectrolyserCommitmentProfile, PlantConfig

from ...controllers.base import Constraints, ControlDecision
from .boundary import ControlStateV2
from .weather import PublicForecastIssueV2


def _non_negative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class DeliveryObservation:
    """Narrow current-deadline projection of the private delivery ledger."""

    steps_to_next: int | None
    delivery_period_steps: int
    next_requested_kg: float
    cumulative_requested_kg: float
    cumulative_delivered_kg: float

    def __post_init__(self) -> None:
        if self.steps_to_next is not None and (
            isinstance(self.steps_to_next, bool)
            or not isinstance(self.steps_to_next, Integral)
            or self.steps_to_next < 0
        ):
            raise ValueError("steps_to_next must be a non-negative integer or None")
        if (
            isinstance(self.delivery_period_steps, bool)
            or not isinstance(self.delivery_period_steps, Integral)
            or self.delivery_period_steps <= 0
        ):
            raise ValueError("delivery_period_steps must be a positive integer")
        requested = _non_negative_finite(
            "next_requested_kg", self.next_requested_kg
        )
        cumulative_requested = _non_negative_finite(
            "cumulative_requested_kg", self.cumulative_requested_kg
        )
        cumulative_delivered = _non_negative_finite(
            "cumulative_delivered_kg", self.cumulative_delivered_kg
        )
        if cumulative_delivered > cumulative_requested + 1e-9:
            raise ValueError(
                "cumulative_delivered_kg cannot exceed cumulative_requested_kg"
            )
        if self.steps_to_next is None and requested != 0.0:
            raise ValueError("next_requested_kg must be zero when no deadline remains")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "steps_to_next": self.steps_to_next,
            "delivery_period_steps": int(self.delivery_period_steps),
            "next_requested_kg": float(self.next_requested_kg),
            "cumulative_requested_kg": float(self.cumulative_requested_kg),
            "cumulative_delivered_kg": float(self.cumulative_delivered_kg),
        }


@dataclass(frozen=True)
class ControllerObservationV2:
    """Detached causal state available at exactly one decision boundary.

    ``forecast=None`` is an explicit information-policy ablation, not missing
    data.  Static equipment design and the active safety envelope remain
    explicit arguments of the controller method rather than being duplicated
    into every observation.
    """

    state: ControlStateV2
    delivery: DeliveryObservation
    forecast: PublicForecastIssueV2 | None
    progress: float
    decision_interval_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, ControlStateV2):
            raise TypeError("state must be a ControlStateV2")
        if not isinstance(self.delivery, DeliveryObservation):
            raise TypeError("delivery must be a DeliveryObservation")
        if self.forecast is not None and not isinstance(
            self.forecast, PublicForecastIssueV2
        ):
            raise TypeError("forecast must be a PublicForecastIssueV2 or None")
        if (
            isinstance(self.progress, bool)
            or not isinstance(self.progress, Real)
            or not math.isfinite(float(self.progress))
            or not 0.0 <= self.progress <= 1.0
        ):
            raise ValueError("progress must be finite and within [0, 1]")
        if (
            isinstance(self.decision_interval_s, bool)
            or not isinstance(self.decision_interval_s, Real)
            or not math.isfinite(float(self.decision_interval_s))
            or self.decision_interval_s <= 0.0
        ):
            raise ValueError("decision_interval_s must be positive and finite")

    def without_forecast(self) -> "ControllerObservationV2":
        """Return the declared forecast-blind view of the same boundary."""
        return replace(self, forecast=None)

    def to_dict(self) -> dict[str, object]:
        """Serialize only declared online fields through their safe projections."""
        return {
            "state": self.state.to_dict(),
            "delivery": self.delivery.to_dict(),
            "forecast": None if self.forecast is None else self.forecast.to_dict(),
            "progress": float(self.progress),
            "decision_interval_s": float(self.decision_interval_s),
        }


@runtime_checkable
class ControllerV2(Protocol):
    """Structural contract shared by transparent and learned V2 policies."""

    id: str
    label: str
    description: str

    def compute_action(
        self,
        observation: ControllerObservationV2,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: ElectrolyserCommitmentProfile | None,
    ) -> ControlDecision:
        """Choose one held action without access to private environment state."""

    def reset(self) -> None:
        """Drop controller-local history before a new episode."""


def action_from_controller(
    controller: ControllerV2,
    observation: ControllerObservationV2,
    *,
    constraints: Constraints,
    config: PlantConfig,
    commitment_profile: ElectrolyserCommitmentProfile | None,
) -> Action:
    """Small typed adapter used by evaluation runners and future online loops."""
    return controller.compute_action(
        observation,
        constraints=constraints,
        config=config,
        commitment_profile=commitment_profile,
    ).action


__all__ = [
    "ControllerObservationV2",
    "ControllerV2",
    "DeliveryObservation",
    "action_from_controller",
]
