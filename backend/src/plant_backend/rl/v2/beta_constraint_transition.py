"""Canonical per-transition constraint costs for the cyclic-market beta.

The beta settlement ledger is cumulative episode evidence.  Constrained
optimizers instead need exactly one sparse cost vector for each controller
transition, without charging an earlier settlement again on later steps.  This
module is the immutable boundary between those two responsibilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Sequence

from .beta_config import RESOURCE_NAMES, CycleSettlementV1, canonical_sha256


CYCLE_CONSTRAINT_TRANSITION_SCHEMA = "cycle-constraint-transition-v1"
CYCLE_CONSTRAINT_RESOURCE_ORDER = RESOURCE_NAMES


def _non_negative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    # A unique positive-zero representation keeps zero-transition digests
    # canonical even if a caller supplies IEEE negative zero.
    return 0.0 if number == 0.0 else number


def _cost_vector(name: str, values: object) -> tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a four-value sequence")
    if len(values) != len(CYCLE_CONSTRAINT_RESOURCE_ORDER):
        raise ValueError(f"{name} must contain four resource costs")
    normalized = tuple(
        _non_negative_finite(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    return (normalized[0], normalized[1], normalized[2], normalized[3])


@dataclass(frozen=True, slots=True)
class CycleConstraintTransitionV1:
    """One exact, resource-ordered cost record at a Gym transition.

    ``raw_cost`` is non-zero only when ``settlement`` occurred on this exact
    transition.  ``training_cost`` divides each raw channel by the number of
    cycles in the episode, making an episode sum equal to mean normalized
    violation per cycle.  The scalar sums are reporting checksums, never an
    alternative source for a four-channel constrained optimizer.
    """

    decision_index: int
    cycle_settled: bool
    cycle_index: int | None
    episode_cycle_count: int
    resource_order: tuple[str, str, str, str]
    raw_cost: tuple[float, float, float, float]
    training_cost: tuple[float, float, float, float]

    schema: ClassVar[str] = CYCLE_CONSTRAINT_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, Integral)
            or self.decision_index < 0
        ):
            raise ValueError("decision_index must be a non-negative integer")
        object.__setattr__(self, "decision_index", int(self.decision_index))

        if not isinstance(self.cycle_settled, bool):
            raise TypeError("cycle_settled must be a bool")
        if self.cycle_settled:
            if (
                isinstance(self.cycle_index, bool)
                or not isinstance(self.cycle_index, Integral)
                or self.cycle_index <= 0
            ):
                raise ValueError("cycle_index must be a positive integer at settlement")
            object.__setattr__(self, "cycle_index", int(self.cycle_index))
        elif self.cycle_index is not None:
            raise ValueError("cycle_index must be None without a settlement")

        if (
            isinstance(self.episode_cycle_count, bool)
            or not isinstance(self.episode_cycle_count, Integral)
            or self.episode_cycle_count <= 0
        ):
            raise ValueError("episode_cycle_count must be a positive integer")
        object.__setattr__(
            self,
            "episode_cycle_count",
            int(self.episode_cycle_count),
        )

        resource_order = tuple(self.resource_order)
        if resource_order != CYCLE_CONSTRAINT_RESOURCE_ORDER:
            raise ValueError("resource_order must be exactly battery, h2, co2, ch4")
        object.__setattr__(self, "resource_order", resource_order)

        raw_cost = _cost_vector("raw_cost", self.raw_cost)
        training_cost = _cost_vector("training_cost", self.training_cost)
        expected_training = tuple(
            value / self.episode_cycle_count for value in raw_cost
        )
        if training_cost != expected_training:
            raise ValueError(
                "training_cost must equal raw_cost divided by episode_cycle_count"
            )
        if not self.cycle_settled and any(value != 0.0 for value in raw_cost):
            raise ValueError("non-settlement raw_cost must be exactly zero")
        object.__setattr__(self, "raw_cost", raw_cost)
        object.__setattr__(self, "training_cost", training_cost)

    @classmethod
    def from_settlement(
        cls,
        *,
        decision_index: int,
        settlement: CycleSettlementV1 | None,
        episode_cycle_count: int,
    ) -> "CycleConstraintTransitionV1":
        """Construct the transition directly from authoritative settlement data."""

        if (
            isinstance(episode_cycle_count, bool)
            or not isinstance(episode_cycle_count, Integral)
            or episode_cycle_count <= 0
        ):
            raise ValueError("episode_cycle_count must be a positive integer")
        if settlement is None:
            raw_cost = (0.0, 0.0, 0.0, 0.0)
            cycle_index = None
        else:
            if not isinstance(settlement, CycleSettlementV1):
                raise TypeError("settlement must be CycleSettlementV1 or None")
            if settlement.boundary_decision != decision_index:
                raise ValueError(
                    "settlement boundary_decision must equal transition decision_index"
                )
            raw_cost = tuple(
                max(
                    0.0,
                    float(settlement.deficits[name])
                    - float(settlement.tolerances[name]),
                )
                / max(1.0, float(settlement.targets[name]))
                for name in CYCLE_CONSTRAINT_RESOURCE_ORDER
            )
            cycle_index = settlement.cycle_index

        training_cost = tuple(value / episode_cycle_count for value in raw_cost)
        return cls(
            decision_index=decision_index,
            cycle_settled=settlement is not None,
            cycle_index=cycle_index,
            episode_cycle_count=episode_cycle_count,
            resource_order=CYCLE_CONSTRAINT_RESOURCE_ORDER,
            raw_cost=(raw_cost[0], raw_cost[1], raw_cost[2], raw_cost[3]),
            training_cost=(
                training_cost[0],
                training_cost[1],
                training_cost[2],
                training_cost[3],
            ),
        )

    @property
    def raw_cost_sum(self) -> float:
        return math.fsum(self.raw_cost)

    @property
    def training_cost_sum(self) -> float:
        return math.fsum(self.training_cost)

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision_index": self.decision_index,
            "cycle_settled": self.cycle_settled,
            "cycle_index": self.cycle_index,
            "episode_cycle_count": self.episode_cycle_count,
            "resource_order": list(self.resource_order),
            "raw_cost": list(self.raw_cost),
            "training_cost": list(self.training_cost),
            "raw_cost_sum": self.raw_cost_sum,
            "training_cost_sum": self.training_cost_sum,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


__all__ = [
    "CYCLE_CONSTRAINT_RESOURCE_ORDER",
    "CYCLE_CONSTRAINT_TRANSITION_SCHEMA",
    "CycleConstraintTransitionV1",
]
