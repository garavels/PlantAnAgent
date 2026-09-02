"""Tick-indexed methane delivery commitments and their settlement ledger.

The delivery contract deliberately lives beside :class:`PlantState`, rather
than inside it.  V1 callers therefore keep the same state/checkpoint schema,
while an opt-in V2 environment can inspect contractual delivery accounting
through :class:`DeliveryLedger`.

Deadlines are integer physics ticks relative to the most recent simulator
``reset()`` or ``load_scenario()``.  This keeps boundary handling exact: a
commitment due at tick ``n`` is settled after interval ``[n - 1, n)`` has
produced methane and before the tick-``n`` state is built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Mapping


def _tick(value: object, *, name: str, minimum: int) -> int:
    """Validate an integer tick without accepting ``True`` as tick one."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    tick = int(value)
    if tick < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return tick


def _mass(value: object, *, name: str) -> float:
    """Return a finite, non-negative mass or fail instead of sanitising it."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        mass = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(mass):
        raise ValueError(f"{name} must be finite")
    if mass < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return mass


@dataclass(frozen=True, slots=True)
class DeliveryCommitment:
    """One immutable contractual methane delivery.

    ``due_step`` is an episode-relative physics tick, never a floating-point
    timestamp.  ``requested_kg=0`` is valid and still produces one auditable
    settlement at its deadline.
    """

    id: str
    due_step: int
    requested_kg: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("delivery commitment id must be a non-empty string")
        object.__setattr__(self, "due_step", _tick(self.due_step, name="due_step", minimum=1))
        object.__setattr__(
            self,
            "requested_kg",
            _mass(self.requested_kg, name="requested_kg"),
        )

    @classmethod
    def parse(cls, raw: "DeliveryCommitment | Mapping[str, Any]") -> "DeliveryCommitment":
        """Accept a commitment or its exact dictionary representation."""
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("delivery commitments must be DeliveryCommitment objects or mappings")
        expected = {"id", "due_step", "requested_kg"}
        missing = expected - set(raw)
        unknown = set(raw) - expected
        if missing:
            raise ValueError(f"delivery commitment missing field(s): {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"unknown delivery commitment field(s): {', '.join(sorted(unknown))}")
        return cls(
            id=raw["id"],
            due_step=raw["due_step"],
            requested_kg=raw["requested_kg"],
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.id,
            "due_step": self.due_step,
            "requested_kg": self.requested_kg,
        }


@dataclass(frozen=True, slots=True)
class DeliverySettlement:
    """The immutable result of settling one commitment exactly once."""

    id: str
    due_step: int
    requested_kg: float
    delivered_kg: float
    unmet_kg: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.id,
            "due_step": self.due_step,
            "requested_kg": self.requested_kg,
            "delivered_kg": self.delivered_kg,
            "unmet_kg": self.unmet_kg,
        }


@dataclass(frozen=True, slots=True)
class DeliveryStepResult:
    """Delivery accounting at one tick, including episode-to-date totals.

    A result exists for every simulator boundary.  On a tick with no due
    commitment, ``settlements`` is empty and all three step aggregates are
    zero, while the cumulative values and next commitment remain informative.
    """

    step: int
    settlements: tuple[DeliverySettlement, ...] = ()
    requested_kg: float = 0.0
    delivered_kg: float = 0.0
    unmet_kg: float = 0.0
    cumulative_requested_kg: float = 0.0
    cumulative_delivered_kg: float = 0.0
    cumulative_unmet_kg: float = 0.0
    next_commitment: DeliveryCommitment | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "settlements": [settlement.to_dict() for settlement in self.settlements],
            "requested_kg": self.requested_kg,
            "delivered_kg": self.delivered_kg,
            "unmet_kg": self.unmet_kg,
            "cumulative_requested_kg": self.cumulative_requested_kg,
            "cumulative_delivered_kg": self.cumulative_delivered_kg,
            "cumulative_unmet_kg": self.cumulative_unmet_kg,
            "next_commitment": (
                None if self.next_commitment is None else self.next_commitment.to_dict()
            ),
        }


class DeliveryLedger:
    """Own a schedule and settle its commitments at exact physics ticks.

    Installing a schedule is atomic: every item and the complete id set are
    validated before the previous schedule or accounting is changed.  Equal
    tick commitments settle in the order supplied by the caller.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self, *, current_step: int = 0) -> None:
        """Remove the schedule, settlements, and all cumulative accounting."""
        step = _tick(current_step, name="current_step", minimum=0)
        self._commitments: tuple[DeliveryCommitment, ...] = ()
        self._pending: dict[str, DeliveryCommitment] = {}
        self._by_step: dict[int, tuple[DeliveryCommitment, ...]] = {}
        self._order: dict[str, int] = {}
        self._settlements: list[DeliverySettlement] = []
        self._cumulative_requested_kg = 0.0
        self._cumulative_delivered_kg = 0.0
        self._cumulative_unmet_kg = 0.0
        self._current = DeliveryStepResult(step=step)

    def set_schedule(
        self,
        commitments: Iterable[DeliveryCommitment | Mapping[str, Any]],
        *,
        current_step: int = 0,
    ) -> DeliveryStepResult:
        """Replace the schedule after validating it completely.

        Deadlines are relative to the current episode, not relative to this
        call.  Consequently, a schedule installed after tick zero may only
        contain commitments whose deadlines are still in the future.
        """
        step = _tick(current_step, name="current_step", minimum=0)
        try:
            parsed = tuple(DeliveryCommitment.parse(item) for item in commitments)
        except TypeError as exc:
            if isinstance(commitments, (str, bytes)):
                raise TypeError("delivery schedule must be an iterable of commitments") from exc
            raise

        seen: set[str] = set()
        duplicate_ids: set[str] = set()
        for commitment in parsed:
            if commitment.id in seen:
                duplicate_ids.add(commitment.id)
            seen.add(commitment.id)
            if commitment.due_step <= step:
                raise ValueError(
                    f"delivery {commitment.id!r} is due at step {commitment.due_step}; "
                    f"current episode step is {step}"
                )
        if duplicate_ids:
            duplicate_list = ", ".join(repr(value) for value in sorted(duplicate_ids))
            raise ValueError(f"delivery commitment ids must be unique: {duplicate_list}")

        # All validation is complete.  Only now replace the prior ledger.
        by_step: dict[int, list[DeliveryCommitment]] = {}
        for commitment in parsed:
            by_step.setdefault(commitment.due_step, []).append(commitment)
        self._commitments = parsed
        self._pending = {commitment.id: commitment for commitment in parsed}
        self._by_step = {
            due_step: tuple(due_commitments)
            for due_step, due_commitments in by_step.items()
        }
        self._order = {commitment.id: index for index, commitment in enumerate(parsed)}
        self._settlements = []
        self._cumulative_requested_kg = 0.0
        self._cumulative_delivered_kg = 0.0
        self._cumulative_unmet_kg = 0.0
        self._current = DeliveryStepResult(
            step=step,
            next_commitment=self._find_next_pending(),
        )
        return self._current

    def settle(self, step: int, *, available_kg: float) -> DeliveryStepResult:
        """Settle all commitments due at ``step`` against available methane.

        Calls advance one integer tick at a time.  Repeating the latest tick is
        idempotent and returns its original result, so no commitment can be
        withdrawn twice accidentally.
        """
        tick = _tick(step, name="step", minimum=0)
        available = _mass(available_kg, name="available_kg")
        if tick == self._current.step:
            return self._current
        if tick != self._current.step + 1:
            raise ValueError(
                f"delivery settlement must advance one tick: "
                f"current={self._current.step}, requested={tick}"
            )

        due = self._by_step.pop(tick, ())
        remaining_kg = available
        settlements: list[DeliverySettlement] = []
        for commitment in due:
            delivered_kg = min(remaining_kg, commitment.requested_kg)
            remaining_kg -= delivered_kg
            unmet_kg = commitment.requested_kg - delivered_kg
            settlement = DeliverySettlement(
                id=commitment.id,
                due_step=commitment.due_step,
                requested_kg=commitment.requested_kg,
                delivered_kg=delivered_kg,
                unmet_kg=unmet_kg,
            )
            settlements.append(settlement)
            self._settlements.append(settlement)
            self._pending.pop(commitment.id)

        requested_kg = sum(item.requested_kg for item in settlements)
        # This subtraction is the authoritative tank withdrawal and cannot
        # exceed ``available`` even after several same-tick allocations.
        delivered_kg = available - remaining_kg
        unmet_kg = requested_kg - delivered_kg
        self._cumulative_requested_kg += requested_kg
        self._cumulative_delivered_kg += delivered_kg
        self._cumulative_unmet_kg += unmet_kg
        self._current = DeliveryStepResult(
            step=tick,
            settlements=tuple(settlements),
            requested_kg=requested_kg,
            delivered_kg=delivered_kg,
            unmet_kg=unmet_kg,
            cumulative_requested_kg=self._cumulative_requested_kg,
            cumulative_delivered_kg=self._cumulative_delivered_kg,
            cumulative_unmet_kg=self._cumulative_unmet_kg,
            next_commitment=self._find_next_pending(),
        )
        return self._current

    @property
    def current(self) -> DeliveryStepResult:
        """Immutable accounting view for the latest settled physics tick."""
        return self._current

    @property
    def next_pending(self) -> DeliveryCommitment | None:
        """The next contractual deadline, without exposing the future tape."""
        return self._current.next_commitment

    @property
    def commitments(self) -> tuple[DeliveryCommitment, ...]:
        """Immutable installed schedule, primarily for diagnostics and tests."""
        return self._commitments

    @property
    def settlements(self) -> tuple[DeliverySettlement, ...]:
        """Immutable settlement history, oldest first."""
        return tuple(self._settlements)

    def audit(self) -> dict[str, Any]:
        """Serialize the complete schedule and accounting for offline audit."""
        return {
            "schedule": [commitment.to_dict() for commitment in self._commitments],
            "settlements": [settlement.to_dict() for settlement in self._settlements],
            "current": self._current.to_dict(),
            "cumulative_requested_kg": self._cumulative_requested_kg,
            "cumulative_delivered_kg": self._cumulative_delivered_kg,
            "cumulative_unmet_kg": self._cumulative_unmet_kg,
            "pending_count": len(self._pending),
            "next_pending": (
                None if self.next_pending is None else self.next_pending.to_dict()
            ),
        }

    def _find_next_pending(self) -> DeliveryCommitment | None:
        if not self._pending:
            return None
        return min(
            self._pending.values(),
            key=lambda commitment: (commitment.due_step, self._order[commitment.id]),
        )


__all__ = [
    "DeliveryCommitment",
    "DeliveryLedger",
    "DeliverySettlement",
    "DeliveryStepResult",
]
