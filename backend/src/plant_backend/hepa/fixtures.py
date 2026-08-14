"""Deterministic HEPA stand-ins. **Test fixtures, not a model.**

Nothing in this module predicts anything. Every number it returns was written by
hand, and it exists so that the V10 integration path -- forecast, level, trend,
trigger, investigation, intervention, forecast again -- can be tested on a machine
with no GPU, no checkpoint and no torch:

    forecaster = ScriptedForecaster([
        {"co2_starvation": {"6h": 0.18, "12h": 0.76}},   # fixture, not a prediction
    ])

Any output that reaches a user through one of these classes is labelled: the
`Forecast.source` is `fixture:<name>`, the supervisor snapshot carries it through
to the control room, and the evaluation harness refuses to call a fixture run
evidence. A real answer only ever comes from `service.CheckpointForecaster`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .events import EVENT_NAMES
from .horizons import HorizonGrid
from .service import ACTIVE, Forecast

#: Marks every forecast produced here, all the way to the frontend.
SOURCE_PREFIX = "fixture"


def _filled(
    probabilities: Mapping[str, Mapping[str, float]],
    horizons: Sequence[str],
) -> dict[str, dict[str, float]]:
    """The scripted numbers, with every other event/horizon at a flat 1%.

    A partial script stays readable ("CO2 starvation is the story here") while the
    shape handed to the supervisor is still the full event table it would get from
    a checkpoint.
    """
    table: dict[str, dict[str, float]] = {}
    for event in EVENT_NAMES:
        scripted = probabilities.get(event, {})
        table[event] = {
            horizon: float(scripted.get(horizon, 0.01)) for horizon in horizons
        }
    return table


def _horizons(requested: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(requested) if requested else HorizonGrid.of_hours().labels[:4]


@dataclass
class FixedForecaster:
    """Always the same answer. The "nothing is happening" fixture."""

    probabilities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    name: str = "fixed"

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast:
        labels = _horizons(horizons)
        return Forecast(
            status=ACTIVE,
            probabilities=_filled(self.probabilities, labels),
            horizons=labels,
            observed_at_s=float(frames[-1]["time_s"]) if frames else None,
            source=f"{SOURCE_PREFIX}:{self.name}",
        )

    def status(self) -> dict[str, Any]:
        return {"available": True, "fixture": self.name}


@dataclass
class ScriptedForecaster:
    """One hand-written answer per call; the last one repeats forever.

    This is how a *rising* risk is tested: the script is the story, and the
    monitor's trend logic is what turns two consecutive entries into `rising`.
    """

    script: Sequence[Mapping[str, Mapping[str, float]]] = field(default_factory=list)
    name: str = "scripted"

    def __post_init__(self) -> None:
        self.calls = 0

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast:
        if not self.script:
            raise ValueError("ScriptedForecaster needs at least one scripted answer")
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        labels = _horizons(horizons)
        return Forecast(
            status=ACTIVE,
            probabilities=_filled(step, labels),
            horizons=labels,
            observed_at_s=float(frames[-1]["time_s"]) if frames else None,
            source=f"{SOURCE_PREFIX}:{self.name}",
        )

    def status(self) -> dict[str, Any]:
        return {"available": True, "fixture": self.name, "calls": self.calls}


@dataclass
class UnavailableForecaster:
    """No model loaded. What the plant does without a checkpoint."""

    reason: str = "no HEPA checkpoint configured (fixture)"

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast:
        return Forecast.unavailable(self.reason, source=f"{SOURCE_PREFIX}:unavailable")

    def status(self) -> dict[str, Any]:
        return {"available": False, "reason": self.reason}


@dataclass
class BrokenForecaster:
    """Raises. The supervisor is expected to survive this without a scratch."""

    message: str = "fixture inference failure"

    def forecast(
        self,
        frames: Sequence[dict[str, Any]],
        horizons: Sequence[str] | None = None,
    ) -> Forecast:
        raise RuntimeError(self.message)

    def status(self) -> dict[str, Any]:
        return {"available": False, "reason": self.message}


__all__ = [
    "BrokenForecaster",
    "FixedForecaster",
    "ScriptedForecaster",
    "SOURCE_PREFIX",
    "UnavailableForecaster",
]
