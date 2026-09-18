"""The contract every plant component honours.

Components own their internal state and enforce their own physical limits: the
simulator only decides *how much* to ask of them, never what they are allowed to
do. Adding DAC, methanation or a CH4 tank later means adding a `Component`
subclass and wiring it into `PlantSimulator`, nothing else.

Faults arrive the same way limits do: the simulator hands each component a
`ComponentEffect` every timestep and the component decides what a derating or
an outage means for *its* physics -- less yield, less capacity, no output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..faults import NO_EFFECT, ComponentEffect


class Component(ABC):
    """Base class for a piece of plant equipment.

    Subclasses add their own domain verbs (`charge`, `run`, `store`, ...) and are
    responsible for clamping requests to what they can physically accept.
    """

    #: Stable identifier, used as the key in `PlantSimulator.components`.
    name: str = "component"

    #: Current fault effect: 0 degradation, not offline, unless a fault says so.
    effect: ComponentEffect = NO_EFFECT

    def apply_effect(self, effect: ComponentEffect) -> None:
        """Adopt this timestep's fault effect. Override to react to the change."""
        self.effect = effect

    @property
    def health(self) -> float:
        """Fraction of nominal performance available, 0..1 (0 while offline)."""
        return self.effect.health

    @property
    def offline(self) -> bool:
        return self.effect.offline

    @property
    def status(self) -> str:
        """`nominal`, `degraded` or `offline` -- the plant-status building block."""
        if self.effect.offline:
            return "offline"
        return "degraded" if self.effect.degradation > 0.0 else "nominal"

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "degradation": self.effect.degradation,
            "offline": self.effect.offline,
            "health": self.health,
        }

    @abstractmethod
    def reset(self) -> None:
        """Return to the configured initial condition."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Component-local telemetry, for debugging and future per-unit views."""

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.snapshot().items())
        return f"{type(self).__name__}({fields})"
