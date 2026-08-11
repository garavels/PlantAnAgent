"""The contract every plant component honours.

Components own their internal state and enforce their own physical limits: the
simulator only decides *how much* to ask of them, never what they are allowed to
do. Adding DAC, methanation or a CH4 tank later means adding a `Component`
subclass and wiring it into `PlantSimulator`, nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Base class for a piece of plant equipment.

    Subclasses add their own domain verbs (`charge`, `run`, `store`, ...) and are
    responsible for clamping requests to what they can physically accept.
    """

    #: Stable identifier, used as the key in `PlantSimulator.components`.
    name: str = "component"

    @abstractmethod
    def reset(self) -> None:
        """Return to the configured initial condition."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Component-local telemetry, for debugging and future per-unit views."""

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.snapshot().items())
        return f"{type(self).__name__}({fields})"
