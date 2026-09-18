"""The controller contract: telemetry in, action out.

    decision = controller.compute_action(observation, constraints=..., config=...)
    state = simulator.step(decision.action)

A controller sees exactly what the control room sees -- `PlantState` telemetry,
which a sensor fault is allowed to distort -- and answers with an `Action`. It
never touches the simulator, so a heuristic, an RL policy and an LLM-driven
supervisor are all interchangeable here.

`Constraints` is the other half of the seam, and the more important one for what
comes next: a supervisor is expected to *retune the envelope* (lower the DAC cap,
raise the battery reserve) rather than micromanage every timestep. That keeps the
fast loop deterministic and cheap while the slow loop stays free to think.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Any

from plant_sim import Action, PlantConfig, PlantState


def _finite(value: Any) -> float | None:
    """Coerce an optional JSON number; `None` and junk both mean "no limit"."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


@dataclass(frozen=True)
class Constraints:
    """The operating envelope every controller and manual command lives inside.

    Power caps are absolute (kW) and default to "whatever the equipment is
    rated for". `battery_min_soc` is the reserve nothing is allowed to eat
    into; it is applied on top of the battery's own configured floor, never
    below it.
    """

    electrolyser_max_kw: float | None = None
    dac_max_kw: float | None = None
    methanator_max_kw: float | None = None
    battery_support_max_kw: float | None = None
    battery_min_soc: float = 0.15

    # --- resolved limits --------------------------------------------------
    def electrolyser_limit_kw(self, config: PlantConfig) -> float:
        return _cap(config.electrolyser.rated_kw, self.electrolyser_max_kw)

    def dac_limit_kw(self, config: PlantConfig) -> float:
        return _cap(config.dac.rated_kw, self.dac_max_kw)

    def methanator_limit_kw(self, config: PlantConfig) -> float:
        return _cap(config.methanator.rated_kw, self.methanator_max_kw)

    def support_limit_kw(self, config: PlantConfig) -> float:
        return _cap(config.battery.max_discharge_kw, self.battery_support_max_kw)

    def reserve_soc(self, config: PlantConfig) -> float:
        """The SOC floor in force: the operator's reserve or the pack's own."""
        return max(config.battery.min_soc, self.battery_min_soc)

    # --- configuration ----------------------------------------------------
    def merged(self, changes: dict[str, Any]) -> "Constraints":
        """A copy with the given fields replaced. Unknown keys raise."""
        unknown = set(changes) - set(asdict(self))
        if unknown:
            raise ValueError(f"unknown constraints: {', '.join(sorted(unknown))}")
        updates: dict[str, Any] = {}
        for key, value in changes.items():
            if key == "battery_min_soc":
                soc = _finite(value)
                if soc is None or not 0.0 <= soc <= 0.95:
                    raise ValueError("battery_min_soc must be between 0 and 0.95")
                updates[key] = soc
            else:
                limit = _finite(value)
                if limit is not None and limit < 0.0:
                    raise ValueError(f"{key} cannot be negative")
                updates[key] = limit
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cap(rated_kw: float, limit_kw: float | None) -> float:
    """A constraint can only ever tighten what the equipment can do."""
    if limit_kw is None:
        return rated_kw
    return min(rated_kw, max(0.0, limit_kw))


@dataclass(frozen=True)
class ControlDecision:
    """An action plus why: `notes` is what the control room shows the operator.

    `invalid` means the controller asked for something outside its own range and
    had to be corrected before the action was even built -- a clipped policy
    vector, not a clamped power. It is a statement about the controller, which is
    why controller health counts it (see `controllers.health`).
    """

    action: Action
    notes: tuple[str, ...] = ()
    invalid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "notes": list(self.notes), "invalid": self.invalid}


class Controller(ABC):
    """Decides what the plant should do next, from telemetry alone.

    Implementations today: `HeuristicController`. Implementations later: an RL
    policy, an optimiser, or a supervisor-tuned variant of either -- all reached
    through this same method.
    """

    #: Stable identifier used by the API.
    id: str = "controller"
    #: Short human name for the control room.
    label: str = "Controller"
    description: str = ""

    @abstractmethod
    def compute_action(
        self,
        observation: PlantState,
        *,
        constraints: Constraints,
        config: PlantConfig,
    ) -> ControlDecision:
        """Choose this timestep's action from the observed plant."""

    def reset(self) -> None:
        """Drop any internal state. Stateless controllers need do nothing."""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "description": self.description}
