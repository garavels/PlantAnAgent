"""Owns the running plant: one simulator instance plus a rolling history.

This is the layer that will later grow into the agent/controller stack. Today
the action arrives from the UI and is passed straight through:

    action = request.action              # manual control (today)
    action = agent.decide(self.state)    # autonomous control (later)
    state = self.simulator.step(action)

The simulator itself stays unaware of who is driving it.

What leaves this layer is *telemetry*, never the simulator's private truth:
`PlantSimulator.get_telemetry()` is the observation a sensor fault is allowed
to distort. Anything that wants the ground truth has to be inside the
simulator process and ask for `simulator.state` on purpose (the tests do).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from plant_sim import Action, PlantConfig, PlantSimulator, Scenario
from plant_sim import faults as fault_library
from plant_sim import scenarios as scenario_library

#: ~2 simulated days at the default 60 s timestep.
HISTORY_LIMIT = 2880
#: Events handed back with each response.
EVENT_LIMIT = 60


class PlantSession:
    """A single in-memory plant session. One process, one plant, no database."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.simulator = PlantSimulator(config)
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self.history.append(self.state)

    @property
    def state(self) -> dict[str, Any]:
        """The observed plant: telemetry, not the simulator's hidden truth."""
        return self.simulator.get_telemetry().to_dict()

    @property
    def config(self) -> dict[str, Any]:
        config = self.simulator.config
        return {
            **config.to_dict(),
            "dt_h": config.dt_h,
            "steps_per_day": config.steps_per_day,
            "electrolyser_min_load_kw": self.simulator.electrolyser.min_load_kw,
        }

    # --- conditions -------------------------------------------------------
    @property
    def faults(self) -> list[dict[str, Any]]:
        return self.simulator.faults.to_list()

    @property
    def scenario(self) -> str | None:
        return self.simulator.scenario

    def events(self, limit: int = EVENT_LIMIT) -> list[dict[str, Any]]:
        return self.simulator.events.recent(limit)

    def conditions(self, events: int = EVENT_LIMIT) -> dict[str, Any]:
        """The abnormal-operation side of the session, in one payload."""
        return {"faults": self.faults, "events": self.events(events), "scenario": self.scenario}

    # --- driving ----------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        self.simulator.reset()
        state = self.state
        self.history.clear()
        self.history.append(state)
        return state

    def step(self, action: Action | dict[str, Any] | None, repeat: int = 1) -> list[dict[str, Any]]:
        """Advance `repeat` timesteps under the same action; returns the new frames."""
        frames = []
        for _ in range(max(1, min(repeat, 1000))):
            self.simulator.step(action)
            frames.append(self.state)
        self.history.extend(frames)
        return frames

    def recent(self, limit: int = 240) -> list[dict[str, Any]]:
        frames = list(self.history)
        return frames[-max(1, limit) :]

    # --- abnormal operation -----------------------------------------------
    def inject_fault(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Inject a fault or disturbance; raises `ValueError` on unknown types."""
        return self.simulator.inject_fault(spec).to_dict()

    def clear_fault(self, fault_id: str) -> bool:
        return self.simulator.clear_fault(fault_id)

    def clear_faults(self) -> None:
        self.simulator.clear_faults()

    def load_scenario(self, scenario: Scenario | str) -> dict[str, Any]:
        """Reset the plant into a predefined situation."""
        self.simulator.load_scenario(scenario)
        state = self.state
        self.history.clear()
        self.history.append(state)
        return state

    @staticmethod
    def fault_catalogue() -> list[dict[str, Any]]:
        return fault_library.catalogue()

    @staticmethod
    def scenario_catalogue() -> list[dict[str, Any]]:
        return scenario_library.catalogue()
