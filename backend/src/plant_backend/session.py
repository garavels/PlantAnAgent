"""Owns the running plant: one simulator instance plus a rolling history.

This is the layer that will later grow into the agent/controller stack. Today
the action arrives from the UI and is passed straight through:

    action = request.action              # manual control (today)
    action = agent.decide(self.state)    # autonomous control (later)
    state = self.simulator.step(action)

The simulator itself stays unaware of who is driving it.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from plant_sim import Action, PlantConfig, PlantSimulator

#: ~2 simulated days at the default 60 s timestep.
HISTORY_LIMIT = 2880


class PlantSession:
    """A single in-memory plant session. One process, one plant, no database."""

    def __init__(self, config: PlantConfig | None = None) -> None:
        self.simulator = PlantSimulator(config)
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self.history.append(self.simulator.state.to_dict())

    @property
    def state(self) -> dict[str, Any]:
        return self.simulator.state.to_dict()

    @property
    def config(self) -> dict[str, Any]:
        config = self.simulator.config
        return {
            **config.to_dict(),
            "dt_h": config.dt_h,
            "steps_per_day": config.steps_per_day,
            "electrolyser_min_load_kw": self.simulator.electrolyser.min_load_kw,
        }

    def reset(self) -> dict[str, Any]:
        state = self.simulator.reset().to_dict()
        self.history.clear()
        self.history.append(state)
        return state

    def step(self, action: Action | dict[str, Any] | None, repeat: int = 1) -> list[dict[str, Any]]:
        """Advance `repeat` timesteps under the same action; returns the new frames."""
        frames = [self.simulator.step(action).to_dict() for _ in range(max(1, min(repeat, 1000)))]
        self.history.extend(frames)
        return frames

    def recent(self, limit: int = 240) -> list[dict[str, Any]]:
        frames = list(self.history)
        return frames[-max(1, limit) :]
