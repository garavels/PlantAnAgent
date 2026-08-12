"""A policy's vector -> the `Action` the existing guard already knows how to check.

    policy output (5 floats in [-1, 1])
            |
            v
      ActionAdapter.decode        <- this file: range conversion only
            |
            v
      safety.validate(...)        <- the one guard, unchanged
            |
            v
      PlantSimulator.step(...)

This file deliberately contains no safety logic. It converts ranges, and it
records what was asked for so the difference between "requested" and "applied"
can be reported. Every envelope decision -- component caps, offline units, the
battery reserve -- stays in `safety/validation.py`, because two places deciding
what is allowed is how they end up disagreeing.

The one clamp here is the policy's own range: a Gaussian policy can emit values
outside [-1, 1], and mapping those linearly would let a single large output
dominate the share normalisation. `Action.sanitised` then normalises shares that
sum above 1.0, exactly as it does for the heuristic controller and the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from plant_sim import Action, PlantConfig

from ..controllers.base import Constraints
from .spaces import ACTION_NAMES, ACTION_SIZE, action_high, action_low


@dataclass(frozen=True)
class DecodedAction:
    """What the policy asked for, in simulator units, before validation."""

    action: Action
    #: True when the raw vector had to be clipped into the policy's own range.
    clipped: bool
    #: Requested power per consumer (kW), for logging and telemetry.
    requested_kw: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "clipped": self.clipped,
            "requested_kw": dict(self.requested_kw),
        }


class ActionAdapter:
    """Converts policy output to `Action`, and reports what was requested."""

    def __init__(self, config: PlantConfig) -> None:
        self.config = config
        self._low = action_low()
        self._high = action_high()

    def decode(
        self,
        vector: Sequence[float] | np.ndarray,
        *,
        constraints: Constraints,
        solar_available_kw: float,
    ) -> DecodedAction:
        raw = np.nan_to_num(np.asarray(vector, dtype=np.float64).reshape(-1), nan=0.0)
        if raw.size != ACTION_SIZE:
            raise ValueError(
                f"action must have {ACTION_SIZE} entries ({', '.join(ACTION_NAMES)}), got {raw.size}"
            )
        clipped = bool(np.any(raw < self._low) or np.any(raw > self._high))
        unit = (np.clip(raw, self._low, self._high) + 1.0) / 2.0

        support_kw = float(unit[4]) * constraints.support_limit_kw(self.config)
        action = Action.sanitised(
            {
                "battery_share": float(unit[0]),
                "electrolyser_share": float(unit[1]),
                "dac_share": float(unit[2]),
                "methanator_share": float(unit[3]),
                "battery_support_kw": support_kw,
            }
        )
        bus_kw = max(0.0, solar_available_kw)
        return DecodedAction(
            action=action,
            clipped=clipped,
            requested_kw={
                "battery": action.battery_share * bus_kw,
                "electrolyser": action.electrolyser_share * bus_kw + action.battery_support_kw,
                "dac": action.dac_share * bus_kw,
                "methanator": action.methanator_share * bus_kw,
                "battery_support": action.battery_support_kw,
            },
        )

    def encode(self, action: Action, *, constraints: Constraints) -> np.ndarray:
        """The inverse, for tests and for replaying a recorded action as a vector."""
        support_limit_kw = constraints.support_limit_kw(self.config)
        shares = (
            action.battery_share,
            action.electrolyser_share,
            action.dac_share,
            action.methanator_share,
            action.battery_support_kw / support_limit_kw if support_limit_kw > 0.0 else 0.0,
        )
        return np.clip(
            np.asarray([2.0 * share - 1.0 for share in shares], dtype=np.float32),
            self._low,
            self._high,
        )


__all__ = ["ActionAdapter", "DecodedAction"]
