"""The simulator as a Gymnasium environment. One plant, two ways in.

    env = PlantEnv()
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(policy(observation))

Nothing here models the plant: `PlantSimulator` does, and this file is the
translation layer around it -- vector in, `Action` out, telemetry in, vector out.
The step path is the same one the backend uses, in the same order:

    decide -> safety.validate(...) -> simulator.step(...)

`step()` is the RL door (a vector from a policy) and `step_action()` is the
controller door (an `Action` from anything implementing `Controller`). They share
one body, which is what makes the evaluation in `evaluation/` a fair comparison
rather than two rollout loops that drifted apart: the heuristic controller and a
policy are validated by the same guard, scored by the same reward and measured by
the same metrics.

Truth versus telemetry is respected in both directions. The observation comes
from `get_telemetry()` -- what the instruments say, distortions included -- and
the reward comes from `simulator.state` -- what the plant actually did. A policy
that learns to trust a drifting sensor is therefore penalised by the same
mechanism a real plant would penalise it: production it did not make.

Episodes end by truncation, never termination: an industrial plant has no
absorbing state, a critical condition is something to recover from rather than a
game over, so `terminated` is always False and `truncated` becomes True at the
episode horizon. Bootstrapping stays correct at the cut, which is what the
distinction is for.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from plant_sim import Action, PlantConfig, PlantSimulator, PlantState

from ..controllers.base import Constraints
from ..safety.validation import validate
from . import episodes as episode_library
from . import reward as reward_module
from .actions import ActionAdapter, DecodedAction
from .metrics import EpisodeMetrics
from .observation import ObservationEncoder
from .spaces import (
    ACTION_SIZE,
    action_high,
    action_low,
    observation_high,
    observation_low,
)


class PlantEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium view of the plant: normalised observation, share-based action."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: PlantConfig | None = None,
        episode: episode_library.EpisodeConfig | None = None,
        constraints: Constraints | None = None,
        weights: reward_module.RewardWeights | None = None,
    ) -> None:
        self.config = config or PlantConfig()
        self.episode_config = (episode or episode_library.EpisodeConfig()).validate()
        #: The envelope the policy operates inside. The supervisor's lever in the
        #: running plant; a fixed training condition here.
        self.constraints = constraints or Constraints()
        self.weights = weights or reward_module.RewardWeights()

        self.simulator = PlantSimulator(self.config)
        self.encoder = ObservationEncoder(self.config)
        self.adapter = ActionAdapter(self.config)

        self.observation_space = Box(
            low=observation_low(), high=observation_high(), dtype=np.float32
        )
        self.action_space = Box(
            low=action_low(), high=action_high(), shape=(ACTION_SIZE,), dtype=np.float32
        )

        self.episode: episode_library.Episode | None = None
        self.metrics = EpisodeMetrics()
        self.steps_taken = 0
        self.episode_steps = self.episode_config.steps(self.config)

    # --- gymnasium interface ----------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Draw an episode and put the plant into it. Same seed, same episode."""
        super().reset(seed=seed)
        drawn = episode_library.sample(
            self.episode_config,
            self.config,
            self.np_random,
            seed=-1 if seed is None else int(seed),
        )
        self.episode = drawn
        self.episode_steps = drawn.steps
        self.simulator.load_scenario(drawn.scenario)
        self.encoder.reset()
        self.metrics = EpisodeMetrics()
        self.steps_taken = 0
        observation = self._observe()
        return observation, {
            "episode": drawn.to_dict(),
            "clock": self.telemetry.clock,
            "scenario": self.simulator.scenario,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one policy action: decode, validate, advance, score."""
        decoded = self.adapter.decode(
            action,
            constraints=self.constraints,
            solar_available_kw=self.telemetry.solar_available_kw,
        )
        return self._apply(decoded)

    # --- the controller door ----------------------------------------------
    def step_action(
        self, action: Action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an `Action` straight from a `Controller`, on the same path.

        Used by the evaluation harness so the heuristic baseline is rolled out,
        validated, rewarded and measured identically to a policy.
        """
        bus_kw = self.telemetry.solar_available_kw
        return self._apply(
            DecodedAction(
                action=Action.sanitised(action),
                clipped=False,
                requested_kw={
                    "battery": action.battery_share * max(0.0, bus_kw),
                    "electrolyser": action.electrolyser_share * max(0.0, bus_kw)
                    + action.battery_support_kw,
                    "dac": action.dac_share * max(0.0, bus_kw),
                    "methanator": action.methanator_share * max(0.0, bus_kw),
                    "battery_support": action.battery_support_kw,
                },
            )
        )

    # --- shared body ------------------------------------------------------
    def _apply(
        self, decoded: DecodedAction
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation = self.telemetry
        validated = validate(
            decoded.action, observation, constraints=self.constraints, config=self.config
        )
        before = self.simulator.state
        after = self.simulator.step(validated.action)

        breakdown = reward_module.compute(
            before,
            after,
            config=self.config,
            constraints=self.constraints,
            adjustments=validated.adjustments,
            out_of_range=decoded.clipped,
            weights=self.weights,
        )
        self.metrics.record(
            before,
            after,
            config=self.config,
            reward=breakdown.total,
            components=breakdown.components,
            adjustments=validated.adjustments,
            out_of_range=decoded.clipped,
            commanded_methanation=validated.action.methanator_share > 0.0,
        )
        self.steps_taken += 1
        truncated = self.steps_taken >= self.episode_steps
        info: dict[str, Any] = {
            "clock": after.clock,
            "requested": decoded.to_dict(),
            "applied_action": validated.action.to_dict(),
            "adjustments": list(validated.adjustments),
            "corrected": validated.clamped,
            "reward": breakdown.to_dict(),
            "metrics": self.metrics.to_dict(),
            "plant_status": after.plant_status,
        }
        return self._observe(), breakdown.total, False, truncated, info

    # --- views ------------------------------------------------------------
    @property
    def telemetry(self) -> PlantState:
        """What the plant reports about itself: the controller's only input."""
        return self.simulator.get_telemetry()

    @property
    def progress(self) -> float:
        return min(1.0, self.steps_taken / self.episode_steps) if self.episode_steps else 0.0

    def _observe(self) -> np.ndarray:
        return self.encoder.encode(
            self.telemetry, constraints=self.constraints, progress=self.progress
        )


__all__ = ["PlantEnv"]
