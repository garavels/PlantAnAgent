"""The learned controller, behind the same seam as the heuristic one.

    decision = controller.compute_action(observation, constraints=..., config=...)

That is the whole integration: `RLController` encodes the telemetry it is given,
asks the policy for a vector, converts it back into an `Action`, and hands it to
the caller. The runtime then validates it exactly as it validates the heuristic's
action, so the guard, the envelope and the supervisor apply unchanged -- an RL
policy is a different opinion about allocation, not a different path to the plant.

Two things this file refuses to do:

* **Pretend.** Without a checkpoint there is no RL controller. `RLControllerSlot`
  holds the *possibility* of one and reports it as unavailable with a reason; it
  never constructs an untrained policy to have something to select. A random
  network driving a plant while the control room says "RL Policy" is the single
  worst outcome available here.
* **Fail quietly.** If inference raises, the controller returns an idle action,
  records the failure, and the runtime reverts to the heuristic and says so. A
  controller that throws every 60 s is not a controller.

The `episode_progress` observation slot needs care in a live plant: training
episodes are finite and the plant is not. The controller therefore reports its
own step count modulo the training horizon, so the value stays in the range the
policy saw rather than pinned at one end of it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plant_sim import Action, PlantConfig, PlantState

from ..rl import checkpoints
from ..rl.actions import ActionAdapter
from ..rl.episodes import EpisodeConfig
from ..rl.observation import ObservationEncoder
from ..rl.policies import LoadedPolicy, load_policy
from .base import Constraints, ControlDecision, Controller

#: What the control room shows for RL even when no checkpoint is loaded.
RL_CONTROLLER_ID = "rl"
RL_CONTROLLER_LABEL = "RL Policy"
RL_CONTROLLER_DESCRIPTION = (
    "Learned allocation policy loaded from a checkpoint. Available only once a "
    "compatible checkpoint has been loaded; its actions pass the same guard as "
    "every other controller's."
)


class RLController(Controller):
    """Runs a loaded policy as a plant controller. One inference per timestep."""

    id = RL_CONTROLLER_ID
    label = RL_CONTROLLER_LABEL
    description = RL_CONTROLLER_DESCRIPTION

    def __init__(self, loaded: LoadedPolicy, *, horizon_h: float | None = None) -> None:
        self.loaded = loaded
        self.horizon_h = horizon_h or _trained_horizon_h(loaded)
        #: Set when inference fails. The runtime reads it and falls back.
        self.failure: str | None = None
        self._steps = 0
        self._config: PlantConfig | None = None
        self._encoder: ObservationEncoder | None = None
        self._adapter: ActionAdapter | None = None

    def reset(self) -> None:
        self._steps = 0
        self.failure = None
        if self._encoder is not None:
            self._encoder.reset()

    def compute_action(
        self,
        observation: PlantState,
        *,
        constraints: Constraints,
        config: PlantConfig,
    ) -> ControlDecision:
        encoder, adapter = self._tools(config)
        try:
            vector = encoder.encode(
                observation, constraints=constraints, progress=self._progress(config)
            )
            decoded = adapter.decode(
                self.loaded.policy.act(vector),
                constraints=constraints,
                solar_available_kw=observation.solar_available_kw,
            )
        except Exception as reason:  # inference is third-party code: never trust it
            self.failure = f"{type(reason).__name__}: {reason}"
            return ControlDecision(
                Action.idle(), (f"RL inference failed ({self.failure}): plant idled",)
            )

        self._steps += 1
        notes = [f"RL policy ({self.loaded.metadata.algorithm})"]
        if decoded.clipped:
            notes.append("Policy output was outside its action range and was clipped")
        if not self.loaded.metadata.trained:
            notes.append("Mock checkpoint: this is not a trained policy")
        return ControlDecision(decoded.action, tuple(notes))

    # --- internals --------------------------------------------------------
    def _tools(self, config: PlantConfig) -> tuple[ObservationEncoder, ActionAdapter]:
        """Encoder/adapter bound to the plant currently being driven."""
        if self._encoder is None or self._adapter is None or self._config is not config:
            self._config = config
            self._encoder = ObservationEncoder(config)
            self._adapter = ActionAdapter(config)
        return self._encoder, self._adapter

    def _progress(self, config: PlantConfig) -> float:
        steps_per_horizon = max(1, int(round(self.horizon_h * 3600.0 / config.dt_s)))
        return (self._steps % steps_per_horizon) / steps_per_horizon

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "checkpoint": str(self.loaded.path),
            "metadata": self.loaded.metadata.to_dict(),
        }


def _trained_horizon_h(loaded: LoadedPolicy) -> float:
    """The episode horizon the policy was trained with, or the current default."""
    episode = loaded.metadata.training_config.get("episode")
    if isinstance(episode, dict):
        horizon = episode.get("horizon_h")
        if isinstance(horizon, (int, float)) and horizon > 0:
            return float(horizon)
    return EpisodeConfig().horizon_h


@dataclass(frozen=True)
class RLStatus:
    """Whether RL can drive the plant right now, and why not when it cannot."""

    #: `unavailable` (no checkpoint), `ready` (loaded), `failed` (load or inference).
    state: str
    reason: str = ""
    checkpoint: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "available": self.available,
            "reason": self.reason,
            "checkpoint": self.checkpoint,
            "metadata": self.metadata,
        }


class RLControllerSlot:
    """The place an RL checkpoint is loaded into, and the honest status of that.

    Empty by default, which is why the heuristic controller is the default: there
    is nothing else to select until a checkpoint is loaded, either from
    `PLANT_RL_CHECKPOINT` at start-up or through the API at runtime.
    """

    def __init__(self, *, checkpoint_env: str = checkpoints.CHECKPOINT_ENV) -> None:
        self.checkpoint_env = checkpoint_env
        self.controller: RLController | None = None
        self._status = RLStatus(
            state="unavailable",
            reason=f"No RL checkpoint loaded. Set {checkpoint_env} or POST /api/controller/rl.",
        )

    @property
    def status(self) -> RLStatus:
        return self._status

    @property
    def available(self) -> bool:
        return self.controller is not None and self._status.available

    def load(self, path: Path | str) -> RLController:
        """Load a checkpoint, or raise `CheckpointError` with the reason.

        The load is atomic: a bad checkpoint never disturbs a policy that is
        already loaded and possibly driving, because a failed *attempt* says
        nothing about the policy that loaded cleanly earlier -- the caller gets
        the reason from the exception. With an empty slot the reason is recorded
        on the status instead, so the control room can say why RL is not
        selectable rather than nothing appearing to happen.
        """
        try:
            loaded = load_policy(path)
        except checkpoints.CheckpointError as reason:
            if self.controller is None:
                self._status = RLStatus(state="failed", reason=str(reason), checkpoint=str(path))
            raise
        self.controller = RLController(loaded)
        self._status = RLStatus(
            state="ready",
            reason=(
                ""
                if loaded.metadata.trained
                else "Mock checkpoint: a constant action, not a trained policy."
            ),
            checkpoint=str(loaded.path),
            metadata=loaded.metadata.to_dict(),
        )
        return self.controller

    def unload(self) -> RLStatus:
        self.controller = None
        self._status = RLStatus(
            state="unavailable",
            reason=f"No RL checkpoint loaded. Set {self.checkpoint_env} or POST /api/controller/rl.",
        )
        return self._status

    def autoload(self) -> RLStatus:
        """Load whatever `PLANT_RL_CHECKPOINT` points at, if anything.

        Start-up must not fail because a checkpoint moved: a bad path is recorded
        as a failed status and the plant runs on the heuristic controller.
        """
        configured = os.environ.get(self.checkpoint_env, "").strip()
        if not configured:
            return self._status
        try:
            self.load(configured)
        except checkpoints.CheckpointError:
            pass  # the reason is already on the status
        return self._status

    def mark_failed(self, reason: str) -> RLStatus:
        """Record an inference failure and empty the slot."""
        checkpoint = self._status.checkpoint
        self.controller = None
        self._status = RLStatus(state="failed", reason=reason, checkpoint=checkpoint)
        return self._status

    def descriptor(self) -> dict[str, Any]:
        """The controller-list entry for RL, loaded or not."""
        return {
            "id": RL_CONTROLLER_ID,
            "label": RL_CONTROLLER_LABEL,
            "description": RL_CONTROLLER_DESCRIPTION,
            **self._status.to_dict(),
        }


__all__ = [
    "RL_CONTROLLER_DESCRIPTION",
    "RL_CONTROLLER_ID",
    "RL_CONTROLLER_LABEL",
    "RLController",
    "RLControllerSlot",
    "RLStatus",
]
