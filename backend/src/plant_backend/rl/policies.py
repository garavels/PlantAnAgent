"""Loading a checkpoint into something callable, with no room for a fake.

    loaded = load_policy("checkpoints/ppo-demo")
    action = loaded.policy.act(observation)

`Policy` is one method wide on purpose: inference is all the running plant needs
from a learned controller, so the backend never imports a training library to
drive the plant, and a test can pass a two-line stub where a network would be.

The training library is imported *inside* the loader, not at module scope. That
keeps Stable-Baselines3 and Torch an optional install: the backend, the tests and
the simulator run without them, and asking for an `sb3-ppo` checkpoint without
them installed produces a message naming the extra to install rather than an
ImportError from three frames deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import checkpoints
from .checkpoints import CheckpointError, CheckpointMetadata


@runtime_checkable
class Policy(Protocol):
    """Observation in, action out. Deterministic: the plant is not exploring."""

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class LoadedPolicy:
    """A callable policy plus the metadata that vouched for it."""

    policy: Policy
    metadata: CheckpointMetadata
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "metadata": self.metadata.to_dict()}


class FixedActionPolicy:
    """Returns the same action every step. A stand-in, never a trained policy."""

    def __init__(self, action: NDArray[np.generic] | list[float]) -> None:
        self._action = cast(
            NDArray[np.float32],
            np.asarray(action, dtype=np.float32).reshape(-1),
        )

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]:
        del observation
        return cast(NDArray[np.float32], self._action.copy())


class SB3Policy:
    """Deterministic inference from a Stable-Baselines3 model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def act(self, observation: NDArray[np.generic]) -> NDArray[np.float32]:
        action, _ = self._model.predict(observation, deterministic=True)
        return cast(
            NDArray[np.float32],
            np.asarray(action, dtype=np.float32).reshape(-1),
        )


def load_policy(
    path: Path | str,
    *,
    contract: checkpoints.PolicyEnvironmentContract | None = None,
    environment: object | None = None,
) -> LoadedPolicy:
    """Load a checkpoint for inference, or raise `CheckpointError` saying why not.

    Supplying the admitted environment additionally checks the concrete Gym
    spaces embedded in a real model artifact.  V2 callers always provide it;
    legacy callers retain their historical metadata-only admission behavior.
    """
    directory = Path(path)
    if environment is not None and contract is None:
        raise ValueError("an admitted environment requires an explicit policy contract")
    metadata = checkpoints.load_metadata(directory, contract=contract)
    if metadata.format == checkpoints.FORMAT_FIXED:
        action = metadata.extra.get("action")
        if not isinstance(action, list) or len(action) != metadata.action_size:
            raise checkpoints.CorruptCheckpointError(
                f"{directory}: a fixed checkpoint needs an 'action' of "
                f"{metadata.action_size} numbers in its metadata"
            )
        return LoadedPolicy(FixedActionPolicy(action), metadata, directory)

    artefact = checkpoints.policy_path(directory)
    try:
        from stable_baselines3 import PPO
    except ImportError as reason:  # pragma: no cover - depends on the optional extra
        raise CheckpointError(
            f"{directory} is a {metadata.format} checkpoint, but stable-baselines3 is not "
            "installed: install the optional training dependencies "
            "(`uv sync --group train`) to load it"
        ) from reason
    try:
        model = PPO.load(artefact, device="cpu")
    except Exception as reason:  # pragma: no cover - depends on the optional extra
        raise checkpoints.CorruptCheckpointError(
            f"cannot load {artefact}: {reason}"
        ) from reason
    if environment is not None:
        observation_space = getattr(environment, "observation_space", None)
        action_space = getattr(environment, "action_space", None)
        if observation_space is None or action_space is None:
            raise ValueError(
                "the admitted environment must expose observation_space and action_space"
            )
        checkpoints.verify_policy_model_spaces(
            model,
            observation_space=observation_space,
            action_space=action_space,
            path=directory,
        )
    return LoadedPolicy(SB3Policy(model), metadata, directory)


__all__ = ["FixedActionPolicy", "LoadedPolicy", "Policy", "SB3Policy", "load_policy"]
