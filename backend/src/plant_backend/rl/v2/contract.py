"""Full checkpoint-compatibility identity for Environment V2.

The static digest in :mod:`spaces` identifies vector layout only.  A V2 policy
also depends on timing, dynamics, episode distribution, constraints, and reward
semantics.  This module makes that wider contract canonical and refuses legacy
or mismatched V2 checkpoint metadata instead of guessing compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from . import spaces
from .config import EpisodeConfigV2
from .reward import OBJECTIVE_VERSION, RewardWeightsV2

CHECKPOINT_CONTRACT_SCHEMA_VERSION = 1
FLOW_SAMPLING_SEMANTICS = "last one-minute PlantState interval at decision boundary"
TIME_SEMANTICS = (
    "observe boundary t; hold one decision over [t,t+decision_interval); "
    "settle a deadline after production in its final physical interval"
)


class CheckpointContractError(ValueError):
    """A checkpoint cannot be shown to implement this exact V2 task."""


def checkpoint_contract_payload(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    """Return every runtime setting that can change learned policy semantics."""
    episode.validate(plant)
    weights.validate()
    return {
        "schema_version": CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        "environment_id": spaces.ENVIRONMENT_ID,
        "interface_contract": spaces.contract_payload(),
        "interface_contract_digest": spaces.ENVIRONMENT_CONTRACT_DIGEST,
        "timing": {
            "physics_timestep_s": plant.dt_s,
            "decision_interval_s": episode.decision_interval_s,
            "time_semantics": TIME_SEMANTICS,
            "flow_sampling_semantics": FLOW_SAMPLING_SEMANTICS,
        },
        "episode_distribution": episode.to_dict(),
        "plant_config": plant.to_dict(),
        "constraints": constraints.to_dict(),
        "objective": {
            "version": OBJECTIVE_VERSION,
            "reward_weights": weights.to_dict(),
        },
    }


def contract_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def describe_checkpoint_contract(
    *,
    plant: PlantConfig,
    episode: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    payload = checkpoint_contract_payload(
        plant=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
    )
    return {
        "checkpoint_contract_digest": contract_digest(payload),
        **payload,
    }


def require_compatible_checkpoint(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    """Fail unless metadata declares the exact current full V2 contract."""
    environment_id = metadata.get("environment_id")
    if environment_id != expected.get("environment_id"):
        raise CheckpointContractError(
            f"checkpoint environment {environment_id!r} is incompatible with "
            f"{expected.get('environment_id')!r}"
        )
    declared = metadata.get("checkpoint_contract_digest")
    if not isinstance(declared, str) or not declared:
        raise CheckpointContractError(
            "V2 checkpoint metadata is missing checkpoint_contract_digest"
        )
    current = expected.get("checkpoint_contract_digest")
    if declared != current:
        raise CheckpointContractError(
            f"V2 checkpoint contract mismatch: checkpoint={declared}, current={current}"
        )


__all__ = [
    "CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "FLOW_SAMPLING_SEMANTICS",
    "TIME_SEMANTICS",
    "CheckpointContractError",
    "checkpoint_contract_payload",
    "contract_digest",
    "describe_checkpoint_contract",
    "require_compatible_checkpoint",
]
